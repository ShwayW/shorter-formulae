# imports
import torch
import numpy as np
from grammar import (V, sample_simplified_pn, relabel_variables, set_operator_weights,
                     _DIST_MAX_OPS, _MAX_UNARY_OPS, _uniform_randPN, add_prefactors,
                     substitute_consts, tokenize_with_floats)
from simplify_backend import simplify as simplify_formula, set_backend as set_simplify_backend
from funcWrappers import wrapExtEvalPN
from scipy.stats import special_ortho_group
import random
from collections import deque


def generate_inputs(N, D, max_centroids=10):
    """
    Generate N raw (unwhitened) input points in D dimensions using a mixture of
    Gaussians or Uniforms, matching the NeurIPS 2022 paper (Kamienny et al.) exactly.

    Key design choices (from paper):
    - Distribution type (Gaussian or Uniform) is chosen once per call, applied
      to all centroids -- not chosen independently per centroid.
    - Up to max_centroids=10 mixture components.
    - Points per centroid allocated via multinomial (not floor-based).
    - Uniform uses scale uniform(-1,1)*sqrt(cov), matching Gaussian variance.
    - One independent SO(D) rotation per centroid; identity for D=1.
    - Whitening is done after formula evaluation, not here.
    """
    # Choose distribution type once for all centroids (50/50)
    is_gaussian = np.random.random() < 0.5

    # Sample number of centroids: randint(1, max_centroids) gives 1..max_centroids-1
    n_centroids = np.random.randint(1, max_centroids + 1)

    # Sample centroid parameters
    means       = np.random.randn(n_centroids, D)
    covariances = np.random.uniform(0, 1, size=(n_centroids, D))

    # One rotation matrix per centroid; identity for D=1 (paper convention)
    if D > 1:
        rotations = [special_ortho_group.rvs(D) for _ in range(n_centroids)]
    else:
        rotations = [np.identity(1) for _ in range(n_centroids)]

    # Allocate points to centroids via multinomial
    weights = np.random.uniform(0, 1, size=n_centroids)
    weights /= np.sum(weights)
    counts = np.random.multinomial(N, weights)

    # Sample points per centroid and rotate
    X_list = []
    for mean, cov, rotation, n in zip(means, covariances, rotations, counts):
        if n == 0:
            continue
        if is_gaussian:
            pts = np.random.multivariate_normal(mean, np.diag(cov), n) @ rotation
        else:
            pts = (mean + np.random.uniform(-1, 1, size=(n, D)) * np.sqrt(cov)) @ rotation
        X_list.append(pts)

    if X_list:
        X_raw = np.concatenate(X_list, axis=0)
    else:
        # All multinomial counts were zero (floating-point edge case in weights).
        # Fall back to a simple uniform draw so the call always succeeds.
        X_raw = np.random.uniform(-1.0, 1.0, size=(N, D))

    return X_raw


def _decaying_choice(lo, hi, decay):
    """Return an int in [lo, hi] with geometric weights favouring the low (simpler) end.

    decay in (0, 1]: smaller -> stronger preference for small values.  decay == 1.0
    gives a uniform draw (distributionally identical to np.random.randint(lo, hi + 1)).
    Used by the optional complexity prior so simpler formulae are sampled more often
    while larger ones remain reachable in the tail.
    """
    if hi <= lo:
        return lo
    ks = np.arange(lo, hi + 1)
    w = decay ** (ks - lo)
    return int(np.random.choice(ks, p=w / w.sum()))


def sample_formula(fml_len_range, complexity_decay=None, length_decay=None,
                   simplify_targets=True, use_prefactors=False, restrict_consts=True):
    """Draw ONE formula from the online generator's formula distribution.

    This is the formula-sampling half of online_batch_iterater (structure budget ->
    grammar draw -> degeneracy/length rejection), factored out so that offline
    analyses can sample from exactly the same distribution the trainer sees without
    also paying for IO-pair generation.  The knob semantics are documented on
    online_batch_iterater; they are passed straight through.

    The caller is responsible for grammar.set_operator_weights(...) and
    simplify_backend.set_backend(...) (both per-process settings) and for seeding
    the RNG.

    Returns (fml_pn, input_dim) for an accepted draw, or None when the draw was
    REJECTED -- an over-long skeleton, or a formula that carries no real variable
    dependence.  A rejection consumes RNG exactly as the inline version did, so the
    caller simply retries; both are single draws, not loops.
    """
    var_dim = len(V)
    min_fml_len_sym, max_fml_len_sym = fml_len_range
    # Sample input dimension first (mirrors symbolicregression), ensuring
    # the formula will use all `input_dim` variables (full coverage guarantee).
    # A formula with input_dim variables needs at least input_dim-1 binary ops,
    # i.e. sizeLimit >= 2*input_dim - 1 structural tokens.
    max_vars_fitting = (max_fml_len_sym + 1) // 2  # largest dim that fits in cap
    max_binary_budget = (max_fml_len_sym - 1) // 2
    if length_decay is not None:
        # Target-LENGTH-first sampling.  Draw the symbolic length directly with a
        # SINGLE mild geometric decay over [min,max], then back out the operator
        # budget.  This fixes the pathology of complexity_decay, which applies its
        # decay to THREE separate draws (input_dim, nb_binary, nb_unary) that
        # compound -- collapsing the tail so that at decay 0.7 the median length is
        # ~9 and length>40 is ~0.1%, even with a cap of 80.  Here the length
        # distribution is an explicit knob (length_decay ~0.97 keeps simple formulae
        # dominant while ~20-25% of mass lands past 40) and length is DECOUPLED from
        # #variables: a long tree is filled by reusing few variables, as real
        # scientific formulae do.
        target_len    = _decaying_choice(max(1, min_fml_len_sym), max_fml_len_sym, length_decay)
        nb_unary_ops  = _decaying_choice(0, min(_MAX_UNARY_OPS, max(0, (target_len - 1) // 2)),
                                         length_decay)

        # The uniform binary-tree generator caps nb_binary at grammar._DIST_MAX_OPS
        # (=40, the precomputed Catalan table) and nb_unary at grammar._MAX_UNARY_OPS
        # (=12), so the effective length ceiling is ~2*40+1+12 ~= 93 tokens.  That now
        # sits ABOVE the fml_len cap of 80, so the cap is what binds (it did not before,
        # when the caps were 30/8 and the ceiling was ~69).
        nb_binary_ops = min(max(0, (target_len - 1 - nb_unary_ops) // 2), _DIST_MAX_OPS)

        # Distinct variables: reuse-friendly -- favour few, capped by leaf count so
        # the coverage guarantee (input_dim <= n_leaves) still holds.
        _var_decay    = complexity_decay if complexity_decay is not None else 0.7
        input_dim     = _decaying_choice(1, min(var_dim, nb_binary_ops + 1), _var_decay)
    elif complexity_decay is None:
        # Uniform sampling (default / original behaviour): all sizes equally likely.
        input_dim     = np.random.randint(1, min(var_dim, max_vars_fitting) + 1)
        nb_binary_ops = np.random.randint(
            max(0, input_dim - 1),
            min(input_dim + 12, max_binary_budget) + 1
        )
        remaining_budget = max_fml_len_sym - (2 * nb_binary_ops + 1)
        nb_unary_ops = np.random.randint(0, min(_MAX_UNARY_OPS, max(0, remaining_budget)) + 1)
    else:
        # Complexity prior: favour fewer variables and fewer operators (simpler
        # formulae) while keeping larger ones reachable in the tail.
        input_dim     = _decaying_choice(1, min(var_dim, max_vars_fitting), complexity_decay)
        nb_binary_ops = _decaying_choice(
            max(0, input_dim - 1),
            min(input_dim + 12, max_binary_budget),
            complexity_decay,
        )
        remaining_budget = max_fml_len_sym - (2 * nb_binary_ops + 1)
        nb_unary_ops = _decaying_choice(0, min(_MAX_UNARY_OPS, max(0, remaining_budget)),
                                        complexity_decay)

    # Now generate a PN
    if use_prefactors:
        # E2E-style order: RAW skeleton -> relabel -> add_prefactors -> simplify.
        # Prefactors must be injected before simplification, since they are what the
        # normalizer then folds into canonical coefficient positions; injecting them
        # after would leave the target in a non-canonical form.
        raw = _uniform_randPN(nb_binary_ops, nb_unary_ops, input_dim,
                              restrict_consts=restrict_consts)
        raw, _ = relabel_variables(raw)
        # fml_len_range caps the SKELETON here (see the docstring note): the
        # post-prefactor length is bounded by target_num_cols instead, so the
        # structural distribution matches the no-prefactor arm exactly.
        if len(raw.split()) > max_fml_len_sym:
            return None
        fml = add_prefactors(raw)
        if simplify_targets:
            fml = simplify_formula(fml)
    else:
        fml = sample_simplified_pn(nb_binary_ops, nb_unary_ops, input_dim,
                                   simplify=simplify_targets,
                                   restrict_consts=restrict_consts)
    # Reject degenerate draws that carry no real variable dependence.  With
    # simplify_targets=True the returned formula is already canonical, so a constant
    # surfaces directly as input_dim==0 in the relabel_variables call below -- no
    # extra work needed.  With simplify_targets=False we train on the RAW draw, which
    # can be constant-valued while still containing variable tokens ('- v1 v1' == 0);
    # only simplify_formula() exposes that, so it runs here purely as the degeneracy
    # filter (its simplified result is intentionally discarded -- the raw fml is kept).
    if not simplify_targets and relabel_variables(simplify_formula(fml))[1] == 0:
        return None
    fml, input_dim = relabel_variables(fml)
    if input_dim == 0:
        return None  # collapsed to a constant (simplify_targets=True) or has no vars
    if not use_prefactors and len(fml.split()) > max_fml_len_sym:
        return None  # simplified symbolic length exceeds cap
                     # (under use_prefactors the cap was already applied to the skeleton)
    return fml, input_dim


def online_batch_iterater(tok2ind, fml_len_range, num_io_pairs_range,
                           batch_size=256, steps_per_epoch=1000, noise_gamma=0.0,
                           noise_clean_frac=0.0,
                           target_tokens_per_batch=10_000, bpe_model=None,
                           complexity_decay=None, operator_weights="uniform",
                           simplify_targets=True, length_decay=None,
                           use_prefactors=False, restrict_consts=True,
                           simplify_backend="ours"):
    """Generate formulae randomly from the grammar and yield (ios_tensor, targets_tensor).

    Formulae are sampled fresh each call via grammar.sample_simplified_pn (uniform tree
    + composite constants + final simplify) -- the formula space is unbounded and no
    pre-built corpus is required.  By default the grammar is mantissa-free: constants are
    restricted to the symbolic set C (0, 1, 2, 3, pi), so each symbolic token maps to a
    single target token (no float expansion).  use_prefactors / restrict_consts (below)
    turn that off and restore the end-to-end float-constant generator.

    simplify_targets=False trains on the RAW uniform draw rather than its canonical form
    (composite constants stay canonical -- only the structure is raw).  Degenerate draws
    are rejected either way: simplify() still runs as a filter, so the two settings differ
    ONLY in the target form and generation cost is unchanged.  Raw targets are ~1.2 tokens
    longer on average, so more of them hit the fml_len_range cap.

    tok2ind               : vocab token -> index dict.
    fml_len_range         : (min_symbolic_tokens, max_symbolic_tokens) formula size limit.
    num_io_pairs_range    : (min_ios, max_ios).
    batch_size            : hardware microbatch cap.  In token-budget mode it is the
                            upper bound on samples-per-forward (effective batch =
                            batch_size * accumulate_gradients in the trainer).  In the
                            legacy path (target_tokens_per_batch=0) it is the exact
                            fixed batch size.  Set to 0 to disable the cap.
    steps_per_epoch       : number of batches to yield before the iterator stops.
    noise_gamma           : if > 0, add output noise scaled by gamma ~ U[0, noise_gamma].
    noise_clean_frac      : fraction of formulae left EXACTLY noiseless when noise_gamma
                            > 0 (default 0.0 = every formula gets some noise).  U[0, g]
                            alone never yields gamma == 0 -- it is a continuous draw, so
                            P(gamma == 0) = 0 and the model never sees a clean target
                            again.  This reserves a real clean share to train against,
                            mixing the two conditions rather than replacing one with the
                            other.  0.5 = half the formulae clean, half noised.
    target_tokens_per_batch  : target IO-point count per batch (paper: 10,000).
                            Samples are sorted by (N, fml_len) and grouped so that
                            batch_size = max(1, target_tokens_per_batch // N_max_in_batch),
                            eliminating wasteful IO-row and target-token padding.
                            Set to 0 to use the legacy fixed batch_size path.
    bpe_model             : optional SubtreeBPE model for subformula compression.
    complexity_decay      : if None (default), formula size is sampled UNIFORMLY
                            (original behaviour).  If a float in (0, 1], the input
                            dimension and the binary/unary operator counts are each
                            drawn from a geometric distribution favouring smaller
                            (simpler) formulae, with larger ones kept in the tail.
                            1.0 is equivalent to None (uniform).
    operator_weights      : "uniform" (default) draws every grammar operator with equal
                            probability (original behaviour); "tiered" uses a domain-
                            general prior favouring common operators (see grammar.py).
                            Applied per process here so it survives DataLoader workers.
    simplify_targets      : True (default) trains on the canonical simplest form
                            (original behaviour).  False trains on the raw draw; see
                            the note above.
    length_decay          : if None (default), size sampling follows complexity_decay
                            above.  If a float in (0, 1], OVERRIDES it with target-
                            length-first sampling: the symbolic length is drawn directly
                            from a single geometric decay over fml_len_range and the
                            operator budget is derived from it, decoupling length from
                            #variables.  ~0.97 keeps simple formulae dominant while giving
                            a genuine tail past 40; 1.0 draws length uniformly.
    use_prefactors        : False (default) generates bare skeletons -- the mantissa-free
                            behaviour, where any numeric structure comes from the grammar's
                            own constants.  True runs grammar.add_prefactors over each raw
                            skeleton before simplification, injecting a fittable float in
                            front of every additive term and unary argument exactly as
                            Kamienny et al. (2022) do.  Prefactors are floats by
                            construction, so this implies the float vocabulary regardless
                            of restrict_consts.
    simplify_backend      : which canonicaliser runs on every draw -- "ours" (default,
                            simplifyFormula.simplify) or "simplipy" (the rule-mined
                            engine of Saegert & Koethe, reached by a round trip through
                            its vocabulary).  Both are process-level settings, set here
                            so they survive into each DataLoader worker.  The two agree
                            closely on the operator distribution but disagree on the
                            canonical form of ~68% of individual formulae, and simplipy
                            is ~2.4x slower; see simplify_backend.py.
    restrict_consts       : True (default) restricts constant LEAVES to the symbolic set C
                            (0, 1, 2, 3, pi), assembling other values as composite
                            subexpressions ('/ 1 2', '* 2 2').  False uses the full E2E-style
                            constant set: a constant leaf is drawn from C_ALL = C + [CONST]
                            and a pow exponent from _POW_EXP_CHOICES_FULL, so leaves may be
                            arbitrary random floats.  Composite constants are skipped in
                            that mode (see grammar._fill_leaves).

    Whenever floats are reachable (use_prefactors=True or restrict_consts=False) each
    CONST placeholder is replaced by an independently sampled float via substitute_consts
    BEFORE evaluation, and the concrete formula is tokenized with tokenize_with_floats
    into 4 tokens per float -- so tok2ind must come from grammar.FlatGramFloat and the
    target cap grows accordingly.  The two settings are independent knobs: use_prefactors
    alone reproduces the E2E-style generator, restrict_consts alone controls only which
    constants the leaves may take.

    NOTE on fml_len_range under use_prefactors: the cap is applied to the SKELETON, before
    prefactor injection, not to the prefactored result.  add_prefactors roughly doubles the
    token count, so capping the output would leave the prefactor arm training on
    structurally simpler formulae than the no-prefactor arm at the same fml_len_range
    (measured: mean skeleton length 18.5 vs 33.7 at cap 80, plus a 44% rejection rate) --
    which would confound any prefactors-only ablation.  Capping the skeleton keeps the
    structural distribution identical across arms; the expanded target is bounded by
    target_num_cols instead.
    """
    set_operator_weights(operator_weights)
    set_simplify_backend(simplify_backend)
    var_dim = len(V)
    patch_dim = var_dim + 1
    pad_index = tok2ind["<pad>"]
    bes_index = tok2ind["<bes>"]
    min_num_ios, max_num_ios = num_io_pairs_range
    min_fml_len_sym, max_fml_len_sym = fml_len_range
    # Float literals are reachable through prefactors or through unrestricted constant
    # leaves; either way each float expands to 4 target tokens (encode_float).
    needs_floats = use_prefactors or not restrict_consts
    # Target cap, in tokens, plus the two <bes> delimiters:
    #   no floats      -> 1 token per symbolic token.
    #   floats only    -> at most 4 tokens per symbolic token.
    #   prefactors     -> the cap bounds the SKELETON, and add_prefactors then inflates it
    #                     before float expansion.  6x is empirical headroom: with
    #                     max_fml_len_sym=80 the worst case measured over 6k adversarial
    #                     draws (largest skeletons, both operator-weight modes) is 292
    #                     tokens = 3.7x, measured back when grammar._DIST_MAX_OPS=30 bounded
    #                     a skeleton at ~69 tokens; at 40/12 the skeleton ceiling is ~93, so
    #                     max_fml_len_sym now binds instead.  Anything that still
    #                     overflows is dropped by the target_num_cols guard below.
    target_num_cols = max_fml_len_sym * (6 if use_prefactors else 4 if needs_floats else 1) + 2

    use_token_budget = target_tokens_per_batch > 0

    # Sort buffer: accumulate this many samples before sorting by (N, fml_len) and
    # draining into batches.  Large enough to give meaningful grouping; ~5* the largest
    # expected batch.
    _SORT_BUFFER = max(500, (target_tokens_per_batch // max(1, min_num_ios)) * 5) if use_token_budget else 0

    buffer = []               # token-budget path: [(ios_2d (N,D+1), enc_raw list, N, fml_len)]
    ios_buff, targets_buff = deque(), deque()  # legacy fixed-batch path
    batches_yielded = 0

    while batches_yielded < steps_per_epoch:
        # Structure budget + grammar draw + degeneracy/length rejection.  Factored
        # into sample_formula() so offline analyses (proximity_vs_ood.py) can draw
        # from the identical distribution; a rejected draw is retried here exactly
        # as the inline version did.
        drawn = sample_formula(fml_len_range,
                               complexity_decay=complexity_decay,
                               length_decay=length_decay,
                               simplify_targets=simplify_targets,
                               use_prefactors=use_prefactors,
                               restrict_consts=restrict_consts)
        if drawn is None:
            continue
        fml, input_dim = drawn
        fml_sym_toks = fml.split()

        num_ios = np.random.randint(min_num_ios, max_num_ios + 1)

        # Gather valid IO pairs, retrying inputs to work around domain failures
        # (e.g. exp overflow, log of a negative).  When floats are reachable the formula
        # still holds CONST placeholders, so the outer loop also retries with fresh
        # constant draws -- one lucky assignment is enough, and the inner loop handles the
        # domain failures that are input-dependent rather than constant-dependent.
        # Without floats the formula is already concrete and one "candidate" reproduces the
        # single-pass behaviour exactly.
        _all_raw = []
        _all_out = []
        fml_concrete = None
        for _const_try in range(5 if needs_floats else 1):
            _cand = substitute_consts(fml) if needs_floats else fml
            _all_raw_try = []
            _all_out_try = []
            for _ in range(10):
                _n = num_ios - sum(len(r) for r in _all_raw_try)
                if _n <= 0:
                    break
                _batch_raw = generate_inputs(_n, input_dim)
                _batch_eval = np.concatenate([_batch_raw, np.full((_n, var_dim - input_dim), np.nan)], axis=1) if input_dim < var_dim else _batch_raw
                _batch_out, _ = wrapExtEvalPN(_cand, _batch_eval)
                _mask = np.isfinite(_batch_out) & (np.abs(_batch_out) < 1e9)
                if _mask.any():
                    _all_raw_try.append(_batch_raw[_mask])
                    _all_out_try.append(_batch_out[_mask])
            if sum(len(r) for r in _all_raw_try) >= num_ios:
                fml_concrete = _cand
                _all_raw = _all_raw_try
                _all_out = _all_out_try
                break

        if fml_concrete is None:
            continue

        # Flatten the per-retry chunks into one array and trim to the requested count.
        # Each retry above contributed only the rows that survived the finite/|y|<1e9
        # mask, so the chunks have ragged first dimensions but sum to >= num_ios.
        #   before: _all_raw = [(n_1, input_dim), ..., (n_k, input_dim)],  sum n_i >= num_ios
        #           _all_out = [(n_1,),           ..., (n_k,)]
        #   after:  inputs_raw  (num_ios, input_dim)  -- ACTIVE variable columns only; the
        #                                                var_dim - input_dim NaN pad columns
        #                                                are appended below, after whitening
        #           outputs     (num_ios,)
        inputs_raw = np.concatenate(_all_raw, axis=0)[:num_ios]
        outputs    = np.concatenate(_all_out, axis=0)[:num_ios]
        num_ios_valid = len(outputs)

        # Whiten inputs AFTER capturing raw outputs; the model sees (x~, f(x_raw)).
        inputs_whitened = (inputs_raw - np.mean(inputs_raw, axis=0)) / (np.std(inputs_raw, axis=0) + 1e-8)
        if input_dim < var_dim:
            # (num_ios, input_dim) + (num_ios, var_dim - input_dim) -> (num_ios, var_dim)
            pad_nan = np.full((num_ios_valid, var_dim - input_dim), np.nan)
            inputs = np.concatenate([inputs_whitened, pad_nan], axis=1)
        else:
            inputs = inputs_whitened          # (num_ios, var_dim)

        # add output noise: y += gamma * (||y|| / sqrt(n)) * N(0,1)
        # A noise_clean_frac share of formulae is held EXACTLY clean (gamma = 0): the
        # U[0, g] draw is continuous, so on its own it never produces a truly noiseless
        # target and the model would only ever see corrupted y again.  Mixing the two
        # keeps the clean objective in the loss the whole way through.
        if noise_gamma > 0.0:
            if noise_clean_frac > 0.0 and np.random.random() < noise_clean_frac:
                gamma = 0.0
            else:
                gamma = np.random.uniform(0.0, noise_gamma)
            if gamma > 0.0:
                norm  = np.linalg.norm(outputs) / np.sqrt(len(outputs))
                outputs = outputs + gamma * norm * np.random.randn(*outputs.shape)

        # Tokenize the target.  Without floats every token is a primitive grammar token
        # (or a BPE compound) and no expansion occurs.  With floats the target is the
        # CONCRETE formula -- the same one the IO pairs were evaluated from -- and each
        # float literal expands into 4 tokens, which can push it past the cap.
        if needs_floats:
            if use_prefactors:
                # A BPE model is learned on the mantissa-free sampler; its compounds do
                # not exist in the prefactored distribution, so it is bypassed here
                # (mirrors the E2E generator).
                fml_toks = tokenize_with_floats(fml_concrete)
            elif bpe_model is not None:
                fml_toks = tokenize_with_floats(" ".join(bpe_model.tokenize(fml_concrete)))
            else:
                fml_toks = tokenize_with_floats(fml_concrete)
        elif bpe_model is not None:
            fml_toks = bpe_model.tokenize(fml)
        else:
            fml_toks = fml_sym_toks
        try:
            encoded = [bes_index] + [tok2ind[tok] for tok in fml_toks] + [bes_index]
        except KeyError:
            continue
        if len(encoded) > target_num_cols:
            continue  # sequence exceeds cap; skip
        fml_len = len(encoded)  # actual token count before global padding

        if use_token_budget:
            # Defer target padding to batch-formation time; pad only to the batch-local
            # maximum formula length (dual-axis N*L bucketing).
            ios_2d = np.concatenate((inputs, outputs.reshape(-1, 1)), axis=1).astype("float32")
            buffer.append((ios_2d, list(encoded), num_ios_valid, fml_len))

            # Compare buffer size against an item-count estimate of remaining
            # work, not the raw batch count.  steps_per_epoch is batches; each
            # batch consumes ~(target_tokens_per_batch // max_N) items, so
            # multiplying converts batches -> items and prevents the buffer from
            # being capped at steps_per_epoch (< _SORT_BUFFER) in multi-worker
            # mode where each worker only sees a fraction of the full epoch.
            est_items_remaining = (steps_per_epoch - batches_yielded) * max(1, target_tokens_per_batch // max(1, max_num_ios))
            if len(buffer) < _SORT_BUFFER and len(buffer) < est_items_remaining:
                continue

            # Sort by (N, fml_len): primary key minimises IO-row padding; secondary key
            # minimises target-token padding within each N-group.
            buffer.sort(key=lambda x: (x[2], x[3]))
            i = 0

            while i < len(buffer) and batches_yielded < steps_per_epoch:
                N_est = buffer[i][2]
                k = max(1, target_tokens_per_batch // N_est)
                if batch_size > 0:
                    k = min(k, batch_size)        # hardware microbatch cap
                k = min(k, len(buffer) - i)
                batch_items = buffer[i:i + k]
                N_batch = batch_items[-1][2]        # max N in batch (N-sorted ascending)
                L_batch = max(it[3] for it in batch_items)  # max fml_len in batch

                padded_ios = []
                padded_tgt = []
                for ios_2d_item, enc_raw, N, L in batch_items:
                    if N < N_batch:
                        ios_2d_item = np.concatenate([
                            ios_2d_item,
                            np.full((N_batch - N, patch_dim), np.nan, dtype="float32"),
                        ], axis=0)
                    padded_ios.append(ios_2d_item.flatten())
                    pad_len = L_batch - L
                    padded_tgt.append(np.asarray(
                        enc_raw + [pad_index] * pad_len if pad_len > 0 else enc_raw,
                        dtype="int64",
                    ))

                yield (
                    torch.from_numpy(np.stack(padded_ios)),
                    torch.from_numpy(np.stack(padded_tgt)),
                )
                batches_yielded += 1
                i += k
            del buffer[:i]   # carry the (< batch_size) remainder into the next sort cycle
        else:
            # Legacy: pad targets to global max and IO rows to max_num_ios.
            encoded += [pad_index] * (target_num_cols - fml_len)
            pad_size = max_num_ios - num_ios_valid
            if pad_size > 0:
                padded_inputs  = np.concatenate((inputs,  np.full((pad_size, var_dim), np.nan)), axis=0)
                padded_outputs = np.concatenate((outputs.reshape(-1, 1), np.full((pad_size, 1), np.nan)), axis=0)
            else:
                padded_inputs  = inputs[:max_num_ios]
                padded_outputs = outputs[:max_num_ios].reshape(-1, 1)
            ios_buff.append(np.concatenate((padded_inputs, padded_outputs), axis=1).flatten().astype("float32"))
            targets_buff.append(np.asarray(encoded, dtype="int64"))
            if len(ios_buff) >= batch_size:
                batch_ios = [ios_buff.popleft() for _ in range(batch_size)]
                batch_tgt = [targets_buff.popleft() for _ in range(batch_size)]
                yield (torch.from_numpy(np.asarray(batch_ios, dtype="float32")),
                       torch.from_numpy(np.asarray(batch_tgt, dtype="int64")))
                batches_yielded += 1


def _passthrough_collate(x):
    """Identity collate_fn -- batches are already formed by the generator."""
    return x


class OnlineBatchDataset(torch.utils.data.IterableDataset):
    """IterableDataset wrapper around online_batch_iterater.

    Enables DataLoader multi-worker prefetching: each worker runs an independent
    copy of the generator with a different RNG seed, yielding roughly
    steps_per_epoch // num_workers batches, so the total across all workers
    equals steps_per_epoch.
    """

    def __init__(self, tok2ind, fml_len_range, num_io_pairs_range,
                 batch_size=256, steps_per_epoch=1000, noise_gamma=0.0,
                 noise_clean_frac=0.0,
                 target_tokens_per_batch=10_000, seed=0, bpe_model=None,
                 complexity_decay=None, operator_weights="uniform",
                 simplify_targets=True, length_decay=None,
                 use_prefactors=False, restrict_consts=True,
                 simplify_backend="ours"):
        super().__init__()
        self.tok2ind = tok2ind
        self.fml_len_range = fml_len_range
        self.num_io_pairs_range = num_io_pairs_range
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.noise_gamma = noise_gamma
        self.noise_clean_frac = noise_clean_frac
        self.target_tokens_per_batch = target_tokens_per_batch
        self.seed = seed
        self.bpe_model = bpe_model
        self.complexity_decay = complexity_decay
        self.operator_weights = operator_weights
        self.simplify_targets = simplify_targets
        self.length_decay = length_decay
        self.use_prefactors = use_prefactors
        self.restrict_consts = restrict_consts
        self.simplify_backend = simplify_backend
        # Counts how many times __iter__ has been entered on this worker's copy of
        # the dataset.  It is the epoch component of the per-worker seed (see
        # __iter__), which is what keeps each epoch's formulae distinct under
        # persistent_workers=True.  (With persistent_workers=False the workers are
        # respawned and this counter resets to 0 every epoch; there the epoch
        # component comes from torch.initial_seed() instead.)
        self._iter_count = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            steps = self.steps_per_epoch
            base = 0
        else:
            base_steps = self.steps_per_epoch // worker_info.num_workers
            # give leftover steps to worker 0
            remainder = self.steps_per_epoch - base_steps * worker_info.num_workers
            steps = base_steps + (remainder if worker_info.id == 0 else 0)
            # torch.initial_seed() is DataLoader's per-worker seed (base_seed +
            # worker_id).  It is redrawn each epoch when workers are respawned, but
            # with persistent_workers=True it stays FIXED across epochs, so
            # _iter_count supplies the epoch component there.
            base = int(torch.initial_seed()) % (2 ** 32)
        # Mix the three streams through SeedSequence instead of ADDING them.
        # Adding aliases: worker w at epoch e gets base+w+seed+e, which equals
        # worker w+1 at epoch e-1 whenever base is fixed (persistent_workers=True) --
        # replaying (num_workers-1)/num_workers of the previous epoch's formulae.
        worker_seed = int(np.random.SeedSequence(
            [base, self.seed, self._iter_count]).generate_state(1, dtype=np.uint32)[0])
        self._iter_count += 1

        np.random.seed(worker_seed)
        random.seed(worker_seed)

        yield from online_batch_iterater(
            self.tok2ind,
            self.fml_len_range,
            self.num_io_pairs_range,
            batch_size=self.batch_size,
            steps_per_epoch=steps,
            noise_gamma=self.noise_gamma,
            noise_clean_frac=self.noise_clean_frac,
            target_tokens_per_batch=self.target_tokens_per_batch,
            bpe_model=self.bpe_model,
            complexity_decay=self.complexity_decay,
            operator_weights=self.operator_weights,
            simplify_targets=self.simplify_targets,
            length_decay=self.length_decay,
            use_prefactors=self.use_prefactors,
            restrict_consts=self.restrict_consts,
            simplify_backend=self.simplify_backend,
        )


if (__name__ == "__main__"):
    '''
    # Run the verification test natively
    max_num_ios = 24
    num_ios = 16
    var_dim = 9

    pad_size = max_num_ios - num_ios
    inputs_raw = generate_inputs(num_ios, var_dim)
    print(f"input dim: {inputs_raw.shape}")

    [outputs, _] = wrapExtEvalPN("0", inputs_raw)
    inputs = (inputs_raw - np.mean(inputs_raw, axis=0)) / (np.std(inputs_raw, axis=0) + 1e-8)

    inputs_pad = np.full((pad_size, var_dim), np.nan)
    print(f"inputs_pad dim: {inputs_pad.shape}")

    padded_inputs = np.concatenate((inputs, inputs_pad), axis = 0)
    print(f"padded_inputs dim: {padded_inputs.shape}")
    print(padded_inputs)
                    
    outputs_pad = np.full((pad_size, 1), np.nan)
    print(f"outputs_pad dim: {outputs_pad.shape}")
    padded_outputs = np.concatenate((outputs.reshape(-1, 1), outputs_pad), axis = 0)
    print(f"padded_outputs dim: {padded_outputs.shape}")
    print(padded_outputs)

    cur_ios_flat = torch.asarray(np.concatenate((padded_inputs, padded_outputs), axis=1).flatten())
    print(cur_ios_flat.shape)
    print(cur_ios_flat)

    ios = cur_ios_flat.view(-1, 10)
    print(ios)
    print(ios.shape)

    ios_padding_mask = torch.isnan(ios).any(dim = -1)
    print(ios_padding_mask)
    print(ios_padding_mask.shape)

    valid_mask = (~ios_padding_mask).unsqueeze(-1).float()
    print(valid_mask)

    ios = torch.nan_to_num(ios, nan = 0.0)
    print(ios)
    '''

    ans = _decaying_choice(1, 7, 1.0)
    print(ans)





