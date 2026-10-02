#!/usr/bin/env python3
"""eval_phyedc.py -- OUR transformer decoded with PHYE2E'S divide-and-conquer Oracle.

This is the cross arm of the D&C comparison.  Holding the backbone fixed and swapping
only the decomposition is what makes a difference attributable to the ALGORITHM:

    eval_devncon.py  =  our checkpoints + OUR decomposition   (results/mymodels_devncon)
    eval_phyedc.py   =  our checkpoints + THEIR decomposition (results/mymodels_phyedc)
    eval_phye2e.py --use-divide = their checkpoint + their decomposition

WHY THIS ARM IS WORTH RUNNING
-----------------------------
devncon.py is a faithful but RESTRICTED reimplementation of the published strategy: it
implements sigma in {id, log} where the paper sweeps ten operators, and it partitions the
variables where the paper's Definition 2 allows a cover (overlapping groups, which is what
the inclusion-exclusion in Theorem 3 exists to unwind).  Driving their Oracle directly
gets the full sigma set, the overlapping division and the real `reverse()` for free, so
the gap between this curve and the devncon curve measures exactly what our restriction
costs.

HOW THE SWAP IS POSSIBLE
------------------------
Their Oracle is two-phase with plain numpy in between, and the middle phase is the only
part that needs a symbolic model:

    res_x, res_y, res_hints = oracle.oracle_fit([X], [y], [0], hints)   # no model
        <solve each (res_x[i], res_y[i])>                               # <- we substitute
    res_exprs = oracle.reverse(original_gens, oracle_gens)              # no model

`oracle_gens` is a FLAT list indexed by single-expression index, aligned exactly with the
order of `res_x` -- their own fit() relies on that alignment when it slices one batch into
`best_gens_noref[:len(x)]` (originals) and `best_gens_noref[len(x):]` (subproblems).  So
we solve the subproblems in the order given and hand back a list of the same length.

`reverse(..., eliminate=True)` returns TWO entries per problem -- (winner, best-oracle) --
and their fit() keeps `[0::2]`.  We do the same, then score on real held-out data.

Every predicted formula crosses the grammar boundary through phye2e_bridge, in BOTH
directions: `pn_to_node` on the way in (an untranslatable one becomes
`predicted_tree=None`, which their code already treats as a failed subproblem rather than
a crash) and `node_to_pn` on the way back, because every consumer downstream of this
script speaks our prefix notation.

FIVE THINGS THEIR CODE DOES THAT THIS SCRIPT HAS TO HANDLE
----------------------------------------------------------
* `y` MUST be (N, 1).  Their fit() reshapes it before calling oracle_fit; their RMSE loss
  compares a (B, 1) prediction against whatever it is given, so a 1-D y broadcasts to a
  (B, B) loss and trains the surrogate on nonsense without ever raising.
* `hints` MUST be a real structure, not None -- `sample_oracle_hints` dereferences
  `hints[0]` unconditionally.  We build the same all-dimensionless hint their fit()
  builds for `units=None` (see `_hints`).
* `use_seperate_type` DEFAULTS to `["id"]` inside oracle_fit, which is add/mul only --
  i.e. exactly devncon's restriction, which would make this arm pointless.  We pass their
  checkpoint's own `params.oracle_seperation_type` instead.
* The surrogate is CACHED BY FILENAME and reloaded whenever the path exists, regardless of
  `save_model`.  With their fixed default (`./Oracle_model/test/test_0.pth`) every problem
  after the first would silently reuse the first problem's surrogate, so each call gets a
  unique path.
* `train_oracle` hardcodes `tqdm_bar=True`, one bar per problem over N_reg_lr*epochs
  steps.  Silenced unless --verbose.

Usage:
    python eval_phyedc.py --model 145M_40_simp1 --dataset feynman_I_12_11 --verbose
    python eval_phyedc.py --model 145M_40_simp1 --n_seeds 10 --target-noise 0 \
        --results-dir results/mymodels_phyedc/seed42
    python eval_phyedc.py --benchmark llmsrbench --model 145M_40_simp1 --split lsr_transform
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import phye2e_bridge                                                          # noqa: E402


# --------------------------------------------------------------------------------------
# their machinery
# --------------------------------------------------------------------------------------
class _NoBar:
    """Stand-in for tqdm.tqdm -- their train_oracle asks for a bar unconditionally."""

    def __init__(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    def close(self, *a, **k):
        pass


def build_oracle(weights_path: str, device, phyreg_device: str = "cpu",
                 quiet: bool = True):
    """(oracle, params) -- their Oracle plus the params object its Nodes need.

    Built through PhyReg exactly as eval_phye2e.py does: the Oracle constructor wants
    (env, env.generator, params), and those come from their pretrained checkpoint.  We
    never call their MODEL here -- only the Oracle -- but PhyReg is what assembles the
    env/generator/params triple, so it is the cheapest correct way in.

    Their transformer is therefore loaded onto `phyreg_device` (cpu by default) so its
    ~370 MB does not sit on the GPU beside OUR model for the whole sweep, while
    `params.device` -- which is what the surrogate net and the Hessians actually run on --
    is set to our device afterwards.
    """
    import eval_phye2e
    eval_phye2e._setup_imports()
    from PhysicsRegression import PhyReg
    import Oracle.oracle as oracle_mod
    from Oracle.oracle import Oracle

    if quiet:
        # oracle.py does `import tqdm` then `tqdm.tqdm(...)`, so rebinding the name in
        # ITS namespace silences it without touching the real tqdm module.
        oracle_mod.tqdm = types.SimpleNamespace(tqdm=_NoBar)

    phyreg = PhyReg(path=weights_path, device=phyreg_device)
    env = phyreg.env
    params = phyreg.params
    params.device = str(device)
    return Oracle(env, env.generator, params), params


def _hints(params, n_vars: int):
    """The hint structure oracle_fit expects, with every variable DIMENSIONLESS.

    Mirrors what their fit() builds for `units=complexitys=unarys=consts=None`, keyed off
    `params.use_hints` so it stays correct if a checkpoint declares a different set.
    `sample_oracle_hints` treats entry 0 as units and the last entry as consts, which is
    the order `use_hints` already has.

    Units are all-zero on purpose: that is the `--units none` convention every other arm
    in this comparison runs under.  Note it is not neutral here either -- their
    arcsin/arccos branches are gated on `np.all(hints[0][i][-1] == 0)`, so an
    all-dimensionless hint ENABLES those separations rather than suppressing them.
    """
    out = []
    for h in params.use_hints.split(","):
        if h == "units":
            out.append([[np.zeros(5) for _ in range(n_vars + 1)]])
        elif h == "complexity":
            out.append([[0]])
        else:                                   # unarys, consts, and anything else
            out.append([[]])
    return out


def _r2(y_true, y_pred) -> float:
    """Plain R^2 on arrays -- their winners are Node trees, not PN strings."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_pred.shape != y_true.shape or not np.all(np.isfinite(y_pred)):
        return float("-inf")
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return float("-inf")
    return float(1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot)


# --------------------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------------------
def phyedc_solve(X, y, leaf_solver, r2_fn, oracle, params, holdout=None,
                 sep_types=None, oracle_epochs=400, oracle_lr=0.005, oracle_bs=None,
                 oracle_file=None, verbose=False):
    """Decompose (X, y) with THEIR oracle, solve each piece with OUR model, recombine.

    Returns (pn_string, r2) for the best surviving candidate, scored on `holdout` when
    given -- their own ordering uses the training data, and the baseline has to be
    re-scored on the same rows or the two are not comparable.
    """
    def log(msg):
        if verbose:
            print(f"    [phyedc] {msg}", flush=True)

    X = np.asarray(X, dtype=np.float64)
    y1 = np.asarray(y, dtype=np.float64).reshape(-1)
    if oracle_bs is None:
        oracle_bs = max(1, len(X) // 3)

    # -- the full-problem baseline, first: it is also the floor `reverse()` competes with.
    try:
        base_pn, _ = leaf_solver(X, y1, baseline=True)
    except Exception as exc:
        log(f"baseline failed ({type(exc).__name__}: {exc})")
        base_pn = ""
    Xh, yh = holdout if holdout is not None else (X, y1)
    best_pn = base_pn
    best_r2 = r2_fn(base_pn, Xh, yh) if base_pn else float("-inf")
    log(f"baseline R^2 = {best_r2:.4f}")

    # -- phase 1: their surrogate + separability + pseudo-data (no symbolic model) --
    t0 = time.perf_counter()
    res_x, res_y, _ = oracle.oracle_fit(
        X, y1.reshape(-1, 1), 0, hints=_hints(params, X.shape[1]),
        use_seperate_type=sep_types,
        lr=oracle_lr, batch_size=oracle_bs, epochs=oracle_epochs,
        oracle_file=oracle_file, save_model=False, verbose=False,
    )
    log(f"oracle_fit -> {len(res_x)} subproblem(s) in {time.perf_counter()-t0:.1f}s")

    # -- phase 2: OUR transformer on every subproblem, in the order they were emitted --
    oracle_gens = []
    for i, (rx, ry) in enumerate(zip(res_x, res_y)):
        try:
            pn, _ = leaf_solver(np.asarray(rx, dtype=float),
                                np.asarray(ry, dtype=float).reshape(-1))
        except Exception as exc:                       # a dead subproblem must not kill the run
            log(f"  sub {i}: solver failed ({type(exc).__name__})")
            pn = ""
        gen = phye2e_bridge.as_oracle_gen(pn, params) if pn else {
            "predicted_tree": None, "relabed_predicted_tree": None, "message": "no formula"}
        if gen["predicted_tree"] is None and pn:
            log(f"  sub {i}: untranslatable -> {pn!r}")
        oracle_gens.append(gen)

    original_gens = [phye2e_bridge.as_oracle_gen(base_pn, params)] if base_pn else [
        {"predicted_tree": None, "relabed_predicted_tree": None, "message": "no baseline"}]

    # -- phase 3: their recombination + BFGS refinement --
    res_exprs = oracle.reverse(original_gens, oracle_gens, eliminate=True)
    winners = res_exprs[0::2]                          # same slice their fit() takes
    log(f"reverse -> {len(res_exprs)} expr(s), {len(winners)} winner(s)")

    for w in winners:
        tree = w.get("predicted_tree")
        if tree is None:
            continue
        try:
            pn = phye2e_bridge.node_to_pn(tree)
        except phye2e_bridge.BridgeError as exc:
            log(f"  winner untranslatable: {exc}")
            continue
        # Scored through OUR evaluator on the PN we will actually record, so the number
        # in the artifact is the number every downstream script will recompute.
        r2 = r2_fn(pn, Xh, yh)
        if np.isfinite(r2) and r2 > best_r2:
            best_r2, best_pn = r2, pn
    log(f"best R^2 = {best_r2:.4f}")
    return best_pn, (best_r2 if np.isfinite(best_r2) else 0.0)


def _solve(X_all, y_all, model, vocab, device, seed, args):
    """One (problem, seed) cell.  Signature matches eval_devncon._solve."""
    from eval_mymodels import compute_r2, evaluate_xy
    import eval_devncon

    rng = np.random.default_rng(seed)
    X_tr, y_tr, X_te, y_te, X_or, y_or = eval_devncon._prepare(
        X_all, y_all, rng, args.n_bags, args.sample_size, args.oracle_rows,
        args.target_noise)

    def leaf_solver(Xs, ys, baseline=False):
        # baseline=True -> evaluate_xy does its OWN 75/25 split and selects candidates
        # out-of-sample, exactly as a standalone beam run does.  Sub-problems fit on all
        # of their pseudo-data (no_split=True), which has no held-out concept.
        return evaluate_xy(Xs, ys, model, vocab, device, rng,
                           n_bags=args.n_bags, beam_size=args.beam_size,
                           no_split=not baseline, use_unscaling=args.unscale,
                           max_forward_batch=args.max_forward_batch)

    # Their oracle_fit's (x, y) does double duty: it trains the surrogate AND becomes the
    # `original_xs/original_ys` that safely_refine and order_candidates use.  Handing it
    # X_or keeps both on the train side of the outer split -- X_te never enters, so the
    # reported R^2 stays out-of-sample -- and matches the oracle budget devncon gets.
    t0 = time.perf_counter()
    formula, r2 = phyedc_solve(
        X_or, y_or, leaf_solver=leaf_solver, r2_fn=compute_r2,
        oracle=args._oracle, params=args._params,
        holdout=(X_te, y_te), sep_types=args.sep_types,
        oracle_epochs=args.oracle_epochs, oracle_lr=args.oracle_lr,
        oracle_bs=args.oracle_bs, oracle_file=args._oracle_file(),
        verbose=args.verbose)
    return formula, float(r2), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate our checkpoints with PhyE2E's divide-and-conquer Oracle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--benchmark", "--bench", dest="benchmark",
                    choices=["srbench", "llmsrbench", "llmsr"], default="srbench")
    ap.add_argument("--model", default="89M_40_simp1",
                    help="OUR checkpoint that solves the subproblems")
    ap.add_argument("--checkpoint", default=None, help="explicit .pth (overrides --model)")
    ap.add_argument("--checkpoints-dir", default="./checkpoints")
    ap.add_argument("--weights-path", default="weights/phye2e/phye2e_model.pt",
                    help="THEIR checkpoint -- only the env/generator/params are used")
    ap.add_argument("--phyreg-device", default="cpu",
                    help="where THEIR (unused) transformer is parked; the Oracle itself "
                         "runs on --device")
    ap.add_argument("--device", default=None)

    ap.add_argument("--datasets", default="./datasets/pmlb/datasets",
                    help="SRBench dataset root")
    ap.add_argument("--dataset", default=None, help="single SRBench dataset")
    ap.add_argument("--split", default="lsr_transform", help="LLM-SRBench split")
    ap.add_argument("--problem", default=None, help="single LLM-SRBench equation")
    ap.add_argument("--max-problems", type=int, default=None)

    ap.add_argument("--seed", type=int, default=42, help="first seed (42..42+n-1)")
    ap.add_argument("--n_seeds", "--n-seeds", dest="n_seeds", type=int, default=1)
    ap.add_argument("--target-noise", type=float, default=0.0, metavar="TAU",
                    help="additive RMS noise on TRAIN targets (SRBench convention)")
    ap.add_argument("--results-dir", default="./results",
                    help="root; the noise subdir is composed from --target-noise")

    ap.add_argument("--n-bags", type=int, default=100)
    ap.add_argument("--sample-size", type=int, default=None, metavar="N",
                    help="IO pairs per bag; default = eval_mymodels.SAMPLE_SIZE (200)")
    ap.add_argument("--beam-size", type=int, default=10)
    ap.add_argument("--unscale", action="store_true")
    ap.add_argument("--max-forward-batch", type=int, default=None)

    ap.add_argument("--oracle-rows", type=int, default=20000,
                    help="raw rows their oracle_fit gets; same budget devncon's oracle "
                         "gets, so the two D&C arms differ only in algorithm")
    ap.add_argument("--oracle-epochs", type=int, default=400,
                    help="their fit() default; run N_reg_lr(=4) times, so this is the "
                         "dominant per-problem cost")
    ap.add_argument("--oracle-lr", type=float, default=0.005, help="their fit() default")
    ap.add_argument("--oracle-bs", type=int, default=None,
                    help="default len(X)//3, as their fit() computes it")
    ap.add_argument("--separation-types", dest="sep_types", default=None,
                    help="comma-separated sigma family for oracle_fit; default is their "
                         "checkpoint's own oracle_seperation_type. NB their code tests "
                         "for the misspelling 'arccin', so the shipped default's "
                         "'arcsin' never activates -- pass it to enable that branch.")
    ap.add_argument("--verbose", action="store_true",
                    help="per-problem trace (subproblem count, per-candidate R^2)")
    args = ap.parse_args()

    # Our modules FIRST: _setup_imports() prepends PhysicsRegression/ to sys.path and
    # evicts colliding packages, so anything of ours must already be resolved.
    from eval_mymodels import (DEFAULT_FORWARD_BATCH, SAMPLE_SIZE, load_model,
                               resolve_model_checkpoint)
    import eval_devncon

    if args.max_forward_batch is None:
        args.max_forward_batch = DEFAULT_FORWARD_BATCH
    if args.sample_size is None:
        args.sample_size = SAMPLE_SIZE
    else:
        import eval_mymodels as _tf
        _tf.SAMPLE_SIZE = int(args.sample_size)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = args.checkpoint or resolve_model_checkpoint(args.model, args.checkpoints_dir)
    model_name = args.model or os.path.basename(os.path.dirname(ckpt)).replace("_res", "")
    print(f"Loading {ckpt} ...", flush=True)
    model, vocab = load_model(ckpt, device)[:2]

    oracle, params = build_oracle(args.weights_path, device, args.phyreg_device,
                                  quiet=not args.verbose)
    if args.sep_types is None:
        args.sep_types = params.oracle_seperation_type
    args.sep_types = [t.strip() for t in args.sep_types.split(",") if t.strip()]
    args._oracle, args._params = oracle, params

    # A path that never exists, so their filename-keyed surrogate cache can never serve
    # one problem's oracle to the next.  save_model=False means it is never created.
    _n = [0]

    def _oracle_file():
        _n[0] += 1
        return os.path.join(SCRIPT_DIR, ".phyedc_cache_never",
                            f"{os.getpid()}_{_n[0]}.pth")
    args._oracle_file = _oracle_file

    print(f"model {model_name!r} + PhyE2E Oracle on {device}\n"
          f"separations: {','.join(args.sep_types)}\n", flush=True)

    if args.benchmark == "srbench":
        eval_devncon.run_srbench(model, vocab, device, model_name, args,
                                 solve_fn=_solve, tag="phyedc")
    else:
        eval_devncon.run_llmsr(model, vocab, device, model_name, args,
                               solve_fn=_solve, tag="phyedc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
