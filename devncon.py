#!/usr/bin/env python3
"""devncon.py -- oracle-network + Hessian decomposition, in the style of
"A neural symbolic model for space physics" (PhysicsRegression/Oracle/oracle.py).

THE IDEA
--------
Symbolic regression gets harder fast as the variable count grows.  So: fit a plain
neural network to the data first (the "oracle"), read the *structure* of f off that
network's second derivatives, use the structure to split the problem into independent
low-dimensional pieces, solve each piece with the pre-trained transformer's ordinary
beam search, and multiply/add the pieces back together.

The oracle is not the answer and is never returned -- it is a differentiable stand-in
for f that can be queried anywhere, which is what makes both the Hessian test and the
pseudo-data construction possible.

WHY THE HESSIAN
---------------
For a twice-differentiable f, with H = d^2 f / dx_i dx_j:

    f = g(x_A) + h(x_B)   <=>   H_ij = 0                for all i in A, j in B
    f = g(x_A) * h(x_B)   <=>   [d^2 log|f|]_ij = 0     for all i in A, j in B

and  d^2 log|f| / dx_i dx_j  =  H_ij / f  -  (df/dx_i)(df/dx_j) / f^2.

So one Hessian per sample point yields BOTH tests.  Averaging |H_ij| over sample points
(median, to survive outliers) gives an n x n coupling matrix per link type; entries near
zero mark variable pairs that separate.  Grouping the variables by the pairs that do NOT
separate gives the decomposition.

PIPELINE
--------
    0. SAFETY NET.  Plain beam search on the full problem.  If it already solves the
       data, return it -- no oracle, no decomposition.  Nothing below can then make
       things worse, and the expensive path is skipped on easy problems.
    1. ORACLE.  Train the feed-forward net (below) on 80% of the data.
    2. HESSIAN.  Batched exact second derivatives at sample points -> H_add, H_mul.
    3. GROUPS.  Threshold each matrix, take connected components of the "still coupled"
       graph -> a partition of the variables, per link type.
    4. PSEUDO-DATA.  For group G_i, freeze every other variable at its median and read
       t_i = f(x_Gi, median) off the ORACLE.  Exactly recoverable:
           additive        f = sum_i t_i - (k-1)*c
           multiplicative  f = prod_i t_i / c^(k-1)        with c = f(median)
    5. SUBFORMULAS.  Beam-search each (X[:, G_i], t_i) with the transformer.
    6. COMBINE.  Remap each subformula's variables back to parent indices, join with the
       link operator and the constant, BFGS-refine against the REAL (X, y).
    7. Keep whichever of {decomposition, baseline} scores higher on real data.

ORACLE NETWORK (as specified)
-----------------------------
Fully connected, 4 hidden layers of 128 / 128 / 64 / 64, tanh throughout, linear output.
80/20 train/validation split.  RMSE loss.  Adam with betas (0.9, 0.999).  Learning rate
starts at 1e-2 and is divided by 10 whenever validation loss has not improved for more
than 20 epochs; training stops once it would drop below 1e-5.

KNOWN LIMITS
------------
* Only additive and multiplicative structure is tested.  A scalar-bottleneck function
  such as sqrt(x0^2/x1^2 + x2) has neither and is correctly left to the baseline.
* The oracle's SECOND derivatives are far noisier than its values: it reaches R^2 = 1.0000
  on the data while its Hessian still carries ~20% relative error.  Near-linear functions
  are therefore missed (their true H is 0, so noise dominates) -- but those are exactly
  the ones the baseline beam search solves outright, so the safety net covers them.
* More training data sharpens the Hessian markedly (2k -> 20k rows moved a known-zero
  entry from 0.32 to 0.10), so the oracle is trained on all supplied rows, not on the
  subsample handed to the transformer.

USAGE
    python devncon.py --dataset feynman_I_29_16              # 89M model by default
    python devncon.py --dataset feynman_I_12_11 --model 145M_40_simp1 --verbose

    from devncon import devncon_solve                          # as a library
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from grammar import V

_VSET = set(V)
_NV = len(V)


# ===========================================================================
# 1 -- the oracle network
# ===========================================================================
class OracleNet(nn.Module):
    """128 -> 128 -> 64 -> 64 -> 1, tanh on every hidden layer."""

    def __init__(self, n_in: int):
        super().__init__()
        self.l1 = nn.Linear(n_in, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = torch.tanh(self.l4(x))
        return self.out(x)


class Oracle:
    """A trained OracleNet plus the standardisation it was fitted under.

    Inputs are whitened and the target standardised, both for training stability and
    because the Hessian test wants dimensionless derivatives -- an entry of H must not
    look "small" merely because that variable is measured in large units.  ``f``,
    ``grad`` and ``hessian`` below are w.r.t. the WHITENED inputs but return the target
    in its ORIGINAL units, which is what the log-Hessian needs (log|f| is not invariant
    to shifting f).
    """

    def __init__(self, net, x_mu, x_sd, y_mu, y_sd, device):
        self.net, self.device = net, device
        self.x_mu, self.x_sd, self.y_mu, self.y_sd = x_mu, x_sd, y_mu, y_sd

    def _whiten(self, X):
        return (np.asarray(X, dtype=np.float64) - self.x_mu) / self.x_sd

    def predict(self, X) -> np.ndarray:
        """f(x) in original target units, for raw (unwhitened) X."""
        Z = torch.as_tensor(self._whiten(X), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            out = self.net(Z).squeeze(-1).cpu().numpy().astype(np.float64)
        return out * self.y_sd + self.y_mu

    def derivatives(self, X):
        """(f, g, H) at raw points X.  Shapes (B,), (B, n), (B, n, n).

        Exact second derivatives, computed with n backward passes for the WHOLE batch
        (differentiate the summed gradient component-by-component) rather than one
        autograd.functional.hessian call per point, which is what the reference
        implementation does and is ~B times slower.
        """
        Z = torch.as_tensor(self._whiten(X), dtype=torch.float32,
                            device=self.device).requires_grad_(True)
        out = self.net(Z).squeeze(-1)
        g = torch.autograd.grad(out.sum(), Z, create_graph=True)[0]
        rows = [torch.autograd.grad(g[:, j].sum(), Z, retain_graph=True)[0]
                for j in range(Z.shape[1])]
        H = torch.stack(rows, dim=1)
        f = out.detach().cpu().numpy().astype(np.float64) * self.y_sd + self.y_mu
        gn = g.detach().cpu().numpy().astype(np.float64) * self.y_sd
        Hn = H.detach().cpu().numpy().astype(np.float64) * self.y_sd
        return f, gn, Hn


def train_oracle(X: np.ndarray, y: np.ndarray, device,
                 lr0: float = 1e-2, lr_min: float = 1e-5, lr_factor: float = 10.0,
                 patience: int = 20, val_frac: float = 0.2,
                 batch_size: int = 256, max_epochs: int = 3000,
                 seed: int = 0, verbose: bool = False):
    """Fit the oracle to (X, y).  Returns (Oracle, val_rmse_standardised).

    Follows the specified protocol exactly: 80/20 split, RMSE loss, Adam(0.9, 0.999),
    lr 1e-2 divided by 10 after ``patience`` epochs without validation improvement,
    stopping once the next reduction would fall below 1e-5.  ``max_epochs`` is only a
    backstop -- the learning-rate floor is the real termination condition.
    """
    torch.manual_seed(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    x_mu, x_sd = X.mean(0), X.std(0) + 1e-12
    y_mu, y_sd = float(y.mean()), float(y.std()) + 1e-12
    Z = (X - x_mu) / x_sd
    t = (y - y_mu) / y_sd

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(t))
    n_val = max(1, int(round(val_frac * len(t))))
    va, tr = perm[:n_val], perm[n_val:]

    Ztr = torch.as_tensor(Z[tr], dtype=torch.float32, device=device)
    ttr = torch.as_tensor(t[tr], dtype=torch.float32, device=device).unsqueeze(-1)
    Zva = torch.as_tensor(Z[va], dtype=torch.float32, device=device)
    tva = torch.as_tensor(t[va], dtype=torch.float32, device=device).unsqueeze(-1)

    net = OracleNet(X.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr0, betas=(0.9, 0.999))

    def rmse(p, q):
        return torch.sqrt(torch.mean((p - q) ** 2) + 1e-30)

    lr = lr0
    best_val, best_state, since_improve = np.inf, None, 0
    n_tr = Ztr.shape[0]
    for epoch in range(1, max_epochs + 1):
        net.train()
        idx = torch.randperm(n_tr, device=device)
        for s in range(0, n_tr, batch_size):
            b = idx[s:s + batch_size]
            loss = rmse(net(Ztr[b]), ttr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            val = float(rmse(net(Zva), tva))

        if val < best_val - 1e-9:
            best_val, since_improve = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            since_improve += 1
            if since_improve > patience:
                lr /= lr_factor
                if lr < lr_min:
                    if verbose:
                        print(f"    oracle: stop at epoch {epoch}, lr floor reached, "
                              f"val RMSE {best_val:.5f}", flush=True)
                    break
                for gparam in opt.param_groups:
                    gparam["lr"] = lr
                since_improve = 0
                if verbose:
                    print(f"    oracle: epoch {epoch}  lr -> {lr:.1e}  "
                          f"best val RMSE {best_val:.5f}", flush=True)

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return Oracle(net, x_mu, x_sd, y_mu, y_sd, device), best_val


# ===========================================================================
# 2 -- Hessian coupling matrices
# ===========================================================================
def coupling_matrices(oracle: Oracle, X: np.ndarray, n_points: int = 1000,
                      rng: np.random.Generator | None = None,
                      batch: int = 256):
    """(H_add, H_mul): median |second derivative| over sample points, per variable pair.

    H_add[i,j] = median |d^2 f   / dx_i dx_j|      -> zero iff additively separable
    H_mul[i,j] = median |d^2 ln|f| / dx_i dx_j|    -> zero iff multiplicatively separable

    Probing at the data's own rows rather than a uniform box: the oracle is only
    trustworthy where it saw data, and a box in whitened space wanders off the joint
    distribution as soon as the inputs are correlated.

    Each matrix is normalised by its own largest off-diagonal entry, so the thresholds
    downstream are scale-free.
    """
    rng = rng or np.random.default_rng(0)
    n = X.shape[1]
    idx = (rng.choice(X.shape[0], size=min(n_points, X.shape[0]), replace=False))
    P = np.asarray(X, dtype=np.float64)[idx]

    fs, gs, Hs = [], [], []
    for s in range(0, P.shape[0], batch):
        f, g, H = oracle.derivatives(P[s:s + batch])
        fs.append(f); gs.append(g); Hs.append(H)
    f = np.concatenate(fs); g = np.concatenate(gs); H = np.concatenate(Hs)

    # log|f| is undefined where f ~ 0; drop those rows from the multiplicative test only
    scale = np.mean(np.abs(f)) + 1e-30
    ok = np.abs(f) > 1e-6 * scale

    # Additive coupling, made DIMENSIONLESS by the gradient scale:
    #     r_ij = |H_ij| * |f| / (|df/dx_i| * |df/dx_j|)
    # H_ij ~ f/(x_i x_j) and g_i ~ f/x_i carry the same units, so this is scale-free,
    # exactly 0 when the pair is additively separable and O(1) when it is not.
    # A plain max-normalised |H_ij| CANNOT work: a fully separable function has no large
    # entry to normalise against, so dividing by the max turns pure noise into apparent
    # full coupling and x0+x1+x2 yields no decomposition at all.
    gi = np.abs(g)[:, :, None] * np.abs(g)[:, None, :]
    H_add = np.median(np.abs(H) * np.abs(f)[:, None, None] / (gi + 1e-30), axis=0)

    if ok.sum() >= 8:
        fo, go, Ho = f[ok], g[ok], H[ok]
        # d^2 ln|f| = (f*H_ij - g_i g_j) / f^2.  Scored as RELATIVE cancellation rather
        # than as an absolute magnitude, because the numerator is a difference of two
        # nearly equal terms whenever the pair really is separable: for f=(x0+x1)*x2,
        # f*H_02 = g_0 g_2 exactly, so the true value is 0 only via cancellation, and
        # any oracle derivative error survives at the same O(1/f) size as a genuinely
        # coupled entry.  Measured on that function, the absolute form gives
        # 0.923/0.961 against 1.000 -- no signal at all -- while the relative form gives
        # 0.212/0.212 against 1.000.  The ratio below is dimensionless, 0 for exact
        # cancellation and O(1) otherwise, so it is immune to that amplification.
        t1 = fo[:, None, None] * Ho
        t2 = go[:, :, None] * go[:, None, :]
        H_mul = np.median(np.abs(t1 - t2) / (np.abs(t1) + np.abs(t2) + 1e-30), axis=0)
    else:
        H_mul = np.full((n, n), np.inf)

    # Both measures are already absolute and dimensionless (0 = separable, O(1) = not),
    # so they are returned unnormalised -- see the note above.
    return H_add, H_mul


def variable_groups(M: np.ndarray, cutoff: float):
    """Partition the variables: i and j stay together unless M[i,j] <= cutoff.

    Connected components of the "still coupled" graph.  Components are used rather than
    the reference implementation's iterative group-splitting because they are guaranteed
    to be a partition -- which the pseudo-data construction below requires -- and because
    when the separability relation is consistent the two agree.
    """
    n = M.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in itertools.combinations(range(n), 2):
        if M[i, j] > cutoff:                       # still coupled -> same group
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

    buckets: dict = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    return sorted((sorted(v) for v in buckets.values()), key=lambda g: (len(g), g))


def decomposition_candidates(H_add: np.ndarray, H_mul: np.ndarray,
                             coeffs=(1.5, 2.0, 3.0, 5.0, 10.0),
                             abs_cuts=(0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85),
                             rel_cap: float = 0.9):
    """[(groups, link, cutoff)] with >= 2 groups.

    The cutoff is set RELATIVE to the smallest off-diagonal entry, as in the reference
    implementation: that entry is the oracle's own noise floor for this problem, and it
    varies by an order of magnitude between problems, so a fixed absolute threshold
    either splits everything or nothing.  ``rel_cap`` additionally refuses to call a
    pair separable when it is a large fraction of the LARGEST entry, which is what stops
    a fully-entangled function (every entry within 2x of every other) from being declared
    completely separable.

    Proposals are deliberately loose -- each is checked against the oracle reconstruction
    before any beam search is spent, and against the real data after.  The cut ladder runs
    high (to 0.85) because how well the two measures separate is problem-dependent: on
    lsr_transform/III.19.51_1_0 the separable pairs sit below 0.5, but on II.13.23_1_0 the
    same measure puts them at 0.83 against 1.00 for the coupled pair -- correctly ranked,
    just compressed.  Since the reconstruction filter is free and tight, a wide ladder
    costs nothing and a narrow one silently loses decompositions.
    """
    out, seen = [], set()
    for link, M in (("+", H_add), ("*", H_mul)):
        if not np.all(np.isfinite(M)):
            continue
        off = M[~np.eye(M.shape[0], dtype=bool)]
        if off.size == 0:
            continue
        lo = float(off.min())
        # Adaptive cuts (relative to this problem's own noise floor, as in the reference)
        # PLUS fixed absolute cuts, which the adaptive rule alone cannot express: when
        # every pair separates there is no coupled entry to be relative to.
        cuts = [min(lo * k, rel_cap) for k in coeffs] + list(abs_cuts)
        for cut in cuts:
            groups = variable_groups(M, cut)
            if len(groups) < 2:
                continue
            key = (link, tuple(tuple(g) for g in groups))
            if key not in seen:
                seen.add(key)
                out.append((groups, link, cut))
    return out


def reconstruction_r2(targets, c: float, link: str, y: np.ndarray) -> float:
    """R^2 of the decomposition rebuilt from ORACLE pseudo-targets alone.

    Free (no transformer) and it is the decisive filter: if summing/multiplying the
    frozen-slice targets cannot reproduce y, the proposed grouping is wrong and there is
    no point beam-searching its parts.

    The default threshold is TIGHT (0.999) on purpose.  When the decomposition is real
    the identity is exact, so the reconstruction is limited only by oracle error and
    scores 1.0000 in practice; a loose bar lets through functions that merely happen to
    be nearly separable on the sampled domain.  x0*x1 + x1*x2 + x0*x2 over [1,3]^3 is the
    case in point -- it has no multiplicative split at all, yet an all-singleton
    multiplicative reconstruction still reaches 0.9973.
    """
    y = np.asarray(y, dtype=np.float64)
    k = len(targets)
    with np.errstate(all="ignore"):
        if link == "+":
            rec = np.sum(targets, axis=0) - (k - 1) * c
        else:
            rec = np.prod(targets, axis=0) / (c ** (k - 1))
    if not np.all(np.isfinite(rec)):
        return 0.0
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return float(max(0.0, 1.0 - float(np.sum((y - rec) ** 2)) / ss_tot))


# ===========================================================================
# 3 -- pseudo-data and recombination
# ===========================================================================
def group_targets(oracle: Oracle, X: np.ndarray, groups, link: str):
    """(targets, c): the per-group pseudo-target and the median-point constant.

    With every variable outside G_i frozen at its median,

        additive        f = sum_i t_i - (k-1) * c
        multiplicative  f = prod_i t_i / c^(k-1)          c = f(median)

    both exact when the decomposition holds (substitute t_i = h_i(x) * c / h_i(m)).
    Returns None when c is too close to zero for the multiplicative form.
    """
    med = np.median(np.asarray(X, dtype=np.float64), axis=0)
    c = float(oracle.predict(med[None, :])[0])
    if link == "*" and abs(c) < 1e-12:
        return None, c

    targets = []
    for g in groups:
        Xi = np.array(X, dtype=np.float64, copy=True)
        mask = np.ones(X.shape[1], dtype=bool)
        mask[list(g)] = False
        Xi[:, mask] = med[mask]
        targets.append(oracle.predict(Xi))
    return targets, c


def remap_vars(formula: str, group) -> str:
    """Rewrite a subformula's local v1..v|group| to the parent's variable indices.

    Two-phase via ``@i@`` placeholders so a rename cannot collide with a token that a
    later rename would pick up again.
    """
    group = list(group)
    out = []
    for tok in formula.split():
        if tok in _VSET:
            j = V.index(tok)
            out.append(f"@{group[j]}@" if j < len(group) else tok)
        else:
            out.append(tok)
    return " ".join(
        (f"v{int(t[1:-1]) + 1}" if (t.startswith("@") and t.endswith("@")) else t)
        for t in out
    )


def combine(subformulas, groups, link: str, c: float) -> str:
    """Join the remapped subformulas into one prefix expression."""
    parts = [remap_vars(f, g) for f, g in zip(subformulas, groups)]
    expr = parts[-1]
    for p in reversed(parts[:-1]):                      # right-folded n-ary join
        expr = f"{link} {p} {expr}"
    k = len(parts)
    if link == "+":
        return f"- {expr} {(k - 1) * c:.10g}" if k > 1 else expr
    return f"/ {expr} {c ** (k - 1):.10g}" if k > 1 else expr


def _pad_to_nv(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] < _NV:
        X = np.concatenate(
            [X, np.zeros((X.shape[0], _NV - X.shape[1]), dtype=np.float64)], axis=1)
    return X


# ===========================================================================
# 4 -- the driver
# ===========================================================================
def devncon_solve(X: np.ndarray, y: np.ndarray,
                 leaf_solver, r2_fn, refine_fn,
                 rng: np.random.Generator | None = None,
                 device=None,
                 holdout: tuple | None = None,
                 oracle_data: tuple | None = None,
                 safety_threshold: float = 0.9999,
                 baseline_draws: int = 1,
                 oracle_points: int = 1000,
                 recon_threshold: float = 0.999,
                 min_points: int = 50,
                 verbose: bool = True,
                 **oracle_kwargs):
    """Solve (X, y) by oracle-guided decomposition.  Returns (formula, r2).

    Dependency-injected exactly like the rest of this repo's solvers:
      leaf_solver(Xs, ys, baseline=False) -> (formula, r2)
          Plain transformer beam search.  ``baseline=True`` marks the ONE top-level
          safety-net solve, which must be run the way a standalone beam run would be --
          with the searcher's own internal split and held-out candidate selection.
          Sub-problems are called with baseline=False: their targets are oracle
          pseudo-data, which has no held-out concept.  Getting this backwards is not
          cosmetic -- selecting the baseline in-sample and then reporting it on the outer
          held-out rows dropped feynman_I_29_16 from R^2 0.89 to 0.00, and it is the same
          regression the retired AIF2 path documented (_old/aif2.py).
      r2_fn(formula, X, y) -> float
      refine_fn(formula, X, y) -> formula    BFGS constant refinement

    ``holdout=(X_te, y_te)`` makes the FINAL choice between the baseline and each
    decomposition on held-out rows instead of the fitting rows, and the returned R^2 is
    then the held-out one.  Evaluation should always pass it: a decomposition has more
    moving parts than a single beam result, so selecting in-sample lets it win on rows it
    was fitted to and then lose on the reported split.  eval_mymodels wraps exactly this guard
    around the retired AIF2 path for the same reason.

    Any failure falls back to the baseline beam result, so this can never do worse than
    plain beam search except through sampling noise.
    """
    rng = rng or np.random.default_rng(0)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = X.shape[1]

    def log(msg):
        if verbose:
            print(msg, flush=True)

    # ---- step 0: safety net ------------------------------------------------
    # Best-of-k draws.  Beam search is high-variance on hard problems -- one draw can
    # score ~0 where another solves outright -- so the floor this method is measured
    # against is itself a coin flip.  The retired AIF2 path drew 3 (_old/aif2.py);
    # devncon drew once, which is most of
    # why the two diverge under noise.  Re-draw ONLY while short of safety_threshold, so
    # problems the first draw already solves still cost exactly one beam search, and
    # each draw is scored by the searcher's own held-out split (baseline=True), never
    # in-sample.  Default 1 keeps every existing result reproducible; pass 3 to match
    # what the retired AIF2 path did.
    t0 = time.time()
    base_f, base_r = leaf_solver(X, y, baseline=True)
    for _ in range(max(0, baseline_draws - 1)):
        if base_r >= safety_threshold:
            break
        alt_f, alt_r = leaf_solver(X, y, baseline=True)
        if alt_r > base_r:
            base_f, base_r = alt_f, alt_r
    log(f"[devncon] baseline beam R^2={base_r:.6f}  (best of <={baseline_draws} draw(s), "
        f"{time.time() - t0:.1f}s)  {base_f}")
    if base_r >= safety_threshold:
        log(f"[devncon] baseline already solves it (>= {safety_threshold}) -> done")
        return base_f, base_r
    if n < 2 or X.shape[0] < min_points:
        return base_f, base_r

    try:
        # ---- step 1: oracle ------------------------------------------------
        # The oracle is trained on ALL available rows, not the subsample the transformer
        # sees: its Hessian is what the decomposition rests on, and second-derivative
        # accuracy improves sharply with data (a known-zero entry went 0.32 -> 0.10
        # going from 2k to 20k rows) while the transformer is capped by its context.
        Xo, yo = oracle_data if oracle_data is not None else (X, y)
        t0 = time.time()
        log(f"[devncon] training oracle on {Xo.shape[0]} rows ...")
        oracle, val = train_oracle(Xo, yo, device, verbose=verbose, **oracle_kwargs)
        log(f"[devncon] oracle trained: val RMSE {val:.5f} (standardised)  "
            f"({time.time() - t0:.1f}s)")

        # ---- step 2: Hessian ------------------------------------------------
        H_add, H_mul = coupling_matrices(oracle, Xo, n_points=oracle_points, rng=rng)
        if verbose:
            np.set_printoptions(precision=3, suppress=True)
            log(f"[devncon] H_add (normalised)\n{H_add}")
            log(f"[devncon] H_mul (normalised)\n{H_mul}")

        # ---- step 3: candidate decompositions --------------------------------
        cands = decomposition_candidates(H_add, H_mul)
        if not cands:
            log("[devncon] no decomposition found -> baseline")
            return base_f, base_r
        log(f"[devncon] {len(cands)} candidate decomposition(s)")

        Xp = _pad_to_nv(X)
        if holdout is not None:
            Xh, yh = holdout
            Xhp, yh = _pad_to_nv(Xh), np.asarray(yh, dtype=np.float64).reshape(-1)
            score = lambda f: (r2_fn(f, Xhp, yh) if f else 0.0)
        else:
            score = lambda f: (r2_fn(f, Xp, y) if f else 0.0)
        best = (base_f, score(base_f))
        for groups, link, cutoff in cands:
            # ---- step 4: pseudo-data ---------------------------------------
            targets, c = group_targets(oracle, X, groups, link)
            if targets is None:
                log(f"[devncon]   {groups} link='{link}': degenerate constant -> skip")
                continue
            rec = reconstruction_r2(targets, c, link, y)
            if rec < recon_threshold:
                log(f"[devncon]   {groups} link='{link}': oracle reconstruction "
                    f"R^2={rec:.4f} < {recon_threshold} -> skip (no beam search spent)")
                continue
            log(f"[devncon]   {groups} link='{link}' cut={cutoff:.3f}: "
                f"oracle reconstruction R^2={rec:.4f}")

            # ---- step 5: one beam search per group -------------------------
            subs = []
            for g, t in zip(groups, targets):
                sf, sr = leaf_solver(X[:, list(g)], t)
                log(f"[devncon]   group {g} (cut {cutoff}): R^2={sr:.4f}  {sf}")
                if not sf:
                    break
                subs.append(sf)
            if len(subs) != len(groups):
                continue

            # ---- step 6: combine + refine on the REAL data -----------------
            # Constants are always refined against the FITTING rows; only the
            # comparison below uses the held-out ones.
            formula = combine(subs, groups, link, c)
            try:
                ref = refine_fn(formula, Xp, y)
                if ref and r2_fn(ref, Xp, y) > r2_fn(formula, Xp, y):
                    formula = ref
            except Exception:
                pass
            r = score(formula)
            log(f"[devncon]   -> combined link='{link}' R^2={r:.6f}  {formula}")
            if r > best[1]:
                best = (formula, r)

        # ---- step 7: never lose to the baseline -----------------------------
        if best[0] is not base_f:
            log(f"[devncon] decomposition wins: R^2={best[1]:.6f}")
        else:
            log(f"[devncon] baseline retained: R^2={best[1]:.6f}")
        return best

    except Exception as e:
        log(f"[devncon] ERROR {e!r} -> baseline")
        return base_f, base_r


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Oracle-network + Hessian decomposition symbolic regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", default="feynman_I_29_16",
                    help="PMLB/Feynman dataset folder name")
    ap.add_argument("--datasets-dir", default="./datasets/pmlb/datasets")
    ap.add_argument("--llmsrbench", default=None, metavar="SPLIT",
                    help="load from LLM-SRBench instead (e.g. lsr_transform); "
                         "use --problem to name the equation")
    ap.add_argument("--problem", default=None, metavar="NAME",
                    help="LLM-SRBench equation id, e.g. III.9.52_0_0")
    ap.add_argument("--model", default="89M_40_simp1",
                    help="89M_40_simp1 = 89M (default), 145M_40_simp1 = 145M, m89float")
    ap.add_argument("--checkpoints-dir", default="./checkpoints")
    ap.add_argument("--n-points", type=int, default=200,
                    help="rows handed to the transformer per beam search")
    ap.add_argument("--oracle-rows", type=int, default=20000,
                    help="rows used to TRAIN the oracle (its Hessian needs far more "
                         "data than the transformer's context allows)")
    ap.add_argument("--oracle-points", type=int, default=1000,
                    help="sample points for the Hessian estimate")
    ap.add_argument("--safety-threshold", type=float, default=0.9999,
                    help="baseline R^2 at or above which decomposition is skipped")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import os
    from eval_mymodels import (bfgs_refine_mfg, compute_r2, evaluate_xy, load_dataset,
                         load_model, resolve_model_checkpoint)

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = resolve_model_checkpoint(args.model, args.checkpoints_dir)
    model, vocab = load_model(ckpt, device)
    print(f"model {args.model!r} ({sum(p.numel() for p in model.parameters()):,} "
          f"weights) on {device}")

    held_out = None
    if args.llmsrbench:
        from eval_mymodels import load_problems
        probs = {p["name"]: p for p in load_problems(args.llmsrbench)}
        if args.problem not in probs:
            print(f"unknown problem {args.problem!r}; e.g. "
                  f"{sorted(probs)[:3]}", file=sys.stderr)
            return 1
        p = probs[args.problem]
        tr = p["train"]
        y_full, X_full = tr[:, 0], tr[:, 1:]
        if p.get("test") is not None and len(p["test"]):
            held_out = (p["test"][:, 1:], p["test"][:, 0])
        label = f"{args.llmsrbench}/{args.problem}"
        print(f"ground truth: {p['expression']}")
        print(f"symbols: {list(p['symbols'])}")
    else:
        tsv = os.path.join(args.datasets_dir, args.dataset, f"{args.dataset}.tsv.gz")
        X_full, y_full = load_dataset(tsv)
        label = args.dataset
    rng = np.random.default_rng(args.seed)
    if X_full.shape[0] > args.oracle_rows:
        idx = rng.choice(X_full.shape[0], args.oracle_rows, replace=False)
        X_full, y_full = X_full[idx], y_full[idx]
    if X_full.shape[0] > args.n_points:
        idx = rng.choice(X_full.shape[0], args.n_points, replace=False)
        X, y = X_full[idx], y_full[idx]
    else:
        X, y = X_full, y_full
    print(f"dataset {label!r}: transformer sees X{X.shape}, "
          f"oracle sees X{X_full.shape}")

    def leaf_solver(Xs, ys, baseline=False):
        return evaluate_xy(Xs, ys, model, vocab, device, rng,
                           no_split=not baseline)

    formula, r2 = devncon_solve(
        X, y, leaf_solver=leaf_solver, r2_fn=compute_r2, refine_fn=bfgs_refine_mfg,
        rng=rng, device=device, safety_threshold=args.safety_threshold,
        oracle_data=(X_full, y_full),
        oracle_points=args.oracle_points, verbose=not args.quiet)
    print(f"\n=== {label}: train R^2 = {r2:.6f} ===\n{formula}")
    if held_out is not None and formula:
        Xh, yh = held_out
        print(f"=== held-out test R^2 = "
              f"{compute_r2(formula, _pad_to_nv(Xh), yh):.6f} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
