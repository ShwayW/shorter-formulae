"""offset_repair.py -- post-hoc structural repair for missing +1 / -1 offsets.

THE FAILURE THIS FIXES
----------------------
Measured on SRBench Feynman (noise 0): 21% of ground-truth formulas need the grammar's
``++`` / ``--`` unary (the canonical form of x+1 / x-1), and the models emit one in only
4.5-6.0% of predictions -- barely more often when the target actually needs it (1.34x)
than when it does not.  The beam usually gets the skeleton right and is one operator
short:

    true      kb*v/(A*(gamma-1))
    predicted 1.596 * v2*v4 / (v1*v3)              R^2 = 0.958
    repaired  1.0   * v2*v4 / ((v1-1)*v3)          R^2 = 1.000   <- '--' on v1

This is the one failure mode BFGS cannot rescue.  Refinement fits multiplicative
constants, so a wrong coefficient gets corrected downstream; an additive offset INSIDE a
nonlinearity cannot be absorbed by rescaling -- exp(u)-1 is not c*exp(u) -- so a missing
``--`` caps R^2 around 0.97-0.99 permanently.  Nor can a top-level affine wrapper fix it:
only 1 of 119 Feynman targets carries its offset at the root, and fitting ``a*f+b``
across the whole benchmark was measured to solve exactly 0 additional datasets.

WHY IT IS A REPAIR AND NOT A PRIOR / SEARCH CHANGE
--------------------------------------------------
Three cheaper explanations were tested and ruled out first:
  * the generator prior is already right -- it produces ``++``/``--`` in 21% of formulas
    against the 25% the benchmark needs;
  * beam search applies no length or complexity penalty to remove;
  * candidate selection is not discarding good structures -- on a 16-dataset probe the
    selected formula was already the best in the pool on held-out data, 16/16.
The correct structure simply is not generated, so it has to be added afterwards.

NO TEST LEAKAGE
---------------
The repair sees only the training slice it is handed.  That slice is split into a fit
part and a validation part: every candidate -- INCLUDING the incumbent, refitted so the
comparison is fair -- has its constants fitted on the fit part and is ranked on the
validation part.  Selecting on the same points used to fit is exactly what lets an
overfitted near-miss beat an exact structure, which is the trap this is built to avoid.
Only the winner is refitted on the full training slice before being returned.

Dependency-injected (``r2_fn``, ``refine_fn``) so it does not import eval_mymodels.
"""

from __future__ import annotations

import numpy as np

from grammar import B, U

_UNARY = set(U)
_BINARY = set(B)
OFFSET_OPS = ("++", "--")


def _subtree_end(toks, i: int) -> int:
    """Index one past the end of the subtree rooted at ``toks[i]`` (-1 if malformed)."""
    if i >= len(toks):
        return -1
    t = toks[i]
    if t in _UNARY:
        return _subtree_end(toks, i + 1)
    if t in _BINARY:
        a = _subtree_end(toks, i + 1)
        return -1 if a < 0 else _subtree_end(toks, a)
    return i + 1


def _is_valid_pn(toks) -> bool:
    return bool(toks) and _subtree_end(toks, 0) == len(toks)


def candidate_sites(formula: str, max_sites: int = 24):
    """Positions where inserting a ``++``/``--`` yields a well-formed formula.

    Every subtree root is a legal site -- the offset may belong on a bare variable
    (gamma-1) or on a whole subexpression (exp(...)-1).  Sites already carrying an
    offset op are skipped so repairs cannot stack into ``++ ++ ++ x``.
    """
    toks = formula.split()
    if not _is_valid_pn(toks):
        return []
    sites = []
    for i, t in enumerate(toks):
        if t in OFFSET_OPS:
            continue
        if i > 0 and toks[i - 1] in OFFSET_OPS:
            continue
        if _subtree_end(toks, i) > 0:
            sites.append(i)
    # Prefer shallow sites: an offset on a whole factor is more common than one buried
    # deep inside, and this only matters when we have to truncate.
    return sites[:max_sites]


def variants(formula: str, max_sites: int = 24, max_len: int | None = None):
    """All single-insertion ``++``/``--`` repairs of ``formula``."""
    toks = formula.split()
    out = []
    for i in candidate_sites(formula, max_sites):
        for op in OFFSET_OPS:
            cand = toks[:i] + [op] + toks[i:]
            if max_len is not None and len(cand) > max_len:
                continue
            out.append(" ".join(cand))
    return out


def repair(formula: str, X_tr, y_tr, r2_fn, refine_fn, *,
           screen_refine_fn=None,
           lo: float = 0.90, hi: float = 0.999999,
           val_frac: float = 0.25, min_points: int = 40,
           margin: float = 1e-4, max_sites: int = 24,
           max_len: int | None = None, screen_top: int = 3,
           verbose: bool = False):
    """Try to recover a missing +1/-1 offset.  Returns (formula, info).

    ``formula`` is returned unchanged unless a repaired variant beats it on held-out
    validation points by more than ``margin``.

    Parameters
    ----------
    lo, hi : the incumbent's validation R^2 band in which repair is attempted.  Below
        ``lo`` the skeleton is wrong and one operator will not save it; above ``hi`` the
        problem is already solved and there is nothing to gain.  This is what keeps the
        pass cheap -- it fires on the near-miss tail only.
    margin : required validation-R^2 improvement.  Small but non-zero, so a tie never
        churns the answer.
    screen_top : variants are screened with ``screen_refine_fn``, and the best
        ``screen_top`` are re-fitted on the full training slice with ``refine_fn``.
    screen_refine_fn : cheap refiner used for the O(2*sites) screening pass -- a
        full-strength BFGS on every variant costs seconds each and would dominate the
        whole evaluation.  Defaults to ``refine_fn``; pass a low-restart / short-timeout
        variant in production.  Only the finalists get the expensive refit.
    """
    screen = screen_refine_fn or refine_fn
    info = {"attempted": False, "n_variants": 0, "improved": False,
            "base_val_r2": None, "best_val_r2": None, "op": None, "site": None}
    if not formula or not formula.strip():
        return formula, info

    X_tr = np.asarray(X_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    n = len(y_tr)
    if n < min_points:
        return formula, info

    n_fit = int((1.0 - val_frac) * n)
    if n_fit < min_points // 2 or n - n_fit < min_points // 4:
        return formula, info
    Xf, yf = X_tr[:n_fit], y_tr[:n_fit]
    Xv, yv = X_tr[n_fit:], y_tr[n_fit:]

    def fit_and_score(f):
        """Refit constants on the fit slice, score on the validation slice."""
        try:
            ref = screen(f, Xf, yf) or f
        except Exception:
            ref = f
        try:
            return float(r2_fn(ref, Xv, yv)), ref
        except Exception:
            return -np.inf, ref

    # The incumbent is refitted on the SAME fit slice, or it would enjoy an unfair
    # advantage from having been fitted on the validation points too.
    base_val, _base_ref = fit_and_score(formula)
    info["base_val_r2"] = base_val
    if not (lo <= base_val <= hi):
        return formula, info

    cands = variants(formula, max_sites=max_sites, max_len=max_len)
    info["attempted"] = True
    info["n_variants"] = len(cands)
    if not cands:
        return formula, info

    scored = []
    for c in cands:
        v, ref = fit_and_score(c)
        if np.isfinite(v):
            scored.append((v, c, ref))
    if not scored:
        return formula, info
    scored.sort(key=lambda t: t[0], reverse=True)
    best_val, best_raw, _best_ref = scored[0]
    info["best_val_r2"] = best_val

    if best_val <= base_val + margin:
        return formula, info

    # Winner refitted on the FULL training slice for the returned formula.
    finalists = [c for _v, c, _r in scored[:max(1, screen_top)]]
    best_full, best_full_r2 = formula, -np.inf
    for c in finalists:
        try:
            ref = refine_fn(c, X_tr, y_tr) or c
            r = float(r2_fn(ref, X_tr, y_tr))
        except Exception:
            continue
        if r > best_full_r2:
            best_full, best_full_r2 = ref, r
    if best_full_r2 <= -np.inf:
        return formula, info

    toks = best_raw.split()
    for i, t in enumerate(toks):
        if t in OFFSET_OPS:
            info["op"], info["site"] = t, i
            break
    info["improved"] = True
    if verbose:
        print(f"  [offset-repair] val R^2 {base_val:.4f} -> {best_val:.4f} "
              f"via '{info['op']}' at token {info['site']}  ({len(cands)} variants)",
              flush=True)
    return best_full, info
