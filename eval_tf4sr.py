#!/usr/bin/env python3
"""
eval_tf4sr.py -- Evaluate the pretrained Transformer of Lalande et al. (2023,
"A Transformer Model for Symbolic Regression towards Scientific Discovery",
vendored in tf4sr/) on the SRBench Feynman benchmark and on LLM-SRBench.

    # SRBench (Feynman PMLB datasets, R^2 >= 0.99 on the held-out slice = solved)
    python eval_tf4sr.py --benchmark srbench

    # LLM-SRBench (default split lsr_transform)
    python eval_tf4sr.py --benchmark llmsrbench --split lsr_transform

    # the canonical sweep
    python eval_tf4sr.py --benchmark both --seeds 42 43 ... 51 \
        --target-noise 0 0.001 0.01 0.1

Artifacts follow the usual results_io layout, so the existing plotting picks this
up as one more method:
    results/tf4sr/seed<N>/noise_<tau>/eval_tf4sr.pkl.gz                 (srbench)
    results/tf4sr/seed<N>/noise_<tau>/llmsrbench/tf4sr_<split>/results.pkl.gz

WHAT THIS ADDS OVER tf4sr/evaluate_model.py
-------------------------------------------
Upstream's own harness is not comparable to our other methods, in three ways that
this script has to close:

  1. It scores a normalised TREE EDIT DISTANCE (zss) against the ground-truth
     expression -- never R^2, and never on held-out data.  We fit and score R^2 on
     a train/test split like every other eval_*.py.
  2. Its `C` is a single placeholder token: the decoder emits a SKELETON, not
     numeric constants (upstream even multiplies the ground truth by `C` before
     comparing).  We give every occurrence of `C` its own free parameter and fit
     them with BFGS, exactly as run_nesymres.py does for NeSymReS skeletons.
  3. It reads SRSD .txt files from a hardcoded ../../srsd_datasets/ path.  We feed
     it our PMLB mirror and the LLM-SRBench HDF5 instead.

MODEL CONSTRAINTS (both are properties of the pretrained checkpoint, not choices)
  * <= 6 input variables: the decoder vocabulary is x1..x6 (the model's
    `max_nb_var=7` counts the target column too).  --max-vars defaults to 6.
  * strictly positive inputs: upstream masks to rows with all x > 0 before
    sampling, because the encoder's log10 rescaling is undefined otherwise, and
    its sympy symbols are declared positive=True.  We keep that mask; a problem is
    attempted when it has >= --n-sample-points such rows.
  Measured coverage under both: 97/99 Feynman, 89/111 LSR-Transform.

ENCODER INPUT.  Upstream normalises each column by its geometric mean --
x' = 10^(log10 x - mean(log10 x)) -- and the target likewise, keeping its sign.
The decoded expression therefore lives in that rescaled space, so predictions are
mapped back as  y = 10^shift_y * f(X / 10^shift_x)  before anything is scored.
(Upstream skips the shift for columns whose SI unit is radian; we have no unit
metadata for these benchmarks, so the shift is applied uniformly.)
"""

import argparse
import gzip
import os
import pickle
import sys
import time
import warnings

import numpy as np
import sympy
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
# tf4sr imports itself as top-level `model` / `datasets` packages.
sys.path.insert(0, os.path.join(_HERE, "tf4sr"))

import results_io                                                    # noqa: E402
from eval_nesymres import (feynman_problems, llmsrbench_problems,    # noqa: E402
                           add_target_noise, seed_everything, shard)

METHOD_NAME = "tf4sr"
SRBENCH_ARTIFACT = "eval_tf4sr.pkl.gz"

# The six variable symbols the decoder can emit, in order.
MAX_MODEL_VARS = 6
DEFAULT_WEIGHTS_DIR = os.path.join(_HERE, "tf4sr", "best_model_weights")


def _clipped_r2(y, yp):
    """R^2 clipped at 0, ignoring non-finite predictions (same as run_nesymres.py).

    Non-finite predictions are expected here rather than exceptional: the model
    emits log/sqrt/inv freely, and the test slice may contain inputs outside the
    positive region the encoder was fed.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yp = np.asarray(yp, dtype=np.float64).reshape(-1)
    m = np.isfinite(y) & np.isfinite(yp)
    if m.sum() == 0:
        return 0.0
    try:
        return float(max(0.0, r2_score(y[m], yp[m])))
    except Exception:
        return 0.0


# -- skeleton -> parameterised callable -----------------------------------------

def _split_constants(expr, counter):
    """Give every OCCURRENCE of the placeholder `C` its own symbol c0, c1, ...

    The decoder has a single `C` token, so a skeleton like C*x1 + C*x2 parses into
    one sympy symbol used twice -- which would tie the two constants together and
    make most equations unfittable.  Every other skeleton method we compare against
    (NeSymReS, E2E) fits each constant independently, so we rebuild the tree with a
    fresh symbol per leaf.
    """
    if expr.is_Symbol and expr.name == "C":
        s = sympy.Symbol(f"c{counter[0]}", real=True)
        counter[0] += 1
        return s
    if not expr.args:
        return expr
    return expr.func(*[_split_constants(a, counter) for a in expr.args])


class _Fitted:
    """A decoded skeleton plus the rescaling that maps it back to data space."""

    def __init__(self, expr, params, var_syms, shift_x, shift_y, n_cols):
        self.expr = expr
        self.params = params
        self.var_syms = var_syms
        self.shift_x = shift_x          # per-column log10 shift, len == n_cols
        self.shift_y = shift_y          # scalar log10 shift for the target
        self.n_cols = n_cols
        self._f = sympy.lambdify(list(var_syms) + list(params), expr, "numpy")

    def __call__(self, X, theta):
        Xs = np.asarray(X, dtype=np.float64)[:, :self.n_cols]
        # into the encoder's rescaled space, then back out again
        Xn = Xs / np.power(10.0, self.shift_x)[None, :]
        with np.errstate(all="ignore"):
            out = self._f(*[Xn[:, i] for i in range(self.n_cols)], *theta)
            out = np.asarray(out, dtype=np.float64)
            if out.ndim == 0:
                out = np.full(len(Xs), float(out))
            return out * (10.0 ** self.shift_y)


def _fit_constants(fitted, X, y, rng, n_restarts):
    """BFGS on the free constants, best of n_restarts.  Returns (theta, train_r2)."""
    from scipy.optimize import minimize

    k = len(fitted.params)
    if k == 0:
        return np.zeros(0), _clipped_r2(y, fitted(X, []))

    def loss(theta):
        with np.errstate(all="ignore"):
            p = fitted(X, theta)
        if not np.all(np.isfinite(p)):
            p = np.where(np.isfinite(p), p, 0.0)
        return float(np.mean(np.square(p - y)))

    best, best_loss = np.ones(k), np.inf
    for i in range(max(1, n_restarts)):
        x0 = np.ones(k) if i == 0 else rng.normal(0.0, 1.0, size=k)
        try:
            res = minimize(loss, x0, method="BFGS", options={"maxiter": 300})
            if np.isfinite(res.fun) and res.fun < best_loss:
                best_loss, best = float(res.fun), np.asarray(res.x, dtype=np.float64)
        except Exception:
            continue
    return best, _clipped_r2(y, fitted(X, best))


# -- runner ---------------------------------------------------------------------

class Tf4srRunner:
    """Loads one tf4sr checkpoint once; fit()/predict() per problem."""

    def __init__(self, enc_type, label_smoothing, weights_dir, device,
                 n_sample_points, n_restarts, n_draws):
        import torch
        from model.transformer_model import TransformerModel

        self.torch = torch
        self.device = device
        self.n_sample_points = n_sample_points
        self.n_restarts = n_restarts
        self.n_draws = n_draws

        name = enc_type + ("_label_smoothing" if label_smoothing else "")
        self.checkpoint = os.path.join(weights_dir, name, "model_weights.pt")
        if not os.path.exists(self.checkpoint):
            sys.exit(f"[fatal] checkpoint not found: {self.checkpoint}")

        # Architecture is fixed by the released weights (evaluate_model.py:31-43).
        self.model = TransformerModel(
            enc_type=enc_type, nb_samples=n_sample_points, max_nb_var=7,
            d_model=256, vocab_size=18 + 2, seq_length=30, h=4,
            N_enc=4, N_dec=8, dropout=0.25,
        )
        sd = torch.load(self.checkpoint, map_location="cpu")
        # Upstream's release is a DataParallel state dict ("module." prefixed).
        stripped = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        self.model.load_state_dict(stripped, strict=True)
        self.model.eval().to(device)

        self.fitted_ = None
        self.top_k_idx_ = None          # kept for artifact-schema parity; no selection

    # -- encoder input ---------------------------------------------------------

    def _encode(self, X, y, rng):
        """Upstream's format_srsd_dataset, over an in-memory array.

        Returns (tensor, shift_x, shift_y) or None when the problem has too few
        strictly-positive rows for the encoder.
        """
        n = self.n_sample_points
        mask = np.all(X > 0.0, axis=1) & np.isfinite(y) & (y != 0.0)
        if int(mask.sum()) < n:
            return None
        Xv, yv = X[mask], y[mask]
        idx = rng.choice(len(Xv), n, replace=False)
        Xs, ys = Xv[idx], yv[idx]

        n_cols = Xs.shape[1]
        grid = np.zeros((n, 7), dtype=np.float64)
        shift_x = np.zeros(n_cols, dtype=np.float64)
        for k in range(n_cols):
            shift_x[k] = float(np.mean(np.log10(Xs[:, k])))
            grid[:, k + 1] = np.power(10.0, np.log10(Xs[:, k]) - shift_x[k])
        shift_y = float(np.mean(np.log10(np.abs(ys))))
        grid[:, 0] = np.power(10.0, np.log10(np.abs(ys)) - shift_y) * np.sign(ys)

        if not np.all(np.isfinite(grid)):
            return None
        t = self.torch.tensor(grid, dtype=self.torch.float32)
        return t.unsqueeze(0).unsqueeze(-1).to(self.device), shift_x, shift_y

    # -- decoder ---------------------------------------------------------------

    def _decode(self, enc_input):
        """Greedy decode until the expression tree closes (evaluate_model.py:63)."""
        torch = self.torch
        from model._utils import is_tree_complete

        with torch.no_grad():
            enc_out = self.model.encoder(enc_input)
            L = self.model.decoder.positional_encoding.seq_length
            out = torch.zeros((enc_input.shape[0], L + 1), dtype=torch.int64,
                              device=self.device)
            out[:, 0] = 1
            done = torch.zeros(enc_input.shape[0], dtype=torch.bool, device=self.device)
            future = torch.triu(torch.ones(L, L, device=self.device), diagonal=1).bool()
            for i in range(L):
                pad = torch.eq(out[:, :-1], 0).unsqueeze(1).unsqueeze(1)
                dec = self.model.decoder(target_seq=out[:, :-1],
                                         mask_dec=torch.logical_or(pad, future),
                                         output_enc=enc_out)
                logits = self.model.last_layer(dec)
                out[:, i + 1] = torch.where(done, torch.zeros_like(done, dtype=torch.int64),
                                            torch.argmax(logits[:, i], axis=-1))
                for b in range(enc_input.shape[0]):
                    if is_tree_complete(out[b, 1:].cpu()):
                        done[b] = True
                if bool(done.all()):
                    break
            return out.cpu()

    def _to_sympy(self, seq):
        from model._utils import translate_integers_into_tokens
        from datasets._utils import from_sequence_to_sympy

        if int(seq[1:].sum()) == 0:
            return None
        toks = translate_integers_into_tokens(seq)
        expr = from_sequence_to_sympy(toks)
        # NB: upstream also calls first_variables_first() here -- a canonical
        # RENAMING of variables used for tree-edit-distance only.  Applying it
        # would break the x_j <-> column j correspondence and silently score the
        # wrong columns, so it is deliberately not used.
        return sympy.sympify(expr)

    # -- public API ------------------------------------------------------------

    def fit(self, X, y, rng):
        """Decode + fit constants.  Returns the formula string, or None on failure.

        With --n-draws > 1 the 50-row encoder subsample is redrawn and the best
        candidate is kept.  Selection is on TRAIN R^2 -- never the reported test
        slice, which would leak.
        """
        self.fitted_ = None
        self.theta_ = None
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_cols = min(X.shape[1], MAX_MODEL_VARS)

        best = None
        for _ in range(max(1, self.n_draws)):
            enc = self._encode(X[:, :n_cols], y, rng)
            if enc is None:
                continue
            enc_input, shift_x, shift_y = enc
            try:
                expr = self._to_sympy(self._decode(enc_input)[0])
                if expr is None:
                    continue
                params = []
                counter = [0]
                expr = _split_constants(expr, counter)
                params = sorted(
                    [s for s in expr.free_symbols if s.name.startswith("c")],
                    key=lambda s: int(s.name[1:]))
                var_syms = [sympy.Symbol(f"x{i + 1}", real=True, positive=True)
                            for i in range(n_cols)]
                fitted = _Fitted(expr, params, var_syms, shift_x, shift_y, n_cols)
                theta, tr_r2 = _fit_constants(fitted, X[:, :n_cols], y, rng,
                                              self.n_restarts)
            except Exception:
                continue
            if best is None or tr_r2 > best[0]:
                best = (tr_r2, fitted, theta)

        if best is None:
            return None
        _, self.fitted_, self.theta_ = best
        return self._formula_string()

    def _formula_string(self):
        """The fitted expression in data space, written over v1..vn."""
        e = self.fitted_.expr
        subs = {p: sympy.Float(float(v)) for p, v in zip(self.fitted_.params,
                                                         self.theta_)}
        # undo the encoder rescaling symbolically so the artifact is directly
        # comparable with every other method's `predicted_formula`
        for i, s in enumerate(self.fitted_.var_syms):
            subs[s] = sympy.Symbol(f"v{i + 1}") / sympy.Float(
                float(10.0 ** self.fitted_.shift_x[i]))
        try:
            out = sympy.sympify(e).xreplace(subs) * sympy.Float(
                float(10.0 ** self.fitted_.shift_y))
            # plain str(): nsimplify() here turns every fitted Float into a huge
            # exact Rational (0.5735... -> 573513537804351/1000000000000000),
            # which is unreadable and breaks downstream sympify round-trips.
            return str(sympy.expand(out))
        except Exception:
            return str(e)

    def predict(self, X):
        if self.fitted_ is None:
            return np.full(len(X), np.nan)
        return self.fitted_(np.asarray(X, dtype=np.float64), self.theta_)


# -- SRBench (Feynman) ----------------------------------------------------------

def _load_ood(path):
    if not path or not os.path.exists(path):
        return {}
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh).get("datasets", {})


def run_srbench(runner, problems, seed, tau, args):
    out_path = os.path.join(
        results_io.noise_dir(
            results_io.seed_dir(os.path.join(args.output, METHOD_NAME), seed), tau),
        SRBENCH_ARTIFACT)

    meta, rows = results_io.load_results(out_path)
    done = {r["dataset"] for r in rows}
    if done:
        print(f"  resuming: {len(done)} datasets already in {out_path}", flush=True)

    ood = _load_ood(args.ood_data)
    import pandas as pd

    for i, (name, n_vars, formula, target_pn) in enumerate(problems, 1):
        if name in done:
            continue
        rng = seed_everything(name, seed)
        try:
            df = pd.read_csv(
                os.path.join(args.pmlb_dir, "datasets", name, f"{name}.tsv.gz"),
                sep="\t", compression="gzip")
        except Exception as exc:
            print(f"  [{i}/{len(problems)}] {name}: load error {exc}", flush=True)
            continue

        if len(df) > args.n_points:
            df = df.iloc[rng.choice(len(df), args.n_points, replace=False)]
            df = df.reset_index(drop=True)
        feat_cols = [c for c in df.columns if c != "target"]
        X = df[feat_cols].values.astype(np.float64)
        y = df["target"].values.astype(np.float64)
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            print(f"  [{i}/{len(problems)}] {name}: non-finite data, skipped", flush=True)
            continue

        n_train = int(0.75 * len(df))
        X_tr, X_te = X[:n_train], X[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]
        y_tr = add_target_noise(y_tr, tau, rng)

        t0 = time.perf_counter()
        pred = runner.fit(X_tr, y_tr, rng)
        dt = time.perf_counter() - t0

        r2 = _clipped_r2(y_te, runner.predict(X_te)) if pred else 0.0
        ood_r2 = None
        if pred and name in ood:
            ood_r2 = _clipped_r2(ood[name]["y"], runner.predict(ood[name]["X"]))

        rows.append({
            "dataset": name, "seed": seed, "round": 1,
            "r2": r2, "ood_r2": ood_r2, "time": dt,
            "predicted_formula": pred, "predicted_formula_raw": pred,
            "feature_idx": list(range(min(X.shape[1], MAX_MODEL_VARS))),
            "target_formula": formula, "target_pn": target_pn,
            "n_vars": n_vars,
        })
        results_io.save_results(
            out_path, rows, model_name=METHOD_NAME, checkpoint=runner.checkpoint,
            target_noise=tau, seed=seed, max_vars=args.max_vars,
            n_draws=args.n_draws, n_restarts=args.n_restarts,
            n_sample_points=args.n_sample_points)
        print(f"  [{i}/{len(problems)}] {name:<28} R^2={r2:.4f} "
              f"{'solved' if r2 >= args.r2_threshold else '      '} "
              f"t={dt:.1f}s | {pred}", flush=True)
    return rows


# -- LLM-SRBench ----------------------------------------------------------------

def run_llmsrbench(runner, problems, seed, tau, args):
    import h5py
    from eval_mymodels import resolve_llmsrbench_file

    method_dir = f"{METHOD_NAME}_{args.split}"
    out_path = os.path.join(
        results_io.llmsr_dir(
            results_io.seed_dir(os.path.join(args.output, METHOD_NAME), seed),
            tau, method_dir),
        results_io.RESULTS_NAME)

    meta, rows = results_io.load_results(out_path)
    done = {r["equation_id"] for r in rows}
    if done:
        print(f"  resuming: {len(done)} problems already in {out_path}", flush=True)

    hdf5_path = resolve_llmsrbench_file("lsr_bench_data.hdf5")
    if args.split == "lsr_transform":
        group_of = lambda n: f"/lsr_transform/{n}"
    else:
        group_of = lambda n: f"/lsr_synth/{args.split[len('lsr_synth_'):]}/{n}"

    with h5py.File(hdf5_path, "r") as fh:
        for i, (name, n_vars, expression, syms) in enumerate(problems, 1):
            if name in done:
                continue
            rng = seed_everything(name, seed)
            try:
                g = fh[group_of(name)]
                samples = {k: g[k][...].astype(np.float64) for k in g.keys()}
            except Exception as exc:
                print(f"  [{i}/{len(problems)}] {name}: load error {exc}", flush=True)
                continue

            train, test = samples.get("train"), samples.get("test")
            if train is None or test is None:
                continue
            # Column 0 is y, columns 1: are X.
            X_tr, y_tr = train[:, 1:], train[:, 0]
            X_te, y_te = test[:, 1:], test[:, 0]
            y_tr = add_target_noise(y_tr, tau, rng)

            t0 = time.perf_counter()
            pred = runner.fit(X_tr, y_tr, rng)
            dt = time.perf_counter() - t0

            r2 = _clipped_r2(y_te, runner.predict(X_te)) if pred else 0.0
            ood = samples.get("ood_test")
            ood_metrics = None
            if pred and ood is not None:
                ood_metrics = {"r2": _clipped_r2(ood[:, 0], runner.predict(ood[:, 1:]))}

            rows.append({
                "equation_id": name, "gt_equation": expression,
                "discovered_equation": pred, "discovered_equation_raw": pred,
                "feature_idx": list(range(min(X_tr.shape[1], MAX_MODEL_VARS))),
                "n_vars": n_vars,
                "num_datapoints": int(len(train)), "num_eval_datapoints": int(len(test)),
                "search_time": dt, "seed": seed,
                "id_metrics": {"r2": r2}, "ood_metrics": ood_metrics,
                "symbols": list(map(str, syms)),
            })
            results_io.save_results(
                out_path, rows, model_name=METHOD_NAME, split=args.split,
                checkpoint=runner.checkpoint, target_noise=tau, seed=seed,
                max_vars=args.max_vars, n_draws=args.n_draws,
                n_restarts=args.n_restarts, n_sample_points=args.n_sample_points)
            print(f"  [{i}/{len(problems)}] {name:<28} R^2={r2:.4f} "
                  f"{'solved' if r2 >= args.r2_threshold else '      '} "
                  f"t={dt:.1f}s | {pred}", flush=True)
    return rows


# -- Reporting ------------------------------------------------------------------

def summarise(tag, rows, thr):
    vals = []
    for r in rows:
        v = r.get("r2")
        if v is None and isinstance(r.get("id_metrics"), dict):
            v = r["id_metrics"].get("r2")
        if v is not None:
            vals.append(float(v))
    a = np.asarray(vals, dtype=np.float64)
    print(f"  {tag}: n={a.size}"
          + (f"  solve={100 * np.mean(a >= thr):.1f}%  meanR2={np.mean(a):.4f}"
             if a.size else ""), flush=True)


# -- Main -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=["srbench", "llmsrbench", "both"],
                    default="both")
    ap.add_argument("--split", default="lsr_transform", help="LLM-SRBench split.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Seeds to run. The canonical sweep is 42..51.")
    ap.add_argument("--target-noise", type=float, nargs="+", default=[0.0],
                    dest="noises", help="Noise level(s) tau. Canonical: 0 0.001 0.01 0.1")
    ap.add_argument("--max-vars", type=int, default=MAX_MODEL_VARS,
                    help=f"Keep equations with <= this many variables. The decoder "
                         f"vocabulary is x1..x{MAX_MODEL_VARS}, so {MAX_MODEL_VARS} "
                         f"is the model's ceiling. 0 disables the filter (wider "
                         f"problems are then truncated to the first "
                         f"{MAX_MODEL_VARS} columns).")
    ap.add_argument("--output", default="results", help="Results root.")
    ap.add_argument("--pmlb-dir", default="datasets/pmlb")
    ap.add_argument("--feynman-csv", default="datasets/feynman/FeynmanEquations.csv")
    ap.add_argument("--ood-data", default=None,
                    help="feynman_ood_g*.pkl.gz to also score the formula on.")
    ap.add_argument("--n-points", type=int, default=10000,
                    help="Rows subsampled per Feynman equation before the 75/25 split.")
    ap.add_argument("--r2-threshold", type=float, default=0.99)
    # Model knobs
    ap.add_argument("--enc-type", default="mix", choices=["mix", "att", "mlp"],
                    help="Which released encoder variant. Upstream's best is mix.")
    ap.add_argument("--no-label-smoothing", action="store_true",
                    help="Use the plain checkpoint instead of *_label_smoothing.")
    ap.add_argument("--weights-dir", default=os.environ.get(
        "TF4SR_WEIGHTS", DEFAULT_WEIGHTS_DIR))
    ap.add_argument("--device", default=None, help="cuda/cpu. Default: cuda if available.")
    ap.add_argument("--n-sample-points", type=int, default=50,
                    help="Rows fed to the encoder. 50 is what the weights were "
                         "trained with; changing it will not load.")
    ap.add_argument("--n-draws", type=int, default=1,
                    help="Encoder subsamples per problem; the best by TRAIN R^2 is "
                         "kept. Upstream's own protocol repeats 30 times.")
    ap.add_argument("--n-restarts", type=int, default=10,
                    help="BFGS restarts when fitting the skeleton constants.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--list-problems", action="store_true")
    args = ap.parse_args()

    do_sr = args.benchmark in ("srbench", "both")
    do_lsr = args.benchmark in ("llmsrbench", "both")

    sr_problems = (feynman_problems(args.pmlb_dir, args.feynman_csv, args.max_vars)
                   if do_sr else [])
    lsr_problems = (llmsrbench_problems(args.split, args.max_vars) if do_lsr else [])

    if args.list_problems:
        for n, k, f, _ in sr_problems:
            print(f"srbench    {n:<28} {k}v  {f}")
        for n, k, e, _ in lsr_problems:
            print(f"llmsrbench {n:<28} {k}v  {e}")
        print(f"\nsrbench: {len(sr_problems)}   llmsrbench: {len(lsr_problems)}")
        return

    sr_problems = shard(sr_problems, args.shard, args.n_shards)
    lsr_problems = shard(lsr_problems, args.shard, args.n_shards)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    runner = Tf4srRunner(
        enc_type=args.enc_type, label_smoothing=not args.no_label_smoothing,
        weights_dir=args.weights_dir, device=device,
        n_sample_points=args.n_sample_points, n_restarts=args.n_restarts,
        n_draws=args.n_draws)

    print(f"tf4sr ({os.path.basename(os.path.dirname(runner.checkpoint))}) | "
          f"max_vars<={args.max_vars} | device={device} | "
          f"shard {args.shard}/{args.n_shards} | "
          f"srbench {len(sr_problems)} + llmsrbench {len(lsr_problems)} problems | "
          f"seeds {args.seeds} | noise {args.noises}", flush=True)

    for seed in args.seeds:
        for tau in args.noises:
            print(f"\n=== seed {seed} | noise {tau:g} ===", flush=True)
            if sr_problems:
                summarise("srbench",
                          run_srbench(runner, sr_problems, seed, tau, args),
                          args.r2_threshold)
            if lsr_problems:
                summarise("llmsrbench",
                          run_llmsrbench(runner, lsr_problems, seed, tau, args),
                          args.r2_threshold)


if __name__ == "__main__":
    main()
