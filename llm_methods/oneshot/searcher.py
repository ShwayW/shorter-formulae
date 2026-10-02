"""One-shot symbolic regression: one LLM call per problem, no search loop.

The cheapest possible LLM baseline, and the control the other three methods in
llm_methods/ are missing. LLMSR, LaSR and SGA all wrap the LLM in a search loop -- many
calls per problem, each proposal's constants refit by a numeric optimizer. Here the
model sees a table of input-output pairs exactly once, and there is no loop.

Two modes, which together split that loop's contribution in half:

  mode="direct"    (default) The model writes a closed-form expression *including its
                   numeric constants*, and that expression is the answer. No optimizer
                   at all -- the score is attributable purely to the LLM's prior over
                   formulas plus its ability to read a small table.

  mode="skeleton"  The model writes a *skeleton*: structure with the unknown numeric
                   coefficients left as free symbols (c0, c1, ...), which BFGS then fits
                   to the TRAIN split -- exactly LLMSR's parameter-optimization step
                   (llm_methods/llmsr/searcher.py:eval_spec), run once instead of 1000 times.

So: direct = no structure search, no fitting; skeleton = no structure search, fitting;
LLMSR = both. Whatever skeleton mode gains over direct is what constant fitting alone
buys, and whatever LLMSR gains over skeleton is what the search itself buys.

Interface matches llm_methods/llmsr/searcher.py (BaseSearcher.discover -> [SearchResult])
so eval_oneshot.py scores it with the same metrics and writes the same artifacts.
"""

from __future__ import annotations

import re
import time

import numpy as np
import sympy
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        convert_xor)

from bench.searchers.base import BaseSearcher
from bench.dataclasses import SEDTask, Equation, SearchResult


SYSTEM_PROMPT = (
    "You are an exceptional symbolic regression assistant. You analyse numerical "
    "relationships between variables and infer the closed-form mathematical formula "
    "that generated them. You answer with a formula and nothing else."
)

# Functions the parser accepts. Anything else in the reply makes it unparseable and
# triggers a retry (see OneShotSearcher._parse). numpy-style aliases are included
# because models emit both spellings.
_ALLOWED = {
    "exp": sympy.exp, "log": sympy.log, "ln": sympy.log, "log10": lambda x: sympy.log(x, 10),
    "sqrt": sympy.sqrt, "cbrt": sympy.cbrt,
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
    "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan, "atan2": sympy.atan2,
    "arcsin": sympy.asin, "arccos": sympy.acos, "arctan": sympy.atan,
    "sinh": sympy.sinh, "cosh": sympy.cosh, "tanh": sympy.tanh,
    "asinh": sympy.asinh, "acosh": sympy.acosh, "atanh": sympy.atanh,
    "abs": sympy.Abs, "Abs": sympy.Abs, "sign": sympy.sign,
    "pi": sympy.pi, "Pi": sympy.pi, "PI": sympy.pi, "E": sympy.E,
}

# Cap on fitted constants in a skeleton, matching LLMSR's MAX_NPARAMS. A reply asking
# for more than this is rejected: past ~10 free parameters BFGS from a single start is
# not fitting a skeleton any more, it is curve-fitting noise.
MAX_PARAMS = 10

_ALLOWED_OPS_TEXT = ("+, -, *, /, ** (power), exp, log, sqrt, sin, cos, tan, "
                     "arcsin, arccos, arctan, sinh, cosh, tanh, abs, and the constant pi")


def _fmt_table(samples: np.ndarray, var_names: list, out_name: str,
               n_rows: int, rng: np.random.Generator, sig: int = 6) -> str:
    """Render up to `n_rows` observations as a fixed-width table (inputs then output).

    Rows are drawn without replacement from `samples` so the prompt sees the whole
    input range rather than whatever the first rows happen to cover."""
    n = samples.shape[0]
    idx = np.arange(n) if n <= n_rows else np.sort(rng.choice(n, size=n_rows, replace=False))
    cols = var_names + [out_name]
    widths = [max(len(c), sig + 7) for c in cols]
    lines = ["  ".join(c.rjust(w) for c, w in zip(cols, widths))]
    for i in idx:
        row = list(samples[i, 1:]) + [samples[i, 0]]
        lines.append("  ".join(f"{v:.{sig}g}".rjust(w) for v, w in zip(row, widths)))
    return "\n".join(lines)


def build_prompt(var_names: list, var_descs: list, out_name: str, out_desc: str,
                 samples: np.ndarray, n_rows: int, rng: np.random.Generator,
                 mode: str = "direct") -> str:
    """The whole method, in one message: variable semantics + a data table + a format rule.

    Only the constant rule differs between modes: "direct" demands explicit numbers
    because nothing downstream will fit them, "skeleton" demands the opposite -- named
    placeholders and no guessed values -- because BFGS fits them afterwards. Asking a
    skeleton-mode model for numbers anyway would waste its effort on digits the fitter
    immediately overwrites."""
    var_lines = "\n".join(f"  {n}: {d}" for n, d in zip(var_names, var_descs))
    _example = ("c0*x1*exp(-c1*x2) + c2" if mode == "skeleton"
                else "2.5*x1*exp(-0.3*x2) + 1")
    if mode == "skeleton":
        const_rule = (
            f"  - Write every unknown numeric coefficient as a placeholder symbol c0, c1, "
            f"c2, ... (at most {MAX_PARAMS}). Their values are fitted to the data afterwards "
            "by a numerical optimizer, so do NOT guess numbers: give the STRUCTURE, e.g. "
            "c0*x1*exp(-c1*x2) + c2.\n"
            "    Structural exponents may stay literal (x1**2, sqrt(x1)).\n"
            "  - Every placeholder must be multiplied/added into the formula somewhere it "
            "can matter; do not write a placeholder you do not need.\n")
    else:
        const_rule = (
            "  - Write every numeric constant explicitly as a number (e.g. 2.5, not C or a).\n"
            "    Do NOT use undetermined parameters -- the formula is used exactly as written.\n")
    return (
        f"Find the mathematical formula for {out_desc} ({out_name}) as a function of "
        f"the following input variables:\n{var_lines}\n\n"
        f"Below are {min(n_rows, samples.shape[0])} observed samples "
        f"(columns: {', '.join(var_names)}, {out_name}):\n\n"
        f"{_fmt_table(samples, var_names, out_name, n_rows, rng)}\n\n"
        "Infer the formula that maps the inputs to the output. It must hold for all the "
        "data above, and it must be physically/mathematically plausible, not a fit of "
        "convenience.\n\n"
        "Rules for your answer:\n"
        f"  - Use only these variables: {', '.join(var_names)}.\n"
        f"  - Use only these operations and functions: {_ALLOWED_OPS_TEXT}.\n"
        f"{const_rule}"
        f"  - Use Python/SymPy infix syntax on a single line, e.g. {_example}.\n"
        f"  - Output ONLY the right-hand side of the expression for {out_name}. No "
        "explanation, no units, no code fences, no '=' sign.\n"
    )


class OneShotSearcher(BaseSearcher):
    """Single LLM call per problem; the returned expression is the final answer.

    Args:
        llm: an object with .complete(system, user) -> str (llm_methods/oneshot/llm.py).
        mode: "direct" (the reply is the answer) or "skeleton" (the reply is structure
            with free constants, fitted to TRAIN by BFGS -- LLMSR's optimizer step, once).
        n_prompt_rows: observations shown in the prompt.
        max_retries: extra calls allowed when a reply does not parse into a valid
            expression over the task's variables. Costs an LLM call, so it is capped
            low; a run that needs many retries is reported via aux["n_llm_calls"].
        num_candidates: >1 asks for that many independent formulas and keeps the one
            with the lowest TRAIN MSE. Default 1 = the pure one-shot protocol; the knob
            exists only to measure what best-of-k buys.
        n_restarts: skeleton mode -- BFGS starts per skeleton (first is all-ones).
        fit_max_rows: skeleton mode -- cap on TRAIN rows used by the fit (not by scoring,
            which always uses the full TEST/OOD splits). LLM-SRBench ships 80k train rows
            and BFGS evaluates the objective thousands of times, so fitting on all of them
            costs minutes per problem for constants that converge on a few thousand.
            None = use every row (strict parity with LLMSR's optimizer).
        fit_tol: stop restarting once a fit reaches this MSE (it is already exact).
        seed: seeds the prompt's row subsample, the fit subsample, and the BFGS restarts.
    """

    def __init__(self, name: str, llm, mode: str = "direct", n_prompt_rows: int = 30,
                 max_retries: int = 2, num_candidates: int = 1, n_restarts: int = 4,
                 fit_max_rows: int | None = 5000, fit_tol: float = 1e-14,
                 seed: int = 0) -> None:
        super().__init__(name)
        if mode not in ("direct", "skeleton"):
            raise ValueError(f"mode must be 'direct' or 'skeleton'; got {mode!r}")
        self.llm = llm
        self.mode = mode
        self.n_prompt_rows = n_prompt_rows
        self.max_retries = max_retries
        self.num_candidates = max(1, int(num_candidates))
        self.n_restarts = max(1, int(n_restarts))
        self.fit_max_rows = fit_max_rows
        self.fit_tol = fit_tol
        self.seed = seed

    # -- reply -> sympy -------------------------------------------------------

    @staticmethod
    def _candidates(text: str):
        """Formula candidates from a raw reply, most likely first.

        Small instruct models pad the answer despite the format rule ("Sure! The
        formula is:\n y = ...\n This fits all points."), so we scan lines bottom-up
        -- the answer is usually the last mathematical line -- after stripping the
        usual wrapping (code fences, LaTeX, a "y =" prefix).
        """
        t = (text or "").strip()
        t = re.sub(r"```[a-zA-Z]*\n?", "", t).replace("```", "")
        t = t.replace("\\cdot", "*").replace("\\times", "*").replace("$", "")
        t = t.replace("\\left", "").replace("\\right", "").replace("^", "**")
        out = []
        for line in reversed([ln.strip() for ln in t.splitlines() if ln.strip()]):
            # Drop a leading "y =" / "f(x) =" (but not a "==" comparison).
            line = re.sub(r"^[A-Za-z_][\w\\{}\(\), ]*\s*=(?!=)\s*", "", line).strip()
            line = line.rstrip(".;, ")
            if line:
                out.append(line)
        return out

    def _parse(self, text: str, var_names: list, allow_params: bool = False):
        """Return (expr, params) or (None, raw_first_candidate) if the reply is unusable.

        `params` is the sorted list of free symbols that are not task variables, i.e. the
        constants BFGS will fit; it is always empty when allow_params is False.

        Rejects anything we cannot use: unknown function names, non-finite constants, and
        -- in direct mode -- undetermined parameters, since nothing downstream would fill
        them in. Skeleton mode instead accepts any non-variable symbol as a parameter
        (models write `A*exp(-B*x)` as readily as `c0*exp(-c1*x)`) up to MAX_PARAMS.
        Parsing runs with an empty global namespace so only the whitelisted functions in
        _ALLOWED resolve -- a model that answers `Heaviside(x)` or `gamma(x)` is rejected
        rather than silently scored on a function the benchmark's grammar does not have.
        """
        syms = {v: sympy.Symbol(v, real=True) for v in var_names}
        local = dict(_ALLOWED)
        local.update(syms)
        # parse_expr's auto_number/auto_symbol transformations emit these constructors
        # into the global namespace, so they have to be present even though nothing else is.
        glob = {"Symbol": sympy.Symbol, "Integer": sympy.Integer, "Float": sympy.Float,
                "Rational": sympy.Rational, "Function": sympy.Function}
        cands = self._candidates(text)
        for cleaned in cands:
            try:
                expr = parse_expr(cleaned, local_dict=local, global_dict=glob,
                                  transformations=standard_transformations + (convert_xor,),
                                  evaluate=True)
            except Exception:
                continue
            if not isinstance(expr, sympy.Expr):
                continue
            extra = sorted(expr.free_symbols - set(syms.values()), key=str)
            if extra and not allow_params:
                continue                      # undetermined parameters / unknown names
            if len(extra) > MAX_PARAMS:
                continue                      # more free parameters than we will fit
            if expr.atoms(AppliedUndef):
                continue                      # unknown function, e.g. heaviside(x)
            if expr.has(sympy.zoo, sympy.oo, sympy.nan, sympy.I):
                continue
            return expr, extra
        return None, (cands[0] if cands else "")

    @staticmethod
    def _lambdify(expr, var_names: list):
        """sympy expr -> f(X) over an (N, D) array, NaN wherever it cannot be evaluated.

        `expr` must be parameter-free: in skeleton mode the fitted values are substituted
        in before this is called, so the scored equation is always a concrete formula."""
        syms = [sympy.Symbol(v, real=True) for v in var_names]
        f = sympy.lambdify(syms, expr, modules=["numpy"])

        def fn(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=np.float64)
            n = X.shape[0]
            try:
                with np.errstate(all="ignore"):
                    out = f(*[X[:, i] for i in range(len(syms))])
                out = np.asarray(out)
                if np.iscomplexobj(out):        # log/sqrt of a negative -> drop those rows
                    out = np.where(np.abs(out.imag) < 1e-12, out.real, np.nan)
                out = np.asarray(out, dtype=np.float64)
                if out.ndim == 0:               # constant expression
                    out = np.full(n, float(out))
                return out.reshape(-1)
            except Exception:
                return np.full(n, np.nan)
        return fn

    def _fit_constants(self, expr, params: list, var_names: list,
                       X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        """Fit `params` in `expr` to (X, y) with BFGS; return (fitted_expr, mse, theta).

        Same optimizer and objective as LLMSR's parameter-optimization step
        (llm_methods/llmsr/searcher.py:eval_spec -- scipy `minimize(..., method="BFGS")` on
        the mean squared error, started from all-ones). The one addition is restarts: a
        search loop that fits 1000 skeletons can afford one start per skeleton because a
        bad basin just costs that proposal its rank, but a single-call method has one
        skeleton and no second chance, so an unlucky start would be scored as if the
        structure were wrong. Restarts after the first are random and seeded, so the run
        stays reproducible.

        Non-finite objectives are mapped to a large finite penalty rather than inf: BFGS
        estimates its gradient by finite differences, and inf makes that gradient NaN,
        which ends the fit on the spot instead of steering back to the feasible region."""
        from scipy.optimize import minimize

        f = sympy.lambdify([sympy.Symbol(v, real=True) for v in var_names] + params,
                           expr, modules=["numpy"])
        cols = [X[:, i] for i in range(X.shape[1])]
        n = len(y)
        PENALTY = 1e30

        def objective(theta):
            try:
                with np.errstate(all="ignore"):
                    pred = f(*cols, *theta)
                pred = np.asarray(pred, dtype=np.float64)
                if pred.ndim == 0:
                    pred = np.full(n, float(pred))
                resid = pred - y
                if not np.all(np.isfinite(resid)):
                    return PENALTY
                return float(np.mean(resid ** 2))
            except Exception:
                return PENALTY

        k = len(params)
        # First start is all-ones (LLMSR's initialisation); the rest are random.
        starts = [np.ones(k)] + [rng.normal(0.0, 2.0, size=k)
                                 for _ in range(max(0, self.n_restarts - 1))]
        best_mse, best_theta = np.inf, np.ones(k)
        for x0 in starts:
            try:
                res = minimize(objective, x0, method="BFGS")
                mse, theta = float(res.fun), np.asarray(res.x, dtype=np.float64)
            except Exception:
                continue
            if np.isfinite(mse) and mse < best_mse and np.all(np.isfinite(theta)):
                best_mse, best_theta = mse, theta
            if best_mse <= self.fit_tol:      # already exact; more restarts cannot help
                break

        if not np.isfinite(best_mse) or best_mse >= PENALTY:
            return None, float("inf"), best_theta
        fitted = expr.subs({p: sympy.Float(v) for p, v in zip(params, best_theta)})
        return fitted, best_mse, best_theta

    # -- search ---------------------------------------------------------------

    def discover(self, task: SEDTask):
        info = vars(task)
        samples = task.samples
        out_name, *var_names = [str(s) for s in info["symbols"]]
        descs = list(info["symbol_descs"])
        out_desc, var_descs = descs[0], descs[1:]
        # Names go into a sympy expression, so they must be plain identifiers.
        out_name = re.sub(r"\W", "_", out_name) or "y"
        var_names = [re.sub(r"\W", "_", v) or f"x{i}" for i, v in enumerate(var_names)]

        rng = np.random.default_rng(self.seed)
        prompt = build_prompt(var_names, var_descs, out_name, out_desc,
                              samples, self.n_prompt_rows, rng, mode=self.mode)

        X_train, y_train = samples[:, 1:], samples[:, 0]
        # Subsample once per problem, so every candidate is fitted on the same rows and
        # their train MSEs stay comparable.
        X_fit, y_fit = X_train, y_train
        if self.fit_max_rows is not None and len(y_train) > self.fit_max_rows:
            sub = rng.choice(len(y_train), size=self.fit_max_rows, replace=False)
            X_fit, y_fit = X_train[sub], y_train[sub]

        skeleton = (self.mode == "skeleton")
        candidates, n_calls, n_parsed, n_fitted, fit_time, raw_last = [], 0, 0, 0, 0.0, ""
        for _ in range(self.num_candidates):
            expr, params = None, []
            for _attempt in range(self.max_retries + 1):
                try:
                    raw = self.llm.complete(SYSTEM_PROMPT, prompt)
                except Exception as e:
                    raw = ""
                    print(f"  [oneshot] LLM call failed: {e}", flush=True)
                n_calls += 1
                raw_last = raw
                expr, params = self._parse(raw, var_names, allow_params=skeleton)
                if expr is not None:
                    break
            if expr is None:
                continue
            n_parsed += 1

            skeleton_str = str(expr)
            if skeleton and params:
                t0 = time.perf_counter()
                fitted, mse, _theta = self._fit_constants(expr, params, var_names,
                                                          X_fit, y_fit, rng)
                fit_time += time.perf_counter() - t0
                if fitted is None:
                    continue              # BFGS never found a finite objective
                expr = fitted
                n_fitted += 1
            else:
                # direct mode, or a skeleton the model wrote with no free constants:
                # nothing to fit, so score the expression as written.
                fn = self._lambdify(expr, var_names)
                with np.errstate(all="ignore"):
                    pred = fn(X_fit)
                m = np.isfinite(pred) & np.isfinite(y_fit)
                mse = (float(np.mean((pred[m] - y_fit[m]) ** 2))
                       if m.sum() >= 0.5 * len(y_fit) else float("inf"))

            fn = self._lambdify(expr, var_names)
            candidates.append((mse, expr, str(expr), fn, skeleton_str, len(params)))

        if not candidates:
            # Nothing usable: report the failure rather than a silent constant. The two
            # ways to get here are distinguished, because they mean different things --
            # an unparseable reply is a prompt/model problem, a fit that never found a
            # finite objective is a skeleton that cannot represent the data at all.
            return [SearchResult(
                equation=Equation(symbols=info["symbols"], symbol_descs=info["symbol_descs"],
                                  symbol_properties=info["symbol_properties"],
                                  expression=None, sympy_format=None, lambda_format=None),
                aux={"n_llm_calls": n_calls, "mode": self.mode,
                     "parse_failed": n_parsed == 0, "fit_failed": n_parsed > 0,
                     "raw_response": (raw_last or "")[:2000]},
            )]

        mse, expr, expr_str, fn, skeleton_str, n_params = min(candidates, key=lambda c: c[0])
        return [SearchResult(
            equation=Equation(symbols=info["symbols"], symbol_descs=info["symbol_descs"],
                              symbol_properties=info["symbol_properties"],
                              expression=expr_str, sympy_format=expr, lambda_format=fn),
            aux={"n_llm_calls": n_calls, "parse_failed": False, "fit_failed": False,
                 "mode": self.mode,
                 "train_mse": mse, "num_candidates": len(candidates),
                 "skeleton": skeleton_str, "n_params": n_params,
                 "n_fitted": n_fitted, "fit_time": fit_time},
        )]
