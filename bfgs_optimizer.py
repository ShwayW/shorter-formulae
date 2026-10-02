import time
import warnings
import numpy as np
import torch
from scipy.optimize import minimize
from collections import defaultdict
from grammar import U, B
from utils import is_number

np.seterr(invalid='ignore')


class _TimedFun:
    """Matches the TimedFun in symbolicregression/model/utils_wrapper.py exactly:
    raises ValueError after stop_after seconds; exposes .fun() and .best_x."""
    def __init__(self, fn, stop_after=10):
        self.fun_in = fn
        self.stop_after = stop_after
        self.started = False
        self.best_fun_value = np.inf
        self.best_x = None

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        val = self.fun_in(x, *args)
        if val < self.best_fun_value:
            self.best_fun_value = val
            self.best_x = x.copy()
        return val


# Torch operator maps for the prefix evaluator (matches extEvalPN.cpp exactly)
_UNARY_OPS = {
    '++':     lambda a: a + 1.0,
    '--':     lambda a: a - 1.0,
    'neg':    lambda a: -a,
    'sqr':    lambda a: a * a,
    'sqrt':   lambda a: torch.sqrt(a.clamp(min=0.0)),
    'exp':    torch.exp,
    'ln':     lambda a: torch.log(a.clamp(min=1e-38)),
    'sin':    torch.sin,
    'cos':    torch.cos,
    'tan':    torch.tan,
    'abs':    torch.abs,
    'arcsin': torch.asin,
    'arccos': torch.acos,
    'arctan': torch.atan,
    'invert': lambda a: 1.0 / a,
    'pow2':   lambda a: a * a,
    'pow3':   lambda a: a * a * a,
}
_BINARY_OPS = {
    '+':   lambda a, b: a + b,
    '-':   lambda a, b: a - b,
    '*':   lambda a, b: a * b,
    '/':   lambda a, b: a / b,
    # clamp base to positive so autograd is defined for non-integer exponents
    'pow': lambda a, b: torch.pow(a.abs().clamp(min=1e-38), b),
}
_VAR_IDX = {f'v{i+1}': i for i in range(10)}
_PI = 3.141592653589793


def _eval_prefix_torch(tokens, X, params, param_names):
    """Evaluate a prefix-notation formula using PyTorch ops (enables autograd).

    tokens:      list of strings (operators, variable names, Ci param names,
                 numeric literals, symbolic constants like 'pi')
    X:           (N, D) torch.float64 tensor -- fixed data
    params:      1-D torch.float64 tensor of free constants (may require_grad)
    param_names: list of strings in the same order as params
    Returns:     (N,) torch.float64 tensor
    """
    param_idx = {name: i for i, name in enumerate(param_names)}
    it = iter(tokens)

    def _parse():
        tok = next(it)
        if tok in param_idx:
            return params[param_idx[tok]].expand(X.shape[0])
        if tok in _VAR_IDX:
            return X[:, _VAR_IDX[tok]]
        if tok in _BINARY_OPS:
            a, b = _parse(), _parse()
            return _BINARY_OPS[tok](a, b)
        if tok in _UNARY_OPS:
            return _UNARY_OPS[tok](_parse())
        # numeric literal or symbolic constant
        val = _PI if tok == 'pi' else float(tok)
        return torch.full((X.shape[0],), val, dtype=X.dtype)

    return _parse()


def _pow_exponent_positions(tokens):
    """Return indices of single-constant exponents inside `pow` nodes.

    In prefix notation, `pow` takes two sub-expressions: base then exponent.
    If the exponent is a single terminal token starting with 'C' (a constant
    placeholder), return its index.  These constants are frozen during BFGS:
    even a tiny perturbation away from an integer value causes C++ pow() to
    return inf/nan for negative bases.
    """
    _BINARY = set(B)
    _UNARY  = set(U)
    positions: set = set()

    def parse(pos):
        if pos >= len(tokens):
            return pos
        tok = tokens[pos]
        if tok == 'pow':
            base_end = parse(pos + 1)
            if base_end < len(tokens) and tokens[base_end].startswith('C'):
                positions.add(base_end)
            return parse(base_end)
        if tok in _BINARY:
            return parse(parse(pos + 1))
        if tok in _UNARY:
            return parse(pos + 1)
        return pos + 1

    parse(0)
    return positions


class BFGSOptimizer:
    """BFGS refinement matching the NeurIPS 2022 paper (Kamienny et al.) and
    symbolicregression/model/utils_wrapper.py BFGSRefinement:

    - Refines only the numeric literals already in the E2E-predicted formula,
      initialized from those predicted values (no affine wrappers).
    - Analytical gradients via torch.func.grad (same as the paper's jac).
    - 10-second time limit with best-params recovery (same as paper).
    - MSE/2 loss (same as paper).
    """

    def __init__(self, formula_str):
        self.param_map = defaultdict(list)
        self.initial_guess = []
        cur_const_idx = 1

        fml_tokens = formula_str.split()
        self.tokens = []
        for token in fml_tokens:
            if is_number(token):
                name = f"C{cur_const_idx}"
                self.tokens.append(name)
                self.param_map[name].append(len(self.tokens) - 1)
                self.initial_guess.append(float(token))
                cur_const_idx += 1
            else:
                self.tokens.append(token)

        # Numeric sort so param_names[i] matches initial_guess[i] (appearance order).
        # Lexicographic sort mis-orders C10+ before C2, scrambling the initial guess.
        self.param_names = sorted(self.param_map.keys(), key=lambda s: int(s[1:]))

        # Freeze pow exponents: bake their initial values directly into the token
        # list so BFGS never touches them.  Even a tiny perturbation away from an
        # integer (e.g. 2 -> 2.00001) makes C++ pow(negative_base, 2.00001) = inf.
        _ci_init = {self.param_names[i]: self.initial_guess[i]
                    for i in range(len(self.param_names))}
        for pos in _pow_exponent_positions(self.tokens):
            ci = self.tokens[pos]
            if ci in _ci_init:
                self.tokens[pos] = str(_ci_init[ci])
                del self.param_map[ci]
        self.param_names = sorted(self.param_map.keys(), key=lambda s: int(s[1:]))
        self.initial_guess = [_ci_init[n] for n in self.param_names]
        self.has_constants = len(self.param_names) > 0

    def opt_consts(self, X, y, stop_after=10, n_restarts=8):
        """Run BFGS with analytical gradients, a time limit, and random restarts.

        Runs 1 + n_restarts BFGS calls within the stop_after budget: the first
        from the formula's predicted constant values, then n_restarts from
        log-uniform random starting points.  Returns the formula for the best
        loss found across all runs.

        Falls back to the original if there are no constants or any error occurs.
        """
        original = " ".join(str(t) for t in self.tokens)
        if not self.has_constants:
            return original

        rng = np.random.default_rng()
        x0 = np.array(self.initial_guess, dtype=np.float64)
        n_params = len(x0)
        X_t = torch.tensor(X, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.float64)
        tokens = self.tokens
        param_names = self.param_names
        total_runs = 1 + n_restarts
        per_run_budget = stop_after / total_runs

        def objective_torch(params):
            y_pred = _eval_prefix_torch(tokens, X_t, params, param_names)
            return (y_t - y_pred).pow(2).mean().div(2)

        def objective_numpy(coeffs):
            params = torch.tensor(coeffs, dtype=torch.float64)
            val = objective_torch(params).item()
            # Cap at 1e38 whether the value is non-finite OR merely very large.
            # Large-but-finite objectives (e.g. 1e300) produce enormous finite
            # gradients that nan_to_num cannot detect, corrupting the Hessian.
            return val if (np.isfinite(val) and val < 1e38) else 1e38

        def gradient_numpy(coeffs):
            params = torch.tensor(coeffs, dtype=torch.float64, requires_grad=True)
            g = torch.func.grad(objective_torch)(params)
            g_np = np.nan_to_num(g.detach().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
            # Clip gradient by global norm.  Large-but-finite gradients (from
            # formula predictions like 1e150) are finite so nan_to_num is a
            # no-op on them, but they cause BFGS to build a pk where
            # xk + s*pk overflows float64 itself (overflow in scipy's multiply).
            norm = np.linalg.norm(g_np)
            if norm > 1e6:
                g_np *= 1e6 / norm
            return g_np

        t_start = time.time()
        best_val = np.inf
        best_x = x0.copy()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "overflow encountered",    RuntimeWarning)
            warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
            warnings.filterwarnings("ignore", "The line search algorithm", RuntimeWarning)

            for run_idx in range(total_runs):
                elapsed = time.time() - t_start
                remaining = stop_after - elapsed
                if remaining < 0.1:
                    break

                if run_idx == 0:
                    start = x0.copy()
                else:
                    # Log-uniform random restart: sign * 10^U(-3, 2)
                    signs = rng.choice([-1.0, 1.0], size=n_params)
                    exponents = rng.uniform(-3.0, 2.0, size=n_params)
                    start = signs * np.power(10.0, exponents)

                budget = min(per_run_budget, remaining)
                timed = _TimedFun(objective_numpy, stop_after=budget)
                try:
                    minimize(
                        timed.fun,
                        start,
                        method="BFGS",
                        jac=gradient_numpy,
                        options={"disp": False},
                    )
                except ValueError:
                    pass  # time limit hit -- best_x already tracked
                except Exception:
                    continue

                run_best_x = timed.best_x if timed.best_x is not None else start
                run_best_val = timed.best_fun_value
                if run_best_val < best_val:
                    best_val = run_best_val
                    best_x = run_best_x

        const_map = {param_names[i]: str(best_x[i]) for i in range(len(param_names))}
        final_tokens = [const_map.get(t, t) for t in tokens]
        return " ".join(final_tokens)



