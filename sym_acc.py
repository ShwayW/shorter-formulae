#!/usr/bin/env python3
"""
sym_acc.py -- Symbolic accuracy, ported from PhysicsRegression's `cal_sym_acc`.

Upstream: PhysicsRegression/symbolicregression/metrics.py:259 (Meta's E2E
metrics.py with the authors' sympy equivalence checker added on top).

WHAT IT MEASURES
    A prediction counts as correct when it is equivalent to the ground truth
    UP TO A GLOBAL ADDITIVE OR MULTIPLICATIVE CONSTANT:

        simplify(pred - true).is_number   or   simplify(pred / true).is_number

    i.e. `pred = true + c` and `pred = c * true` both score 1.  That leniency is
    deliberate upstream (their decoder emits a skeleton whose constants are fitted
    afterwards) but it is looser than "recovered the equation", so we also report
    a STRICT column -- simplify(pred - true) == 0 -- computed from the same
    intermediate results at no extra cost.  See sym_acc_pair().

    Note this is NOT the `is_symbolic_solution` metric in symbolicregression/
    metrics.py:126: that one is numerical (std of y-residual and y-ratio) and
    requires BOTH to be constant, which collapses to pred == true.

WHAT WE CHANGED, AND WHY
  1. No string munging on the way in.  Upstream normalises with raw
     str.replace("add","+"), ("mul","*"), ("inv","1/"), ("pow","**") ... which is
     substring-based and would corrupt our grammar's tokens ("invert" -> "1/ert",
     "pow2" -> "**2").  We convert our Polish-notation formulas to sympy through
     compare_srbench._prefix_to_sympy -- the same converter the complexity metric
     uses -- so the two metrics see identical trees.
  2. No in-place mutation.  Upstream rewrites the caller's `pred_exprs` /
     `true_exprs` list entries as it goes; ours is pure.
  3. Timeouts are reported, not silently scored 0.  Upstream wraps every sympy
     call in @timeout(10) and swallows the exception as `_sym_acc = 0`, which
     biases the rate down on exactly the long formulas we care about.  We return
     an explicit status so callers can separate "not equivalent" from "gave up".
  4. timeout_decorator (upstream's dependency) is not installed here; we use a
     SIGALRM-based equivalent, see _timeout.

The canonicalisation itself -- eps-rounding of numeric atoms, the prefix/infix
round-trip with the sqrt/abs rewrite rules, the eps=3 retry with Abs stripped --
is a faithful port of the upstream algorithm.
"""
import os
import signal
import sys

import sympy as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Reuse the PN -> sympy converter that compare_srbench's complexity metric uses,
# so complexity and symbolic accuracy are computed on the very same tree.
from compare_srbench import _prefix_to_sympy  # noqa: E402

__all__ = [
    "TIMEOUT_SECONDS", "set_timeout", "SymAccResult",
    "formula_to_sympy", "sym_acc_pair", "cal_sym_acc",
]

# Upstream's @timeout(10) on every sympy call.  Read at CALL time, not at import
# time, so callers can raise it: long predictions (our *_unscale models emit them)
# blow the 10s budget on ~6% of rows, and every one of those is scored 0.
TIMEOUT_SECONDS = 10


def set_timeout(seconds: int) -> None:
    """Set the per-sympy-call budget.  Call before forking workers."""
    global TIMEOUT_SECONDS
    TIMEOUT_SECONDS = int(seconds)


class _Timeout(Exception):
    pass


def _timeout(fn):
    """SIGALRM-based stand-in for upstream's timeout_decorator.timeout.

    Only arms the alarm when we are on the main thread of a process with SIGALRM
    (true for the CLI and for multiprocessing workers, false inside a thread
    pool); elsewhere it degrades to running the call unguarded rather than
    raising, so a threaded caller still gets an answer.
    """
    def wrapped(*args, **kwargs):
        try:
            prev = signal.signal(signal.SIGALRM, _raise_timeout)
        except (ValueError, AttributeError):
            return fn(*args, **kwargs)              # not on the main thread
        signal.setitimer(signal.ITIMER_REAL, TIMEOUT_SECONDS)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev)
    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapped


def _raise_timeout(signum, frame):
    raise _Timeout("Timed Out")


# -- Upstream helpers (metrics.py:163-257), ported -----------------------------

@_timeout
def _parse(expr, local_dic=None):
    return sp.parse_expr(expr, local_dict=local_dic)


@_timeout
def _simplify(expr):
    return sp.simplify(expr)


@_timeout
def value_approximate(expr, eps=6):
    """Round every numeric atom to `eps` decimals (snap to an integer when within
    10^-eps of one).  Upstream returns the literal 0 on failure; we keep that so
    the downstream "pred == 0 -> score 0" guard still fires."""
    if expr == sp.nan:
        return sp.Integer(0)
    try:
        v = expr.xreplace({
            n: round(n, eps)
            if abs(n - round(n)) > 10 ** (-eps)
            else round(n)
            for n in expr.atoms(sp.Number)
        })
    except Exception:
        return sp.Integer(0)
    if v == sp.nan:
        return sp.Integer(0)
    return v


@_timeout
def reduce_abs(expr):
    return expr.replace(sp.Abs, lambda x: x)


@_timeout
def expr_to_prefix(expr):
    """sympy expr -> comma-separated prefix string.

    The rewrite rules below are upstream's hand-rolled canonicaliser: sympy leaves
    sqrt(x**2) as Abs(x)-flavoured forms that block `simplify(p/t).is_number`, so
    the round-trip through prefix and back flattens them.  Ported verbatim.
    """
    if isinstance(expr, str):
        expr = _parse(expr)
    if expr.is_Atom:
        return str(expr)
    operator_map = {sp.Add: "+", sp.Mul: "*", sp.Pow: "**"}
    operator = operator_map.get(expr.func, str(expr.func))

    # sqrt(x**2) = x
    if operator == "**" and expr.args[0].func == sp.Pow:
        operands = expr_to_prefix(expr.args[0].args[0])
        coefficient = expr.args[1] * expr.args[0].args[1]
        return f"**,{operands},{coefficient}"

    # sqrt(exp(-2*x**2)) = exp(0.5 * -2*x**2)
    elif operator == "**" and expr.args[-1] == 0.5 and expr.args[0].func == sp.exp:
        operands = expr_to_prefix(expr.args[0].args[0])
        return f"exp,*,0.5,{operands}"

    # sqrt(abs(exp(-2*x**2))) = exp(0.5 * -2*x**2)
    elif (operator == "**" and expr.args[-1] == 0.5
          and expr.args[0].func == sp.Abs and expr.args[0].args[0].func == sp.exp):
        operands = expr_to_prefix(expr.args[0].args[0].args[0])
        return f"exp,*,0.5,{operands}"

    # sqrt(x**2/y**2) = x/y
    elif operator == "**" and expr.args[-1] == 0.5 and expr.args[0].func == sp.Mul:
        operands = [expr_to_prefix(sp.sqrt(arg)) for arg in expr.args[0].args]
        return f"{','.join(['*'] * max(1, (len(operands) - 1)))},{','.join(operands)}"

    # abs(sqrt(x)) = sqrt(x)
    elif operator == "Abs" and expr.args[0].func == sp.Pow:
        return expr_to_prefix(expr.args[0])

    # sqrt(exp(x)) = exp(re(x))
    elif expr.func == sp.re:
        return expr_to_prefix(expr.args[0])

    elif operator == "**" and expr.args[-1] == 0.5:
        operands = [expr_to_prefix(sp.Abs(expr.args[0])), "0.5"]
        return f"{operator},{','.join(operands)}"

    operands = [expr_to_prefix(arg) for arg in expr.args]
    return f"{','.join([operator] * max(1, (len(operands) - 1)))},{','.join(operands)}"


_PREFIX_BINARY = ("+", "*", "**", "-", "/")
_PREFIX_UNARY = ("sin", "cos", "sqrt", "tan", "log", "arcsin", "arccos", "arctan",
                 "inv", "neg", "exp", "Abs", "abs", "sinh", "cosh", "tanh")

# Atoms prefix_to_infix must NOT try to apply as a function.
_PREFIX_ATOMS = {"pi", "E", "I", "oo", "zoo", "nan"}


def _sympy_arity(token: str):
    """Arity of `token` if it names a sympy function, else None.

    Upstream's expr_to_prefix falls back to str(expr.func) for any operator it has
    no symbol for, which emits SYMPY's spelling -- `asin`, `acos`, `atan`, `sign`,
    `floor` ... -- while prefix_to_infix only recognises the `arcsin` spelling.
    The unmatched token was then read back as a bare atom, so `parse` returned the
    function CLASS and the next arithmetic op raised TypeError, which upstream's
    bare `except` scores as "not equivalent".  On our Feynman results that silently
    zeroed 37/990 rows, every one of them an arcsin/arctan/tanh equation.
    Resolving the token against sympy instead of a fixed list closes the round-trip
    for the whole family.
    """
    if token in _PREFIX_ATOMS or not token[:1].isalpha():
        return None
    fn = getattr(sp, token, None)
    if fn is None or not isinstance(fn, sp.FunctionClass):
        return None
    try:
        nargs = fn.nargs
        return min(int(n) for n in nargs) if nargs else 1
    except (TypeError, ValueError):
        return 1


@_timeout
def prefix_to_infix(prefix):
    tokens = prefix.split(",")

    def _dfs(toks):
        token = toks.pop(0)
        if token in _PREFIX_BINARY:
            left = _dfs(toks)
            right = _dfs(toks)
            return f"(({left}) {token} ({right}))"
        elif token in _PREFIX_UNARY:
            operand = _dfs(toks)
            return f"{token}({operand})"
        arity = _sympy_arity(token)
        if arity:
            args = ", ".join(_dfs(toks) for _ in range(arity))
            return f"{token}({args})"
        return token

    return _dfs(tokens)


# -- Input conversion (our formulas -> sympy) ----------------------------------

def formula_to_sympy(formula):
    """Our Polish-notation formula (grammar tokens, space separated) -> sympy expr.

    Falls back to sympify for infix strings (eval_e2e.py output, LLM-SRBench
    ground truth).  Returns None when the formula is missing or unparseable --
    callers score that as 0, matching upstream's treatment of a NaN prediction.
    """
    if formula is None:
        return None
    s = str(formula).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    toks = s.split()
    try:
        expr, n = _prefix_to_sympy(toks)
        if n == len(toks):
            return expr
    except (ValueError, RecursionError, IndexError):
        pass
    try:
        return sp.sympify(s)
    except Exception:
        return None


def _canonicalise(expr, eps=5, rounds=2):
    """Upstream's normalisation: `rounds` passes of
    value_approximate -> expr_to_prefix -> prefix_to_infix -> parse.

    Upstream applies two passes to the prediction and one to the target; we do
    the same number of passes for both so the comparison is symmetric (an
    asymmetry there can only ever cost recall, never add it).
    """
    out = value_approximate(expr, eps=eps)
    for _ in range(rounds):
        out = _parse(prefix_to_infix(expr_to_prefix(out)))
        out = value_approximate(out, eps=eps)
    return out


# -- The metric ----------------------------------------------------------------

class SymAccResult:
    """Outcome for one (prediction, target) pair.

    acc        -- 1/0, upstream's criterion (equivalent up to +c or *c)
    strict     -- 1/0, pred - true simplifies to exactly 0
    status     -- "ok" | "unparseable" | "timeout" | "error"
    """
    __slots__ = ("acc", "strict", "status", "detail")

    def __init__(self, acc, strict, status, detail=""):
        self.acc = acc
        self.strict = strict
        self.status = status
        self.detail = detail

    def __repr__(self):
        return (f"SymAccResult(acc={self.acc}, strict={self.strict}, "
                f"status={self.status!r})")


def sym_acc_pair(pred, true):
    """Symbolic accuracy for a single pair.  `pred`/`true` are formula strings
    (PN or infix) or sympy expressions.  Returns a SymAccResult.

    Mirrors metrics.py:290-351: first-pass check on the eps=5 canonical forms,
    then -- only if that fails -- upstream's eps=3 retry with a further
    prefix/infix round-trip and Abs stripped.
    """
    p_expr = pred if isinstance(pred, sp.Basic) else formula_to_sympy(pred)
    t_expr = true if isinstance(true, sp.Basic) else formula_to_sympy(true)

    if t_expr is None:
        return SymAccResult(float("nan"), float("nan"), "unparseable", "target")
    if p_expr is None:
        return SymAccResult(0, 0, "unparseable", "prediction")

    try:
        p = _canonicalise(p_expr, eps=5)
        t = _canonicalise(t_expr, eps=5)

        # Upstream scores a prediction that collapsed to the literal 0 as wrong
        # (metrics.py:292) -- value_approximate returns 0 on any failure, so this
        # is really "the prediction did not survive normalisation".
        if p == 0:
            return SymAccResult(0, 0, "ok", "pred collapsed to 0")

        d1 = _simplify(p - t).evalf()
        d2 = _simplify(p / t).evalf()

        d1_s = _simplify(_parse(str(value_approximate(d1, 3))))
        d2_s = _simplify(_parse(str(value_approximate(d2, 3))))

        strict = int(bool(d1_s == 0))
        acc = int(d1_s.is_number or d2_s.is_number)
        if d2_s == 0:
            # pred is identically 0: p/t == 0 is "constant" but meaningless.
            acc = 0
        elif acc == 0:
            # Upstream retry: another round-trip at eps=3, then drop Abs.
            d1_r = _parse(str(value_approximate(
                _parse(prefix_to_infix(expr_to_prefix(d1_s))), eps=3)))
            d2_r = _parse(str(value_approximate(
                _parse(prefix_to_infix(expr_to_prefix(d2_s))), eps=3)))
            d1_r = _simplify(reduce_abs(_simplify(d1_r)))
            d2_r = _simplify(reduce_abs(_simplify(d2_r)))
            if d2_r == 0:
                acc = 0
            else:
                acc = int(d1_r.is_number or d2_r.is_number)
            strict = max(strict, int(bool(d1_r == 0)))

        return SymAccResult(acc, strict if acc else 0, "ok")

    except _Timeout:
        # Upstream scores this 0 and moves on; we flag it so the caller can
        # report how much of the "wrong" bucket is really "unknown".
        return SymAccResult(0, 0, "timeout")
    except (RecursionError, MemoryError) as e:
        return SymAccResult(0, 0, "error", type(e).__name__)
    except Exception as e:                     # sympy raises a wide variety here
        return SymAccResult(0, 0, "error", f"{type(e).__name__}: {e}"[:120])


def cal_sym_acc(pred_exprs, true_exprs, use_tqdm_bar=False, detailed=False):
    """Batch entry point, signature-compatible with upstream's cal_sym_acc.

    Returns a list of 0/1 (with nan where the TARGET was unparseable), or a list
    of SymAccResult when detailed=True.  Unlike upstream this does not mutate
    the input lists.
    """
    if len(pred_exprs) != len(true_exprs):
        raise ValueError(
            f"length mismatch: {len(pred_exprs)} predictions, {len(true_exprs)} targets")

    pbar = None
    if use_tqdm_bar:
        import tqdm
        pbar = tqdm.tqdm(total=len(pred_exprs))

    out = []
    for pred, true in zip(pred_exprs, true_exprs):
        res = sym_acc_pair(pred, true)
        out.append(res if detailed else res.acc)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    return out


if __name__ == "__main__":
    # Smoke test: the cases upstream's own demo CSV exercises, in our grammar.
    cases = [
        # (prediction PN, target PN, expected acc, note)
        ("* v1 sqrt + sqr v2 + sqr v3 sqr v4",
         "* v1 sqrt + sqr v2 + sqr v3 sqr v4", 1, "identical"),
        ("* 3.14159265 * v1 v2", "* pi * v1 v2", 1, "pi vs its decimal"),
        # 1.30451 * 10.08765 = 13.159 = 4*pi**2/3, the ratio is pi -- this is the
        # row that exercises the multiplicative branch in their demo CSV.
        ("/ * 13.15947 * v1 sqr v2 v3", "/ * 4.18879 * v1 sqr v2 v3", 1, "ratio pi"),
        ("+ 5.0 * v1 v2", "* v1 v2", 1, "additive constant (lenient by design)"),
        ("* v1 v2", "+ v1 v2", 0, "genuinely different"),
        ("* v1 sqr v2", "* v1 v2", 0, "wrong power"),
    ]
    print(f"{'acc':>4} {'strict':>7} {'exp':>4} {'status':<12} note")
    ok = True
    for pred, true, expected, note in cases:
        r = sym_acc_pair(pred, true)
        flag = "" if r.acc == expected else "   <-- MISMATCH"
        ok &= (r.acc == expected)
        print(f"{r.acc:>4} {r.strict:>7} {expected:>4} {r.status:<12} {note}{flag}")
    sys.exit(0 if ok else 1)
