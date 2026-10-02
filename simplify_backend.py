"""simplify_backend.py -- which canonicaliser the online data generator runs.

Every formula the trainer sees is canonicalised before it becomes a target, and the
choice of canonicaliser is a property of the TARGET DISTRIBUTION, not an implementation
detail: it decides which of several equivalent spellings the decoder is asked to
produce.  This module makes that choice a knob.

    ours      simplifyFormula.simplify -- a constructive normal form.  One structural
              recursion into a fixed algebra (sum-of-products over the grammar), no
              rules and no search.  105 us/formula.
    simplipy  SimpliPy (Saegert & Koethe) -- a rule set mined offline and applied
              online by hash lookup + subtree matching, iterated to a fixpoint.  It
              speaks a different language, so a formula makes a round trip:
              our binary PN -> their 23-operator vocabulary -> simplify -> our PN.
              ~250 us/formula including both translations, i.e. 2.4x slower than ours.

Measured over 20,000 draws from the production sampler (plot_operator_weights.py), the
two agree closely on the operator distribution they produce -- 16 of 18 unary tokens
within 0.5 percentage points -- diverging only on `invert` (ours 6.6% vs SimpliPy 9.9%,
SimpliPy preferring `inv` where ours writes a `/`) and on the +/- split (ours 24.5/23.4
vs SimpliPy 27.8/20.5, SimpliPy's AC add-bag absorbing subtractions ours keeps
explicit).  They nevertheless disagree on the CANONICAL FORM of ~68% of individual
formulae, which is why this is worth an ablation and not a free swap.

Usage.  The backend is a process-level setting, exactly like
grammar.set_operator_weights, because it is read on every draw:

    import simplify_backend
    simplify_backend.set_backend("simplipy")     # once per process / DataLoader worker
    pn = simplify_backend.simplify(raw_pn)

data_process.online_batch_iterater calls set_backend for you (it runs inside each
worker); train_transformer.py exposes it as the "simplify_backend" config knob and the
SIMPLIFY_BACKEND environment override.

Failure contract, identical to simplifyFormula.simplify's: this is called on every
generated draw, so it must NEVER raise into the training loop.  Anything that cannot
be canonicalised -- an undefined subtree, a formula SimpliPy folds to a pole it has a
spelling for and we do not (float("inf"), float("nan")) -- comes back UNCHANGED.  That
is not a special case for SimpliPy: ours already returns '/ v1 0' unchanged for the
same reason.  It affects ~5% of draws under the simplipy backend (the pole rate on our
corpus); `stats()` reports the running counts so a run can be audited.  Note that we
deliberately do NOT fall back to the other backend on failure -- silently mixing
canonicalisers would defeat the point of the flag.

SimpliPy is an optional dependency (MIT, `pip install simplipy`).  It is imported
lazily, so nothing here costs anything under the default "ours" backend.
"""
import os
from fractions import Fraction

from grammar import V
from simplifyFormula import simplify as _simplify_ours, rational_to_pn

BACKENDS = ("ours", "simplipy")

_backend = "ours"
_mode = "lossy"          # SOUND refuses x/x -> 1, which our simplify does unconditionally
_engine_name = "acj-4-3-llm"
_engine = None           # lazily loaded, per process
_stats = {"calls": 0, "unchanged": 0}

_V_SET = set(V)


# =====================================================================================
# our PN <-> SimpliPy PN
#
# Ours (grammar.U/B):  ++ -- neg sqr sqrt exp ln sin cos tan abs arcsin arccos arctan
#                      invert pow2 pow3 tanh   |   + - * / pow
# Theirs:              + - * / abs acos acosh asin asinh atan atanh cos cosh exp inv
#                      log neg pow rootn sin sinh tan tanh
#
# NOTE: the forward table is duplicated in compare_simplipy.py on purpose.  That script
# re-executes itself in a bare venv (--simplipy-python) where neither grammar nor
# simplifyFormula can be imported, so it cannot import this module.  Keep the two in
# sync if the grammar's operator set ever changes.
# =====================================================================================
_TO_SP_UNARY = {
    "neg": "neg", "abs": "abs", "exp": "exp", "ln": "log", "sin": "sin", "cos": "cos",
    "tan": "tan", "tanh": "tanh", "arcsin": "asin", "arccos": "acos", "arctan": "atan",
    "invert": "inv",
}
# unary in ours, binary-with-literal in theirs
_TO_SP_EXPAND = {"sqr": ("pow", "2"), "pow2": ("pow", "2"), "pow3": ("pow", "3"),
                 "sqrt": ("rootn", "2")}
_BINARY = {"+", "-", "*", "/", "pow"}


def to_simplipy(toks, i=0):
    """Our PN token list -> SimpliPy PN token list.  Returns (tokens, next_index)."""
    t = toks[i]
    if t in _BINARY:
        a, i = to_simplipy(toks, i + 1)
        b, i = to_simplipy(toks, i)
        return [t] + a + b, i
    if t == "++":
        a, i = to_simplipy(toks, i + 1)
        return ["+"] + a + ["1"], i
    if t == "--":
        a, i = to_simplipy(toks, i + 1)
        return ["-"] + a + ["1"], i
    if t in _TO_SP_EXPAND:
        op, lit = _TO_SP_EXPAND[t]
        a, i = to_simplipy(toks, i + 1)
        return [op] + a + [lit], i
    if t in _TO_SP_UNARY:
        a, i = to_simplipy(toks, i + 1)
        return [_TO_SP_UNARY[t]] + a, i
    return [t], i + 1                                   # variable or numeric literal


_FROM_SP_UNARY = {"neg": "neg", "abs": "abs", "inv": "invert", "exp": "exp", "log": "ln",
                  "sin": "sin", "cos": "cos", "tan": "tan", "asin": "arcsin",
                  "acos": "arccos", "atan": "arctan", "tanh": "tanh"}
_FROM_SP_LEAF = {"pi": ["pi"], "np.pi": ["pi"], "np.e": ["exp", "1"], "CONST": ["CONST"]}
_LIT_LIMIT = 1000               # beyond this rational_to_pn emits an absurdly long chain

# Peephole tables, keyed by the ALREADY-BACK-TRANSLATED exponent, i.e. by what
# rational_to_pn writes for it: 4 -> "sqr 2", 1/2 -> "invert 2", 1/4 -> "invert sqr 2".
# Without them SimpliPy's AC form leaves every power spelled as `pow`, and sqr / sqrt /
# pow3 / ++ / -- go dead in the output -- a 17-point swing in the token distribution
# that is pure spelling, not simplification.  See plot_operator_weights.py --no-peephole.
_POW_PEEP = {("2",): ["sqr"], ("3",): ["pow3"], ("sqr", "2"): ["sqr", "sqr"],
             ("invert", "2"): ["sqrt"], ("/", "1", "2"): ["sqrt"],
             ("invert", "sqr", "2"): ["sqrt", "sqrt"], ("/", "3", "2"): ["pow3", "sqrt"]}
_ROOTN_PEEP = {("2",): ["sqrt"], ("sqr", "2"): ["sqrt", "sqrt"]}


class Unrepresentable(Exception):
    """SimpliPy produced something the grammar has no spelling for."""


def from_simplipy(toks, i=0, peephole=True):
    """SimpliPy PN (form='explicit') -> our PN.  Returns (tokens, next_index).

    Raises Unrepresentable for the five hyperbolics beyond tanh (no grammar
    counterpart) and for pole tokens -- float("inf"), float("-inf"), float("nan") --
    which SimpliPy folds to but the grammar cannot write.
    """
    t = toks[i]

    if t in _BINARY:
        a, i = from_simplipy(toks, i + 1, peephole)
        b, i = from_simplipy(toks, i, peephole)
        if peephole:
            if t == "pow":
                chain = _POW_PEEP.get(tuple(b))
                if chain is not None:
                    return [*chain, *a], i
            elif t == "+":
                if b == ["1"]:
                    return ["++", *a], i
                if a == ["1"]:
                    return ["++", *b], i
            elif t == "-" and b == ["1"]:
                return ["--", *a], i
        return [t, *a, *b], i

    if t == "rootn":                                  # rootn(a, k) == a ** (1/k)
        a, i = from_simplipy(toks, i + 1, peephole)
        b, i = from_simplipy(toks, i, peephole)
        chain = _ROOTN_PEEP.get(tuple(b)) if peephole else None
        if chain is not None:
            return [*chain, *a], i
        return ["pow", *a, "invert", *b], i

    if t in _FROM_SP_UNARY:
        a, i = from_simplipy(toks, i + 1, peephole)
        return [_FROM_SP_UNARY[t], *a], i

    if t in _V_SET or t in ("0", "1", "2", "3"):
        return [t], i + 1
    if t in _FROM_SP_LEAF:
        return list(_FROM_SP_LEAF[t]), i + 1
    try:
        f = Fraction(t)                               # float("inf") / nan land here
    except (ValueError, ZeroDivisionError):
        raise Unrepresentable(t)
    if abs(f.numerator) > _LIT_LIMIT or f.denominator > _LIT_LIMIT:
        raise Unrepresentable(t)
    return rational_to_pn(f).split(), i + 1


# =====================================================================================
# the backend switch
# =====================================================================================
def set_backend(name, mode=None, engine=None):
    """Select the canonicaliser for this process.  Call once per DataLoader worker.

    Raises ValueError on an unknown name, and ImportError if "simplipy" is requested
    but the package is not installed -- both at CONFIG time, so a misconfigured run
    fails immediately instead of silently training on the wrong target form.
    """
    global _backend, _mode, _engine_name, _engine
    if name not in BACKENDS:
        raise ValueError(f"unknown simplify backend: {name!r} (use one of {BACKENDS})")
    if mode is not None and mode not in ("sound", "lossy"):
        raise ValueError(f"unknown SimpliPy mode: {mode!r} (use 'sound' or 'lossy')")
    if name != _backend or (engine is not None and engine != _engine_name):
        _engine = None                                # force a reload on next use
    _backend = name
    if mode is not None:
        _mode = mode
    if engine is not None:
        _engine_name = engine
    _stats["calls"] = _stats["unchanged"] = 0
    if name == "simplipy":
        _get_engine()                                 # fail loudly here, not mid-epoch


def get_backend():
    return _backend


def stats():
    """Running counts since the last set_backend: total calls and how many came back
    unchanged because the formula could not be canonicalised."""
    return dict(_stats)


def _get_engine():
    global _engine
    if _engine is None:
        # SimpliPy ships its mined rule set as an HF Hub dataset, so load() resolves the
        # engine name through huggingface_hub.  That put the network on the training hot
        # path: one request per DataLoader worker, repeated every epoch because
        # persistent_workers=False respawns them -- a wall of "unauthenticated requests"
        # warnings, and a live failure path, since this function is NOT inside
        # simplify()'s try/except (a 429 or a node without egress raises out of
        # set_backend and kills the worker).  The assets cache under $HF_HOME once; point
        # the hub at it and forbid requests.  Set either variable yourself to override.
        os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import simplipy                               # optional dependency
        try:
            _engine = simplipy.SimpliPyEngine.load(_engine_name, install=False)
        except Exception:
            # Cold cache (first run on a new cluster): allow exactly one online attempt,
            # then keep the offline default for every worker after it.  Still loud if
            # this fails too -- a misconfigured backend must not train silently.
            os.environ["HF_HUB_OFFLINE"] = "0"
            try:
                _engine = simplipy.SimpliPyEngine.load(_engine_name, install=True)
            finally:
                os.environ["HF_HUB_OFFLINE"] = "1"
        # The Rust core builds its rule index on the first simplify, not in load():
        # measured 21 ms for call #1 against 12 us for every call after it.  set_backend
        # calls us at CONFIG time, so paying it here keeps that 21 ms out of the first
        # formula of the first batch, where it would read as a throughput outlier.
        # Not a correctness requirement -- dropping this line only relocates the cost.
        _engine_simplify(_engine, ["+", "v1", "1"])
    return _engine


# SimpliPy changed its simplify() signature between the release this module was
# written against and 0.14.x, and the new one raises TypeError on the old call:
#   old:  simplify(expr, mode="sound"|"lossy", form="explicit")
#   new:  simplify(expr, *, mode="f64"|"real"|"permissive", max_passes=, effort=)
# `form` is gone because the AC core now always returns explicit binary PN -- which is
# exactly what from_simplipy() already expects, so dropping the kwarg is a no-op on the
# output.  The mode names changed but the SEMANTICS we care about map one-to-one: the
# old LOSSY (performs x/x -> 1, matching our simplify()) is the new f64, and the old
# SOUND (refuses it) is the new real.  Probe once and cache, so an older SimpliPy in
# some other env keeps working unchanged rather than being locked out by a hard pin.
_NEW_MODE = {"lossy": "f64", "sound": "real"}
_call_style = None            # None = not yet probed, "legacy" or "modern"


def _engine_simplify(engine, sp):
    global _call_style
    if _call_style != "modern":
        try:
            out = engine.simplify(sp, mode=_mode, form="explicit")
            _call_style = "legacy"
            return out
        except TypeError:
            if _call_style == "legacy":
                raise                                 # a real error, not the signature
            _call_style = "modern"
    return engine.simplify(sp, mode=_NEW_MODE.get(_mode, _mode))


def _simplify_simplipy(pn):
    toks = pn.split()
    if not toks:
        return pn
    sp, _ = to_simplipy(toks)
    out = _engine_simplify(_get_engine(), sp)
    back, j = from_simplipy(list(out))
    if j != len(out):                                 # trailing tokens: malformed
        return pn
    return " ".join(back)


def simplify(pn):
    """Canonicalise a PN formula string under the active backend.

    Returns the input unchanged on any failure -- see the module docstring.
    """
    _stats["calls"] += 1
    if _backend == "ours":
        return _simplify_ours(pn)
    try:
        return _simplify_simplipy(pn)
    except Exception:
        _stats["unchanged"] += 1
        return pn
