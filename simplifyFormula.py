"""simplifyFormula.py -- deterministic canonical normal form for PN formulae.

`simplify(pn)` maps a Polish-notation formula (grammar tokens only) to a canonical
normal form.  It is designed to be:

  * value-preserving  -- eval(simplify(f)) == eval(f) on the formula's domain
                        (generic equivalence: x/x->1, x-x->0, etc.);
  * idempotent        -- simplify(simplify(f)) == simplify(f);
  * canonical-ish     -- equivalent formulae collapse to the same string for the
                        common cases (commutative reordering, like-term/like-factor
                        collection, rational-constant folding).

The implementation is a small sum-of-products CAS specialised to the grammar:

  additive part   : flattened over {+, -, ++, --, neg} into a Sum (rational coeffs
                    + a rational constant), with like terms collected.
  multiplicative  : flattened over {*, /, invert, sqr, pow2, pow3, pow^int} into a
                    Product (rational coefficient + integer exponents), like factors
                    collected.  Only INTEGER exponents are merged -- this is always
                    value-safe (x^m*x^n = x^(m+n) for integer m,n).
  opaque atoms    : sqrt, abs, pow with non-integer/symbolic exponent, and the
                    transcendental unaries (exp, ln, sin, cos, tan, arcsin, arccos,
                    arctan, tanh) are kept as opaque sub-expressions (their argument
                    is normalised recursively).  A few safe inverse identities are
                    applied: exp.ln, ln.exp, neg.neg, invert.invert, abs.abs.
  constants       : every all-rational sub-tree folds to a Fraction and is re-emitted
                    via rational_to_pn(), a deterministic writer over {0,1,2,3}
                    with +,*,sqr,pow3,++,-- that is shortest-token for integers up
                    to _INT_LIMIT (a precomputed DP table) and a deterministic --
                    not necessarily minimal -- fallback beyond (e.g. 4 -> 'sqr 2').  pi is an
                    opaque atom, so c*pi^k collapses through the product machinery.

Public API: simplify(pn_str) -> pn_str.  Returns the input unchanged on any error.

-------------------------------------------------------------------------------
Reading guide.  simplify() is three passes, and the file is laid out in that order,
one banner-marked section each:

    simplify(pn)                                              [SECTION 5]
      |  parse_pn    PN string   -> raw (tok, kids) tree      [SECTION 1]
      |  _norm       raw tree     -> canonical IR             [SECTION 3]
      |  _emit       canonical IR -> PN string                [SECTION 4]

    SECTION 2 defines the IR shapes and their constructors (_make_sum / _make_prod).

Two families of cross-calls are INHERENT to a recursive normaliser -- they are the
design, not accidental spaghetti, so expect them:

    _norm  <->  _collect_sum / _collect_prod
        _norm hands an entire additive (or multiplicative) layer to a collector; the
        collector recurses back into _norm for every non-additive (non-multiplicative)
        child.  That is how a whole +/- (or */ ) layer is flattened in one pass.

    _make_sum  <->  _make_prod
        the canonical-form rules ("a lone coeff*base is spelled as a product", "a
        scalar over a lone sum distributes into it") make each constructor sometimes
        defer to the other so one value never has two spellings.

The one non-obvious forward reference: _sort_key (SECTION 2) calls _emit (SECTION 4),
only to order opaque terms deterministically by their printed form.
-------------------------------------------------------------------------------
"""

import functools
import math
from fractions import Fraction

from grammar import V, C, U, B

# =============================================================================
# DEBUG tracing -- flip this on to watch the pipeline run, stage by stage.
# =============================================================================
# When DEBUG is True, every @_trace-decorated pipeline function prints its inputs on
# entry and its result on return, indented by call depth so the recursion is visible
# (see test_simplify.py).  It changes NOTHING about the result: when DEBUG is False the
# decorator is a one-bool-check passthrough.  Toggle it here, or at runtime:
#     import simplifyFormula; simplifyFormula.DEBUG = True
DEBUG = False
_trace_depth = 0          # current nesting level, for indentation


def _trace(fn):
    """Decorator that logs `-> name(args)` on entry and `<- name = result` on return
    (indented by nesting depth) whenever DEBUG is on; a transparent passthrough when off."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _trace_depth
        if not DEBUG:
            return fn(*args, **kwargs)
        pad = '    ' * _trace_depth
        shown = ', '.join([repr(a) for a in args]
                          + [f'{k}={v!r}' for k, v in kwargs.items()])
        print(f'{pad}-> {fn.__name__}({shown})')
        _trace_depth += 1
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:                     # show the raise, then re-raise unchanged
            _trace_depth -= 1
            print(f'{pad}<- {fn.__name__} raised {type(exc).__name__}')
            raise
        _trace_depth -= 1
        print(f'{pad}<- {fn.__name__} = {result!r}')
        return result
    return wrapper


# =============================================================================
# SECTION 1 -- grammar tokens & the PN parser  (PN string -> raw (tok, kids) tree)
# =============================================================================

# -- token classification ------------------------------------------------------
_VARS = set(V)
_BIN  = set(B)            # + - * / pow
_UN   = set(U)
_NUMC = {"0", "1", "2", "3"}     # numeric symbolic constants (pi handled separately)

_TRANSCENDENTAL = {"exp", "ln", "sin", "cos", "tan",
                   "arcsin", "arccos", "arctan", "tanh", "abs"}

# Grammar tokens grouped by the IR layer _norm folds them into.  Naming the groups
# lets _norm dispatch a whole layer to one collector instead of matching bare tuples:
#   additive       -- binary +/- plus the unary sugar (++ x = x+1, -- x = x-1, neg x = -x)
#   multiplicative -- * / invert (1/x) and the integer-power shorthands (sqr/pow2 = ^2, pow3 = ^3)
_ADDITIVE_OPS       = {"+", "-", "++", "--", "neg"}
_MULTIPLICATIVE_OPS = {"*", "/", "invert", "sqr", "pow2", "pow3"}


class _Undefined(Exception):
    """Raised when a sub-expression is mathematically undefined (e.g. /0); makes
    simplify() leave the whole formula unchanged so it is filtered downstream."""


def _arity(tok):
    if tok in _VARS or tok in C:   # C includes 0,1,2,3,pi
        return 0
    if tok in _UN:
        return 1
    if tok in _BIN:
        return 2
    return 0                       # unknown tokens treated as nullary leaves


# -- PN parser -> raw tree (tok, [children]) ------------------------------------
@_trace
def parse_pn(toks, i=0):
    """Return (node, next_index). node = (tok, [child_nodes])."""
    tok = toks[i]
    n = _arity(tok)
    j = i + 1
    kids = []
    for _ in range(n):
        kid, j = parse_pn(toks, j)
        kids.append(kid)
    return (tok, kids), j


# =============================================================================
# SECTION 2 -- the canonical IR: its shapes, and the constructors that build it
#              (_make_sum / _make_prod).  No parsing or emission here.
# =============================================================================
# -------------------------------------------------------------------------------
# Canonical IR (all immutable / hashable):
#   ('n', Fraction)                          number
#   ('v', name)                              variable
#   ('pi',)                                  pi
#   ('f', name, arg_ir)                      opaque unary application (sqrt/sin/...)
#   ('pw', base_ir, exp_ir)                  opaque power (symbolic/non-integer exp)
#   ('p', Fraction coeff, ((base_ir, int_exp), ...) sorted)   product
#   ('s', Fraction const, ((Fraction coeff, base_ir), ...) sorted)   sum
#
# Note: binary +/* have NO IR node of their own -- an entire additive layer folds into
#   one n-ary 's' (sum) and an entire multiplicative layer into one n-ary 'p' (product),
#   so (a+b)+c and a+(b+c) reach the same canonical form; see _collect_sum / _collect_prod.
#   ('pw' is the only binary-shaped node, and only for opaque powers.)
# -------------------------------------------------------------------------------

_ONE = ('n', Fraction(1))
_ZERO = ('n', Fraction(0))

def _sort_key(ir):
    # Deterministic ordering for terms/factors.  (Forward ref: the non-variable branch
    # calls _emit from SECTION 4 to order opaque terms by their printed form.)
    # Variables are ordered by NUMERIC index (not lexicographically) so canonicalisation
    # commutes with relabel_variables (which renumbers vars by numeric order): simplify-then-relabel
    # still yields a canonical form (otherwise v10 sorts before v2 and relabelling
    # breaks the order).  Variables sort before everything else.
    if ir[0] == 'v':
        s = ir[1]
        return (0, int(s[1:]) if s[1:].isdigit() else 0, s)
    return (1, 0, _emit(ir))


@_trace
def _make_prod(coeff, factors):
    """Rebuild a canonical product IR from a coefficient (Fraction) and a
    {base_ir: int exponent} map.  Exponent-0 factors are dropped and the rest sorted
    by base (_sort_key).  Special shapes, checked in this order:

        coeff == 0                     -> 0 (a zero coefficient kills the product);
        no factors left                -> a plain number ('n', coeff);
        coeff * (single sum factor)^1  -> distribute the scalar INTO the sum;
        1 * (single base)^1            -> just that base, unwrapped;
        otherwise                      -> a genuine product ('p', coeff, factors).

    The distribute case is what keeps the CAS idempotent: a leading -1 left outside a
    sum would emit as 'neg (Sigma ...)', which re-parses (neg is additive) straight
    back into a distributed sum, so the two would never converge.
    """
    if coeff == 0:
        return _ZERO
    # (base, exponent) pairs with a non-zero exponent, sorted by base.
    sorted_factors = tuple(sorted(((base, exp) for base, exp in factors.items() if exp != 0),
                                  key=lambda be: _sort_key(be[0])))
    if not sorted_factors:
        return ('n', Fraction(coeff))

    # The two unwrapping shapes only apply to a single factor raised to the first power.
    if len(sorted_factors) == 1 and sorted_factors[0][1] == 1:
        base = sorted_factors[0][0]
        # coeff * (a sum)  ->  push coeff through the sum's constant and every term.
        if coeff != 1 and base[0] == 's':
            sum_const, sum_terms = base[1], base[2]
            return _make_sum(Fraction(coeff) * sum_const,
                             {b: Fraction(coeff) * c for (c, b) in sum_terms})
        # 1 * base  ->  just the base.
        if coeff == 1:
            return base

    return ('p', Fraction(coeff), sorted_factors)


@_trace
def _make_sum(const, terms):
    """Rebuild a canonical sum IR from a constant (Fraction) and a {base_ir: Fraction
    coeff} map.  Zero-coefficient terms are dropped and the rest sorted by base
    (_sort_key).  Two shapes are deliberately NOT emitted as sums:

        no terms left                        -> a plain number ('n', const);
        one term and no constant (coeff*base) -> a PRODUCT, not a single-term sum.

    That second rule exists so a value like 2*v1 has ONE IR form.  Reached additively
    (v1+v1) it would otherwise be ('s',0,((2,v1),)); reached multiplicatively it is
    ('p',2,((v1,1),)).  Those print identically but are unequal as dict keys, so opaque
    atoms keyed on their argument -- sin(v1+v1) vs sin(2*v1) -- would fail to collect,
    breaking canonicity and idempotence.  Funnelling both through _make_prod avoids it.
    """
    # (base, coeff) pairs with a non-zero coeff, sorted by base.
    sorted_terms = tuple(sorted(((base, coeff) for base, coeff in terms.items() if coeff != 0),
                                key=lambda bc: _sort_key(bc[0])))
    if not sorted_terms:
        return ('n', Fraction(const))

    # Lone coeff*base with no constant -> build it as a product (see the docstring).
    if const == 0 and len(sorted_terms) == 1:
        base, coeff = sorted_terms[0]
        if base[0] == 'p':                    # fold coeff into the product base's own factors
            return _make_prod(coeff * base[1], dict(base[2]))
        return _make_prod(coeff, {base: 1})   # coeff == 1 collapses to `base` inside _make_prod

    # Genuine sum: store as (coeff, base) pairs (already sorted by base).
    return ('s', Fraction(const), tuple((coeff, base) for base, coeff in sorted_terms))


@_trace
def _as_scalar_times_base(ir):
    """Decompose ir into (scalar: Fraction, base_ir or None) with ir == scalar * base.

    `base` is the coefficient-stripped core of the expression: the canonical sub-IR
    that remains after factoring out any leading rational scalar, so `base` itself
    carries no leading rational coefficient (it is "monic").  `base is None` is the
    degenerate case where nothing but the number is left (ir is a pure constant).
    """
    if ir[0] == 'n':
        # pure number: value is entirely the scalar, no base sub-IR remains
        return ir[1], None
    if ir[0] == 'p':
        # product: its coefficient is the scalar; base = the same product with
        # coefficient reset to 1 (just the factors)
        return ir[1], _make_prod(Fraction(1), {b: e for b, e in ir[2]})
    if ir[0] == 's' and ir[1] == 0 and len(ir[2]) == 1:
        # single-term sum, no constant: it already is coeff * base, so peel them apart
        coeff, base = ir[2][0]
        return coeff, base
    # variable, opaque func/power, or multi-term sum: no scalar to factor out,
    # so scalar = 1 and base is ir itself
    return Fraction(1), ir


# =============================================================================
# SECTION 3 -- normalisation: raw (tok, kids) tree -> canonical IR.
#   Entry point _norm; it delegates additive/multiplicative layers to the mutually
#   recursive _collect_sum / _collect_prod, and powers/opaque funcs to _norm_pow /
#   _norm_func.  _frac_pow below is the allocation-guarded rational power they share.
# =============================================================================

# Ceiling on the bit-length of any integer the CAS will materialise.  Power
# folding (`pow`, `sqr`, `pow3`) over the symbolic constants can otherwise build
# astronomically large Python ints: e.g. `pow 2 pow3 pow3 pow3 3` folds to
# 2**(3**27), a ~950 GB integer.  A single such formula, drawn in one DataLoader
# worker, OOM-killed a 192 GB job (the worker ballooned to ~198 GB while its
# siblings sat at 0.6 GB).  Any power whose result would exceed this many bits is
# refused; simplify() then returns the formula unchanged (its `except Exception`
# path) and it is handled normally downstream -- overflowing IO rows evaluate to
# inf and are masked out in data_process.  2**14 bits (~2 KB) is far above any
# constant that survives the formula-length cap, so no real canonicalisation is
# lost.
_MAX_INT_BITS = 1 << 14


def _frac_pow(c, k):
    c = Fraction(c)
    k = int(k)
    # Predict the result's bit-length from the inputs and bail BEFORE computing:
    # the exponentiation itself is the allocation that would OOM, so it must never
    # start.  bits(c ** k) ~= |k| * bits(c).
    span = max(c.numerator.bit_length(), c.denominator.bit_length(), 1)
    if span * abs(k) > _MAX_INT_BITS:
        raise _Undefined
    return c ** k if k >= 0 else Fraction(1) / (c ** (-k))


@_trace
def _norm(node):
    """Recursively normalise a raw parse tree `(tok, kids)` into canonical IR.

    Format bridge -- this is the one function that spans the two internal formats:
        IN : a raw parse-tree node `(token_string, [child_nodes])` from parse_pn,
             mirroring the grammar 1:1 (one node per token);
        OUT: a canonical IR tuple `(tag, ...)` -- one of the seven shapes listed in
             the "Canonical IR" table above (`('v', name)`, `('s', const, terms)`,
             ...), the collected/folded form that _emit later turns back into a PN
             string.  The tag (tuple[0]) is what every consumer dispatches on.

    This is the dispatch heart of the CAS: it maps each grammar token to the IR
    shape documented at the top of the file, recursing bottom-up so every
    sub-expression is normalised before its parent.  The branches are:

      * leaves          -- variables -> ('v', name); pi -> ('pi',); the numeric
                           constants 0..3 -> ('n', Fraction).
      * additive ops    -- +, -, ++, --, neg are handed to _collect_sum, which
                           flattens the whole additive layer into one (const, terms)
                           pair (recursing into _norm for any non-additive child)
                           and is rebuilt by _make_sum.  So `- (+ a b) b` collapses
                           in a single pass rather than op-by-op.
      * multiplicative  -- *, /, invert, sqr, pow2, pow3 are handed to _collect_prod
                           for the same flatten-and-collect treatment, rebuilt by
                           _make_prod.  Only integer exponents are merged there.
      * pow             -- delegated to _norm_pow, which routes integer exponents
                           into the product machinery, 1/2 into sqrt, and leaves a
                           genuinely symbolic exponent as an opaque ('pw', ...).
      * sqrt / transcendentals -- kept opaque via _norm_func (which also applies the
                           safe inverse identities exp.ln, ln.exp, abs.abs, |const|);
                           the argument is normalised recursively.

    The final two lines are the fallback for tokens outside the known grammar: a
    lone unary is wrapped as an opaque ('f', tok, arg); anything else is treated as
    a nullary leaf (mirroring _arity's "unknown -> nullary" convention).
    """
    # unpack the node
    tok, kids = node

    # -- leaves: a variable, pi, or a numeric constant 0..3 --
    if tok in _VARS:
        return ('v', tok)
    if tok == 'pi':
        return ('pi',)
    if tok in _NUMC:
        return ('n', Fraction(int(tok)))

    # -- additive layer: flatten the entire +/-/++/--/neg subtree into a single
    #    (const, {base: coeff}) pair in one pass, then rebuild the canonical sum --
    if tok in _ADDITIVE_OPS:
        const, terms = _collect_sum(node)
        return _make_sum(const, terms)

    # -- multiplicative layer: likewise flatten the whole */invert/sqr/... subtree
    #    into (coeff, {base: int exponent}) and rebuild the canonical product --
    if tok in _MULTIPLICATIVE_OPS:
        coeff, factors = _collect_prod(node)
        return _make_prod(coeff, factors)

    # -- power: _norm_pow routes an integer exponent into the product machinery,
    #    1/2 into sqrt, and a genuinely symbolic exponent into an opaque ('pw', ...) --
    if tok == 'pow':
        return _norm_pow(kids[0], kids[1])

    # -- sqrt and the transcendentals stay opaque; only their argument is
    #    normalised.  _norm_func also folds the safe inverse identities
    #    (exp(ln x)=x, ln(exp x)=x, abs(abs x)=abs x, |const|) --
    if tok == 'sqrt':
        return _norm_func('sqrt', _norm(kids[0]))
    if tok in _TRANSCENDENTAL:
        return _norm_func(tok, _norm(kids[0]))

    # -- token outside the known grammar: a unary is wrapped opaque as ('f', ...);
    #    anything else (nullary, or unexpected arity) becomes a bare leaf, matching
    #    _arity's "unknown -> nullary" convention --
    if len(kids) == 1:
        return ('f', tok, _norm(kids[0]))
    return ('v', tok)


@_trace
def _norm_pow(base_node, exp_node):
    """Normalise `pow base exp`.  When the exponent is a rational constant:
        exp == 0     -> 1;
        exp integer  -> fold into the product machinery (base^n = the base's factors
                        with every exponent multiplied by n, coefficient raised to n);
        exp == 1/2   -> sqrt(base).
    Any other exponent (symbolic, or a non-1/2 fraction) stays an opaque ('pw', ...).
    """
    exp_ir = _norm(exp_node)
    if exp_ir[0] == 'n':                            # exponent is a rational constant
        exp_val = exp_ir[1]
        if exp_val == 0:                            # base^0 = 1
            return _ONE
        if exp_val.denominator == 1:               # integer exponent -> product power
            exp_int = int(exp_val)
            # Reject huge exponents up front.  For a constant base _frac_pow would
            # OOM (guarded there too); for a variable base the exponent instead
            # feeds int_to_pn, whose recursion depth ~ log2(exponent) -- bound it so
            # a power-tower exponent cannot blow the Python stack either.
            if abs(exp_int) > _MAX_INT_BITS:
                raise _Undefined
            coeff, factors = _collect_prod(base_node)
            return _make_prod(_frac_pow(coeff, exp_int),
                              {base: exp * exp_int for base, exp in factors.items()})
        if exp_val == Fraction(1, 2):              # base^(1/2) = sqrt(base)
            return _norm_func('sqrt', _norm(base_node))
    return ('pw', _norm(base_node), exp_ir)        # symbolic / non-integer exponent: opaque


def _is_call(ir, fn):
    """True if `ir` is an opaque unary application of `fn`, i.e. ('f', fn, <arg_ir>).
    The `ir[0] == 'f'` guard short-circuits before touching ir[1], so a shorter tuple
    like ('pi',) is safe to pass in."""
    return ir[0] == 'f' and ir[1] == fn


@_trace
def _norm_func(name, arg):
    """Wrap an opaque unary function around its (already normalised) argument, applying
    the few value-safe simplifications the CAS allows.  `arg` is IR; the result is IR.

        exp(ln x) = x,  ln(exp x) = x   -- the two inverse pairs cancel;
        |constant|                       -- folds to the non-negative number;
        abs(abs x) = abs x               -- nested abs is redundant.
    Everything else stays wrapped as ('f', name, arg).
    """
    if name == 'exp' and _is_call(arg, 'ln'):
        return arg[2]                               # exp(ln x) = x
    if name == 'ln' and _is_call(arg, 'exp'):
        return arg[2]                               # ln(exp x) = x
    if name == 'abs':
        if arg[0] == 'n':                           # |constant| -> its absolute value
            return ('n', abs(arg[1]))
        if _is_call(arg, 'abs'):                    # abs(abs x) = abs x
            return arg
    return ('f', name, arg)


@_trace
def _collect_sum(node):
    """Flatten an additive subtree into (const, terms).

    Recurses through the +/-/++/--/neg layer, accumulating a running rational `const`
    and a {base_ir: Fraction coeff} map that sums the coefficient of every distinct
    base.  A non-additive child (product, variable, opaque call, ...) is normalised by
    _norm and contributed as a single term.  _make_sum later rebuilds the IR from this.
    """
    tok, kids = node

    # +/- : recurse on both sides, then combine.  Subtraction flips the right side's
    # signs (constant and every coefficient) before merging.
    if tok == '+':
        const_l, terms_l = _collect_sum(kids[0])
        const_r, terms_r = _collect_sum(kids[1])
        return const_l + const_r, _merge_sum(terms_l, terms_r)
    if tok == '-':
        const_l, terms_l = _collect_sum(kids[0])
        const_r, terms_r = _collect_sum(kids[1])
        return const_l - const_r, _merge_sum(terms_l, _negate_values(terms_r))

    # Unary sugar: ++ x = x+1, -- x = x-1, neg x = -x.
    if tok == '++':
        const, terms = _collect_sum(kids[0])
        return const + 1, terms
    if tok == '--':
        const, terms = _collect_sum(kids[0])
        return const - 1, terms
    if tok == 'neg':
        const, terms = _collect_sum(kids[0])
        return -const, _negate_values(terms)

    # Non-additive node: normalise it and contribute it to the sum.
    ir = _norm(node)
    if ir[0] == 's':                          # already a sum -> adopt its const and terms
        return ir[1], {base: coeff for (coeff, base) in ir[2]}
    scalar, base = _as_scalar_times_base(ir)
    if base is None:                          # a pure number -> it is entirely constant
        return scalar, {}
    return Fraction(0), {base: scalar}        # scalar * base -> a single term


@_trace
def _collect_prod(node):
    """Flatten a multiplicative subtree into (coeff, factors).

    Recurses through the * / invert / sqr / pow2 / pow3 layer, accumulating a running
    rational `coeff` and a {base_ir: int exponent} map.  Multiplying two sub-results
    ADDS the exponents of shared bases -- that is exactly what _merge_sum does, so it is
    reused here on the exponent maps.  A non-multiplicative child is normalised by _norm
    and contributed as a single factor.  _make_prod later rebuilds the IR.

    Division or inversion by a zero constant is mathematically undefined and aborts via
    _Undefined, which makes simplify() leave the whole formula unchanged.
    """
    tok, kids = node

    # * : multiply the coefficients; add the exponents of shared bases.
    if tok == '*':
        coeff_l, factors_l = _collect_prod(kids[0])
        coeff_r, factors_r = _collect_prod(kids[1])
        return coeff_l * coeff_r, _merge_sum(factors_l, factors_r)
    # / : divide the coefficients; dividing subtracts the divisor's exponents.
    if tok == '/':
        coeff_l, factors_l = _collect_prod(kids[0])
        coeff_r, factors_r = _collect_prod(kids[1])
        if coeff_r == 0:
            raise _Undefined                        # x/0 is undefined -> leave formula unchanged
        return coeff_l / coeff_r, _merge_sum(factors_l, _negate_values(factors_r))
    # invert x = 1/x : reciprocal coefficient, every exponent negated.
    if tok == 'invert':
        coeff, factors = _collect_prod(kids[0])
        if coeff == 0:
            raise _Undefined                        # 1/0 is undefined -> leave formula unchanged
        return Fraction(1) / coeff, _negate_values(factors)
    # sqr / pow2 = (...)^2, pow3 = (...)^3 : power the coefficient, scale the exponents.
    if tok in ('sqr', 'pow2'):
        coeff, factors = _collect_prod(kids[0])
        return _frac_pow(coeff, 2), {base: exp * 2 for base, exp in factors.items()}
    if tok == 'pow3':
        coeff, factors = _collect_prod(kids[0])
        return _frac_pow(coeff, 3), {base: exp * 3 for base, exp in factors.items()}

    # Non-multiplicative node: normalise it and contribute it as a factor.
    ir = _norm(node)
    scalar, base = _as_scalar_times_base(ir)
    if base is None:                                # a pure number -> it is entirely coefficient
        return scalar, {}
    if base[0] == 'p':                              # a product base -> absorb its coeff and factors
        return scalar * base[1], dict(base[2])
    return scalar, {base: 1}                        # any other base -> that base to the first power


def _merge_sum(a, b):
    """Merge two {base: value} dicts by adding values on shared keys.

    Serves both additive collection (values are Fraction coeffs) and multiplicative
    collection (values are int exponents); the `0` default is type-neutral, since
    0 + Fraction is a Fraction and 0 + int is an int.
    """
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def _negate_values(d):
    """Negate every value in a {base: value} dict (used to subtract terms / invert
    factors -- flips coeff signs for sums, exponent signs for products)."""
    return {k: -v for k, v in d.items()}


# =============================================================================
# SECTION 4 -- emission: canonical IR -> canonical PN string.
#   Entry point _emit; products/sums go to _emit_prod / _emit_sum, and every rational
#   constant is written by rational_to_pn / int_to_pn (the DP-backed constant writer
#   at the end of this section).  This is the inverse of SECTION 3.
# =============================================================================
@_trace
def _emit(ir):
    """
    Serialise canonical IR back to a PN string.

    Dispatches on the IR tag (the first tuple element).  The scalar leaves and
    opaque wrappers are written inline here; the two compound forms are delegated:

      * 'n'  -- a Fraction, written by rational_to_pn as the shortest canonical
                token sequence over {0,1,2,3} (e.g. 4 -> 'sqr 2').
      * 'v'  -- the variable name verbatim; 'pi' -> the literal 'pi'.
      * 'f'  -- opaque unary application: name followed by the emitted argument.
      * 'pw' -- opaque power: 'pow' base exp, both re-emitted.
      * 'p'  -- product, handed to _emit_prod (coefficient sign -> optional 'neg',
                numerator/denominator each built constant-first, unit numerator ->
                'invert <den>').
      * 's'  -- sum, handed to _emit_sum (splits terms by sign so subtractions emit
                as '- ... ...' rather than adding negatives; +/-1 constants fold to
                the '++'/'--' increment tokens).

    _emit is a total function over well-formed IR; an unrecognised tag signals an
    internal bug rather than bad user input, so it raises (simplify()'s except-guard
    then returns the original formula unchanged).
    """
    kind = ir[0]
    if kind == 'n':
        return rational_to_pn(ir[1])
    if kind == 'v':
        return ir[1]
    if kind == 'pi':
        return 'pi'
    if kind == 'f':
        return ir[1] + ' ' + _emit(ir[2])
    if kind == 'pw':
        return 'pow ' + _emit(ir[1]) + ' ' + _emit(ir[2])
    if kind == 'p':
        return _emit_prod(ir[1], ir[2])
    if kind == 's':
        return _emit_sum(ir[1], ir[2])
    raise ValueError(f"bad IR: {ir!r}")


def _emit_power(base_pn, e):
    if e == 1:
        return base_pn
    if e == 2:
        return 'sqr ' + base_pn
    if e == 3:
        return 'pow3 ' + base_pn
    return 'pow ' + base_pn + ' ' + int_to_pn(e)


def _fold_bin(op, parts):
    """Right-fold a non-empty list of PN strings with a binary op."""
    res = parts[-1]
    for p in reversed(parts[:-1]):
        res = op + ' ' + p + ' ' + res
    return res


def _emit_pos_prod(coeff, factors):
    """Emit coeff(>0 Fraction) * Pi base^exp  as a canonical monomial.

    Numerator and denominator are each built constant-first (constant-left), and a
    unit numerator collapses to `invert <denominator>`.
    """
    p, q = coeff.numerator, coeff.denominator
    num, den = [], []
    if p != 1:
        num.append(int_to_pn(p))
    for base, e in factors:
        bpn = _emit(base)
        if e > 0:
            num.append(_emit_power(bpn, e))
        else:
            den.append(_emit_power(bpn, -e))
    if q != 1:
        den = [int_to_pn(q)] + den
    num_pn = _fold_bin('*', num) if num else '1'
    if not den:
        return num_pn
    den_pn = _fold_bin('*', den)
    if num_pn == '1':
        return 'invert ' + den_pn
    return '/ ' + num_pn + ' ' + den_pn


@_trace
def _emit_prod(coeff, factors):
    if coeff < 0:
        return 'neg ' + _emit_pos_prod(-coeff, factors)
    return _emit_pos_prod(coeff, factors)


def _emit_term(coeff, base):
    """Emit coeff(>0 Fraction) * base  (base is a non-constant IR)."""
    if base[0] == 'p':
        return _emit_pos_prod(coeff * base[1], base[2])
    if base[0] == 'n':
        return rational_to_pn(coeff * base[1])
    return _emit_pos_prod(coeff, ((base, 1),))


@_trace
def _emit_sum(const, terms):
    pos = [(c, b) for (c, b) in terms if c > 0]
    neg = [(-c, b) for (c, b) in terms if c < 0]
    has_rest = bool(pos or neg)
    if const == 1 and has_rest:
        return '++ ' + _emit_sum(Fraction(0), terms)
    if const == -1 and has_rest:
        return '-- ' + _emit_sum(Fraction(0), terms)
    pos_pn = [_emit_term(c, b) for (c, b) in pos]
    neg_pn = [_emit_term(c, b) for (c, b) in neg]
    if const > 0:
        pos_pn = [rational_to_pn(const)] + pos_pn
    elif const < 0:
        neg_pn = neg_pn + [rational_to_pn(-const)]
    if not pos_pn and not neg_pn:
        return '0'
    if not pos_pn:
        res = 'neg ' + neg_pn[0]
        for n in neg_pn[1:]:
            res = '- ' + res + ' ' + n
        return res
    res = _fold_bin('+', pos_pn)
    for n in neg_pn:
        res = '- ' + res + ' ' + n
    return res


# -- canonical constant writer (deterministic PN; shortest-token up to _INT_LIMIT) --
_INT_LIMIT = 100
_int_cost = {}
_int_pn = {}


def _build_int_table():
    INF = float('inf')
    cost = {n: INF for n in range(_INT_LIMIT + 1)}
    pn = {n: None for n in range(_INT_LIMIT + 1)}
    for b in range(4):
        cost[b] = 1
        pn[b] = str(b)

    def _key(c, s):
        # prefer: fewer tokens, then fewer ++/-- (more "natural"), then lexicographic
        return (c, s.count('++') + s.count('--'), s)

    def better(v, c, s):
        if 0 <= v <= _INT_LIMIT and (pn[v] is None or _key(c, s) < _key(cost[v], pn[v])):
            cost[v] = c
            pn[v] = s
            return True
        return False

    changed = True
    while changed:
        changed = False
        for v in range(_INT_LIMIT + 1):
            if v - 1 >= 0 and cost[v - 1] < INF:
                changed |= better(v, cost[v - 1] + 1, '++ ' + pn[v - 1])
            if v + 1 <= _INT_LIMIT and cost[v + 1] < INF:
                changed |= better(v, cost[v + 1] + 1, '-- ' + pn[v + 1])
            r = math.isqrt(v)
            if r * r == v and cost[r] < INF:
                changed |= better(v, cost[r] + 1, 'sqr ' + pn[r])
            cr = round(v ** (1 / 3)) if v >= 0 else 0
            for cc in (cr - 1, cr, cr + 1):
                if cc >= 0 and cc ** 3 == v and cost[cc] < INF:
                    changed |= better(v, cost[cc] + 1, 'pow3 ' + pn[cc])
        for a in range(_INT_LIMIT + 1):
            if cost[a] == INF:
                continue
            for bb in range(a, _INT_LIMIT + 1):
                if cost[bb] == INF:
                    continue
                base_c = cost[a] + cost[bb] + 1
                s = a + bb
                if s <= _INT_LIMIT:
                    changed |= better(s, base_c, '+ ' + pn[a] + ' ' + pn[bb])
                pr = a * bb
                if pr <= _INT_LIMIT:
                    changed |= better(pr, base_c, '* ' + pn[a] + ' ' + pn[bb])
    _int_cost.update(cost)
    _int_pn.update(pn)


_build_int_table()


def int_to_pn(n):
    """Canonical PN for an integer n over {0,1,2,3} with +,*,sqr,pow3,++,--.

    Shortest-token for 0 <= n <= _INT_LIMIT (served from the DP table); for larger
    n a deterministic even/odd decomposition, which is canonical but not minimal."""
    n = int(n)
    if n < 0:
        return 'neg ' + int_to_pn(-n)
    if n <= _INT_LIMIT and _int_pn.get(n) is not None:
        return _int_pn[n]
    # fallback for n > table limit: deterministic even/odd decomposition
    if n % 2 == 0:
        return '* 2 ' + int_to_pn(n // 2)
    return '++ ' + int_to_pn(n - 1)


def rational_to_pn(f):
    """Canonical PN for a rational Fraction."""
    f = Fraction(f)
    if f == 0:
        return '0'
    if f < 0:
        return 'neg ' + rational_to_pn(-f)
    p, q = f.numerator, f.denominator
    if q == 1:
        return int_to_pn(p)
    if p == 1:
        return 'invert ' + int_to_pn(q)
    return '/ ' + int_to_pn(p) + ' ' + int_to_pn(q)


# =============================================================================
# SECTION 5 -- public API: the three-pass pipeline, wrapped in error guards.
# =============================================================================
@_trace
def simplify(pn):
    """Canonicalise a PN formula string.  Returns the input unchanged on error.

    The public entry point.  It runs the three-stage pipeline
    parse_pn -> _norm -> _emit:

        parse_pn   splits the string and builds a raw (tok, kids) tree, using
                   _arity to know each token's child count;
        _norm      folds that tree into canonical intermediate representation (IR)
                   (like-term/like-factor collection, constant folding, safe inverse identities);
        _emit      writes the IR back out as a canonical PN string.

    The result is value-preserving, idempotent, and canonical for the common cases
    (see the module docstring).  Robustness is deliberate: this is called on every
    generated draw in the data path, so it must never raise into the training loop.
    Three guards enforce "leave the formula untouched rather than fail":

      * empty / whitespace-only input                       -> "" ;
      * trailing tokens after a complete parse (malformed
        or multi-expression input)                          -> the original string ;
      * ANY exception (a mathematically undefined sub-tree
        raising _Undefined -- e.g. x/0 -- or an over-large
        constant tripping the _MAX_INT_BITS guard, or any
        other internal error)                               -> the original string .

    Downstream code treats an un-simplified (or degenerate) formula normally: the
    degeneracy filter in data_process rejects collapsed draws, and inputs that
    overflow evaluate to inf and are masked out.

    Why this exists instead of SymPy.  The e2e/symbolicregression pipeline has an
    optional SymPy pass and ships it DISABLED (use_sympy=False), for reasons that
    apply doubly here.  SymPy is the wrong tool on four counts:
      1. It is a heuristic, not a normal form -- sympy.simplify() is not guaranteed
         to map equivalent expressions to the same output (or to be idempotent, or
         stable across versions), whereas the cross-entropy target must be canonical
         so the loss ranks FUNCTIONS, not spellings.
      2. It emits arbitrary literals (4, 17, 1/2); our grammar is mantissa-free, so
         constants must be rebuilt from {0,1,2,3,pi} (rational_to_pn: 4 -> 'sqr 2').
      3. Its default rewrites are domain-agnostic without explicit assumptions; we
         only apply value-safe ones (integer-exponent merges, a whitelist of inverse
         identities) and keep sqrt/abs/trig opaque.
      4. It is orders of magnitude slower and can OOM/hang on pathological constants;
         this runs on every generated draw in the data loader, so it must be fast,
         allocation-bounded (_MAX_INT_BITS), and never raise into the training loop.
    This module is a small sum-of-products CAS that buys those four properties at the
    cost of SymPy's general algebraic power (no trig identities, factoring, etc.),
    which the canonical-target use case does not need.
    """
    if not pn or not pn.strip():
        return ""
    try:
        toks = pn.split()
        node, idx = parse_pn(toks, 0)
        if idx != len(toks):
            return pn                       # trailing tokens -> leave untouched
        return _emit(_norm(node))
    except Exception:
        return pn


# -----------------------------------------------------------------------------
# Test harness: unit cases + idempotence + numerical equivalence on random formulae
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import random as _random

    # 1) Unit cases (canonical-form spec).
    test_cases = [
        ("+ v1 0", "v1"), ("+ 0 v1", "v1"), ("- v1 0", "v1"), ("- v1 v1", "0"),
        ("* v1 1", "v1"), ("* 1 v1", "v1"), ("* v1 0", "0"), ("* 0 v1", "0"),
        ("/ v1 1", "v1"), ("/ v1 v1", "1"),
        ("+ 1 1", "2"), ("+ 1 2", "3"), ("+ 2 1", "3"), ("- 3 1", "2"),
        ("- 3 2", "1"), ("- 2 2", "0"), ("* 1 2", "2"),
        ("+ v1 1", "++ v1"), ("+ 1 v1", "++ v1"), ("- v1 1", "-- v1"),
        ("- 0 v1", "neg v1"), ("/ 1 v1", "invert v1"), ("* v1 v1", "sqr v1"),
        ("neg neg v1", "v1"), ("invert invert v1", "v1"),
        ("exp ln v1", "v1"), ("ln exp v1", "v1"), ("neg 0", "0"),
        ("+ - v1 v1 v2", "v2"), ("* / v1 v1 v2", "v2"), ("neg - 0 v1", "v1"),
        ("+ + 1 1 v1", "+ 2 v1"), ("+ + 2 2 v1", "+ sqr 2 v1"),
        ("* v1 2", "* 2 v1"),
        ("+ * v1 v1 - v2 v2", "sqr v1"),
        # collection / canonical-order extras
        ("* v2 v1", "* v1 v2"), ("+ v2 v1", "+ v1 v2"),
        ("+ v1 v1", "* 2 v1"), ("* v1 * v1 v1", "pow3 v1"),
        ("pow v1 2", "sqr v1"), ("pow v1 3", "pow3 v1"),
        ("/ 1 * 2 v1", "invert * 2 v1"), ("* 2 3", "* 2 3"),
        # scalar distributed over a lone sum (needed for idempotence + confluence)
        ("* 2 + v1 v2", "+ * 2 v1 * 2 v2"), ("neg + v1 v2", "- neg v1 v2"),
        ("/ + * 2 v1 * 2 v2 2", "+ v1 v2"),
    ]
    print(f"{'IDX':<4} | {'STATUS':<6} | {'INPUT':<24} | {'OUTPUT':<22} | EXPECTED")
    print("-" * 92)
    failures = []
    for i, (inp, expected) in enumerate(test_cases):
        out = simplify(inp)
        status = "PASS" if out == expected else "FAIL"
        if status == "FAIL":
            failures.append((i, inp, expected, out))
        print(f"{i:<4} | {status:<6} | {inp:<24} | {out:<22} | {expected}")
    print("-" * 92)
    print(f"unit: {len(test_cases) - len(failures)}/{len(test_cases)} passed")

    # 2) Idempotence + 3) numerical equivalence on random generator formulae.
    # Draw from TWO generators: randPN (sizeLimit) and the data-path distribution
    # (_uniform_randPN with composite constants + the trainer's op ranges), which
    # exercises structures bare randPN under-samples (e.g. scalar*sum nesting).
    from grammar import randPN, _uniform_randPN
    from funcWrappers import wrapExtEvalPN

    def _gen_formula(i):
        if i % 2 == 0:
            return randPN(sizeLimit=_random.randint(1, 18))
        d = np.random.randint(1, 8)
        return _uniform_randPN(np.random.randint(max(0, d - 1), d + 5),
                               np.random.randint(0, 5), d)

    np.random.seed(0); _random.seed(0)
    n_total = 6000
    n_idem_fail = 0
    n_val_fail = 0
    n_val_checked = 0
    examples_idem = []
    examples_val = []
    for _i in range(n_total):
        f = _gen_formula(_i)
        s1 = simplify(f)
        s2 = simplify(s1)
        if s1 != s2:
            n_idem_fail += 1
            if len(examples_idem) < 5:
                examples_idem.append((f, s1, s2))
        # numerical equivalence on positive inputs (keeps sqrt/ln/pow in-domain).
        # Two guards keep this from false-flagging correct rewrites:
        #   - skip domain-degenerate formulae (mostly nan/inf, e.g. sqrt of a
        #     negative): they aren't real-valued functions, and the C evaluator's
        #     non-IEEE nan handling can make algebraically-equal forms diverge;
        #   - judge by MEDIAN relative error, so float blow-up near a few poles
        #     (deeply nested formulae) doesn't masquerade as a value bug.
        try:
            X = np.abs(np.random.randn(128, len(V))) + 0.3
            y0, e0 = wrapExtEvalPN(f, X)
            y1, e1 = wrapExtEvalPN(s1, X)
            fin0 = np.isfinite(y0) & (np.abs(y0) < 1e6)
            fin1 = np.isfinite(y1) & (np.abs(y1) < 1e6)
            if np.mean(fin0) < 0.8 or np.mean(fin1) < 0.8:
                continue                                   # ill-defined formula -> skip
            m = fin0 & fin1
            if m.sum() >= 8:
                n_val_checked += 1
                rel = np.abs(y0[m] - y1[m]) / (np.abs(y0[m]) + np.abs(y1[m]) + 1e-9)
                if float(np.median(rel)) > 1e-6:
                    n_val_fail += 1
                    if len(examples_val) < 8:
                        examples_val.append((f, s1, float(np.median(rel))))
        except Exception:
            pass

    print(f"\nidempotence: {n_total - n_idem_fail}/{n_total} stable "
          f"(simplify.simplify == simplify)")
    for f, s1, s2 in examples_idem:
        print(f"   NON-IDEMPOTENT: {f}\n       -> {s1}\n       -> {s2}")
    print(f"numerical equivalence: {n_val_checked - n_val_fail}/{n_val_checked} match "
          f"(eval(f) == eval(simplify(f)) on positive inputs)")
    for f, s1, mr in examples_val:
        print(f"   VALUE MISMATCH (median rel {mr:.2e}): {f}\n       -> {s1}")

    print()
    if failures or n_idem_fail or n_val_fail:
        print(f"PROBLEMS: {len(failures)} unit, {n_idem_fail} idempotence, {n_val_fail} value")
    else:
        print("ALL CHECKS PASSED")
