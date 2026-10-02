# The grammar module contains the definition of the grammar and some relavent functions

from random import choice, randint, sample
import numpy as np
from funcWrappers import wrapExtGetHeadAndInputs
from math import log10, floor

############################### Public Constants #############################

# Terminal:
V = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]
C = ["0", "1", "2", "3", "pi"]   # symbolic constants -- the only constants under restrict_consts
C_ALL = C + ["CONST"]            # unrestricted generation set: adds the random-float placeholder
T = V + C                        # terminals used during formula generation

# Unary:
U = ["++", "--", "neg", "sqr", "sqrt", "exp", "ln", "sin", "cos", "tan", "abs", "arcsin", "arccos", "arctan", "invert", "pow2", "pow3", "tanh"]

# Binaries:
B = ["+", "-", "*", "/", "pow"]

# --------------------------- Operator sampling weights ---------------------------
# Two selectable modes (see set_operator_weights / the "operator_weights" config):
#
#   "uniform" -- every operator equally likely (the ORIGINAL behaviour; default).
#   "tiered"  -- a domain-GENERAL prior on how often each operator tends to appear in
#               real mathematical / scientific expressions.  It is NOT fit to any
#               benchmark; it only encodes the universal ordering
#                   arithmetic > squares/roots/reciprocal > core transcendentals
#                   > less common (tan, abs, x^3, x+/-1) > rare (inverse trig, tanh)
#                   > redundant (pow2 is an exact alias of sqr).
#               Every operator keeps a NONZERO weight so all stay reachable: division
#               and inverse trig are downweighted, never disabled -- real formulae need
#               ratios and the occasional arcsin.  Weights are relative (L1-normalised).
_U_RAW_WEIGHTS_UNIFORM = {op: 1.0 for op in U}
_B_RAW_WEIGHTS_UNIFORM = {op: 1.0 for op in B}

_U_RAW_WEIGHTS_TIERED = {
    "sqr": 1.0,                                   # x^2 -- ubiquitous
    "sqrt": 0.7, "invert": 0.6, "neg": 0.6,       # roots / 1*x^-1 / unary minus
    "sin": 0.5, "cos": 0.5,                       # core trig
    "exp": 0.4, "ln": 0.4,                         # exp / log
    "++": 0.35, "--": 0.35,                       # x+1 / x-1 (canonical form of +/-1)
    "pow3": 0.2, "abs": 0.2,                       # cube / abs
    "tan": 0.12,
    "arcsin": 0.1, "arctan": 0.1, "tanh": 0.1, "arccos": 0.08,   # rare
    "pow2": 0.05,                                  # redundant alias of sqr
}
_B_RAW_WEIGHTS_TIERED = {
    "*": 1.0, "+": 1.0, "-": 0.8,
    "/": 0.6,                                      # ratios are common -- downweighted, NOT disabled
    "pow": 0.1,                                    # general power rare (integer powers -> sqr/pow3/sqrt)
}


def _normalise_weights(raw, ops):
    w = np.array([raw[op] for op in ops], dtype=np.float64)
    return w / w.sum()


# Active probability vectors, read by the generators (np.random.choice) at call time.
# Default to UNIFORM so a bare `import grammar` reproduces the original behaviour;
# switch with set_operator_weights("tiered").
_U_WEIGHTS = _normalise_weights(_U_RAW_WEIGHTS_UNIFORM, U)
_B_WEIGHTS = _normalise_weights(_B_RAW_WEIGHTS_UNIFORM, B)


def set_operator_weights(mode):
    """Select the active operator-sampling distribution: "uniform" or "tiered".

    Mutates the module-level _U_WEIGHTS / _B_WEIGHTS that the generators read, so it
    is safe (and cheap) to call once per process -- including inside each DataLoader
    worker, which is how the setting survives multiprocessing.
    """
    global _U_WEIGHTS, _B_WEIGHTS
    if mode == "uniform":
        _U_WEIGHTS = _normalise_weights(_U_RAW_WEIGHTS_UNIFORM, U)
        _B_WEIGHTS = _normalise_weights(_B_RAW_WEIGHTS_UNIFORM, B)
    elif mode == "tiered":
        _U_WEIGHTS = _normalise_weights(_U_RAW_WEIGHTS_TIERED, U)
        _B_WEIGHTS = _normalise_weights(_B_RAW_WEIGHTS_TIERED, B)
    else:
        raise ValueError(f"unknown operator_weights mode: {mode!r} (use 'uniform' or 'tiered')")

# Floating-point number encoding tokens (mantissa_len=2 scheme).
# These are NOT part of the default grammar (FlatGram): under restrict_consts=True the
# grammar is mantissa-free and constants are restricted to the symbolic set C.  They ARE
# part of FlatGramFloat, the vocabulary used when random float constants are reachable --
# i.e. when prefactors are enabled (use_prefactors=True) or the constant set is
# unrestricted (restrict_consts=False).  See data_process.online_batch_iterater.
NUM_SIGN     = ["NUM+", "NUM-"]                              # 2 tokens
NUM_MANTISSA = [f"M{i:02d}" for i in range(100)]            # 100 tokens (M00 .. M99)
NUM_EXP      = [f"E{i:+d}" for i in range(-100, 101)]       # 201 tokens (E-100 .. E+100)
FLOAT_TOKENS = NUM_SIGN + NUM_MANTISSA + NUM_EXP            # 303 tokens

# The flattened grammar: terminals (variables + symbolic constants), unaries and
# binaries only.  No mantissa/float tokens.
FlatGram = V + C + U + B

# The float-capable grammar: FlatGram plus the mantissa/exponent tokens, so a target
# formula may carry arbitrary float literals (written by substitute_consts and expanded
# by tokenize_with_floats).  Note CONST itself is never a vocabulary token -- it is a
# generation-time placeholder that is always substituted before tokenization.
FlatGramFloat = FlatGram + FLOAT_TOKENS

# fast lookup sets (used by encoding/decoding helpers)
_NUM_SIGN_SET     = set(NUM_SIGN)
_NUM_MANTISSA_SET = set(NUM_MANTISSA)
_NUM_EXP_SET      = set(NUM_EXP)
_SYMBOLIC_SET     = set(V + C + U + B)
_V_SET            = set(V)

############################### Public Functions #############################

# ---------- floating-point token helpers (Kamienny et al. 2022 encoding scheme) ----------

def is_num_sign(tok):
    return tok in _NUM_SIGN_SET

def is_num_mantissa(tok):
    return tok in _NUM_MANTISSA_SET

def is_num_exp(tok):
    return tok in _NUM_EXP_SET

def encode_float(x: float) -> list:
    """Encode a float as [sign_tok, M_hi, M_lo, exp_tok] (4 tokens, mantissa_len=2).

    The 4-digit mantissa (0000-9999) is split into two 2-digit chunks so that
    the mantissa vocabulary is 100 tokens (M00-M99) instead of 10,000.
    """
    if x == 0.0:
        return ["NUM+", "M00", "M00", "E+0"]
    sign = "NUM+" if x > 0 else "NUM-"
    x_abs = abs(x)
    exp = int(floor(log10(x_abs)))
    exp = max(-100, min(100, exp))
    mantissa = int(round(x_abs * 10 ** (3 - exp)))
    mantissa = max(0, min(9999, mantissa))
    m_hi = mantissa // 100
    m_lo = mantissa % 100
    return [sign, f"M{m_hi:02d}", f"M{m_lo:02d}", f"E{exp:+d}"]

def decode_float(sign_tok: str, m_hi_tok: str, m_lo_tok: str, exp_tok: str) -> float:
    """Decode a (sign, M_hi, M_lo, exponent) quadruplet back to a float."""
    sign     = 1.0 if sign_tok == "NUM+" else -1.0
    mantissa = (int(m_hi_tok[1:]) * 100 + int(m_lo_tok[1:])) * 1e-3  # e.g. M31 M41 -> 3.141
    exp      = int(exp_tok[1:])
    return sign * mantissa * (10 ** exp)

def tokenize_with_floats(fml_str: str) -> list:
    """Tokenize a formula string, expanding raw float literals into 4-token sequences.

    Tokens already in the grammar pass through unchanged; any token that parses
    as a float (e.g. "2.4242", "-0.001") is expanded via encode_float().
    """
    result = []
    for tok in fml_str.split():
        if tok in _SYMBOLIC_SET:
            result.append(tok)
        else:
            try:
                result.extend(encode_float(float(tok)))
            except ValueError:
                result.append(tok)  # special tokens like <bes>/<pad> pass through
    return result

def detokenize_floats(toks: list) -> list:
    """Collapse (NUM+/NUM-, M_hi, M_lo, E+/-xx) quadruplets back into float strings.

    Returns a new token list where each float quadruplet is replaced by a single
    string representation of the decoded float value.
    """
    out = []
    i = 0
    while i < len(toks):
        if (is_num_sign(toks[i]) and i + 3 < len(toks)
                and is_num_mantissa(toks[i + 1])
                and is_num_mantissa(toks[i + 2])
                and is_num_exp(toks[i + 3])):
            out.append(repr(decode_float(toks[i], toks[i + 1], toks[i + 2], toks[i + 3])))
            i += 4
        else:
            out.append(toks[i])
            i += 1
    return out

# ---------- end float helpers ----------

# ---------- CONST placeholder helpers ----------

def sample_const() -> str:
    """Sample a random float constant log-uniformly over +/-[1e-3, 1e3].

    Log-uniform sampling gives full M_lo and E token coverage: the mantissa is
    no longer quantized to integer multiples of powers-of-10, so all 100 M_lo
    tokens are reachable, and the exponent spans E-3..E+2 evenly.
    """
    sign    = np.random.choice([-1, 1])
    log_abs = np.random.uniform(-3, 3)
    return repr(float(sign * (10.0 ** log_abs)))


def substitute_consts(fml_str: str) -> str:
    """Replace every CONST token with a freshly sampled float string.

    Call this before wrapExtEvalPN or tokenize_with_floats; each call
    samples independent values so different data multipliers get different
    concrete constants for the same formula skeleton.
    """
    return " ".join(sample_const() if tok == "CONST" else tok
                    for tok in fml_str.split())

# ---------- end CONST placeholder helpers ----------

# ---------- variable relabelling (Kamienny et al. 2022 equivalent) ----------

def relabel_variables(fml: str) -> tuple:
    """Rename variables to consecutive v1, v2, ... sorted by original index.

    Mirrors relabel_variables() from Kamienny et al. 2022: collects all
    variable tokens that appear in the formula, sorts them by numeric index,
    and renames them v1, v2, ... so there are no gaps.

    Returns (relabelled_fml, input_dimension) where input_dimension is the
    number of distinct variables actually used.
    """
    toks = fml.split()
    used = sorted(
        {tok for tok in toks if tok in _V_SET},
        key=lambda t: int(t[1:])
    )
    if not used:
        return fml, 0
    rename = {v: f"v{i + 1}" for i, v in enumerate(used)}
    return " ".join(rename.get(tok, tok) for tok in toks), len(used)

# ---------- end variable relabelling ----------

# ---------- add_prefactors (Kamienny et al. 2022 equivalent) ----------
#
# Injects a fittable multiplicative/additive constant in front of every additive term
# and every unary argument, which is how the end-to-end SR generator
# produces formulae.  It is OFF by default here: the mantissa-free grammar builds its
# constants from {0,1,2,3,pi} instead.  Enable it with use_prefactors=True to reproduce
# the e2e generation distribution -- see data_process.online_batch_iterater.

_ADD_SUB = frozenset(('+', '-'))
_U_SET   = frozenset(U)


def _ap_rec(tokens: list, i: int):
    """Recursively transform one PN subtree starting at tokens[i].

    Returns (transformed_str, next_index, changed).
    Mirrors the paper's _add_prefactors logic:
      +/- node  : each non-+/- child gets wrapped in  '* CONST <child>'
      unary node: if child root is not +/-, wrap as   '+ CONST * CONST <child>'
      other     : recurse unchanged
    The check is always on the *original* token at that position (before recursion).
    """
    tok = tokens[i]

    if tok in _ADD_SUB:                             # + or -
        left_orig  = tokens[i + 1]
        left_str,  j, _ = _ap_rec(tokens, i + 1)
        right_orig = tokens[j]
        right_str, k, _ = _ap_rec(tokens, j)

        if left_orig not in _ADD_SUB:
            left_str  = f"* CONST {left_str}"
        if right_orig not in _ADD_SUB:
            right_str = f"* CONST {right_str}"

        return f"{tok} {left_str} {right_str}", k, True

    elif tok in ('*', '/', 'pow'):                   # other binary
        left_str,  j, lc = _ap_rec(tokens, i + 1)
        right_str, k, rc = _ap_rec(tokens, j)
        return f"{tok} {left_str} {right_str}", k, lc or rc

    elif tok in _U_SET:                              # unary
        child_orig = tokens[i + 1]
        child_str, j, child_changed = _ap_rec(tokens, i + 1)

        if child_orig not in _ADD_SUB:
            child_str = f"+ CONST * CONST {child_str}"
            return f"{tok} {child_str}", j, True
        else:
            return f"{tok} {child_str}", j, child_changed

    else:                                            # terminal (V, C, CONST)
        return tok, i + 1, False


def add_prefactors(fml: str) -> str:
    """Inject float prefactors into a PN formula skeleton, matching the paper's
    add_prefactors() / _add_prefactors() methods.

    CONST placeholders are inserted (not concrete floats) so the caller can
    substitute independent values per data multiplier via substitute_consts().

    Example:
      'sin v1'          -> '+ CONST sin + CONST * CONST v1'
      '+ v1 v2'         -> '+ CONST + * CONST v1 * CONST v2'
      '* v1 v2'         -> '+ CONST * CONST * v1 v2'  (unchanged inner)
    """
    tokens = fml.split()
    result, _, changed = _ap_rec(tokens, 0)
    if not changed:
        result = f"* CONST {result}"
    return f"+ CONST {result}"

# ---------- end add_prefactors ----------

# ---------- lisp / PN / RPN format converters ----------

def lispToPN(lispStr: str) -> str:
    """Convert a Lisp-notation formula to Polish notation by stripping parentheses."""
    return lispStr.replace("(", "").replace(")", "")

# ---------- end format converters ----------


def get_arity(op):
    if (op in T or op == "CONST"): return 0
    elif (op in U): return 1
    elif (op in B): return 2


# the function mutates a formula that is in polish notation
def fMutate(fml, num_mutations = 3, fmlSizeLim = 10):
    num_attempts = 0
    attempt_lim = 10
    # mutate until condition fulfilled
    while (True):
        # restart with the original formula
        curFml = fml

        # mutate for a number of times
        for mI in range(num_mutations):
            # randomly choose an option
            n = randint(0, 3)

            # do different things for different options
            match n:
                # replace a random element with one of the same arity
                case 0:
                    # randomly select a node
                    fInd = randint(0, fSize(curFml) - 1)
                    curFml_toks = curFml.split()
                    cur_node = curFml_toks[fInd]

                    # randomly get an operator with the same arity
                    if (cur_node in T):
                        newNode = choice(T)
                    elif (cur_node in U):
                        newNode = choice(U)
                    elif (cur_node in B):
                        newNode = choice(B)

                    # replace the chosen node with the new node
                    curFml_toks[fInd] = newNode
                    curFml = " ".join(curFml_toks)

                # add a random element at a random location
                case 1:
                    if (len(curFml.split())):
                        gramI = randint(0, len(FlatGram) - 1)
                        fmlI = randint(0, fSize(curFml) - 1)
                        curFml_toks = curFml.split()
                        curFml_toks = curFml_toks[:fmlI] + [FlatGram[gramI]] + curFml_toks[fmlI:]
                        curFml = " ".join(curFml_toks)

                # remove a random element
                case 2:
                    if (len(curFml.split()) > 1):
                        fmlI = randint(0, fSize(curFml) - 1)
                        curFml_toks = curFml.split()
                        curFml_toks.pop(fmlI)
                        curFml = " ".join(curFml_toks)

                # swap two elements (ideally of the same arity)
                case 3:
                    curFml_toks = curFml.split()
                    if (len(curFml_toks) > 1):
                        # Determine arity of all formula elements
                        # Using a list comprehension for a clean Pythonic approach
                        arities = np.array([get_arity(tok) for tok in curFml_toks])
                        
                        # Check if all arities are unique
                        alldifferent = (len(curFml_toks) == len(np.unique(arities)))

                        # Find two indices of the same arity (if possible)
                        while (True):
                            i = sample(range(len(curFml_toks)), 2)
                            
                            # Accessing indices i[0] and i[1]
                            if (alldifferent or arities[i[0]] == arities[i[1]]):
                                break

                        # Swap elements (Python allows elegant one-line swapping)
                        curFml_toks[i[0]], curFml_toks[i[1]] = curFml_toks[i[1]], curFml_toks[i[0]]
                        curFml = " ".join(curFml_toks)


        # check for condition
        if (fSize(curFml) <= fmlSizeLim and curFml != fml):
            # done
            break
        elif (num_attempts < attempt_lim):
            # try again
            num_attempts += 1
        else:
            # tried enough, return a random formula
            curFml = randPN()
            break

    # return
    return curFml
    

# the function recieves a formula and outputs its size by walking through it recursively
def fSize(fml):
    # Recursive Case:
    return len(fml.split())


# ---------- uniform tree generation (combinatorial / Kamienny et al. 2022, after Lample & Charton 2020) ----------

def _build_dist_table(max_ops: int) -> list:
    """D[n][e] = number of binary trees with n binary ops from e empty nodes.

    Recurrence (binary-only):
        D[0][0] = 0,  D[0][e] = 1  for e >= 1
        D[n][e] = D[n][e-1] + D[n-1][e+1]
    """
    D = [[0] + [1] * (2 * max_ops)]
    for n in range(1, 2 * max_ops + 1):
        row = [0]
        for e in range(1, 2 * max_ops - n + 1):
            row.append(row[e - 1] + D[n - 1][e + 1])
        D.append(row)
    return D


# DIST stands for distribution
_DIST_MAX_OPS = 60
_DIST_TABLE   = _build_dist_table(_DIST_MAX_OPS)

# Companion cap on the OTHER half of the operator budget.  _uniform_randPN takes
# (nb_binary_ops, nb_unary_ops): _DIST_MAX_OPS bounds the first because the Catalan
# table above is precomputed only to that size, while the unary count has no such
# structural limit -- it is purely a distributional choice about how deeply nested a
# single draw may get.  It used to live as a bare `8` repeated in every caller that
# samples an operator budget (data_process.sample_formula plus the study scripts that
# mirror its three branches), so raising it meant finding all of them and the mirrors
# silently drifted.  Both caps now live here, and every sampler imports them.
_MAX_UNARY_OPS = 20


def _sample_next_pos_uniform(nb_empty: int, nb_ops: int) -> int:
    """Sample the next expansion slot weighted by the number of valid completions."""
    weights = np.array(
        [_DIST_TABLE[nb_ops - 1][nb_empty - i + 1] for i in range(nb_empty)],
        dtype=np.float64,
    )
    weights /= _DIST_TABLE[nb_ops][nb_empty]
    return int(np.random.choice(nb_empty, p=weights))


class _Node:
    """Minimal tree node used internally during formula generation."""
    __slots__ = ("value", "children", "is_pow_exp")

    def __init__(self):
        self.value      = None
        self.children   = []
        self.is_pow_exp = False   # True when this leaf is the exponent slot of a `pow` node

    def push_child(self, c: "_Node") -> None:
        self.children.append(c)

    def to_pn(self) -> str:
        if not self.children:
            return self.value
        return self.value + " " + " ".join(c.to_pn() for c in self.children)


_PROB_CONST = 0.2   # probability of a non-coverage leaf being a constant

# Constants used as the exponent of `pow`: small integers dominate in physics (x^2, x^3, ...).
_POW_EXP_CHOICES = ["2", "3"]   # 50 / 50 split (restrict_consts=True)
# Unrestricted variant (restrict_consts=False), matching the E2E generator: a CONST exponent lets
# BFGS fit fractional or negative exponents at inference time.
_POW_EXP_CHOICES_FULL = ["2", "3", "2", "3", "CONST"]   # 40 / 40 / 20 % split


# ---------- composite (multi-token) constant generation ----------
#
# A mantissa-free grammar cannot emit arbitrary floats, so numeric constants other
# than the symbolic tokens {0,1,2,3,pi} must be ASSEMBLED from them with arithmetic:
#   1/2 -> '/ 1 2'      4 -> '* 2 2'      3/5 -> '/ 3 (+ 2 3)'      -1/2 -> 'neg / 1 2'
# These helpers let constant leaves (and pow exponents) expand into such small
# subexpressions so the model is trained to build the composite constants that
# pervade physics formulae instead of relying on a single token.
_PROB_COMPOSITE_CONST = 0.5      # P(a constant leaf is composite rather than a single token)
_PROB_COMPOSITE_POW   = 0.35     # P(a pow exponent is composite rather than drawn from _POW_EXP_CHOICES)
_PROB_NEG_CONST       = 0.25     # P(a composite constant is wrapped in neg -> negative constant)
_CONST_LEAF_STOP      = 0.45     # P(stop at a leaf at each level of a constant subtree)
_CONST_MAX_BIN_OPS    = 3        # max binary ops inside a composite constant leaf
_CONST_MAX_POW_OPS    = 1        # max binary ops inside a composite pow exponent (keep exponents small)
_CONST_LEAVES         = ["1", "2", "3", "pi"]   # 0 excluded: avoids 0-denominators / degenerate consts
_CONST_BIN_OPS        = ["+", "-", "*", "/"]


def _gen_const_subtree(max_bin_ops: int, allow_neg: bool = True) -> "_Node":
    """Return a _Node holding a constant expression over {1,2,3,pi}.

    Uses up to max_bin_ops binary operators from {+,-,*,/}; with probability
    _PROB_NEG_CONST the whole expression is wrapped in `neg`, so negative constants
    (-1/2, -5, ...) are reachable.  0 is excluded from the leaves so denominators are
    rarely zero; the occasional 0-denominator that still slips through (e.g. '- 2 2')
    is dropped downstream by the finite-output filter, and simplify_formula folds the
    trivial cases (e.g. '+ 1 1' -> '2', '* 2 2' -> 'sqr 2').
    """
    if max_bin_ops <= 0 or np.random.random() < _CONST_LEAF_STOP:
        node = _Node()
        node.value = choice(_CONST_LEAVES)
    else:
        node = _Node()
        node.value = choice(_CONST_BIN_OPS)
        left_ops  = np.random.randint(0, max_bin_ops)      # 0 .. max_bin_ops-1
        right_ops = max_bin_ops - 1 - left_ops
        node.push_child(_gen_const_subtree(left_ops, allow_neg=False))
        node.push_child(_gen_const_subtree(right_ops, allow_neg=False))
    if allow_neg and np.random.random() < _PROB_NEG_CONST:
        wrapper = _Node()
        wrapper.value = "neg"
        wrapper.push_child(node)
        return wrapper
    return node


def _pn_to_node(pn: str) -> "_Node":
    """Parse a Polish-notation string into a _Node tree using grammar arities."""
    toks = pn.split()
    pos = 0

    def build():
        nonlocal pos
        tok = toks[pos]
        pos += 1
        node = _Node()
        node.value = tok
        for _ in range(get_arity(tok) or 0):
            node.push_child(build())
        return node

    return build()


def _expand_const_leaf(node: "_Node", max_bin_ops: int) -> None:
    """Expand an existing leaf node in place into a composite constant subtree.

    The freshly sampled subtree is run through simplify() so the embedded constant
    is its canonical simplest form (e.g. '* 2 2' -> 'sqr 2', '+ 1 1' -> '2', folding
    away degenerate draws).  simplify is value-preserving, so the constant's value is
    unchanged.  Imported lazily to avoid the grammar <-> simplifyFormula import cycle.

    It goes through simplify_backend, not simplifyFormula directly, so that constant
    leaves are canonicalised by the SAME engine as the formula around them.  This is
    not cosmetic: the two backends agree on every purely rational subtree (both route
    integers through rational_to_pn) but disagree on 14% of pi-bearing ones, almost all
    of it term order ('+ sqr 2 invert pi' vs '+ invert pi sqr 2').  Leaving this call
    pinned to ours would leave those leaves in our order inside an otherwise-SimpliPy
    target.
    """
    from simplify_backend import simplify
    sub = _pn_to_node(simplify(_gen_const_subtree(max_bin_ops).to_pn()))
    node.value    = sub.value
    node.children = sub.children


def _fill_leaves(leaf_nodes: list, input_dimension: int, restrict_consts: bool = True) -> None:
    """Fill leaf nodes with terminals.

    Pow-exponent leaves (is_pow_exp=True) are filled with a constant -- usually a
    single small integer from _POW_EXP_CHOICES, but with probability
    _PROB_COMPOSITE_POW a small composite exponent ('* 2 2' = 4, '/ 3 2' = 3/2,
    'neg 2' = -2).  They are processed last so they never consume variable-coverage
    slots.

    Remaining leaves mirror generate_leaf from Kamienny et al. 2022: all
    input_dimension variables appear at least once before repeats are allowed.  A
    non-coverage leaf becomes a constant with probability _PROB_CONST, and such a
    constant is itself a composite subexpression ('/ 1 2', '/ 3 (+ 2 3)') with
    probability _PROB_COMPOSITE_CONST.  leaf_nodes should already be shuffled by the
    caller.

    restrict_consts=False switches to the UNRESTRICTED constant set (the E2E
    behaviour): a constant leaf is drawn uniformly from C_ALL = C + [CONST] and a
    pow exponent from _POW_EXP_CHOICES_FULL, so a random float placeholder is
    reachable directly.  Composite constants are then unnecessary and are skipped --
    arbitrary values are one CONST away rather than an assembled subexpression.
    """
    pow_exp = [n for n in leaf_nodes if n.is_pow_exp]
    regular = [n for n in leaf_nodes if not n.is_pow_exp]
    ordered = regular + pow_exp   # coverage-guarantee leaves first

    n_used = 0
    for n in ordered:
        if n.is_pow_exp:
            if not restrict_consts:
                n.value = np.random.choice(_POW_EXP_CHOICES_FULL)
            elif np.random.random() < _PROB_COMPOSITE_POW:
                _expand_const_leaf(n, _CONST_MAX_POW_OPS)   # composite exponent: 4, 3/2, -2, ...
            else:
                n.value = np.random.choice(_POW_EXP_CHOICES)
        elif n_used < input_dimension:
            n.value = f"v{n_used + 1}"
            n_used += 1
        elif np.random.random() < _PROB_CONST:
            if not restrict_consts:
                n.value = choice(C_ALL)
            elif np.random.random() < _PROB_COMPOSITE_CONST:
                _expand_const_leaf(n, _CONST_MAX_BIN_OPS)   # composite constant: 1/2, 4, 3/5, ...
            else:
                n.value = choice(C)
        else:
            n.value = f"v{np.random.randint(1, input_dimension + 1)}"


def _generate_binary_tree(nb_binary_ops: int, input_dimension: int,
                          restrict_consts: bool = True) -> _Node:
    """Uniformly sample a binary tree with exactly nb_binary_ops binary operators.

    Implements the placement algorithm from the end-to-end SR paper (Kamienny et al.,
    NeurIPS 2022; symbolicregression repo), which itself follows the unary-binary tree
    sampling of Lample & Charton (ICLR 2020): each operator position is drawn
    proportionally to the number of valid completions of the remaining subtree, ensuring
    a uniform distribution over all structurally distinct binary trees of this size.
    Leaves are filled with variables v1..v{input_dimension} (coverage-first) or constants.
    """
    root = _Node()
    if nb_binary_ops == 0:
        root.value = "v1" if input_dimension >= 1 else choice(C if restrict_consts else C_ALL)
        return root

    assert nb_binary_ops <= _DIST_MAX_OPS, \
        f"nb_binary_ops={nb_binary_ops} exceeds _DIST_MAX_OPS={_DIST_MAX_OPS}"

    empty_nodes = [root]
    next_en     = 0
    nb_empty    = 1
    remaining   = nb_binary_ops

    while remaining > 0:
        pos          = _sample_next_pos_uniform(nb_empty, remaining)
        next_en     += pos
        node         = empty_nodes[next_en]
        node.value   = str(np.random.choice(B, p=_B_WEIGHTS))
        for _ in range(2):
            child = _Node()
            node.push_child(child)
            empty_nodes.append(child)
        if node.value == "pow":
            node.children[1].is_pow_exp = True  # exponent slot -> always a constant
        next_en  += 1
        nb_empty += 1 - pos
        remaining -= 1

    leaf_nodes = [n for n in empty_nodes if not n.children]
    np.random.shuffle(leaf_nodes)   # randomise which leaf gets which coverage slot
    _fill_leaves(leaf_nodes, input_dimension, restrict_consts)
    return root


def _collect_child_slots(node: _Node) -> list:
    """Return all (parent, child_index) pairs in the subtree -- potential unary insertion points."""
    slots = []
    for i, child in enumerate(node.children):
        slots.append((node, i))
        slots.extend(_collect_child_slots(child))
    return slots


def _insert_unaries(root: _Node, nb_unaries: int) -> None:
    """Insert nb_unaries unary operators at randomly chosen child slots."""
    if nb_unaries == 0:
        return
    slots    = _collect_child_slots(root)
    n_insert = min(nb_unaries, len(slots))
    chosen   = np.random.choice(len(slots), size=n_insert, replace=True)
    for idx in chosen:
        parent, child_idx          = slots[idx]
        wrapper                    = _Node()
        wrapper.value              = str(np.random.choice(U, p=_U_WEIGHTS))
        wrapper.push_child(parent.children[child_idx])
        parent.children[child_idx] = wrapper


def _uniform_randPN(nb_binary_ops: int, nb_unary_ops: int, input_dimension: int,
                    restrict_consts: bool = True) -> str:
    """Return a random PN formula string via combinatorial uniform sampling.

    Builds the binary skeleton uniformly over all valid binary trees with
    nb_binary_ops operators, then inserts nb_unary_ops unaries at random child slots.
    Leaves use variables v1..v{input_dimension} with coverage guarantee.

    restrict_consts=False draws constant leaves from C_ALL (see _fill_leaves), so the
    result may contain CONST placeholders that the caller must substitute.
    """
    if nb_binary_ops == 0:
        node = _Node()
        node.value = "v1" if input_dimension >= 1 else choice(C if restrict_consts else C_ALL)
        for _ in range(nb_unary_ops):
            wrapper       = _Node()
            wrapper.value = str(np.random.choice(U, p=_U_WEIGHTS))
            wrapper.push_child(node)
            node          = wrapper
        return node.to_pn()
    root = _generate_binary_tree(nb_binary_ops, input_dimension, restrict_consts)
    _insert_unaries(root, nb_unary_ops)
    return root.to_pn()


def sample_simplified_pn(nb_binary_ops: int, nb_unary_ops: int, input_dimension: int,
                         simplify: bool = True, restrict_consts: bool = True) -> str:
    """Uniform-sample a PN formula and return its canonical simplest form.

    This is the generator's FINAL formula-level pass.  Constants are already
    canonicalised per-constant during leaf-filling (see _expand_const_leaf); this
    runs simplify() once over the whole assembled formula so callers (data_process)
    receive a simplest-form formula directly and do not run a simplify step of their
    own.  WHICH canonicaliser runs is the process-level simplify_backend setting
    ("ours" or "simplipy" -- see simplify_backend.py); it is imported lazily to avoid
    the grammar <-> simplifyFormula cycle.

    simplify=False returns the RAW uniform draw instead, for the target-form ablation
    (see "simplify_targets" in train_transformer.py).  Composite constant leaves are
    still canonicalised per-leaf by _expand_const_leaf, so only the formula STRUCTURE
    is left unsimplified.  Note the raw draw may collapse to a constant ('- v1 v1');
    it is the caller's job to reject those -- data_process.online_batch_iterater does.

    restrict_consts=False draws constant leaves from C_ALL, so the returned formula may
    contain CONST placeholders; simplify() treats CONST as an opaque atom and leaves it
    in place, and the caller substitutes concrete floats afterwards.
    """
    from simplify_backend import simplify as _simplify
    pn = _uniform_randPN(nb_binary_ops, nb_unary_ops, input_dimension, restrict_consts)
    return _simplify(pn) if simplify else pn

# ---------- end uniform tree generation ----------


def randPN(sizeLimit: int = 10, input_dimension: int = None,
           restrict_consts: bool = True) -> str:
    """Random PN formula with approximately sizeLimit total tokens.

    input_dimension controls how many variables (v1..vN) the formula may use.
    If None, N is sampled uniformly from [1, n_leaves] where n_leaves = nb_binary_ops+1,
    ensuring all N variables are guaranteed to appear at least once (coverage guarantee).

    When input_dimension is given, nb_binary_ops is sampled from
    [input_dimension-1, (sizeLimit-1)//2] so the tree is guaranteed to have at
    least input_dimension leaves and all requested variables appear (full coverage).
    When input_dimension is None it is sampled from [1, n_leaves] as before.
    """
    assert sizeLimit >= 1, "sizeLimit must not be lower than 1"
    max_binary_ops = (sizeLimit - 1) // 2
    if input_dimension is not None:
        # Guarantee the tree has at least input_dimension leaves so all requested
        # variables can appear.  nb_binary_ops must be >= input_dimension - 1.
        min_binary_ops = min(input_dimension - 1, max_binary_ops)
        nb_binary_ops  = randint(min_binary_ops, max_binary_ops)
    else:
        nb_binary_ops  = randint(0, max_binary_ops)
    n_leaves      = nb_binary_ops + 1
    if input_dimension is None:
        # Cap at available leaves so _fill_leaves can guarantee full coverage.
        input_dimension = int(np.random.randint(1, min(len(V), n_leaves) + 1))
    else:
        input_dimension = min(input_dimension, n_leaves)  # safety; shouldn't trigger
    remaining    = sizeLimit - (2 * nb_binary_ops + 1)
    nb_unary_ops = randint(0, max(0, remaining))
    return _uniform_randPN(nb_binary_ops, nb_unary_ops, input_dimension, restrict_consts)


def randFML(sizeLimit: int = 10, input_dimension: int = None,
            restrict_consts: bool = True) -> str:
    """Random formula in Polish notation (no parentheses).

    Backward-compatible wrapper around randPN. Callers that apply lispToPN()
    on the result are unaffected since lispToPN is a no-op on paren-free strings.
    """
    return randPN(sizeLimit=sizeLimit, input_dimension=input_dimension,
                  restrict_consts=restrict_consts)


##################### Private Functions #########################

# Replace a subfml from fml specified by ind
def _replaceFmlNodeByInd(fml, ind, newNode):
    # the index of the element we want to replace cannot exceed the size of fml
    if (ind >= fSize(fml)): return fml

    # analyze the current element hct
    [head, inputs] = wrapExtGetHeadAndInputs(fml);

    # Handle base cases
    if (ind == 0): return f"({newNode} {' '.join(inputs)})"

    # if there is no inputs, just return head
    if (len(inputs) == 0): return head

    # decrement index by 1
    ind -= 1

    # depth-first index search
    for inputI in range(len(inputs)):
        # depth-first search the subFml
        subfml = inputs[inputI]
        newSfml = _replaceFmlNodeByInd(subfml, ind, newNode)

        # index is reduced by size of subfml
        ind -= fSize(subfml)
        
        # if element is found
        if (newSfml != subfml or ind < 0):
            inputs[inputI] = newSfml
            return f"({head} {' '.join(inputs)})"


# Replace a subfml from fml specified by ind
def _replaceSubfmlByInd(fml, ind, newSubfml):
    # Handle base cases
    if (ind == 0): return newSubfml

    # the index of the element we want to replace cannot exceed the size of fml
    if (ind >= fSize(fml)): return fml

    # decrement index by 1
    ind -= 1

    # analyze the current element hct
    [head, inputs] = wrapExtGetHeadAndInputs(fml);

    # if there is no inputs, just return head
    if (len(inputs) == 0): return head

    # depth-first index search
    for inputI in range(len(inputs)):
        # depth-first search the subFml
        subfml = inputs[inputI]
        newSfml = _replaceSubfmlByInd(subfml, ind, newSubfml)

        # index is reduced by size of subfml
        ind -= fSize(subfml)
        
        # if element is found
        if (newSfml != subfml or ind < 0):
            inputs[inputI] = newSfml
            return f"({head} {' '.join(inputs)})"


if (__name__ == "__main__"):
    print(sample_const())
    '''
    print("grammar is invoked")
    fml = randFML(sizeLimit = randint(1, 11))
    print(fml)
    fml.replace("(", "").replace(")", "")
    print(len(fml.split()))
    print(fMutate(fml))
    print(fMutate(fml))
    print(fMutate(fml))
    '''

