# imports
import os
import numpy as np
import torch
import random
from collections import Counter
from grammar import is_num_sign, is_num_mantissa, is_num_exp, decode_float
# Aliased: 'B' is already used as a local batch-size name inside beam_search.
from grammar import U as _PN_UNARY_OPS, B as _PN_BINARY_OPS, V as _PN_VARIABLES

# checks if num is a floating point number
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


# the function recieves a formula and outputs its size by walking through it recursively
def fSize(fml):
    return len(fml.split())


# computes the majority class ratio
def mca(num_lst):
    _, count = Counter(num_lst).most_common(1)[0]
    return float(count) / len(num_lst)


# a quick compute of literal accuracy of the predicted RPN compared to the target RPN
def model_acc(pred, target, ignore_index):
    padding_mask = (target == ignore_index)
    correct_bool = torch.eq(torch.argmax(pred, dim = -1), target)
    correct_bool = torch.logical_and(correct_bool, torch.logical_not(padding_mask))
    nb_correct = torch.sum(correct_bool)
    return nb_correct / torch.sum(torch.logical_not(padding_mask))


def inference(model, inputs, full_vocab, max_fml_len, device = "cpu", batch_size = 1024):
    # get the index for the <bes> as initial prompt
    bes_index = full_vocab.index("<bes>")

    if (isinstance(inputs, np.ndarray)):
        inputs = torch.from_numpy(inputs).float()
    else:
        inputs = inputs.float()

    # batch loop
    startI = 0
    prompt = None
    model.eval()
    with torch.no_grad():
        while (startI < inputs.shape[0]):
            # generate the initial prompt
            cur_prompt = None
            cur_inputs = None
            if (batch_size < inputs.shape[0] - startI):
                cur_prompt = torch.full((batch_size, 1), bes_index, dtype = torch.int64).to(device)
                cur_inputs = inputs[startI:startI + batch_size, :].to(device)
            else:
                cur_prompt = torch.full((inputs.shape[0] - startI, 1), bes_index, dtype = torch.int64).to(device)
                cur_inputs = inputs[startI:, :].to(device)

            # inference loop
            for tokI in range(max_fml_len):
                # generate the next prediction
                pred_next = model(cur_inputs, cur_prompt)

                # convert the predicted vectors into token indices
                next_tok = pred_next[:, -1, :].argmax(dim = -1).view(-1, 1)
                cur_prompt = torch.cat((cur_prompt, next_tok), dim = -1)

            # increment startI
            startI += batch_size

            # concatentate the entire prompt
            if (prompt is None):
                prompt = cur_prompt
            else:
                prompt = torch.cat((prompt, cur_prompt), dim = 0)
    return prompt, pred_next


def beam_search(model, inputs, full_vocab, max_fml_len, device="cpu", batch_size=64, beam_size=10):
    """
    Beam search decoding matching the NeurIPS 2022 paper (Kamienny et al.; beam_size=10 default).

    For each input example, maintains beam_size candidate sequences scored by
    cumulative log-probability. Returns the highest-scoring sequence per example.

    batch_size is the number of *examples* per chunk; the model sees
    batch_size * beam_size sequences simultaneously, so use a smaller value
    here than in greedy decoding to keep GPU memory stable.
    """
    bes_index = full_vocab.index("<bes>")
    pad_index = full_vocab.index("<pad>")

    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).float()
    else:
        inputs = inputs.float()

    N = inputs.shape[0]
    all_best = []

    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            cur_inputs = inputs[start:start + batch_size].to(device)    # (B, io_dim)
            B = cur_inputs.shape[0]

            # Encode once; expand encoder output for all beams so the
            # (expensive) encoder is never re-run inside the decoding loop.
            enc_output, enc_mask = model.encode(cur_inputs)             # (B, S, d)
            enc_output = enc_output.repeat_interleave(beam_size, dim=0) # (B*beam_size, S, d)
            enc_mask = enc_mask.repeat_interleave(beam_size, dim=0)  # (B*beam_size, S)

            # All beams start with [<bes>]: (B*beam_size, 1)
            seqs = torch.full((B * beam_size, 1), bes_index, dtype=torch.int64, device=device)

            # Cumulative log-prob scores (B, beam_size).
            # Only beam 0 per example is active at start; rest are -inf.
            scores = torch.full((B, beam_size), float('-inf'), device=device)
            scores[:, 0] = 0.0

            # Which beams have emitted a closing <bes>
            done = torch.zeros(B, beam_size, dtype=torch.bool, device=device)

            for _ in range(max_fml_len):
                seq_len = seqs.shape[1]

                # decoder only -- encoder output is cached
                logits = model.decode(seqs, enc_output, enc_mask)       # (B*beam_size, seq_len, V)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)    # (B*beam_size, V)
                V = log_probs.shape[-1]
                log_probs = log_probs.reshape(B, beam_size, V)              # (B, beam_size, V)

                # Finished beams: force <bes> continuation at zero additional cost
                log_probs[done] = float('-inf')
                log_probs[done, bes_index] = 0.0

                # All candidate scores: (B, beam_size, V)
                cand = scores.unsqueeze(-1) + log_probs

                # Pick top beam_size over flattened beam_size*V candidates
                cand_flat = cand.reshape(B, beam_size * V)
                top_scores, top_idx = cand_flat.topk(beam_size, dim=-1)    # (B, beam_size)

                beam_idx = top_idx // V    # which previous beam
                tok_idx  = top_idx % V     # which token

                # Rebuild sequences from chosen beams + new tokens
                seqs_3d     = seqs.reshape(B, beam_size, seq_len)
                beam_idx_exp = beam_idx.unsqueeze(-1).expand(-1, -1, seq_len)
                new_seqs    = seqs_3d.gather(1, beam_idx_exp)               # (B, beam_size, seq_len)
                new_seqs    = torch.cat([new_seqs, tok_idx.unsqueeze(-1)], dim=-1)

                scores = top_scores                                          # (B, beam_size)
                seqs   = new_seqs.reshape(B * beam_size, seq_len + 1)

                # Mark beams that just generated <bes> as done
                done = tok_idx == bes_index                                  # (B, beam_size)
                if done.all():
                    break

            # Select the highest-scoring beam per example
            best_idx  = scores.argmax(dim=-1)                               # (B,)
            seqs_3d   = seqs.reshape(B, beam_size, -1)
            best_seqs = seqs_3d[torch.arange(B, device=device), best_idx]   # (B, final_len)
            all_best.append(best_seqs.cpu())

    # Concatenate; pad to uniform length across chunks
    max_len = max(s.shape[1] for s in all_best)
    padded = []
    for s in all_best:
        if s.shape[1] < max_len:
            pad = torch.full((s.shape[0], max_len - s.shape[1]), pad_index, dtype=torch.int64)
            s = torch.cat([s, pad], dim=1)
        padded.append(s)

    return torch.cat(padded, dim=0), None


def set_seed(seed = 47):
    '''Completely set seeds'''
    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Additional for DataLoader workers
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA >= 10.2

    
def _reassemble_floats(toks: list) -> list:
    """Collapse (NUM+/NUM-, M_hi, M_lo, E+/-xx) quadruplets back into float strings.

    Any complete quadruplet is decoded and replaced with a single float string so
    that downstream formula evaluators receive standard numeric literals.
    Incomplete/mismatched quadruplets are left as-is.
    """
    out = []
    i = 0
    while i < len(toks):
        if (is_num_sign(toks[i])
                and i + 3 < len(toks)
                and is_num_mantissa(toks[i + 1])
                and is_num_mantissa(toks[i + 2])
                and is_num_exp(toks[i + 3])):
            out.append(repr(decode_float(toks[i], toks[i + 1], toks[i + 2], toks[i + 3])))
            i += 4
        else:
            out.append(toks[i])
            i += 1
    return out


def idx_to_toks(idx, full_vocab, num_fml_per_set = 1):
    # Used to convert a predicted index array back to the formula used for evaluation
    toks = []
    rpns = []
    if (len(idx.shape) == 2): # batch data
        for fmlI in range(idx.shape[0]):
            fml_toks = []
            for tokI in range(idx.shape[1]):
                fml_toks.append(full_vocab[idx[fmlI, tokI]])

            # first, strip the fml_toks from the front
            while (fml_toks and ("<" in fml_toks[0] and ">" in fml_toks[0])):
                fml_toks = fml_toks[1:]

            # construct the RPN formula set
            cur_fml_lst = []
            for tok in fml_toks:
                if (len(cur_fml_lst) != 0 and ("<" in tok and ">" in tok)):
                    break
                elif ("<" not in tok and ">" not in tok):
                    cur_fml_lst.append(tok)

            # collapse float triplets -> float string literals
            cur_fml_lst = _reassemble_floats(cur_fml_lst)
            rpns.append(" ".join(cur_fml_lst))

            # append the indices and the tokens
            toks.append(fml_toks)
    else: # single data
        fml_toks = []
        for tokI in range(idx.shape[0]):
            fml_toks.append(full_vocab[idx[tokI]])

        # first, strip the fml_toks from the front
        while (fml_toks and ("<" in fml_toks[0] and ">" in fml_toks[0])):
            fml_toks = fml_toks[1:]

        for tok in fml_toks:
            if ("<" not in tok and ">" not in tok):
                toks.append(tok)
            else:
                break

        # collapse float triplets -> float string literals
        toks = _reassemble_floats(toks)
        rpns = " ".join(toks)

    # return
    return  (toks, rpns)


# Utility function
def list_flatten(lst):
    # base case
    if (len(lst) == 0):
        return lst

    # inductive cases
    if (isinstance(lst[0], list)):
        res = list_flatten(lst[0])
        return res + list_flatten(lst[1:])
    else:
        return [lst[0]] + list_flatten(lst[1:])


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# ---------------------------------------------------------------------------
# Affine-prefactor removal
# ---------------------------------------------------------------------------
# grammar.add_prefactors wraps a skeleton in affine transformations:
#     'sin v1'  ->  '+ CONST sin + CONST * CONST v1'
# remove_affine undoes that, mapping a formula back to its skeleton.
#
# A subtree counts as constant when it contains no variable, so composite
# constants like '* 2 pi' qualify, not just single tokens.
#
# Float literals may arrive either as one token ('2.5') or, after
# grammar.tokenize_with_floats, as the 4-token quadruplet 'NUM+ M25 M00 E+0'.
# Both are single leaves here; missing that would mis-parse every span after one.
#
# Removed (S is the kept subtree, a is any constant subtree):
#     + a S   + S a   - a S   - S a      adding or subtracting a constant
#     * a S   * S a                      scaling by a constant
#     / S a                              dividing by a constant
#     neg S   ++ S   -- S                -S, S+1, S-1
#
# Kept, since these are not affine in S:
#     / a S                              a / S is a reciprocal
#     pow a S   pow S a                  exponentiation

_VAR_SET      = frozenset(_PN_VARIABLES)
_AFFINE_BIN   = frozenset(("+", "-", "*"))
_AFFINE_UNARY = frozenset(("neg", "++", "--"))


def _pn_arity(tok):
    """Operands a PN token takes.  Anything that is not an operator is a leaf,
    which covers variables, symbolic constants, CONST and float literals."""
    if tok in _PN_BINARY_OPS:
        return 2
    if tok in _PN_UNARY_OPS:
        return 1
    return 0


def _pn_leaf_len(tokens, i):
    """Tokens making up the leaf at tokens[i]: 4 for an encoded float, else 1.

    Guarded the same way grammar.detokenize_floats guards, so a stray sign token
    that is not followed by a full quadruplet stays a single leaf.
    """
    if (is_num_sign(tokens[i]) and i + 3 < len(tokens)
            and is_num_mantissa(tokens[i + 1])
            and is_num_mantissa(tokens[i + 2])
            and is_num_exp(tokens[i + 3])):
        return 4
    return 1


def _pn_span(tokens, i):
    """Index one past the end of the subtree rooted at tokens[i]."""
    arity = _pn_arity(tokens[i])
    if arity == 0:
        return i + _pn_leaf_len(tokens, i)
    end = i + 1
    for _ in range(arity):
        end = _pn_span(tokens, end)
    return end


def _pn_is_constant(tokens, i):
    """True when the subtree at tokens[i] holds no variable."""
    return not any(t in _VAR_SET for t in tokens[i:_pn_span(tokens, i)])


def _remove_affine_rec(tokens, i):
    """Strip affine wrappers from the subtree at tokens[i].

    Returns (kept tokens, index one past the ORIGINAL subtree) so the caller can
    keep walking the input even when the subtree shrank.
    """
    end = _pn_span(tokens, i)
    tok = tokens[i]

    # A subtree with no variable is itself a constant; there is nothing to pull out.
    if _pn_is_constant(tokens, i):
        return tokens[i:end], end

    arity = _pn_arity(tok)

    if arity == 2:
        left  = i + 1
        right = _pn_span(tokens, left)

        # Drop the operator when the constant sits on a side that makes it affine.
        # Division only qualifies when the constant is the divisor.
        if tok in _AFFINE_BIN and _pn_is_constant(tokens, left):
            return _remove_affine_rec(tokens, right)[0], end
        if (tok in _AFFINE_BIN or tok == "/") and _pn_is_constant(tokens, right):
            return _remove_affine_rec(tokens, left)[0], end

        return [tok] + _remove_affine_rec(tokens, left)[0] \
                     + _remove_affine_rec(tokens, right)[0], end

    if arity == 1:
        if tok in _AFFINE_UNARY:
            return _remove_affine_rec(tokens, i + 1)[0], end
        return [tok] + _remove_affine_rec(tokens, i + 1)[0], end

    return tokens[i:end], end


def remove_affine(fml):
    """Remove every affine transformation from a formula in polish notation.

    Any '+ * a S b' and its equivalents collapse to 'S'.  Constants a and b are
    whole constant subtrees, and are simply absent in the shapes where they would
    have been 0 or 1 ('* a S', '+ S b', 'neg S', ...).

    A formula holding no variable is returned unchanged.
    """
    tokens = fml.split()
    if not tokens:
        return ""
    return " ".join(_remove_affine_rec(tokens, 0)[0])


# ---------------------------------------------------------------------------
# End-to-end (Kamienny et al.) prefix -> our polish notation
# ---------------------------------------------------------------------------
# The vendored symbolicregression/ model writes prefix formulae in its own
# dialect (see symbolicregression/envs/generators.py):
#
#     separator   comma, not space          Node.prefix() joins with ','
#     binaries    add sub mul div pow       operators_real / operators_extra
#     unaries     abs inv sqrt log exp sin cos tan arcsin arccos arctan pow2 pow3
#     variables   x_0, x_1, ...             one-based v1..v10 here
#     constants   e pi euler_gamma CONSTANT, plus bare float literals
#
# Translating lets remove_affine work on E2E output unchanged, which matters
# because E2E's own _add_prefactors wraps skeletons in exactly the same shapes
# ours does ('add,a,mul,b,<child>').

_E2E_BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "pow"}

_E2E_UNARY = {
    "abs": "abs", "inv": "invert", "sqrt": "sqrt", "log": "ln", "exp": "exp",
    "sin": "sin", "cos": "cos", "tan": "tan",
    "arcsin": "arcsin", "arccos": "arccos", "arctan": "arctan",
    "pow2": "sqr",      # grammar.py calls pow2 an exact alias of sqr; use the canonical one
    "pow3": "pow3",
}

# 'e' and 'euler_gamma' have no token in our C.  They pass through untranslated
# and still behave correctly, since remove_affine treats every non-variable leaf
# as a constant -- which is what they are.
_E2E_LEAF = {"CONSTANT": "CONST", "pi": "pi", "e": "e", "euler_gamma": "euler_gamma"}


def e2e_to_pn(fml):
    """Translate an end-to-end prefix formula into the polish notation used here.

        e2e_to_pn('add,3.2,mul,1.7,sin,x_0')  ->  '+ 3.2 * 1.7 sin v1'
        remove_affine(e2e_to_pn(...))         ->  'sin v1'

    Raises ValueError on a token that is neither a known operator, a variable, a
    known constant nor a float.  Passing an unknown operator through would make it
    look like a leaf and silently mis-parse the rest of the tree.
    """
    out = []
    for tok in fml.replace(",", " ").split():
        if tok in _E2E_BINARY:
            out.append(_E2E_BINARY[tok])
        elif tok in _E2E_UNARY:
            out.append(_E2E_UNARY[tok])
        elif tok in _E2E_LEAF:
            out.append(_E2E_LEAF[tok])
        elif tok.startswith("x_"):
            idx = tok[2:]
            if not idx.isdigit():
                raise ValueError(f"malformed E2E variable {tok!r}")
            if int(idx) >= len(_PN_VARIABLES):
                raise ValueError(
                    f"E2E variable {tok!r} exceeds the {len(_PN_VARIABLES)} variables "
                    f"in this grammar")
            out.append(_PN_VARIABLES[int(idx)])          # x_0 -> v1
        elif is_number(tok):
            out.append(tok)                              # float literal stays a leaf
        else:
            raise ValueError(f"unknown E2E token {tok!r}")
    return " ".join(out)


if (__name__ == "__main__"):
    print("utils is invoked")



