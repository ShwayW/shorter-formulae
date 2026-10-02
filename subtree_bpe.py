"""subtree_bpe.py

BPE vocabulary learner for PN formula strings.

At each step the most frequent contiguous token subsequence (length >= 2) is
merged into a new compound token.  Both complete subformulas (e.g. '+ v1 v2')
and partial prefixes (e.g. '+ v1', analogous to 'tion'/'er' in text BPE) are
eligible, so the learned vocabulary captures all recurring patterns regardless
of tree-boundary alignment.  Formulae come from the trainer's own sampler
(grammar.sample_simplified_pn), so the vocabulary matches the training distribution.

Usage (standalone):
    python subtree_bpe.py --n_formulas 50000 --n_merges 500 --out vocab/bpe_vocab.pkl

Usage (library):
    from subtree_bpe import SubtreeBPE, generate_bpe_corpus

    corpus = generate_bpe_corpus(n_formulas=50_000)
    bpe = SubtreeBPE()
    bpe.fit(corpus, n_merges=500, min_freq=5)
    bpe.save("vocab/bpe_vocab.pkl")

    bpe = SubtreeBPE.load("vocab/bpe_vocab.pkl")
    tokens = bpe.tokenize("sin + v1 v2")   # -> ['<BPE_3>'] or similar
"""

import argparse
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np

from grammar import V, FlatGram, sample_simplified_pn, relabel_variables


def _replace(tokens, target, new_tok):
    """Replace all non-overlapping left-to-right occurrences of target tuple with new_tok."""
    n      = len(target)
    result = []
    i      = 0
    while i < len(tokens):
        if tuple(tokens[i: i + n]) == target:
            result.append(new_tok)
            i += n
        else:
            result.append(tokens[i])
            i += 1
    return result


# -- Corpus generation ---------------------------------------------------------

def generate_bpe_corpus(n_formulas, fml_len_range=(1, 40), seed=0):
    """Generate a list of canonical PN formula strings for BPE training.

    Uses the SAME sampler as the trainer -- grammar.sample_simplified_pn (uniform
    tree + composite constants + the final simplify pass) -- so the learned vocabulary
    matches the distribution of formulae the model is actually trained on.

    Parameters
    ----------
    n_formulas     : number of formulas to generate.
    fml_len_range  : (min, max) symbolic token count after simplification.
    seed           : RNG seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)

    var_dim          = len(V)
    min_sym, max_sym = fml_len_range
    max_vars_fitting = (max_sym + 1) // 2
    corpus           = []

    while len(corpus) < n_formulas:
        input_dim     = np.random.randint(1, min(var_dim, max_vars_fitting) + 1)
        nb_binary_ops = np.random.randint(max(0, input_dim - 1), input_dim + 5)
        nb_unary_ops  = np.random.randint(0, 5)

        fml = sample_simplified_pn(nb_binary_ops, nb_unary_ops, input_dim)
        fml, simp_dim = relabel_variables(fml)
        if simp_dim == 0:
            continue

        sym_len = len(fml.split())
        if not (min_sym <= sym_len <= max_sym):
            continue

        corpus.append(fml)

    return corpus


# -- SubtreeBPE ----------------------------------------------------------------

class SubtreeBPE:
    """BPE vocabulary learner for PN formula strings.

    Each learned token '<BPE_i>' represents a frequently-occurring contiguous
    token subsequence -- either a complete subformula or a partial prefix.
    The full primitive-token expansion is stored in token_to_pn.

    Fitting
    -------
    bpe = SubtreeBPE()
    bpe.fit(corpus, n_merges=500)

    Tokenization
    ------------
    tokens = bpe.tokenize("sin + v1 v2")   # -> primitive + compound tokens, all in extended_vocab()

    Extended vocabulary
    -------------------
    vocab = bpe.extended_vocab()          # FlatGram + compound tokens
    tok2ind = {t: i for i, t in enumerate(vocab + ['<bes>', '<pad>'])}
    """

    def __init__(self):
        self.merges      = []   # [(target_tuple, new_token), ...] in learn order
        self.token_to_pn = {}   # '<BPE_i>' -> 'sin v1'

    # -- learning -------------------------------------------------------------

    def fit(self, corpus, n_merges=200, min_freq=5, verbose=True):
        """Learn BPE merges from a corpus of PN formula strings.

        At each step, finds the most frequent contiguous token subsequence of
        length >= 2 across the corpus and adds it as a new vocabulary token, then
        replaces all occurrences.

        If called on an instance that already has merges (e.g. loaded via
        SubtreeBPE.load), the existing merges are applied first to reconstruct
        the mid-fit corpus state, then fitting continues from where it left off.
        n_merges is the number of *additional* merges to perform in this call.

        Parameters
        ----------
        corpus    : list of PN formula strings (e.g. from generate_bpe_corpus).
        n_merges  : maximum number of *additional* merges to perform.
        min_freq  : stop if the best candidate frequency falls below this.
        verbose   : print a line per merge.
        """
        token_lists = [fml.split() for fml in corpus]

        if self.merges:
            for target_tup, new_tok in self.merges:
                token_lists = [_replace(tl, target_tup, new_tok) for tl in token_lists]
            if verbose:
                avg_len = sum(len(tl) for tl in token_lists) / len(token_lists)
                print(
                    f"[subtree_bpe] Resumed from {len(self.merges)} existing merges "
                    f"(avg seq len {avg_len:.2f})"
                )

        start_i = len(self.merges)
        for merge_i in range(start_i, start_i + n_merges):
            counter = Counter()
            for toks in token_lists:
                L = len(toks)
                for i in range(L):
                    for j in range(i + 2, L + 1):
                        counter[tuple(toks[i:j])] += 1

            if not counter:
                break

            best_tup, best_count = counter.most_common(1)[0]
            if best_count < min_freq:
                if verbose:
                    print(f"[subtree_bpe] Early stop: best freq {best_count} < min_freq {min_freq}")
                break

            new_tok = f"<BPE_{merge_i}>"
            pn_str  = " ".join(best_tup)
            self.merges.append((best_tup, new_tok))
            self.token_to_pn[new_tok] = pn_str

            token_lists = [_replace(tl, best_tup, new_tok) for tl in token_lists]

            if verbose:
                avg_len = sum(len(tl) for tl in token_lists) / len(token_lists)
                print(
                    f"  [{merge_i + 1:4d}] freq={best_count:6d}  "
                    f"'{pn_str}'  ->  {new_tok}  "
                    f"(avg seq len {avg_len:.2f})"
                )

        if verbose:
            avg_final = sum(len(tl) for tl in token_lists) / len(token_lists)
            print(
                f"[subtree_bpe] Done: {len(self.merges)} merges, "
                f"avg final seq len {avg_final:.2f}"
            )

    # -- tokenization ---------------------------------------------------------

    def tokenize(self, fml_str):
        """Tokenize a PN formula string using the learned BPE merges.

        Applies merges in learn order (same as fit-time replacement).
        Returns a list of tokens -- a mix of primitive grammar tokens and
        compound '<BPE_i>' tokens, all present in extended_vocab().
        """
        toks = fml_str.split()
        for target_tup, new_tok in self.merges:
            toks = _replace(toks, target_tup, new_tok)
        return toks

    # -- vocabulary -----------------------------------------------------------

    @property
    def compound_vocab(self):
        """List of all learned compound tokens in merge order."""
        return [new_tok for _, new_tok in self.merges]

    def extended_vocab(self, base_grammar=None):
        """Return base grammar tokens concatenated with learned compound tokens.

        Parameters
        ----------
        base_grammar : list of base token strings.
                       Defaults to FlatGram from grammar.
        """
        if base_grammar is None:
            base_grammar = FlatGram
        return list(base_grammar) + self.compound_vocab

    # -- persistence ----------------------------------------------------------

    def save(self, path, corpus_params=None):
        """Pickle the BPE model to path.

        corpus_params : dict of generate_bpe_corpus kwargs (stored so that
                        --resume can regenerate the identical corpus).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"merges": self.merges, "token_to_pn": self.token_to_pn}
        if corpus_params is not None:
            payload["corpus_params"] = corpus_params
        elif hasattr(self, "corpus_params"):
            payload["corpus_params"] = self.corpus_params
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[subtree_bpe] Saved {len(self.merges)} merges -> {path}")

    @classmethod
    def load(cls, path):
        """Load a pickled BPE model from path."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj             = cls()
        obj.merges      = data["merges"]
        obj.token_to_pn = data["token_to_pn"]
        obj.corpus_params = data.get("corpus_params")
        return obj


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn subtree-BPE vocabulary for PN formulas")
    parser.add_argument("--n_formulas",     type=int, default=100_000)
    parser.add_argument("--n_merges",       type=int, default=200,
                        help="Merges to learn (additional merges when --resume is set)")
    parser.add_argument("--min_freq",       type=int, default=5)
    parser.add_argument("--fml_len_min",    type=int, default=1)
    parser.add_argument("--fml_len_max",    type=int, default=40)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--out",            type=str, default="vocab/bpe_vocab.pkl")
    parser.add_argument("--resume",         action="store_true",
                        help="Load existing --out file and continue fitting")
    args = parser.parse_args()

    if args.resume:
        print(f"[subtree_bpe] Resuming from {args.out} ...", flush=True)
        bpe = SubtreeBPE.load(args.out)
        if bpe.corpus_params is not None:
            cp = bpe.corpus_params
            print(
                f"[subtree_bpe] Using stored corpus params: "
                f"n_formulas={cp['n_formulas']}, seed={cp['seed']}, "
                f"fml_len_range={cp['fml_len_range']}",
                flush=True,
            )
            corpus_params = cp
        else:
            corpus_params = dict(
                n_formulas     = args.n_formulas,
                fml_len_range  = (args.fml_len_min, args.fml_len_max),
                seed           = args.seed,
            )
            print(
                f"[subtree_bpe] WARNING: saved file has no corpus_params (old format). "
                f"Using CLI args -- make sure they match the original run: "
                f"n_formulas={corpus_params['n_formulas']}, seed={corpus_params['seed']}, "
                f"fml_len_range={corpus_params['fml_len_range']}",
                flush=True,
            )
    else:
        corpus_params = dict(
            n_formulas     = args.n_formulas,
            fml_len_range  = (args.fml_len_min, args.fml_len_max),
            seed           = args.seed,
        )
        bpe = SubtreeBPE()

    corpus_params.pop("use_prefactors", None)   # drop stale key from old saved vocabs
    print(f"[subtree_bpe] Generating corpus ({corpus_params['n_formulas']:,} formulas)...", flush=True)
    corpus = generate_bpe_corpus(**corpus_params)
    avg_len = sum(len(f.split()) for f in corpus) / len(corpus)
    print(f"[subtree_bpe] Corpus ready. Avg formula length: {avg_len:.2f} tokens", flush=True)

    bpe.fit(corpus, n_merges=args.n_merges, min_freq=args.min_freq, verbose=True)
    bpe.save(args.out, corpus_params=corpus_params)

    ext_vocab = bpe.extended_vocab()
    print(f"[subtree_bpe] Extended vocab size: {len(ext_vocab)} "
          f"({len(FlatGram)} base + {len(bpe.compound_vocab)} compound)")

    print("\n[subtree_bpe] Sample tokenizations (first 5 formulas):")
    for fml in corpus[:5]:
        toks = bpe.tokenize(fml)
        print(f"  {fml}")
        print(f"    -> {toks}")
