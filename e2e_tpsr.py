"""
e2e_tpsr.py -- TPSR using Meta's e2e symbolic regression transformer as backbone.

Supports uct_alg in {'uct', 'p_uct', 'var_p_uct'} matching dyna_gym's UCT agent.
"""

import math
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# MCTS node (mirrors tpsr.MCTSNode exactly)
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = (
        "seq", "parent", "prior",
        "children", "visit_count", "sampled_returns",
        "is_terminal",
        "cached_kv", "next_logits",
        "unexplored_actions",
        "_hidden",
    )

    def __init__(self, seq: list, parent: Optional["MCTSNode"] = None,
                 prior: float = 0.0) -> None:
        self.seq = seq
        self.parent = parent
        self.prior = prior
        self.children: dict = {}
        self.visit_count: int = 0
        self.sampled_returns: list = []
        self.is_terminal: bool = False
        self.cached_kv = None
        self.next_logits = None
        self.unexplored_actions = None
        self._hidden = None

    @property
    def Q(self) -> float:
        return max(self.sampled_returns) if self.sampled_returns else 0.0

    def uct_score(self, ucb_constant: float) -> float:
        N_parent = self.parent.visit_count if self.parent else 1
        N_child  = len(self.sampled_returns)
        if N_child == 0:
            return float('inf')
        return self.Q + ucb_constant * math.sqrt(math.log(max(N_parent, 1)) / N_child)

    def p_uct_score(self, ucb_constant: float) -> float:
        N_parent = self.parent.visit_count if self.parent else 1
        N_child  = len(self.sampled_returns)
        U = ucb_constant * self.prior * math.sqrt(math.log(max(N_parent, 1))) / max(N_child, 1)
        return self.Q + U

    def var_p_uct_score(self, c_puct: float, ucb_base: float) -> float:
        N_parent = self.parent.visit_count if self.parent else 1
        N_child  = len(self.sampled_returns)
        ucb_param = math.log((N_parent + ucb_base + 1) / ucb_base) + c_puct
        U = ucb_param * self.prior * math.sqrt(math.log(max(N_parent, 1))) / max(N_child, 1)
        return self.Q + U


class E2ETPSR:
    """TPSR adapted for the e2e ModelWrapper backbone.

    Parameters mirror tpsr.TPSR; see that class for full documentation.
    """

    def __init__(
        self,
        model_wrapper,
        device,
        n_simulations: int = 3,
        horizon: int = 200,
        uct_alg: str = 'uct',
        c_puct: float = 1.0,
        ucb_base: float = 10.0,
        top_k: int = 3,
        prior_temperature: float = 1.0,
        rollout_beam_width: int = 1,
        verbose: bool = False,
        reward_lam: float = 0.1,
        warm_start: bool = False,
        warm_start_beam_width: int = 10,
        bfgs_in_search: bool = False,
        bfgs_stop_after: int = 5,
    ):
        self.mw          = model_wrapper
        self.embedder    = model_wrapper.embedder
        self.encoder     = model_wrapper.encoder
        self.decoder     = model_wrapper.decoder
        self.env         = model_wrapper.env
        self.device      = device

        self.n_simulations       = n_simulations
        self.horizon             = horizon
        self.uct_alg             = uct_alg
        self.c_puct              = c_puct
        self.ucb_base            = ucb_base
        self.top_k               = top_k
        self.prior_temperature   = prior_temperature
        self.rollout_beam_width  = rollout_beam_width
        self.verbose             = verbose
        self.reward_lam          = reward_lam
        self.warm_start          = warm_start
        self.warm_start_beam_width = max(1, warm_start_beam_width)
        self.bfgs_in_search      = bfgs_in_search
        self.bfgs_stop_after     = bfgs_stop_after

        self.eos_idx  = self.decoder.eos_index   # 74 -- used as BOS too
        self.max_seq_len = horizon                # max formula length

        # Set during search()
        self._src_enc = None   # (1, enc_slen, dim)
        self._src_len = None   # (1,)
        self._X_rew   = None
        self._y_rew   = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode(self, X: np.ndarray, y: np.ndarray):
        """Embed IO pairs and encode via the e2e encoder."""
        N = len(y)
        sequences = [[(np.array(X[i]), np.array([y[i]])) for i in range(N)]]
        x, x_len = self.embedder(sequences)
        x     = x.to(self.device)
        x_len = x_len.to(self.device)
        enc_out = self.encoder('fwd', x=x, lengths=x_len, causal=False)
        src_enc = enc_out.transpose(0, 1)   # (1, enc_slen, dim)
        return src_enc, x_len               # x_len: (1,)

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    def _init_kv(self) -> dict:
        """Empty KV cache -- no tokens processed yet."""
        return {"slen": 0, "_seqs": [[]]}

    def _decode_one(self, new_toks: list, kv: dict) -> tuple:
        """Process one new token per beam, update the KV cache.

        new_toks  : list[int] of length bs (one new token per beam).
        kv        : KV dict; kv["_seqs"] is list[list[int]], kv["slen"] is int,
                    kv[layer_id] is (K, V) tensors with batch dim = bs.

        Returns (logits [bs, n_words], new_kv).
        """
        bs   = len(new_toks)
        seqs = [kv["_seqs"][b] + [new_toks[b]] for b in range(bs)]
        slen = len(seqs[0])

        x_t     = torch.tensor(seqs, dtype=torch.long, device=self.device).t().contiguous()
        lengths = torch.full((bs,), slen, dtype=torch.long, device=self.device)

        # Restore decoder cache
        cache = {"slen": kv["slen"]}
        for k, v in kv.items():
            if k not in ("slen", "_seqs"):
                K, V = v
                cache[k] = (K.clone(), V.clone())
        self.decoder.cache = cache

        src_enc = self._src_enc.expand(bs, -1, -1)
        src_len = self._src_len.expand(bs)

        with torch.no_grad():
            tensor = self.decoder(
                "fwd", x=x_t, lengths=lengths, causal=True,
                src_enc=src_enc, src_len=src_len, use_cache=True,
            )

        logits = self.decoder.proj(tensor[-1].float())   # (bs, n_words)

        new_kv = {"slen": self.decoder.cache["slen"], "_seqs": seqs}
        for k, v in self.decoder.cache.items():
            if k != "slen":
                K, V = v
                new_kv[k] = (K.detach(), V.detach())

        return logits, new_kv

    def _replicate_kv(self, kv: dict, B: int) -> dict:
        """Replicate KV cache from bs=1 to bs=B (for beam rollout)."""
        new_kv = {"slen": kv["slen"], "_seqs": kv["_seqs"] * B}
        for k, v in kv.items():
            if k not in ("slen", "_seqs"):
                K, V = v
                new_kv[k] = (K.repeat_interleave(B, dim=0),
                              V.repeat_interleave(B, dim=0))
        return new_kv

    def _reorder_kv(self, kv: dict, beam_idx) -> dict:
        """Reorder KV cache according to beam_idx (beam search step)."""
        idx = beam_idx.tolist() if hasattr(beam_idx, 'tolist') else list(beam_idx)
        new_kv = {"slen": kv["slen"], "_seqs": [kv["_seqs"][i] for i in idx]}
        for k, v in kv.items():
            if k not in ("slen", "_seqs"):
                K, V = v
                new_kv[k] = (K[beam_idx], V[beam_idx])
        return new_kv

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        if self.uct_alg == 'uct':
            return max(node.children.values(),
                       key=lambda n: n.uct_score(self.c_puct))
        elif self.uct_alg == 'p_uct':
            return max(node.children.values(),
                       key=lambda n: n.p_uct_score(self.c_puct))
        else:  # var_p_uct
            return max(node.children.values(),
                       key=lambda n: n.var_p_uct_score(self.c_puct, self.ucb_base))

    # ------------------------------------------------------------------
    # MCTS helpers
    # ------------------------------------------------------------------

    def _populate_unexplored(self, node: MCTSNode) -> None:
        if node.unexplored_actions is not None or node.next_logits is None:
            return
        probs = torch.softmax(node.next_logits.squeeze() / self.prior_temperature, dim=-1)
        k = min(self.top_k, probs.shape[-1])
        top_p, top_i = probs.topk(k)
        pairs = sorted(zip(top_i.tolist(), top_p.tolist()), key=lambda x: x[1])
        node.unexplored_actions = pairs   # lowest prob first -> .pop() gives highest

    @torch.no_grad()
    def _ensure_kv(self, node: MCTSNode) -> None:
        """Build cached_kv and next_logits for node and any uncached ancestors."""
        if node.cached_kv is not None:
            return

        # Walk up to find nearest cached ancestor
        chain = []
        cur   = node
        while cur is not None and cur.cached_kv is None:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()   # top-down

        if cur is not None:
            kv   = cur.cached_kv
            step = len(cur.seq)
        else:
            kv   = self._init_kv()
            step = 0

        for n in chain:
            tok = n.seq[step]
            logits, kv = self._decode_one([tok], kv)
            n.cached_kv   = kv
            n.next_logits = logits.float()   # (1, n_words)
            step += 1
            if not n.is_terminal:
                self._populate_unexplored(n)

    @torch.no_grad()
    def _rollout(self, node: MCTSNode, beam_width: int = None) -> list:
        """Beam-search rollout from node, returns list of complete sequences."""
        B = self.rollout_beam_width if beam_width is None else beam_width

        log_probs = torch.log_softmax(node.next_logits.squeeze(), dim=-1)
        B = min(B, log_probs.shape[-1])
        top_lp, top_tok = log_probs.topk(B)

        # Replicate KV for B beams, then take first token
        kv = self._replicate_kv(node.cached_kv, B)
        logits, kv = self._decode_one(top_tok.tolist(), kv)

        step   = len(node.seq) + 1
        scores = top_lp                                         # (B,)
        seqs   = [list(kv["_seqs"][b]) for b in range(B)]
        done   = (top_tok == self.eos_idx)

        while not done.all() and step <= self.max_seq_len:
            lp         = torch.log_softmax(logits.float(), dim=-1)   # (B, n_words)
            n_words    = lp.shape[-1]
            lp[done]               = float('-inf')
            lp[done, self.eos_idx] = 0.0

            cand         = scores.unsqueeze(-1) + lp               # (B, n_words)
            top_s, top_i = cand.reshape(-1).topk(B)
            old_beam     = top_i // n_words
            new_tok      = top_i %  n_words

            kv     = self._reorder_kv(kv, old_beam)
            logits, kv = self._decode_one(new_tok.tolist(), kv)

            step  += 1
            scores = top_s
            seqs   = [list(kv["_seqs"][b]) for b in range(B)]
            done   = (new_tok == self.eos_idx)

        return seqs if seqs else [list(node.seq) + [self.eos_idx]]

    # ------------------------------------------------------------------
    # Formula evaluation
    # ------------------------------------------------------------------

    def _seq_to_toks(self, seq: list) -> list:
        """Convert token-index sequence to e2e word-token list (no BOS/EOS)."""
        toks = []
        for idx in seq[1:]:   # skip leading EOS used as BOS
            word = self.env.equation_id2word.get(idx, "<UNK>")
            if word in ("<EOS>", "<PAD>"):
                break
            toks.append(word)
        return toks

    def _eval_tree(self, tree, X: np.ndarray, y: np.ndarray) -> float:
        """NMSE reward for an e2e tree: 1/(1+NMSE) + length penalty."""
        try:
            fn     = self.env.simplifier.tree_to_numexpr_fn(tree)
            y_pred = fn(X)
            if y_pred is None:
                return 0.0
            y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
            if not np.all(np.isfinite(y_pred)) or len(y_pred) != len(y):
                return 0.0
            nmse = float(np.sqrt(np.mean((y - y_pred) ** 2) /
                                 (np.mean(y ** 2) + 1e-9)))
            if not math.isfinite(nmse):
                return 0.0
            base = 1.0 / (1.0 + nmse)
        except Exception:
            return 0.0
        tlen = len(tree.infix().split())
        return base + self.reward_lam * math.exp(-tlen / 200.0)

    def _seq_reward(self, seq: list) -> float:
        toks = self._seq_to_toks(seq)
        if not toks:
            return 0.0
        try:
            tree = self.env.word_to_infix(toks, is_float=False, str_array=False)
            if tree is None:
                return 0.0
            return self._eval_tree(tree, self._X_rew, self._y_rew)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # MCTS simulation
    # ------------------------------------------------------------------

    def _simulate(self, root: MCTSNode, _record) -> None:
        """One full MCTS simulation: select -> expand -> rollout -> backprop."""
        node = root
        path = [node]

        # SELECT: descend while fully expanded and non-terminal
        while (not node.is_terminal
               and node.unexplored_actions is not None
               and len(node.unexplored_actions) == 0):
            node = self._best_child(node)
            self._ensure_kv(node)
            path.append(node)

        if node.is_terminal:
            reward = _record(node.seq)
            for n in path:
                n.visit_count += 1
                n.sampled_returns.append(reward)
            return

        if node.unexplored_actions is None:
            self._ensure_kv(node)

        if not node.unexplored_actions:
            reward = _record(node.seq)
            for n in path:
                n.visit_count += 1
                n.sampled_returns.append(reward)
            return

        # EXPAND
        tok_idx, prob = node.unexplored_actions.pop()
        child_seq = node.seq + [tok_idx]
        child = MCTSNode(child_seq, parent=node, prior=prob)
        if tok_idx == self.eos_idx or len(child_seq) > self.max_seq_len:
            child.is_terminal = True
        node.children[tok_idx] = child
        path.append(child)

        # ROLLOUT
        if child.is_terminal:
            reward = _record(child.seq)
        else:
            self._ensure_kv(child)
            rollout_seqs = self._rollout(child)
            reward = max(_record(seq) for seq in rollout_seqs)

        # BACKPROP
        for n in path:
            n.visit_count += 1
            n.sampled_returns.append(reward)

    def _select_expand_only(self, root: MCTSNode):
        """SELECT + EXPAND without rollout/backprop (for parallel rollouts)."""
        node = root
        path = [node]

        while (not node.is_terminal
               and node.unexplored_actions is not None
               and len(node.unexplored_actions) == 0):
            node = self._best_child(node)
            self._ensure_kv(node)
            path.append(node)

        if node.is_terminal:
            return path, node

        if node.unexplored_actions is None:
            self._ensure_kv(node)

        if not node.unexplored_actions:
            return path, node

        tok_idx, prob = node.unexplored_actions.pop()
        child_seq = node.seq + [tok_idx]
        child = MCTSNode(child_seq, parent=node, prior=prob)
        if tok_idx == self.eos_idx or len(child_seq) > self.max_seq_len:
            child.is_terminal = True
        node.children[tok_idx] = child
        path.append(child)

        if not child.is_terminal:
            self._ensure_kv(child)

        return path, child

    def _run_simulations(self, root: MCTSNode, _record) -> None:
        """Run n_simulations sequentially (matches paper's mcts_procedure)."""
        for _ in range(self.n_simulations):
            self._simulate(root, _record)

    @torch.no_grad()
    def _seed_path(self, root: MCTSNode, seq: list, reward: float) -> None:
        """Seed the tree with a warm-start beam path."""
        node = root
        self._ensure_kv(node)
        path = [node]
        for depth in range(len(root.seq), len(seq)):
            tok = seq[depth]
            if tok in node.children:
                node = node.children[tok]
            else:
                probs = torch.softmax(
                    node.next_logits.squeeze() / self.prior_temperature, dim=-1)
                child = MCTSNode(node.seq + [tok], parent=node, prior=float(probs[tok]))
                if tok == self.eos_idx or len(child.seq) > self.max_seq_len:
                    child.is_terminal = True
                node.children[tok] = child
                if node.unexplored_actions is not None:
                    node.unexplored_actions = [
                        (t, p) for t, p in node.unexplored_actions if t != tok]
                node = child
            path.append(node)
            if node.is_terminal:
                break
            self._ensure_kv(node)
        for n in path:
            n.visit_count += 1
            n.sampled_returns.append(reward)

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------

    def search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_reward: np.ndarray = None,
        y_reward: np.ndarray = None,
    ) -> list:
        """Run TPSR on one IO bag and return candidate e2e trees sorted by score.

        Parameters
        ----------
        X         : (N, D) float64 input data (used for encoding).
        y         : (N,)  float64 target data.
        X_reward  : held-out data for reward evaluation (defaults to X).
        y_reward  : held-out targets (defaults to y).

        Returns
        -------
        list of (score, tree) tuples sorted descending by score.
        """
        from symbolicregression.model.utils_wrapper import BFGSRefinement

        X_rew = X if X_reward is None else X_reward
        y_rew = y if y_reward is None else y_reward

        self._src_enc, self._src_len = self._encode(X, y)
        self._X_rew  = X_rew
        self._y_rew  = y_rew

        best_by_skeleton: dict = {}   # skeleton_infix -> (score, tree)
        _bfgs_done: set = set()
        _bfgs_n = min(1024, len(X_rew))

        def _record(seq: list) -> float:
            toks = self._seq_to_toks(seq)
            if not toks:
                return 0.0
            try:
                tree = self.env.word_to_infix(toks, is_float=False, str_array=False)
                if tree is None:
                    return 0.0

                skel, consts = self.env.generator.function_to_skeleton(
                    tree, constants_with_idx=True)
                skel_key = skel.infix()
                score    = self._eval_tree(tree, X_rew, y_rew)

                # BFGS refinement (at most once per skeleton)
                if self.bfgs_in_search and skel_key not in _bfgs_done and score > 0.05:
                    _bfgs_done.add(skel_key)
                    try:
                        refined = BFGSRefinement().go(
                            env=self.env, tree=skel, coeffs0=consts,
                            X=X_rew[:_bfgs_n], y=y_rew[:_bfgs_n],
                            downsample=-1, stop_after=self.bfgs_stop_after,
                        )
                        if refined is not None:
                            s_ref = self._eval_tree(refined, X_rew, y_rew)
                            if s_ref > score:
                                score = s_ref
                                tree  = refined
                    except Exception:
                        pass

                if score > best_by_skeleton.get(skel_key, (-1.0, None))[0]:
                    best_by_skeleton[skel_key] = (score, tree)
                return score
            except Exception:
                return 0.0

        root = MCTSNode(seq=[self.eos_idx])
        self._ensure_kv(root)

        if self.verbose:
            print(f"\n  E2E-TPSR (horizon={self.horizon}, sims={self.n_simulations})",
                  flush=True)

        # Outer loop: commit one token per step
        for _step in range(self.horizon):
            if root.is_terminal:
                break

            self._run_simulations(root, _record)

            if not root.children:
                break

            best = max(root.children.values(), key=lambda n: n.Q)

            if self.verbose:
                word  = self.env.equation_id2word.get(best.seq[-1], '?')
                toks  = self._seq_to_toks(best.seq)
                print(f"  step {_step+1:>3}: {word:<12} Q={best.Q:.4f} | {' '.join(toks)}",
                      flush=True)

            # Also record the committed prefix if it's a complete formula
            _record(best.seq)

            committed_tok = best.seq[-1]
            root = root.children[committed_tok]
            root.parent = None

        ranked = sorted(best_by_skeleton.values(), key=lambda x: x[0], reverse=True)
        return ranked   # list of (score, tree)
