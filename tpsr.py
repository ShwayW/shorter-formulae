"""
tpsr.py  --  Transformer-based Planning for Symbolic Regression (TPSR)

Shojaee et al., NeurIPS 2023.  Faithful re-implementation of the original
algorithm using our VanillaTransformer in place of the e2e model.

Algorithm overview
------------------
The outer loop commits tokens one at a time (up to `horizon` steps).
At each step, `n_simulations` MCTS simulations are run from the current
root; the action with the highest Q is committed, and the tree root moves
down to the corresponding child (tree reuse).

Each simulation does:
  SELECT  -- descend via var_p_UCT until a node with unexplored actions
             is found.  A node is "fully expanded" only when every entry
             in its top-k action queue has been tried.
  EXPAND  -- pop ONE action from the node's queue, create exactly one child.
  ROLLOUT -- beam-search the new child to completion; pick the beam with
             the highest R^2 reward.
  BACKPROP -- propagate the reward up the path; each node stores a list of
              sampled returns and Q = max(sampled_returns).

Selection rule -- var_p_UCT (UCT agent, `alg='var_p_uct'`, default in TPSR):
    ucb_param = log((N_parent + ucb_base + 1) / ucb_base) + c_puct
    score(child) = max(child.returns) + ucb_param * P(child) * sqrt(log(N_parent))
                                                              / N(child)

  N_parent  = visit count of parent
  N(child)  = number of times child has been sampled (= len(sampled_returns))
  P(child)  = transformer softmax prior for that action (temperature-scaled)

Performance design
------------------
KV caching: each node stores cached_kv (self-attention KV for its prefix)
and next_logits (logits for the next token).  _ensure_kv walks the uncached
ancestor chain and extends the KV by exactly one decode_step per node,
so the incremental cost is O(1) per node given the parent's cache.
Rollout starts from cached_kv -- no prefix replay.

Usage
-----
    from tpsr import TPSR
    searcher = TPSR(model, vocab, device, n_simulations=3, horizon=200)
    candidates = searcher.search(ios_tensor, X_s64, y_s64,
                                 sample_mean, sample_std, n_active)
    # candidates: list[str] sorted by R^2 (best first).
"""

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from utils import _reassemble_floats
from bfgs_optimizer import BFGSOptimizer
from funcWrappers import wrapExtEvalPN


# ---------------------------------------------------------------------------
# MCTS node
# ---------------------------------------------------------------------------

class MCTSNode:
    """One node in the TPSR search tree.

    seq                : token-index list beginning with the <bes> index.
    prior              : P(this action | parent state) -- set by parent on creation.
    unexplored_actions : sorted list of (tok_idx, prob) not yet expanded,
                         lowest-prob first so .pop() yields the highest-prob
                         action next.  None until _ensure_kv has been called
                         for this node.
    sampled_returns    : list of rewards observed through this node.
    cached_kv          : self-attention KV accumulated over seq.
    next_logits        : logits for the token after seq[-1].
    """

    __slots__ = (
        "seq", "parent", "prior",
        "children", "visit_count", "sampled_returns",
        "is_terminal",
        "cached_kv", "next_logits",
        "unexplored_actions",
        "_hidden",   # decoder hidden state; reserved for TPSRWithValueHead
    )

    def __init__(
        self,
        seq: list,
        parent: Optional["MCTSNode"] = None,
        prior: float = 0.0,
    ) -> None:
        self.seq = seq
        self.parent = parent
        self.prior = prior
        self.children: dict = {}
        self.visit_count: int = 0
        self.sampled_returns: list = []
        self.is_terminal: bool = False
        self.cached_kv = None
        self.next_logits = None
        self.unexplored_actions = None   # None = not yet initialised
        self._hidden = None

    @property
    def Q(self) -> float:
        """Optimistic value: best reward ever seen through this node."""
        return max(self.sampled_returns) if self.sampled_returns else 0.0

    def var_p_uct_score(self, c_puct: float, ucb_base: float) -> float:
        """var_p_UCT score used for child selection (uct.py:102-108)."""
        N_parent = self.parent.visit_count if self.parent else 1
        N_child  = len(self.sampled_returns)
        ucb_param = math.log((N_parent + ucb_base + 1) / ucb_base) + c_puct
        U = ucb_param * self.prior * math.sqrt(math.log(max(N_parent, 1))) / max(N_child, 1)
        return self.Q + U

    def best_child(self, c_puct: float, ucb_base: float) -> "MCTSNode":
        return max(self.children.values(),
                   key=lambda n: n.var_p_uct_score(c_puct, ucb_base))


# ---------------------------------------------------------------------------
# TPSR
# ---------------------------------------------------------------------------

class TPSR:
    """Transformer-based Planning for Symbolic Regression.

    Parameters
    ----------
    model              : VanillaTransformer in eval mode on device.
    vocab              : token list (same order as model output logits).
    device             : torch device.
    n_simulations      : MCTS rollouts per token-selection step (paper: 3).
    horizon            : maximum formula length / outer-loop steps (paper: 200).
    c_puct             : base UCT exploration constant (paper: ~6.36).
    ucb_base           : UCB base for the variable exploration term (paper: 50).
    top_k              : action queue size per node -- top-k tokens considered
                         for expansion (paper: 3, but more is fine).
    prior_temperature  : softmax temperature on logits before computing priors.
                         T > 1 flattens the distribution; useful when our model
                         concentrates >90% mass on a single token at the root.
    rollout_beam_width : beams in the rollout beam search (paper: 3).
    bpe_model          : optional BPE model for token detokenisation.
    warm_start         : if True, run a beam search from the root before the MCTS
                         outer loop, record every completion (so the returned
                         candidate set is a superset of beam search's) and seed
                         the tree along each beam path with its reward.  Makes
                         TPSR >= beam by construction and lets MCTS refine around
                         the beam's hypotheses instead of rediscovering them.
    warm_start_beam_width : beam width for the warm-start beam search.
    """

    def __init__(
        self,
        model,
        vocab: list,
        device: torch.device,
        n_simulations: int = 3,
        horizon: int = 200,
        c_puct: float = 6.36,
        ucb_base: float = 50.0,
        top_k: int = 3,
        prior_temperature: float = 1.0,
        rollout_beam_width: int = 3,
        bpe_model=None,
        verbose: bool = False,
        bfgs_downsample: int = 1024,
        bfgs_stop_after: int = 5,
        reward_lam: float = 0.1,
        use_unscaling: bool = True,
        use_bfgs: bool = True,
        bfgs_in_search: bool = True,
        warm_start: bool = True,
        warm_start_beam_width: int = 10,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.device = device
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.c_puct = c_puct
        self.ucb_base = ucb_base
        self.top_k = top_k
        self.prior_temperature = prior_temperature
        self.rollout_beam_width = rollout_beam_width
        self.bpe_model = bpe_model
        self.verbose = verbose
        self.bfgs_downsample = bfgs_downsample
        self.bfgs_stop_after = bfgs_stop_after
        self.reward_lam = reward_lam
        self.use_unscaling = use_unscaling
        self.use_bfgs = use_bfgs
        self.bfgs_in_search = use_bfgs and bfgs_in_search
        self.warm_start = warm_start
        self.warm_start_beam_width = max(1, warm_start_beam_width)
        # Provenance: the search phase that produced the best-scoring instance of
        # each skeleton ("warm_start" / "rollout@<step>" / "commit@<step>"; <step>
        # is 1-indexed to match the verbose table).  Set by search() and its
        # helpers; surfaced after a search via self.last_skel_source so callers
        # can attribute the winning formula to where it was found.
        self._record_source = "init"
        self.last_skel_source: dict = {}

        self.bes_idx: int = vocab.index("<bes>")
        self.max_fml_len: int = model.max_seq_length - 1
        self._use_amp: bool = device.type == "cuda"
        self.model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _amp(self):
        if self._use_amp:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return torch.amp.autocast("cpu", enabled=False)

    @torch.inference_mode()
    def _encode(self, ios_tensor: torch.Tensor):
        """Encode one IO bag once.  Returns (enc_output, enc_mask, cross_kv)."""
        inputs = ios_tensor.float().to(self.device)
        with self._amp():
            enc_output, enc_mask = self.model.encode(inputs)
            cross_kv = self.model.precompute_cross_kv(enc_output)
        return enc_output, enc_mask, cross_kv

    def _populate_unexplored(self, node: MCTSNode) -> None:
        """Fill node.unexplored_actions from next_logits.

        Stores pairs as (tok_idx, prob) sorted lowest-prob-first so that
        .pop() always yields the highest-probability unexplored action,
        matching the original's list(reversed(top_k_predict)) + .pop().
        """
        if node.unexplored_actions is not None or node.next_logits is None:
            return
        probs = torch.softmax(node.next_logits.squeeze() / self.prior_temperature, dim=-1)
        k = min(self.top_k, probs.shape[-1])
        top_p, top_i = probs.topk(k)
        pairs = sorted(zip(top_i.tolist(), top_p.tolist()), key=lambda x: x[1])
        node.unexplored_actions = pairs   # lowest prob first; pop() -> highest prob

    @torch.inference_mode()
    def _ensure_kv(self, node: MCTSNode, enc_mask, cross_kv) -> None:
        """Build cached_kv and next_logits for node (and uncached ancestors).

        Walks up the ancestor chain to find the nearest cached ancestor, then
        extends the KV by one decode_step per uncached node (top-down).
        After setting next_logits, populates unexplored_actions for non-terminal
        nodes so that expansion can immediately pop the next action.
        """
        if node.cached_kv is not None:
            return

        chain: list = []
        cur = node
        while cur is not None and cur.cached_kv is None:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()   # top-down: chain[0] is closest to root

        with self._amp():
            if cur is not None:
                self_kv = cur.cached_kv
                step    = len(cur.seq)
            else:
                self_kv = self.model.init_self_kv_cache(1, self.device)
                step    = 0

            for n in chain:
                tok_t = torch.tensor(
                    [[n.seq[step]]], dtype=torch.long, device=self.device
                )
                logits, self_kv = self.model.decode_step(
                    tok_t, step, enc_mask, self_kv, cross_kv,
                )
                n.cached_kv    = self_kv
                n.next_logits  = logits.float()
                step += 1
                if not n.is_terminal:
                    self._populate_unexplored(n)

    @torch.inference_mode()
    def _rollout(self, node: MCTSNode, enc_mask_b, cross_kv_b, beam_width: int = None) -> list:
        """Beam-search rollout from node's cached KV state, fully batched.

        enc_mask_b and cross_kv_b are pre-replicated to `beam_width` (default
        rollout_beam_width) and passed in from search() so they are not
        recomputed per rollout.  Warm start reuses this with a wider beam.

        Returns a list of complete token-index sequences (one per beam),
        each starting with node.seq.  The caller picks the best by reward.
        """
        beam_width = self.rollout_beam_width if beam_width is None else beam_width
        step = len(node.seq)

        with self._amp():
            log_probs_init = torch.log_softmax(node.next_logits.squeeze(), dim=-1)
            B = min(beam_width, log_probs_init.shape[-1])
            top_lp, top_tok = log_probs_init.topk(B)

            # Replicate node's self-KV for B beams (enc/cross KV pre-replicated).
            if B > 1:
                self_kv = [(K.repeat_interleave(B, dim=0), V.repeat_interleave(B, dim=0))
                           if K is not None else (None, None)
                           for K, V in node.cached_kv]
            else:
                self_kv = node.cached_kv   # B=1: no copy needed

            tok_batch = top_tok.unsqueeze(1)
            logits, self_kv = self.model.decode_step(
                tok_batch, step, enc_mask_b, self_kv, cross_kv_b)
            step += 1

            scores = top_lp
            seqs   = [list(node.seq) + [int(t)] for t in top_tok.tolist()]
            done   = (top_tok == self.bes_idx)

            while not done.all() and step <= self.max_fml_len:
                lp = torch.log_softmax(logits.float(), dim=-1)
                vocab_size = lp.shape[-1]

                lp[done]               = float('-inf')
                lp[done, self.bes_idx] = 0.0

                cand       = scores.unsqueeze(-1) + lp
                top_s, top_i = cand.reshape(-1).topk(B)
                old_beam   = top_i // vocab_size
                new_tok    = top_i %  vocab_size

                seqs    = [seqs[int(b)] + [int(t)]
                           for b, t in zip(old_beam.tolist(), new_tok.tolist())]
                self_kv = self.model.reorder_kv_cache(self_kv, old_beam)

                tok_batch = new_tok.unsqueeze(1)
                logits, self_kv = self.model.decode_step(
                    tok_batch, step, enc_mask_b, self_kv, cross_kv_b)
                step  += 1
                scores = top_s
                done   = (new_tok == self.bes_idx)

        return seqs if seqs else [list(node.seq) + [self.bes_idx]]

    def _formula_reward(self, formula: str, X_eval: np.ndarray, y_eval: np.ndarray) -> float:
        """1/(1+NMSE) + length penalty -- matches original TPSR reward.py.

        NMSE = sqrt(mean((y - yhat)^2) / mean(y^2)), matching reward.py:compute_reward_e2e.
        Length penalty: lam * exp(-complexity / 200), matching reward.py:43-47.
        """
        if not formula:
            return 0.0
        try:
            preds, stacklefts = wrapExtEvalPN(formula, X_eval)
            if np.any(stacklefts != 0) or not np.all(np.isfinite(preds)):
                return 0.0
            with np.errstate(over='ignore', invalid='ignore'):
                nmse = float(np.sqrt(np.mean((y_eval - preds) ** 2) /
                                     (np.mean(y_eval ** 2) + 1e-9)))
            if not math.isfinite(nmse):
                return 0.0
            base = 1.0 / (1.0 + nmse)
        except Exception:
            return 0.0
        complexity = len(formula.split())
        return base + self.reward_lam * math.exp(-complexity / 200.0)

    # Tokens that terminate a formula sequence.
    _TERMINAL_TOKENS = {"<bes>", "<pad>"}

    def _seq_to_raw_str(self, seq: list) -> str:
        """Token-index list -> raw formula string (before sanitization)."""
        toks: list = []
        for idx in seq[1:]:   # skip leading <bes>
            tok = self.vocab[idx]
            if tok in self._TERMINAL_TOKENS:
                break
            toks.append(tok)
        return " ".join(_reassemble_floats(toks))

    def _simulate(self, root: MCTSNode, enc_mask, cross_kv, enc_mask_b, cross_kv_b, _record) -> None:
        """One full MCTS simulation: select -> expand -> rollout -> backprop.

        Mirrors mcts_procedure (mcts.py:50-143) and UCT.act (uct.py:111-113).
        """
        # ---- SELECT: descend while node is fully expanded and non-terminal ----
        node = root
        path = [node]

        while (not node.is_terminal
               and node.unexplored_actions is not None
               and len(node.unexplored_actions) == 0):
            node = node.best_child(self.c_puct, self.ucb_base)
            self._ensure_kv(node, enc_mask, cross_kv)
            path.append(node)

        # ---- Terminal reached during selection ----
        if node.is_terminal:
            reward = _record(self._seq_to_raw_str(node.seq))
            for n in path:
                n.visit_count += 1
                n.sampled_returns.append(reward)
            return

        # ---- EXPAND: pop one action and create exactly one child ----
        # If unexplored_actions is None the node hasn't been initialised yet
        # (shouldn't happen after _ensure_kv at root, but guard anyway).
        if node.unexplored_actions is None:
            self._ensure_kv(node, enc_mask, cross_kv)

        if not node.unexplored_actions:
            # No actions available (degenerate vocab or max-len node).
            reward = _record(self._seq_to_raw_str(node.seq))
            for n in path:
                n.visit_count += 1
                n.sampled_returns.append(reward)
            return

        tok_idx, prob = node.unexplored_actions.pop()   # highest-prob unexplored action
        child_seq = node.seq + [tok_idx]
        child = MCTSNode(child_seq, parent=node, prior=prob)
        if tok_idx == self.bes_idx or len(child_seq) > self.max_fml_len:
            child.is_terminal = True
        node.children[tok_idx] = child
        path.append(child)

        # ---- ROLLOUT from the newly created child ----
        if child.is_terminal:
            reward = _record(self._seq_to_raw_str(child.seq))
        else:
            self._ensure_kv(child, enc_mask, cross_kv)
            rollout_seqs = self._rollout(child, enc_mask_b, cross_kv_b)
            reward = max(_record(self._seq_to_raw_str(seq)) for seq in rollout_seqs)

        # ---- BACKPROPAGATE ----
        for n in path:
            n.visit_count += 1
            n.sampled_returns.append(reward)

    # ------------------------------------------------------------------
    # Parallel simulation helpers
    # ------------------------------------------------------------------

    def _select_expand_only(self, root: MCTSNode, enc_mask, cross_kv):
        """SELECT + EXPAND only.  Returns (path, leaf) without rollout or backprop."""
        node = root
        path = [node]

        while (not node.is_terminal
               and node.unexplored_actions is not None
               and len(node.unexplored_actions) == 0):
            node = node.best_child(self.c_puct, self.ucb_base)
            self._ensure_kv(node, enc_mask, cross_kv)
            path.append(node)

        if node.is_terminal:
            return path, node

        if node.unexplored_actions is None:
            self._ensure_kv(node, enc_mask, cross_kv)

        if not node.unexplored_actions:
            return path, node

        tok_idx, prob = node.unexplored_actions.pop()
        child_seq = node.seq + [tok_idx]
        child = MCTSNode(child_seq, parent=node, prior=prob)
        if tok_idx == self.bes_idx or len(child_seq) > self.max_fml_len:
            child.is_terminal = True
        node.children[tok_idx] = child
        path.append(child)

        if not child.is_terminal:
            self._ensure_kv(child, enc_mask, cross_kv)

        return path, child

    @torch.inference_mode()
    def _rollout_batched(self, nodes: list, enc_mask, cross_kv) -> list:
        """Batch rollout for N same-depth leaf nodes, B beams each.

        Stacks all N nodes' self-KV caches along the batch dimension and runs
        one combined beam-search pass (N*B beams total).  Beam search is
        per-group: the top-B beams are selected independently within each node's
        B candidates, so there is no cross-node beam merging.

        Returns: list of N lists, each containing B complete token-index sequences.
        """
        N    = len(nodes)
        B0   = self.rollout_beam_width
        step = len(nodes[0].seq)          # all nodes must be at the same depth

        with self._amp():
            # Initial logits for each node -> [N, vocab]
            all_logits     = torch.cat([n.next_logits for n in nodes], dim=0)
            log_probs_init = torch.log_softmax(all_logits.view(N, -1), dim=-1)
            B  = min(B0, log_probs_init.shape[-1])
            NB = N * B

            top_lp, top_tok = log_probs_init.topk(B, dim=-1)   # [N, B]
            top_lp_flat     = top_lp.reshape(NB)
            top_tok_flat    = top_tok.reshape(NB)

            # Stack self-KV caches: cat N nodes then repeat_interleave B times
            # -> [NB, heads, depth, d_head] per layer
            n_layers   = len(nodes[0].cached_kv)
            batched_kv = []
            for li in range(n_layers):
                Ks = [n.cached_kv[li][0] for n in nodes]
                Vs = [n.cached_kv[li][1] for n in nodes]
                if Ks[0] is None:
                    batched_kv.append((None, None))
                    continue
                K = torch.cat(Ks, dim=0).repeat_interleave(B, dim=0)
                V = torch.cat(Vs, dim=0).repeat_interleave(B, dim=0)
                batched_kv.append((K, V))

            # Replicate encoder tensors NB times (computed once here)
            enc_mask_nb = enc_mask.repeat_interleave(NB, dim=0)
            cross_kv_nb = [
                (K.repeat_interleave(NB, dim=0), V.repeat_interleave(NB, dim=0))
                if K is not None else (None, None)
                for K, V in cross_kv
            ]

            tok_batch = top_tok_flat.unsqueeze(1)
            logits, self_kv = self.model.decode_step(
                tok_batch, step, enc_mask_nb, batched_kv, cross_kv_nb)
            step += 1

            scores     = top_lp_flat
            seqs       = [list(nodes[i // B].seq) + [int(top_tok_flat[i])] for i in range(NB)]
            done       = (top_tok_flat == self.bes_idx)
            vocab_size = logits.shape[-1]

            while not done.all() and step <= self.max_fml_len:
                lp = torch.log_softmax(logits.float(), dim=-1)    # [NB, vocab]
                lp[done]               = float('-inf')
                lp[done, self.bes_idx] = 0.0

                cand = scores.unsqueeze(-1) + lp                   # [NB, vocab]

                # Per-group beam search: top-B within each node's B beams
                top_s, top_i = cand.view(N, B * vocab_size).topk(B, dim=-1)  # [N, B]
                old_beam_local  = top_i // vocab_size              # [N, B]
                new_tok         = top_i %  vocab_size              # [N, B]
                node_offsets    = torch.arange(N, device=cand.device).unsqueeze(1) * B
                old_beam_global = (node_offsets + old_beam_local).reshape(NB)
                new_tok_flat    = new_tok.reshape(NB)

                seqs    = [seqs[int(old_beam_global[i])] + [int(new_tok_flat[i])]
                           for i in range(NB)]
                self_kv = self.model.reorder_kv_cache(self_kv, old_beam_global)

                tok_batch = new_tok_flat.unsqueeze(1)
                logits, self_kv = self.model.decode_step(
                    tok_batch, step, enc_mask_nb, self_kv, cross_kv_nb)
                step  += 1
                scores = top_s.reshape(NB)
                done   = (new_tok_flat == self.bes_idx)

        return [seqs[i * B:(i + 1) * B] for i in range(N)]

    def _run_simulations_parallel(
        self, root, enc_mask, cross_kv, enc_mask_b, cross_kv_b, _record
    ):
        """Run n_simulations with batched rollouts.

        SELECT+EXPAND phases are sequential (tree is mutable); ROLLOUT phases
        are batched per depth group, reducing GPU kernel launches by ~n_simulations*
        when all leaves land at the same depth (the common case).
        """
        # Phase 1: SELECT+EXPAND all simulations sequentially
        pending = []
        for _ in range(self.n_simulations):
            path, leaf = self._select_expand_only(root, enc_mask, cross_kv)
            pending.append((path, leaf))

        # Phase 2: Batch rollouts, grouped by leaf depth
        depth_groups = defaultdict(list)
        for idx, (_, leaf) in enumerate(pending):
            depth_groups[len(leaf.seq)].append((idx, leaf))

        rewards = [None] * self.n_simulations

        for _depth, group in depth_groups.items():
            p_indices = [i for i, _ in group]
            leaves    = [leaf for _, leaf in group]

            term_idx  = [i for i, leaf in enumerate(leaves) if leaf.is_terminal]
            nterm_idx = [i for i, leaf in enumerate(leaves) if not leaf.is_terminal]

            for li in term_idx:
                rewards[p_indices[li]] = _record(self._seq_to_raw_str(leaves[li].seq))

            if not nterm_idx:
                continue

            nt_leaves = [leaves[i] for i in nterm_idx]
            nt_pend   = [p_indices[i] for i in nterm_idx]

            if len(nt_leaves) == 1:
                seqs_list = [self._rollout(nt_leaves[0], enc_mask_b, cross_kv_b)]
            else:
                seqs_list = self._rollout_batched(nt_leaves, enc_mask, cross_kv)

            for pi, seqs in zip(nt_pend, seqs_list):
                rewards[pi] = max(_record(self._seq_to_raw_str(seq)) for seq in seqs)

        # Phase 3: Backprop all rewards
        for (path, _), reward in zip(pending, rewards):
            for n in path:
                n.visit_count += 1
                n.sampled_returns.append(reward)

    @torch.inference_mode()
    def _seed_path(self, root: MCTSNode, seq: list, reward: float,
                   enc_mask, cross_kv) -> None:
        """Build the tree path for a complete sequence and backprop its reward.

        Used by warm start: `seq` is a beam-search completion (starts with the
        root's <bes> prefix).  For each token past the root prefix, descend into
        the existing child or create one -- seeding its prior from the parent's
        next_logits and removing it from the parent's unexplored_actions so the
        normal expansion path will not duplicate it (which would overwrite the
        seeded child and lose its statistics).  `reward` is appended to every
        node on the path (and visit_count incremented), so both MCTS selection
        and the outer-loop commit (max-Q) are biased toward the beam's
        hypotheses.  Stops at the first terminal (<bes>) token.
        """
        node = root
        self._ensure_kv(node, enc_mask, cross_kv)
        path = [node]
        for depth in range(len(root.seq), len(seq)):
            tok = seq[depth]
            if tok in node.children:
                node = node.children[tok]
            else:
                probs = torch.softmax(
                    node.next_logits.squeeze() / self.prior_temperature, dim=-1)
                child = MCTSNode(node.seq + [tok], parent=node, prior=float(probs[tok]))
                if tok == self.bes_idx or len(child.seq) > self.max_fml_len:
                    child.is_terminal = True
                node.children[tok] = child
                if node.unexplored_actions is not None:
                    node.unexplored_actions = [
                        (t, p) for (t, p) in node.unexplored_actions if t != tok]
                node = child
            path.append(node)
            if node.is_terminal:
                break
            self._ensure_kv(node, enc_mask, cross_kv)
        for n in path:
            n.visit_count += 1
            n.sampled_returns.append(reward)

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------

    def search(
        self,
        ios_tensor: torch.Tensor,
        X_s64: np.ndarray,
        y_s64: np.ndarray,
        sample_mean: np.ndarray,
        sample_std: np.ndarray,
        n_active: int,
        X_reward: np.ndarray = None,
        y_reward: np.ndarray = None,
    ) -> list:
        """Run TPSR on one IO bag and return candidate formula strings.

        The outer loop commits tokens sequentially (up to `horizon` steps),
        each time running `n_simulations` MCTS simulations and picking the
        child with the highest Q as the next token.  The tree is reused
        across steps.  All rollout completions are collected and returned
        ranked by R^2.

        Parameters
        ----------
        ios_tensor  : (1, N*(patch_dim)) whitened IO tensor for the encoder.
        X_s64       : (N, len(V)) float64 bag data (model context only).
        y_s64       : (N,) float64 bag targets (model context only).
        sample_mean : per-column means of the raw bag (input unscaling).
        sample_std  : per-column stds  of the raw bag (input unscaling).
        n_active    : number of active variable columns in the dataset.
        X_reward    : held-out data for reward evaluation (falls back to X_s64).
        y_reward    : targets matching X_reward (falls back to y_s64).

        Returns
        -------
        list[str] : unique sanitized, input-unscaled formula strings,
                    sorted by R^2 on the reward set (best first).
        """
        from eval_mymodels import (
            sanitize_formula,
            apply_input_unscaling,
            formula_skeleton,
            expand_bpe_tokens,
            formula_is_complete,
        )
        bpe_model = self.bpe_model

        X_rew = X_s64 if X_reward is None else X_reward
        y_rew = y_s64 if y_reward is None else y_reward

        # When unscaling is disabled the formula maps whitened inputs -> raw y,
        # so reward evaluation must also use whitened X.
        # Only whiten the n_active real columns; the rest are zero-padded dummies.
        if not self.use_unscaling:
            X_rew = X_rew.copy()
            X_rew[:, :n_active] = (X_rew[:, :n_active] - sample_mean) / (sample_std + 1e-8)

        enc_output, enc_mask, cross_kv = self._encode(ios_tensor)

        # Pre-replicate encoder tensors for rollout beam width -- done once per
        # bag so _rollout never re-allocates them (saves B allocs per simulation).
        B = self.rollout_beam_width
        if B > 1:
            enc_mask_b = enc_mask.repeat_interleave(B, dim=0)
            cross_kv_b = [(K.repeat_interleave(B, dim=0), V.repeat_interleave(B, dim=0))
                          if K is not None else (None, None)
                          for K, V in cross_kv]
        else:
            enc_mask_b = enc_mask
            cross_kv_b = cross_kv

        root = MCTSNode(seq=[self.bes_idx])
        self._ensure_kv(root, enc_mask, cross_kv)

        best_by_skeleton: dict = {}
        skel_source: dict = {}   # skeleton -> phase that first emitted it

        # Fixed random subsample for BFGS -- computed once for a stable reward signal.
        _bfgs_n   = min(self.bfgs_downsample, X_rew.shape[0])
        _bfgs_idx = np.random.choice(X_rew.shape[0], size=_bfgs_n, replace=False)
        X_bfgs    = X_rew[_bfgs_idx]
        y_bfgs    = y_rew[_bfgs_idx]

        # Skeleton-level BFGS cache: run BFGS at most once per unique formula
        # structure.  The same skeleton recurs many times during tree search
        # (different rollout completions share prefixes); caching cuts BFGS
        # calls from O(n_simulations * horizon) to O(unique skeletons).
        _bfgs_done: set = set()

        def _record(raw_str: str) -> float:
            if bpe_model is not None:
                raw_str = expand_bpe_tokens(raw_str, bpe_model)
            formula = sanitize_formula(raw_str)
            if not formula:
                return 0.0
            if self.use_unscaling:
                formula = apply_input_unscaling(formula, sample_mean, sample_std, n_active)

            # Always compute the base reward for the current formula -- this is
            # fast (one wrapExtEvalPN call) and gives MCTS a meaningful signal
            # even when the same skeleton has been seen before.
            score = self._formula_reward(formula, X_rew, y_rew)
            skel = formula_skeleton(formula)

            # BFGS: run at most once per skeleton (expensive), only if promising.
            # The cache skips BFGS only -- not the reward computation above.
            if self.bfgs_in_search and skel not in _bfgs_done and score > 0.05:
                _bfgs_done.add(skel)
                try:
                    refined = BFGSOptimizer(formula).opt_consts(
                        X_bfgs, y_bfgs,
                        stop_after=self.bfgs_stop_after, n_restarts=1)
                    if refined:
                        score_ref = self._formula_reward(refined, X_rew, y_rew)
                        if score_ref > score:
                            score   = score_ref
                            formula = refined
                except Exception:
                    pass

            if score > best_by_skeleton.get(skel, (-1.0, ""))[0]:
                best_by_skeleton[skel] = (score, formula)
                # Attribute the skeleton to the phase that produced its
                # best-scoring instance (not merely the first to emit it) -- this
                # is the formula that is actually returned/refined downstream.
                skel_source[skel] = self._record_source
            return score

        # ---- Warm start from beam search ----
        # Run a beam search from the root, record every completion (so the
        # returned candidate set is a superset of beam search's -- TPSR >= beam by
        # construction) and seed the tree along each beam path with its reward so
        # MCTS refines around the beam's hypotheses instead of rediscovering them.
        if self.warm_start:
            ws_w = self.warm_start_beam_width
            if ws_w == B:
                ws_mask, ws_ckv = enc_mask_b, cross_kv_b
            elif ws_w > 1:
                ws_mask = enc_mask.repeat_interleave(ws_w, dim=0)
                ws_ckv = [(K.repeat_interleave(ws_w, dim=0), V.repeat_interleave(ws_w, dim=0))
                          if K is not None else (None, None)
                          for K, V in cross_kv]
            else:
                ws_mask, ws_ckv = enc_mask, cross_kv
            beam_seqs = self._rollout(root, ws_mask, ws_ckv, beam_width=ws_w)
            self._record_source = "warm_start"
            seen_ws, best_ws = set(), 0.0
            for seq in beam_seqs:
                key = tuple(seq)
                if key in seen_ws:
                    continue
                seen_ws.add(key)
                r = _record(self._seq_to_raw_str(seq))
                best_ws = max(best_ws, r)
                self._seed_path(root, seq, r, enc_mask, cross_kv)
            if self.verbose:
                print(f"  warm start: seeded {len(seen_ws)} beam path(s), "
                      f"best reward {best_ws:.4f}", flush=True)

        # ---- Outer loop: commit one token per step, reuse tree ----
        if self.verbose:
            print(f"\n  TPSR search (horizon={self.horizon}, sims={self.n_simulations})",
                  flush=True)
            print(f"  {'Step':>4} | {'Token':<14} | {'Q':>6} | Partial sequence", flush=True)
            print(f"  {'-'*4}-+-{'-'*14}-+-{'-'*6}-+-{'-'*40}", flush=True)

        for _step in range(self.horizon):
            if root.is_terminal:
                break

            self._record_source = f"rollout@{_step + 1}"
            self._run_simulations_parallel(
                root, enc_mask, cross_kv, enc_mask_b, cross_kv_b, _record)

            if not root.children:
                if self.verbose:
                    print("  [TPSR] no children after simulations -- stopping early", flush=True)
                break

            # Select best committed action: max Q among visited children
            # (mirrors mcts.py:144 -- max(root.children, key=chance_node_value))
            best = max(root.children.values(), key=lambda n: n.Q)
            committed_raw = self._seq_to_raw_str(best.seq)

            # Record the committed prefix only when it is already a complete
            # expression.  Partial prefixes would be padded with constants by
            # sanitize_formula, injecting degenerate completions into the pool;
            # skipping them does not affect tree statistics or selection (this
            # record only feeds the candidate pool).
            _check = (expand_bpe_tokens(committed_raw, bpe_model)
                      if bpe_model is not None else committed_raw)
            if formula_is_complete(_check):
                self._record_source = f"commit@{_step + 1}"
                _record(committed_raw)

            if self.verbose:
                token = self.vocab[best.seq[-1]]
                print(f"  {_step+1:>4} | {token:<14} | {best.Q:>6.4f} | {committed_raw}",
                      flush=True)

            # Move root down to committed child -- mirrors update_root() in original
            # mcts.py:241-252.  Explicitly look up by action index (matches original's
            # state-matching traversal) then detach so GC can collect pruned subtrees.
            committed_tok = best.seq[-1]
            root = root.children[committed_tok]
            root.parent = None

        self.last_skel_source = skel_source
        ranked = sorted(best_by_skeleton.values(), key=lambda x: x[0], reverse=True)
        return [fml for _, fml in ranked]
