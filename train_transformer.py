# Choose your architecture: "Van" or "Sett" or "Deco"
ARCH = "Van"

# imports
import os
import csv
import math
import torch
import torch.optim as optim
import pickle
import gzip
import copy
import datetime
from pathlib import Path
if (ARCH == "Van"):
    from tfs.van import VanillaTransformer
elif (ARCH == "Sett"):
    from tfs.sett import SetTransformer
elif (ARCH == "Deco"):
    from tfs.deco import DecoderOnlyTransformer
from time import time
if (ARCH == "Van"):
    from van_config import config_space, sample_config, get_supernet_config, get_infinet_config, ga_config
elif (ARCH == "Sett"):
    from sett_config import config_space, sample_config, get_supernet_config, ga_config
elif (ARCH == "Deco"):
    from deco_config import config_space, sample_config, get_supernet_config, get_infinet_config, ga_config
from utils import set_seed, model_acc
from grammar import FlatGram, FlatGramFloat, FLOAT_TOKENS, V
from subtree_bpe import SubtreeBPE
from data_process import online_batch_iterater, OnlineBatchDataset, _passthrough_collate
from torch.utils.data import DataLoader
from losses import SymbolicRegressionLoss

def _atomic_torch_save(obj, path, keep_prev=True):
    """torch.save that cannot leave a truncated file behind.

    torch.save writes in place, so a walltime SIGKILL landing mid-write destroys the
    only copy.  That happened once in practice (a 24h chunk was
    killed while saving; the 680MB fragment of a 1.09GB checkpoint took down the next
    five jobs).  Writing to a temp file in the SAME directory -- so os.replace is a
    same-filesystem rename, and therefore atomic -- means an interrupted save leaves
    the previous checkpoint untouched and only a stray .tmp behind.

    keep_prev additionally hard-links the outgoing checkpoint to <path>.prev before
    the swap.  A hard link is used rather than a rename so there is never an instant
    where <path> does not exist for a concurrently starting job to miss.
    """
    path = Path(path)
    tmp  = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    # fsync before the rename: the rename becoming visible before the data is on
    # disk is precisely the window this function exists to close.
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    if keep_prev and path.exists():
        prev = path.with_name(path.name + ".prev")
        try:
            if prev.exists():
                prev.unlink()
            os.link(path, prev)
        except OSError:
            pass          # no hard links on this filesystem -- the atomic swap still holds
    os.replace(tmp, path)


class TrainingManager():
    # the main file for training
    def __init__(self, configs):
        # unpack the training configurations
        self.approx_correct_tol = configs["approx_correct_tol"]
        # Every artifact this run writes (best-epoch checkpoint, final weights, the
        # training-curve pickle) lives under run_dir, so two runs launched from the
        # SAME repo checkout -- e.g. the two arms of the prefactor ablation -- never
        # overwrite each other.  Default "./checkpoints" reproduces the old flat
        # layout for anyone who does not set it.
        self.run_dir = Path(configs.get("run_dir", "./checkpoints"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = configs["model_path"]
        self.max_epochs = configs["max_epochs"]
        self.checkpoint_path = configs["checkpoint_path"]
        self.grad_accum_chunk_size = configs["grad_accum_chunk_size"]
        self.patience = configs["patience"]
        self.dropout = configs["dropout"]
        self.use_supernet_sampling = configs["use_supernet_sampling"]
        self.train = configs["train"]
        self.device = configs["device"]
        self.verbose = configs["verbose"]
        self.seed = configs["seed"]
        self.num_io_pairs_range = configs["num_io_pairs_range"]
        self.fml_len_range = configs["fml_len_range"]
        self.special_symbols = configs["special_symbols"]
        # The decoder-only model needs a <sep> token to separate the IO context
        # from the formula in its single causal stream.  Append it automatically
        # so ARCH="Deco" works without having to edit special_symbols by hand.
        if (ARCH == "Deco" and "<sep>" not in self.special_symbols):
            self.special_symbols = self.special_symbols + ["<sep>"]
        self.use_lr_warmup = configs["use_lr_warmup"]
        self.warmup_steps = configs["warmup_steps"]
        self.peak_lr = configs.get("peak_lr", 4e-4)
        # Max gradient norm for clipping.  <= 0 disables clipping entirely -- the
        # pre-clip norm is still measured and logged either way, so the clipped and
        # unclipped cells of the norm-position sweep stay directly comparable.
        self.grad_clip = configs.get("grad_clip", 0.5)
        self.log_grad_norms = configs.get("log_grad_norms", False)
        self._opt_steps = 0
        self._gn_rows = []
        self._gn_header_written = False
        self._last_loss = float("nan")
        self.noise_gamma = configs["noise_gamma"]
        self.noise_clean_frac = configs.get("noise_clean_frac", 0.0)
        self.steps_per_epoch = configs["steps_per_epoch"]
        self.target_tokens_per_batch = configs.get("target_tokens_per_batch", 10_000)
        self.num_data_workers = configs.get("num_data_workers", 0)
        self.persistent_workers = configs.get("persistent_workers", False)
        self.val_steps = configs.get("val_steps", 200)
        self.bpe_model_path = configs.get("bpe_model_path", "")
        # Optional complexity prior for online data generation.  None (default) keeps
        # the original uniform formula-size sampling; a float in (0, 1] biases towards
        # simpler formulae (see online_batch_iterater / _decaying_choice in data_process).
        self.complexity_decay = configs.get("complexity_decay", None)
        # Operator-sampling distribution for online data generation.  "uniform"
        # (default) is the original behaviour; "tiered" favours common operators
        # (see set_operator_weights in grammar.py).
        self.operator_weights = configs.get("operator_weights", "uniform")
        # False trains on raw (unsimplified) formula targets -- see online_batch_iterater.
        self.simplify_targets = configs.get("simplify_targets", True)
        # WHICH canonicaliser produces those targets: "ours" (simplifyFormula.simplify,
        # the default and every published run) or "simplipy" (the rule-mined engine of
        # Saegert & Koethe).  Orthogonal to simplify_targets, which decides WHETHER one
        # runs at all.  See simplify_backend.py.
        self.simplify_backend = configs.get("simplify_backend", "ours")
        # If set (float in (0,1]), overrides complexity_decay with target-length-first
        # sampling so long formulae actually reach the fml_len_range cap -- see the
        # length_decay note in data_process.online_batch_iterater.
        self.length_decay = configs.get("length_decay", None)
        # False (default) generates bare skeletons; True injects E2E-style float
        # prefactors into every additive term / unary argument before simplification
        # (grammar.add_prefactors) -- see online_batch_iterater.
        self.use_prefactors = configs.get("use_prefactors", False)
        # True (default) restricts constant leaves to {0,1,2,3,pi} and assembles other
        # values as composite subexpressions; False draws leaves from C_ALL = C + [CONST]
        # so they may be arbitrary random floats, as the E2E generator does.
        self.restrict_consts = configs.get("restrict_consts", False)
        # Either knob makes float literals reachable, which decides both the vocabulary
        # (FlatGramFloat vs FlatGram) and the 4x target-length expansion.
        self.needs_float_tokens = self.use_prefactors or not self.restrict_consts

        # Optionally load a subtree-BPE model for subformula compression.  With no
        # BPE model the vocabulary is the bare grammar.  A BPE model is learned on the
        # mantissa-free sampler, so it is bypassed under use_prefactors (matching
        # data_process); with float leaves alone its compounds still apply and the float
        # tokens are appended to its extended vocabulary.
        self.bpe_model = None
        if self.bpe_model_path and not self.use_prefactors and Path(self.bpe_model_path).exists():
            self.bpe_model = SubtreeBPE.load(self.bpe_model_path)
            print(f"Loaded BPE model: {self.bpe_model_path} ({len(self.bpe_model.merges)} merges)", flush=True)
        if self.bpe_model is not None:
            base_gram = self.bpe_model.extended_vocab()
            if self.needs_float_tokens:
                base_gram = base_gram + FLOAT_TOKENS
        else:
            base_gram = FlatGramFloat if self.needs_float_tokens else FlatGram
        self.vocab = base_gram + self.special_symbols

        self.pad_index = self.vocab.index("<pad>")
        self.vocab_size = len(self.vocab)
        # max_fml_len is the maximum TARGET sequence length.  Under the mantissa-free
        # grammar each symbolic token maps to a single target token, so the cap is
        # fml_len (+ the two <bes> delimiters added by the model); once floats are
        # reachable each literal expands to 4 tokens, and under prefactors fml_len bounds
        # the SKELETON that add_prefactors then inflates before that expansion (6x
        # headroom).  This must match target_num_cols in data_process.py.  BPE compound
        # tokens only shorten it.
        self.max_fml_len = self.fml_len_range[1] * (
            6 if self.use_prefactors else 4 if self.needs_float_tokens else 1)
        self.config_space = config_space
        self.log_gradients = configs["log_gradients"]
        self.supernet_path = configs.get("supernet_path", "")
        self.use_lpe = configs.get("use_lpe", False)
        self.lpe_d_emb = configs.get("lpe_d_emb", 16)
        self.lpe_n_mlp_layers = configs.get("lpe_n_mlp_layers", 2)
        self.lpe_expansion_factor = configs.get("lpe_expansion_factor", 1.0)
        self.activation = configs.get("activation", "gelu")
        self.norm_position = configs.get("norm_position", "pre")
        self.accumulate_gradients = configs.get("accumulate_gradients", 1)

        # initialize the containers
        self.ind2tok = {i: tok for i, tok in enumerate(self.vocab)}
        self.tok2ind = {tok: i for i, tok in enumerate(self.vocab)}


        # manually-set network configuration
        # GA chosen config
        if (self.use_supernet_sampling):
            # get the largest possible model's configuration
            self.config = get_supernet_config(self.config_space, use_lpe=self.use_lpe)
        else:
            #self.config = get_supernet_config(self.config_space)
            # Deep-copied so per-run overrides below never mutate the module-level dict
            # (self.config is also what gets written into the checkpoint).
            self.config = copy.deepcopy(ga_config)
            self._apply_arch_overrides(self.config)

        # Inject LPE settings into the model config so they are saved with the
        # checkpoint and automatically restored when the checkpoint is loaded.
        self.config['use_lpe'] = self.use_lpe
        self.config['lpe_d_emb'] = self.lpe_d_emb
        self.config['lpe_n_mlp_layers'] = self.lpe_n_mlp_layers
        self.config['lpe_expansion_factor'] = self.lpe_expansion_factor
        self.config['activation'] = self.activation
        self.config['norm_position'] = self.norm_position

        # invoke the training procedure
        if (not torch.cuda.is_available() and not torch.backends.mps.is_available()): self.device = "cpu"

        print(f"device for training: {self.device}", flush = True)

        # set the seed
        set_seed(self.seed)

        if self.accumulate_gradients > 1:
            print(f"Using Gradient Accumulation (effective batch = {self.grad_accum_chunk_size * self.accumulate_gradients})", flush = True)

        # Create a Van object
        print(f"size of the vocabulary: {len(self.vocab)}", flush = True)
        #print(self.vocab, flush = True)
        print("pad_index: ", self.pad_index, flush = True)

        self._build_model()

    def _build_model(self):
        """(Re)construct self.model at the CURRENT self.max_fml_len.

        Factored out of __init__ so _adopt_checkpoint_config() can rebuild at the
        cap a resumed checkpoint was actually trained under, using identical
        construction arguments rather than a second hand-written copy of them.
        """
        if (ARCH == "Van"):
            self.model = VanillaTransformer(self.config, len(V) + 1, self.vocab_size, self.max_fml_len + 1, self.pad_index, self.dropout).to(self.device) # + 1 for <bes>
        elif (ARCH == "Sett"):
            self.model = SetTransformer(self.config, len(V) + 1, self.vocab_size, self.max_fml_len + 1, self.pad_index, self.dropout).to(self.device) # + 1 for <bes>
        elif (ARCH == "Deco"):
            # Decoder-only: single causal stream <bes> IO <sep> formula <bes>.
            # Requires "<sep>" in special_symbols so the IO context can be separated
            # from the formula.  Positions only span the formula length, so
            # max_fml_len + 1 is sufficient positional capacity.
            self.model = DecoderOnlyTransformer(self.config, len(V) + 1, self.vocab_size, self.max_fml_len + 1, self.pad_index, self.dropout, sep_idx=self.vocab.index("<sep>")).to(self.device)

        self.model.set_sample_config(self.config)
        print(f"number of parameters in the super transformer: {sum(p.numel() for p in self.model.parameters())}")
        

    @staticmethod
    def _apply_arch_overrides(config):
        """Optional env overrides of the ga_config architecture, for ablation sweeps.

        NUM_ENC_LAYERS / NUM_DEC_LAYERS / D_MODEL / D_FF / N_HEADS.  Unset leaves
        ga_config exactly as written, so a normal production run is untouched.  The
        per-layer list-valued keys are rebuilt to the new depth, which is what makes
        a depth override safe (van_config stores heads/dims as one entry per layer).

        This exists so the norm-position sweep can train a ~10-20M model without
        editing van_config.py -- see the ga_config quote trap, where the active model
        size depends on which of two copies is inside a triple-quoted block.
        """
        def _get(name):
            raw = os.environ.get(name, "")
            return int(raw) if raw.strip() else None

        n_enc = _get("NUM_ENC_LAYERS")
        n_dec = _get("NUM_DEC_LAYERS")
        d_model = _get("D_MODEL")
        d_ff = _get("D_FF")
        n_heads = _get("N_HEADS")
        if not any(v is not None for v in (n_enc, n_dec, d_model, d_ff, n_heads)):
            return

        for prefix, n_new in (("enc", n_enc), ("dec", n_dec)):
            n_old = config[f"num_{prefix}_layers"]
            n = n_new if n_new is not None else n_old
            config[f"num_{prefix}_layers"] = n
            for key, override in (
                (f"num_{prefix}_heads", n_heads),
                (f"qk_dim_{prefix}_embed", None),
                (f"v_dim_{prefix}_embed", None),
                (f"dim_{prefix}_mlp", d_ff),
            ):
                # Re-length the per-layer list, keeping the existing value (or the
                # override) for every layer.
                base = override if override is not None else config[key][0]
                config[key] = [base] * n
            if d_model is not None:
                config[f"in_dim_{prefix}_embed"] = d_model

        print(f"arch overrides applied: enc={config['num_enc_layers']}L "
              f"dec={config['num_dec_layers']}L d_model={config['in_dim_enc_embed']} "
              f"d_ff={config['dim_enc_mlp'][0]} heads={config['num_enc_heads'][0]}", flush=True)

    def _clip_and_record(self, optimizer):
        """Clip the accumulated gradients and record the PRE-clip global grad norm.

        torch.nn.utils.clip_grad_norm_ returns the norm it measured *before* rescaling,
        which is the diagnostic that separates a genuinely stable run from one the
        clipper is silently holding together.  Without logging it, a post-LN arm that
        is spiking to ||g|| = 200 every step looks identical to a pre-LN arm sitting at
        0.3, because both emerge from the clipper at exactly 0.5.

        With clipping disabled (grad_clip <= 0) the norm is measured but the gradients
        are left strictly untouched.  Routing that case through clip_grad_norm_ with
        max_norm=inf would NOT be equivalent: on a step whose gradient is already
        non-finite the clip coefficient becomes inf/(inf + eps) = nan, and every
        parameter's gradient is then multiplied by nan.  That would turn one bad step
        into a permanently poisoned model in exactly the arm -- unclipped -- where the
        divergence behaviour is the thing being measured.
        """
        if self.grad_clip and self.grad_clip > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        else:
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            total_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2) if grads \
                else torch.tensor(0.0)
        self._opt_steps += 1
        if self.log_grad_norms:
            self._record_grad_norm(float(total_norm), optimizer)

    def _record_grad_norm(self, grad_norm, optimizer):
        """Buffer one row of grad-norm telemetry; flush to CSV periodically.

        Flushing as we go (rather than at the end) means a run that diverges and is
        killed -- precisely the outcome the sweep is trying to count -- still leaves
        its telemetry behind.
        """
        finite = math.isfinite(grad_norm)
        self._gn_rows.append({
            "step": self._opt_steps,
            "lr": optimizer.param_groups[0]["lr"],
            "loss": self._last_loss,
            "grad_norm": grad_norm,
            # Did the clipper actually bite on this step?
            "clipped": int(finite and self.grad_clip > 0 and grad_norm > self.grad_clip),
            "nonfinite": int(not finite),
        })
        if len(self._gn_rows) >= 50:
            self._flush_grad_norms()

    def _flush_grad_norms(self):
        if not self._gn_rows:
            return
        path = self.run_dir / "grad_norms.csv"
        fields = list(self._gn_rows[0].keys())
        write_header = not self._gn_header_written and not path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerows(self._gn_rows)
        self._gn_header_written = True
        self._gn_rows = []

    def _step(self, optimizer, n_iter):
        if n_iter % self.accumulate_gradients == 0:
            self._clip_and_record(optimizer)
            optimizer.step()
            if self.use_lr_warmup:
                self.scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    def optimize(self, loss, optimizer, n_iter):
        if self.accumulate_gradients > 1:
            loss = loss / self.accumulate_gradients
        # Kept for the grad-norm log so each recorded norm carries the loss that
        # produced it (rescaled back to the unscaled per-microbatch CE).  Guarded
        # because .item() forces a device sync, and the log is off by default.
        if self.log_grad_norms:
            self._last_loss = float(loss.item()) * max(1, self.accumulate_gradients)
        loss.backward()
        self._step(optimizer, n_iter)

    def _adopt_checkpoint_config(self, checkpoint):
        """Reconcile this run's target settings with the checkpoint it is resuming.

        A checkpoint is the authority on the settings that shaped its weights; the
        config literals in this file only describe a run started from scratch.  The
        two settings handled here were, until now, absent from the saved dict, so a
        resume had to trust whatever the literal happened to say that day.

        fml_len_range sizes dec_positional_encoding.pe, and the literal has moved
        over the project's life (40 -> 80 -> 100).  A cluster tree that picked up a
        newer literal turned every resume of an older run into a bare `size mismatch`
        traceback ~12 s into a GPU allocation.  There is only one workable answer --
        the checkpoint's -- so adopt it and say so loudly.

        simplify_targets is deliberately NOT adopted.  It does not change any tensor
        shape, so a mismatch does not crash: it silently trains a different arm to
        completion.  That is a broken launch (a dropped SIMPLIFY_TARGETS export), and
        papering over it here would hide the same breakage in the sibling runs that
        started from scratch, so refuse and name what to set.

        Checkpoints written before these fields were recorded carry neither; the cap
        is still recoverable from the pe table, and an absent simplify_targets is
        simply unverifiable and passes.
        """
        state_dict = checkpoint[f"{ARCH}_state_dict"]

        # --- target length cap: adopt ------------------------------------------
        ckpt_range = checkpoint.get("fml_len_range")
        ckpt_cap = ckpt_range[1] if ckpt_range else None
        if ckpt_cap is None:
            # Legacy checkpoint: recover the cap from the positional table it saved.
            # Use the checkpoint's OWN generation flags for the expansion factor --
            # they have been recorded for far longer than fml_len_range has.
            pe = state_dict.get("dec_positional_encoding.pe")
            if pe is not None:
                use_pre = checkpoint.get("use_prefactors", self.use_prefactors)
                restrict = checkpoint.get("restrict_consts", self.restrict_consts)
                expand = 6 if use_pre else 4 if (use_pre or not restrict) else 1
                rows = pe.shape[0] - 1
                if rows % expand == 0:
                    ckpt_cap = rows // expand

        if ckpt_cap is not None and ckpt_cap != self.fml_len_range[1]:
            print(f"NOTE: checkpoint was trained at fml_len cap {ckpt_cap}, this config says "
                  f"{self.fml_len_range[1]} -- adopting the checkpoint's and rebuilding the model. "
                  f"(Set the config literal to match if you meant to change it; a cap change "
                  f"needs a fresh run_dir, not a resume.)", flush=True)
            self.fml_len_range = [self.fml_len_range[0], ckpt_cap]
            self.max_fml_len = self.fml_len_range[1] * (
                6 if self.use_prefactors else 4 if self.needs_float_tokens else 1)
            self._build_model()

        # --- target canonicalisation: refuse -----------------------------------
        ckpt_simp = checkpoint.get("simplify_targets")
        if ckpt_simp is not None and bool(ckpt_simp) != bool(self.simplify_targets):
            raise RuntimeError(
                f"refusing to resume: this checkpoint was trained with "
                f"simplify_targets={bool(ckpt_simp)} but this run is configured for "
                f"simplify_targets={bool(self.simplify_targets)}.\n"
                f"Nothing about this mismatch would crash on its own -- it would just "
                f"train a different arm into {self.checkpoint_path}.\n"
                f"Relaunch with SIMPLIFY_TARGETS={int(bool(ckpt_simp))} to continue this "
                f"run, or point RUN_NAME at a fresh run_dir to start the other arm.")

    # train the (super) Van and save the results including the trained weights
    def train_and_save(self):
        # train the Hybrid Transformer on the preprocessed data
        if (self.train or not Path(self.model_path).exists()):
            # training starting from the checkpoint
            if (len(self.checkpoint_path) != 0 and Path(self.checkpoint_path).exists()):
                # A checkpoint interrupted mid-write (pre-_atomic_torch_save, or on a
                # filesystem without hard links) is an unreadable zip.  Fall back to the
                # .prev generation -- one epoch older -- rather than losing the run.
                try:
                    checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(self.device), weights_only=False)
                except Exception as err:
                    prev_path = self.checkpoint_path + ".prev"
                    print(f"WARNING: {self.checkpoint_path} is unreadable ({err}); trying {prev_path}", flush=True)
                    if not Path(prev_path).exists():
                        raise RuntimeError(
                            f"checkpoint {self.checkpoint_path} is corrupt and no .prev exists. "
                            f"Move it aside to restart from scratch, deliberately.") from err
                    checkpoint = torch.load(prev_path, map_location=torch.device(self.device), weights_only=False)
                    print(f"Recovered from {prev_path}", flush=True)
                self._adopt_checkpoint_config(checkpoint)
                self.model.load_state_dict(checkpoint[f"{ARCH}_state_dict"])
                self._resume_optimizer_state = checkpoint.get("optimizer_state_dict")
                self._resume_scheduler_state = checkpoint.get("scheduler_state_dict")
                self._resume_best_val_loss   = float('inf')  # always reset so patience starts from 0
                print(f"Loaded from checkpoint {self.checkpoint_path}")
                # Drop the checkpoint dict now.  It is a local of train_and_save, which
                # goes on to call trainTransformer() below, so without this the loaded
                # weights (already copied into the model) and the saved loss histories
                # stay resident on self.device for the ENTIRE run -- roughly another
                # parameter-count's worth of tensors on the GPU for nothing.  Only the
                # optimizer/scheduler states survive, via the attributes above, and
                # trainTransformer() releases those as soon as they are loaded.
                del checkpoint
            elif (not self.use_supernet_sampling and len(self.supernet_path) != 0 and Path(self.supernet_path).exists()):
                supernet_data = torch.load(self.supernet_path, map_location = torch.device(self.device), weights_only = True)
                supernet_sd = supernet_data[f"{ARCH}_state_dict"]
                model_sd = self.model.state_dict()
                sliced, skipped = 0, 0
                for name, param in model_sd.items():
                    if name not in supernet_sd:
                        skipped += 1
                        continue
                    src = supernet_sd[name]
                    if src.shape == param.shape:
                        model_sd[name] = src
                    elif all(s >= p for s, p in zip(src.shape, param.shape)):
                        # slice the supernet tensor down to sub-network dims
                        slices = tuple(slice(0, d) for d in param.shape)
                        model_sd[name] = src[slices].clone()
                        sliced += 1
                    else:
                        skipped += 1
                self.model.load_state_dict(model_sd)
                print(f"Inherited weights from supernet {self.supernet_path} (sliced: {sliced}, skipped: {skipped})")

            # Enable TF32 tensor cores for ~20% faster float32 matmuls on Ampere+ GPUs.
            torch.set_float32_matmul_precision('high')

            # torch.compile is intentionally disabled: it changes the active
            # sub-graph each step under supernet sampling (defeating the compiled
            # cache) and needs triton on CUDA.  Re-enable here if that changes.

            # train the model
            list_train_loss, list_train_acc, list_vali_loss, list_vali_acc = self.trainTransformer()

            # save the trained weights
            _atomic_torch_save({
                f"{ARCH}_state_dict": self.model.state_dict(),
                "model_config": self.config,
                "vocab": self.vocab,
                # Generation settings that the vocabulary alone cannot distinguish
                # (both prefactor arms share FlatGramFloat) -- recorded so downstream
                # evaluation can tell the two runs apart.
                "use_prefactors": self.use_prefactors,
                "restrict_consts": self.restrict_consts,
                "simplify_backend": self.simplify_backend,
                # Recorded so a RESUME can reconcile against them rather than trusting
                # this file's literals -- see _adopt_checkpoint_config().  fml_len_range
                # sizes dec_positional_encoding.pe; simplify_targets changes the target
                # distribution without changing any shape.
                "fml_len_range": self.fml_len_range,
                "simplify_targets": self.simplify_targets,
                "list_train_loss": list_train_loss,
                "list_train_acc": list_train_acc,
                "list_vali_loss": list_vali_loss,
                "list_vali_acc": list_vali_acc,
            }, self.model_path)
        else:
            # load previously trained model
            startT = time()
            print("Loading previously trained model...", end = "", flush = True)
            model_data = torch.load(self.model_path, map_location = torch.device(self.device), weights_only = True)
            print("done ", end = "", flush = True)
            self.model.load_state_dict(model_data[f"{ARCH}_state_dict"])
            list_train_loss = model_data["list_train_loss"]
            list_train_acc = model_data["list_train_acc"]
            list_vali_loss = model_data["list_vali_loss"]
            list_vali_acc = model_data["list_vali_acc"]
            print(f"({time() - startT:.2f} sec.)", flush = True)

        # save the results
        print("Saving training results...", end = "", flush = True)
        startT = time()
        data = {"list_train_loss": list_train_loss,
                "list_train_acc": list_train_acc,
                "list_vali_loss": list_vali_loss,
                "list_vali_acc": list_vali_acc}
        with gzip.open(self.run_dir / f"train_super_{ARCH}_ios_eval_res.pkl.gz", "wb") as f:
            pickle.dump(data, f)
        print(f"({time() - startT:.2f} sec.)", flush = True)


    # train a model with the generated data with early stop enabled
    def trainTransformer(self):
        # initialize the loss function
        criterion = SymbolicRegressionLoss(self.vocab, self.pad_index).to(self.device)

        # Initialize the optimizer
        peak_lr = self.peak_lr
        start_lr = 1e-7
        #optimizer = optim.AdamW(self.model.parameters(), lr = peak_lr, betas = (0.9, 0.999), eps = 1e-8)
        optimizer = optim.Adam(self.model.parameters(), lr = peak_lr, betas = (0.9, 0.999), eps = 1e-8)

        # Initialize the scheduler: linear warmup then inverse-sqrt decay (mirrors e2e paper)
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                alpha = current_step / self.warmup_steps
                return (start_lr / peak_lr) + alpha * (1 - start_lr / peak_lr)
            return max((self.warmup_steps / current_step) ** 0.5, 0.25)

        if (self.use_lr_warmup):
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        optimizer.zero_grad(set_to_none=True)

        # Restore optimizer / scheduler / best-val-loss from checkpoint if available.
        best_val_loss = getattr(self, '_resume_best_val_loss', float('inf'))
        # Each state is cleared right after it is loaded: load_state_dict() copies what
        # it needs into the optimizer/scheduler, so holding our own reference just keeps
        # a second copy of the Adam moments (2x the parameter count) alive on
        # self.device for the whole run.
        if getattr(self, '_resume_optimizer_state', None) is not None:
            optimizer.load_state_dict(self._resume_optimizer_state)
            self._resume_optimizer_state = None
            print("Restored optimizer state from checkpoint.", flush=True)
        if self.use_lr_warmup and getattr(self, '_resume_scheduler_state', None) is not None:
            self.scheduler.load_state_dict(self._resume_scheduler_state)
            self._resume_scheduler_state = None
            print("Restored scheduler state from checkpoint.", flush=True)

        # stats
        iterations = 0
        num_stall_epoches = 0
        early_stop = False
        verboseFreq = 300
        # Carry the most recent held-out loss into the next epoch's first progress
        # line.  These persist ACROSS epochs (hence initialised here, not inside the
        # epoch loop): validation runs at each epoch's end and sets is_validated so
        # the following epoch's first verbose row can print it, then clears the flag.
        vali_loss = float('nan')
        is_validated = False
        #verboseFreq = 10
        best_model_state = None  # Holds the best model weights in memory
        best_model_config = None # Holds the best model configuration in memory

        # print the minibatch size
        print(f"batch size: {self.grad_accum_chunk_size}  |  accumulate_gradients: {self.accumulate_gradients}  |  effective batch: {self.grad_accum_chunk_size * self.accumulate_gradients}", flush = True)

        # Generate a fixed held-out validation set once before training.
        # If val_steps == 0, skip validation entirely and monitor mean epoch train loss instead.
        val_data = []
        if self.val_steps > 0:
            print(f"Generating held-out validation set ({self.val_steps} batches)...", end="", flush=True)
            startT = time()
            for vInputs, vTargets in online_batch_iterater(
                self.tok2ind,
                self.fml_len_range,
                self.num_io_pairs_range,
                batch_size=self.grad_accum_chunk_size,
                steps_per_epoch=self.val_steps,
                noise_gamma=self.noise_gamma,
                noise_clean_frac=self.noise_clean_frac,
                target_tokens_per_batch=self.target_tokens_per_batch,
                bpe_model=self.bpe_model,
                complexity_decay=self.complexity_decay,
                operator_weights=self.operator_weights,
                simplify_targets=self.simplify_targets,
                length_decay=self.length_decay,
                use_prefactors=self.use_prefactors,
                restrict_consts=self.restrict_consts,
                simplify_backend=self.simplify_backend,
            ):
                val_data.append((vInputs.cpu(), vTargets.cpu()))
            print(f" done ({time() - startT:.1f}s)", flush=True)
        else:
            print("val_steps=0: monitoring mean epoch training loss for early stopping.", flush=True)

        # data recording
        list_train_loss = []
        list_train_acc = []
        list_vali_loss = []
        list_vali_acc = []

        if (self.verbose):
            print("|========================================================================================|", flush = True)
            print("|  Epoch  |  Iteration  |  Time Elapsed  |   Training   |  Validation  |  Base Learning  |", flush = True)
            print("|         |             |   (hh:mm:ss)   |     Loss     |     Loss     |      Rate       |", flush = True)
            print("|========================================================================================|", flush = True)
            train_startT = time()

        # Create the DataLoader once before the epoch loop so forkserver workers
        # are spawned only once and reused across epochs (persistent_workers=True).
        # OnlineBatchDataset._iter_count guarantees distinct data each epoch even
        # when torch.initial_seed() does not change with persistent workers.
        if self.num_data_workers > 0:
            _train_dataset = OnlineBatchDataset(
                self.tok2ind,
                self.fml_len_range,
                self.num_io_pairs_range,
                batch_size=self.grad_accum_chunk_size,
                steps_per_epoch=self.steps_per_epoch,
                noise_gamma=self.noise_gamma,
                noise_clean_frac=self.noise_clean_frac,
                target_tokens_per_batch=self.target_tokens_per_batch,
                seed=self.seed,
                bpe_model=self.bpe_model,
                complexity_decay=self.complexity_decay,
                operator_weights=self.operator_weights,
                simplify_targets=self.simplify_targets,
                length_decay=self.length_decay,
                use_prefactors=self.use_prefactors,
                restrict_consts=self.restrict_consts,
                simplify_backend=self.simplify_backend,
            )
            _train_loader = DataLoader(
                _train_dataset,
                batch_size=None,
                num_workers=self.num_data_workers,
                prefetch_factor=2,
                collate_fn=_passthrough_collate,
                multiprocessing_context="forkserver",
                persistent_workers=self.persistent_workers,
            )

        # the main training loop
        for epoch in range(self.max_epochs):
            # data recording.  Training loss/accuracy are accumulated as running sums
            # rather than per-batch lists: keeping every batch value grew the histories
            # by steps_per_epoch floats per epoch (~0.18 MB/epoch at 3000 steps, i.e.
            # 3.6 GB at max_epochs=20000) AND re-pickled that whole history into every
            # checkpoint save, so save size and time grew linearly with the epoch count.
            # Only the epoch mean is recorded, matching what list_vali_loss already does.
            sum_train_loss  = 0.0
            sum_train_acc   = 0.0
            n_train_batches = 0
            cur_vali_loss = []
            cur_vali_acc = []

            # train one epoch
            if self.num_data_workers > 0:
                train_iter = _train_loader
            else:
                train_iter = online_batch_iterater(
                    self.tok2ind,
                    self.fml_len_range,
                    self.num_io_pairs_range,
                    batch_size=self.grad_accum_chunk_size,
                    steps_per_epoch=self.steps_per_epoch,
                    noise_gamma=self.noise_gamma,
                    noise_clean_frac=self.noise_clean_frac,
                    target_tokens_per_batch=self.target_tokens_per_batch,
                    bpe_model=self.bpe_model,
                    complexity_decay=self.complexity_decay,
                    operator_weights=self.operator_weights,
                    simplify_targets=self.simplify_targets,
                    length_decay=self.length_decay,
                    use_prefactors=self.use_prefactors,
                    restrict_consts=self.restrict_consts,
                    simplify_backend=self.simplify_backend,
                )
            self.model.train()
            for trainInputs, trainTargets in train_iter:
                trainInputs = trainInputs.to(self.device)
                trainTargets = trainTargets.to(self.device)

                iterations += 1

                if self.use_supernet_sampling:
                    # Sandwich rule: first slot of each window -> max (supernet),
                    # last slot -> min (infinet), remaining slots -> random.
                    # When accumulate_gradients==1 there is only one slot, so
                    # both conditions would match; the else branch handles that
                    # case and keeps pure random sampling.
                    n = self.accumulate_gradients
                    window_slot = (iterations - 1) % n
                    if n >= 2 and window_slot == 0:
                        self.config = get_supernet_config(self.config_space, use_lpe=self.use_lpe)
                    elif n >= 2 and window_slot == n - 1:
                        self.config = get_infinet_config(self.config_space, use_lpe=self.use_lpe)
                    else:
                        self.config = sample_config(self.config_space, use_lpe=self.use_lpe)
                    self.model.set_sample_config(self.config)

                train_batch_ios = trainInputs.contiguous()
                if (ARCH == "Deco"):
                    # The decoder-only model builds its own <bes> IO <sep> formula
                    # stream and matching targets (floats -> <pad>, formula shifted right).
                    trainPreds         = self.model(train_batch_ios, trainTargets)
                    train_batch_target = self.model.build_targets(train_batch_ios, trainTargets)
                else:
                    train_batch_fml_in = trainTargets[:, :-1].contiguous()
                    train_batch_target = trainTargets[:, 1:].contiguous()
                    trainPreds         = self.model(train_batch_ios, train_batch_fml_in)
                loss_dict  = criterion(trainPreds, train_batch_target, train_batch_ios)

                training_batch_loss = loss_dict['ce'].item()
                training_batch_acc  = model_acc(trainPreds.contiguous().reshape(-1, self.vocab_size), train_batch_target.reshape(-1), self.pad_index).item()
                sum_train_loss  += training_batch_loss
                sum_train_acc   += training_batch_acc
                n_train_batches += 1
                self.optimize(loss_dict['loss'], optimizer, iterations)

                if (self.verbose and (iterations == 1 or iterations % verboseFreq == 0 or early_stop)):
                    learningRate = optimizer.param_groups[0]['lr']
                    formatted_time = str(datetime.timedelta(seconds = round(time() - train_startT)))
                    if (is_validated):
                        print(f"| {epoch + 1:7} | {iterations:11} | {formatted_time:14} | {training_batch_loss:12.4f} | {vali_loss:12.4f} | {learningRate:15.4} |", flush = True)
                        is_validated = False
                        if (self.log_gradients):
                            self.log_layer_stats() # inspect the gradients
                    else:
                        print(f"| {epoch + 1:7} | {iterations:11} | {formatted_time:14} | {training_batch_loss:12.4f} | ------------ | {learningRate:15.4} |", flush = True)

            # Flush any uncommitted accumulated gradients at the epoch boundary
            # so they are not carried into the next epoch's accumulation window.
            if iterations % self.accumulate_gradients != 0:
                self._clip_and_record(optimizer)
                optimizer.step()
                if self.use_lr_warmup:
                    self.scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Validate on the held-out set after each epoch.
            train_mean = sum_train_loss / n_train_batches if n_train_batches else float('nan')
            train_acc_mean = sum_train_acc / n_train_batches if n_train_batches else float('nan')
            if self.val_steps > 0:
                if self.use_supernet_sampling:
                    self.model.set_sample_config(get_supernet_config(self.config_space, use_lpe=self.use_lpe))
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for vInputs, vTargets in val_data:
                        vInputs  = vInputs.to(self.device)
                        vTargets = vTargets.to(self.device)
                        if (ARCH == "Deco"):
                            vPreds = self.model(vInputs, vTargets)
                            total_val_loss += criterion.ce_loss(vPreds, self.model.build_targets(vInputs, vTargets)).item()
                        else:
                            vPreds = self.model(vInputs, vTargets[:, :-1])
                            total_val_loss += criterion.ce_loss(vPreds, vTargets[:, 1:]).item()
                vali_loss    = total_val_loss / len(val_data)
                is_validated = True
                cur_vali_loss.append(vali_loss)
                if self.verbose:
                    learningRate   = optimizer.param_groups[0]['lr']
                    formatted_time = str(datetime.timedelta(seconds=round(time() - train_startT)))
                    print(f"| {epoch + 1:7} | {iterations:11} | {formatted_time:14} | {train_mean:12.4f} | {vali_loss:12.4f} | {learningRate:15.4} |", flush=True)
            else:
                # val_steps == 0: monitor mean epoch training loss.
                vali_loss    = train_mean
                is_validated = True
                cur_vali_loss.append(vali_loss)
                if self.verbose:
                    learningRate   = optimizer.param_groups[0]['lr']
                    formatted_time = str(datetime.timedelta(seconds=round(time() - train_startT)))
                    print(f"| {epoch + 1:7} | {iterations:11} | {formatted_time:14} | {train_mean:12.4f} | {train_mean:12.4f} | {learningRate:15.4} |", flush=True)

            # Early stop monitors validation loss (or mean train loss when val_steps == 0).
            monitor_loss = vali_loss

            if (monitor_loss < best_val_loss):
                best_val_loss = monitor_loss
                num_stall_epoches = 0
                best_model_state = copy.deepcopy(self.model.state_dict())  # save to RAM

                if (self.use_supernet_sampling):
                    best_model_config = get_supernet_config(self.config_space, use_lpe=self.use_lpe)

                # save the trained model to checkpoint
                if (self.use_supernet_sampling):
                    checkpoint_path = self.run_dir / f"model_super_{ARCH}_checkpoint.pth"
                else:
                    checkpoint_path = self.run_dir / f"model_fixed_{ARCH}_checkpoint.pth"
                _atomic_torch_save({
                    f"{ARCH}_state_dict": best_model_state,
                    "model_config": self.config,
                    "vocab": self.vocab,
                    "use_prefactors": self.use_prefactors,
                    "restrict_consts": self.restrict_consts,
                    "simplify_backend": self.simplify_backend,
                    # See the note at the final-weights save: these two let a resume
                    # reconcile against the checkpoint instead of this file's literals.
                    "fml_len_range": self.fml_len_range,
                    "simplify_targets": self.simplify_targets,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.use_lr_warmup else None,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "list_train_loss": list_train_loss,
                    "list_train_acc": list_train_acc,
                    "list_vali_loss": list_vali_loss,
                    "list_vali_acc": list_vali_acc,
                }, checkpoint_path)
            else:
                num_stall_epoches += 1
                # patience <= 0 disables early stopping entirely, so the run always
                # consumes its full step budget.  An ablation sweep needs that: with
                # early stopping on, a cell whose loss plateaus (a collapsed arm, say)
                # quits sooner than a healthy one and is then scored on a SHORTER budget
                # than the cells it is being compared against.
                if (self.patience > 0 and num_stall_epoches >= self.patience):
                    early_stop = True

            # data recording.  One epoch-mean per entry (wrapped in a list so all four
            # series keep the same list-of-lists shape they have always had).
            list_train_loss.append([train_mean])
            list_train_acc.append([train_acc_mean])
            list_vali_loss.append(cur_vali_loss)
            list_vali_acc.append(cur_vali_acc)

            # break if training ended
            if (early_stop): break

            # Flush telemetry every epoch as well as every 50 steps, so a run killed
            # mid-epoch (or one that NaNs out) still leaves a usable grad_norms.csv.
            self._flush_grad_norms()

        self._flush_grad_norms()

        if (self.verbose):
            print("|========================================================================================|", flush = True)

        # store the model with the lowest validation loss
        if (early_stop and best_model_state is not None):
            if (self.use_supernet_sampling):
                self.model.set_sample_config(best_model_config)
            self.model.load_state_dict(best_model_state)

        # return
        return list_train_loss, list_train_acc, list_vali_loss, list_vali_acc 


    def log_layer_stats(self):
        stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Calculate the L2 norm of the gradients
                grad_norm = torch.norm(param.grad, 2).item()

                # Calculate the L2 norm of the weights
                weight_norm = torch.norm(param.data, 2).item()
                
                stats[name] = {
                    "grad_norm": grad_norm,
                    "weight_norm": weight_norm,
                    "ratio": grad_norm / (weight_norm + 1e-8)
                }
                
                # Print or log to TensorBoard/W&B
                print(f"Layer: {name:20} | Grad Norm: {grad_norm:.6f} | Ratio: {stats[name]['ratio']:.6f}", flush = True)
        return stats


if (__name__ == "__main__"):
    import os

    # ------------------------------------------------------------------
    # Per-run isolation.  Two training runs can share ONE checkout of this
    # repo (e.g. the two arms of the prefactor ablation) as long as they do
    # not write to the same files.  These environment variables are the only
    # thing that has to differ between them:
    #
    #   RUN_NAME        every artifact goes to checkpoints/<RUN_NAME>_res/
    #                   -- the "<name>_res" layout eval_mymodels.py --model
    #                   already expects, so the run is evaluable by name.
    #                   Unset  -> flat ./checkpoints (the old behaviour).
    #   USE_PREFACTORS  1/0 override of the "use_prefactors" knob below.
    #   RESTRICT_CONSTS 1/0 override of the "restrict_consts" knob below.
    #   SIMPLIFY_BACKEND "ours" | "simplipy" override of the "simplify_backend" knob.
    #   NOISE_GAMMA     max training-time output noise (default 0 = clean).  gamma is
    #                   drawn per formula from U[0, NOISE_GAMMA]; 0.1 spans the tau range
    #                   the evaluation sweeps.
    #   NOISE_CLEAN_FRAC fraction of formulae left exactly noiseless when NOISE_GAMMA>0
    #                   (default 0 = all noised).  0.5 = even mix of clean and noised.
    #   RESUME          1 -> continue from THIS run's own best checkpoint
    #                   (checkpoints/<RUN_NAME>_res/model_fixed_<ARCH>_checkpoint.pth)
    #                   instead of starting from scratch.
    #
    # Everything unset reproduces exactly what this file did before.
    # ------------------------------------------------------------------
    def _env_int(name, default):
        """Environment override for a numeric knob; unset/empty keeps the default."""
        raw = os.environ.get(name, "")
        return int(raw) if raw.strip() else default

    def _env_float(name, default):
        """Environment override for a float knob; unset/empty keeps the default."""
        raw = os.environ.get(name, "")
        return float(raw) if raw.strip() else default

    def _env_flag(name, default):
        raw = os.environ.get(name, "")
        if raw.strip() == "":
            return default
        return raw.strip().lower() in ("1", "true", "yes", "on")

    def _default_data_workers():
        """DataLoader workers to use when NUM_DATA_WORKERS is not set.

        One per allocated core minus one for the main process, capped at 23 (RAM).
        The cap was 16 on the assumption that the generator stopped being the
        bottleneck well before that; the 2026-08-28 measurement noted below showed
        it was still the bottleneck at 15 workers, so the cap now tracks the 24
        cores per GPU a typical H100 node offers. Returns the
        historical 2 when not under Slurm or when the var is missing/unparseable, so
        an off-cluster run behaves exactly as it did before.
        """
        raw = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
        try:
            cpus = int(raw)
        except ValueError:
            return 2
        if cpus <= 0:
            return 2
        return max(2, min(23, cpus - 1))

    _run_name = os.environ.get("RUN_NAME", "").strip()
    _run_dir  = Path("./checkpoints") / f"{_run_name}_res" if _run_name else Path("./checkpoints")
    _resume   = _env_flag("RESUME", False)
    # Default "ours" -- simplifyFormula.simplify, what every published run used, and what
    # the BPE model in bpe_model_path was learned from.  This briefly defaulted to
    # "simplipy", which contradicted the "simplify_backend" comment below AND was a live
    # failure: set_backend runs inside each DataLoader worker (data_process.py:299), so on
    # a cluster without the optional simplipy package the run died at the first batch
    # rather than at config time.  Set SIMPLIFY_BACKEND=simplipy to opt into the ablation.
    _simplify_backend = os.environ.get("SIMPLIFY_BACKEND", "").strip() or "ours"
    _resume_from = str(_run_dir / f"model_fixed_{ARCH}_checkpoint.pth") if _resume else ""

    # training configurations
    configs = {
        # Directory for THIS run's artifacts: the per-epoch checkpoint, the final
        # weights, and the training-curve pickle.  Set via RUN_NAME (see above).
        "run_dir": str(_run_dir),
        "num_io_pairs_range": [50, 400],
        "max_epochs": _env_int("MAX_EPOCHS", 20000),
        "special_symbols": ["<bes>", "<pad>"],
        # Symbolic token length; targets stored as up to 3* this (float expansion).
        # FML_LEN_MAX overrides the cap so a len-40/len-80 arm can be trained without
        # editing this literal -- which a git pull silently reverts, and which would
        # also retarget any other run started from the same checkout.  Unset keeps 100,
        # so every existing launch is unchanged.  On RESUME the checkpoint's own
        # fml_len_range still wins over both (see _adopt_checkpoint_config).
        "fml_len_range": [1, _env_int("FML_LEN_MAX", 100)],
        "approx_correct_tol": 0.1,
        "model_path": str(_run_dir / "fixed_weights.pth"),
        # Empty = train from scratch; RESUME=1 points it at this run's own checkpoint.
        "checkpoint_path": _resume_from,
        "supernet_path": "./checkpoints/model_super_Van_checkpoint.pth", # supernet weights to inherit when training a fixed network
        "grad_accum_chunk_size": _env_int("GRAD_ACCUM_CHUNK_SIZE", 128),  # UPPER BOUND on samples per forward; actual B can be smaller when the
                                       # token budget (target_tokens_per_batch // N) bites or on a buffer-tail drain.
        "accumulate_gradients": _env_int("ACCUMULATE_GRADIENTS", 32),    # optimizer.step() runs every `accumulate_gradients` microbatches;
                                       # effective batch <= grad_accum_chunk_size * accumulate_gradients (equality only when
                                       # every microbatch hits the cap, i.e. small N and no buffer-tail in the window).
        # Epochs of no validation improvement before stopping.  PATIENCE=0 disables
        # early stopping so the run always uses its full MAX_EPOCHS budget.
        "patience": _env_int("PATIENCE", 10),
        "dropout": 0.0,
        "train": True,
        "use_supernet_sampling": False,
        "verbose": True,
        "use_lr_warmup": _env_flag("USE_LR_WARMUP", True),
        "warmup_steps": _env_int("WARMUP_STEPS", 3000),
        # Peak LR of the warmup -> inverse-sqrt schedule.  Exposed so the
        # norm-position sweep can vary it; 4e-4 is what every published run used.
        "peak_lr": _env_float("PEAK_LR", 4e-4),
        # Max grad norm for clipping; GRAD_CLIP=0 disables clipping (the pre-clip
        # norm is still logged).  Clipping masks instability, so the sweep needs
        # to be able to turn it off.
        "grad_clip": _env_float("GRAD_CLIP", 0.5),
        # Append per-optimizer-step grad-norm telemetry to <run_dir>/grad_norms.csv.
        "log_grad_norms": _env_flag("LOG_GRAD_NORMS", False),
        # Training-time output noise: y += gamma * RMS(y) * N(0,1) with gamma resampled
        # per formula from U[0, NOISE_GAMMA] (data_process.py:407).  Applied to the RAW y
        # before whitening, so it is the SAME convention the eval tau sweep uses
        # (target_noise * sqrt(mean(y^2))) -- a run at NOISE_GAMMA=0.1 sees the noise the
        # figures measure.  0.0 (default) keeps every existing launch noise-free.
        # Env-exposed rather than edited in place: a literal here is silently reverted by
        # a git pull and would retarget every other run from the same checkout.
        "noise_gamma": _env_float("NOISE_GAMMA", 0.0),
        # Share of formulae held EXACTLY clean while NOISE_GAMMA > 0.  U[0, g] is a
        # continuous draw, so on its own P(gamma == 0) = 0 and a fine-tune would never
        # see a noiseless target again -- the setup for catastrophic forgetting of the
        # clean objective.  0.5 trains on an even mix.  Ignored when NOISE_GAMMA = 0.
        "noise_clean_frac": _env_float("NOISE_CLEAN_FRAC", 0.0),
        "steps_per_epoch": _env_int("STEPS_PER_EPOCH", 3000),
        "target_tokens_per_batch": 20_000,  # PATH TOGGLE only (>0): sort/group batches by similar N to
                                         # cut IO-row padding. Batch SIZE = grad_accum_chunk_size, not this.
        "log_gradients": False,
        "seed": _env_int("SEED", 42),
        "device": "cuda:0",

        # LinearPointEmbedder -- set use_lpe=True to replace the raw-float FNN
        # projection with float-token embeddings.  Requires training from scratch.
        "use_lpe": True,
        "lpe_d_emb": 64,             # embedding dim per float token -- Kamienny et al. 2022 default
        "lpe_n_mlp_layers": 1,      # hidden layers before final fc -- paper default
        "lpe_expansion_factor": 1.0, # hidden = flat_dim * factor -- paper default (flat=240 -> hidden=960)

        # Activation function for encoder/decoder FFN layers: "gelu" or "relu"
        "activation": "gelu",

        # LayerNorm placement in every encoder/decoder layer:
        #   "pre"  -- x + sublayer(norm(x)), plus a final norm on each stack (default)
        #   "post" -- norm(x + sublayer(x)), no final norm (Vaswani et al. 2017)
        # Set NORM_POSITION=post to train the post-LN ablation arm.  Checkpoints are
        # NOT interchangeable between the two (post-LN has no encoder_norm/decoder_norm).
        "norm_position": os.environ.get("NORM_POSITION", "pre"),

        # Path to a BPE model (from subtree_bpe.py) for subformula compression.
        # Leave empty to train on the bare mantissa-free grammar (FlatGram).
        # NOTE: a BPE model must be (re)learned on the current grammar; vocabularies
        # built with the old mantissa grammar are not compatible.
        "bpe_model_path": "",

        # Optional complexity prior for online formula generation.
        #   None (default) -> uniform formula-size sampling (original behaviour).
        #   float in (0, 1] -> bias toward simpler formulae (smaller = stronger bias;
        #                      1.0 == uniform).  Larger formulae remain in the tail.
        # NOTE: with length_decay set (below) this is the VARIABLE-COUNT knob ONLY -- it no
        # longer touches length or the operator budget, which come from length_decay.
        # Without length_decay it is applied to input_dim, nb_binary AND nb_unary separately,
        # so the decay COMPOUNDS and collapses the tail (median len ~9, len>40 ~0.1% even at
        # cap 80) -- which is why length_decay exists.
        # "complexity_decay": None,   # <- baseline: uniform formula-size sampling
        # "complexity_decay": 0.7,    # <- geometric prior on #variables (P(dim=1)~.37, P(dim=10)~.008)
        "complexity_decay": 1.0,      # #variables drawn UNIFORMLY over [1, min(10, n_leaves)],
                                      # matching the e2e generator's flat variable marginal.  Not
                                      # exactly flat: short formulae cannot carry 10 distinct vars
                                      # under the coverage guarantee (only ~64% of draws have
                                      # enough leaves), so low dims keep a mild excess.

        # Target-length-first size sampling.  When set (float in (0,1]) it OVERRIDES
        # complexity_decay: the symbolic length is drawn from a single geometric decay
        # over fml_len_range and the operator budget is derived from it, decoupling
        # length from #variables so long formulae actually reach the cap.
        #   ~0.97 -> simple formulae still dominate but ~20-25% of mass lands past 40;
        #   1.0   -> length uniform over the range.  See data_process.online_batch_iterater.
        "length_decay": 0.99,        # target-length sampling ON (~27% of formulae past len 40)

        # Operator-sampling distribution for online formula generation.
        #   "uniform" (default) -> every operator equally likely (original behaviour).
        #   "tiered"            -> domain-general prior favouring common operators
        #                          (arithmetic > squares/roots > transcendentals > rare);
        #                          every operator stays reachable.  See grammar.py.
        # "operator_weights": "uniform",   # <- baseline: every operator equally likely
        "operator_weights": "tiered",      # first run: domain-general operator prior

        # Target form for online formula generation.
        #   True (default) -> train on the canonical simplest form (original behaviour).
        #   False          -> train on the RAW uniform draw.  Composite constants stay
        #                     canonical; only the formula structure is left unsimplified.
        # Degenerate draws (collapsing to a constant) are rejected either way, so this
        # changes the target form alone -- generation cost is unchanged.  Raw targets run
        # ~1.2 symbolic tokens longer, so more draws hit the fml_len cap.
        # NOTE: a BPE model (bpe_model_path) is learned from the SIMPLIFIED sampler by
        # subtree_bpe.py; relearn it before training with simplify_targets=False.
        "simplify_targets": True,

        # Which canonicaliser produces the target, when simplify_targets is True.
        #   "ours"     (default) -> simplifyFormula.simplify, a constructive normal form.
        #                           Every published run used this.
        #   "simplipy"           -> SimpliPy (Saegert & Koethe), a mined rule set applied
        #                           to a fixpoint.  Requires `pip install simplipy`.
        # This is a TARGET-DISTRIBUTION knob, not an optimisation: the two engines produce
        # near-identical operator distributions (16 of 18 unary tokens within 0.5pp over
        # 20k draws) yet disagree on the canonical form of ~68% of individual formulae, so
        # a model trained under one is not comparable to a model trained under the other.
        # Costs to know before switching: simplipy is ~2.4x slower per formula (250 vs
        # 105 us), and ~5% of draws come back UNCHANGED because it folds them to a pole
        # (inf/nan) the grammar cannot spell -- our simplify leaves those alone too, so
        # the contract matches, but the rate is higher.  Its LOSSY mode also performs
        # sqrt(x^2) -> x, which ours does not; that is sign-correct only because the IO
        # pairs are generated FROM the target.  See simplify_backend.py.
        # NOTE: a BPE model (bpe_model_path) is learned from OUR canonical form by
        # subtree_bpe.py; relearn it before training with simplify_backend="simplipy".
        "simplify_backend": _simplify_backend,

        # ---- Constant / prefactor generation (the two knobs of the prefactor ablation) ----
        # use_prefactors:
        #   False (default) -> bare skeletons.  Any numeric structure in a target comes
        #                      from the grammar's own constant leaves.
        #   True            -> grammar.add_prefactors runs over each RAW skeleton before
        #                      simplification, wrapping every additive term in '* CONST ...'
        #                      and every unary argument in '+ CONST * CONST ...', exactly as
        #                      Kamienny et al. (2022) do.  Prefactors are random
        #                      floats, so this forces the float vocabulary on its own.
        #
        # restrict_consts:
        #   True (default) -> constant leaves come from the symbolic set {0,1,2,3,pi}, with
        #                     other values assembled as composite subexpressions
        #                     ('/ 1 2' = 0.5, '* 2 2' = 4).  Mantissa-free targets.
        #   False          -> constant leaves are drawn from C_ALL = {0,1,2,3,pi,CONST} and
        #                     pow exponents from _POW_EXP_CHOICES_FULL, so a leaf may be an
        #                     arbitrary random float (the E2E constant set).  Composite
        #                     constants are skipped in that mode.
        #
        # Whenever floats are reachable (use_prefactors=True or restrict_consts=False) the
        # vocabulary switches to grammar.FlatGramFloat, each float literal is written as 4
        # tokens, and the target cap grows to 4 * fml_len_range[1] + 2 (6x under
        # use_prefactors).  A BPE model (bpe_model_path) is learned on the mantissa-free
        # sampler and is bypassed under use_prefactors.
        #
        # Under use_prefactors, fml_len_range caps the SKELETON rather than the prefactored
        # result, so both arms below draw from the SAME structural distribution -- capping
        # the output instead would hand the prefactor arm formulae roughly half as long
        # and confound the ablation.  See data_process.online_batch_iterater.
        #
        # Prefactor ablation (isolates prefactors while holding the constant set fixed):
        #   arm 1: "use_prefactors": True,  "restrict_consts": False   # prefactors + full consts
        #   arm 2: "use_prefactors": False, "restrict_consts": False   # no prefactors + full consts
        # Everything else (including num_io_pairs_range) must match between the two arms.
        "use_prefactors": _env_flag("USE_PREFACTORS", False),
        "restrict_consts": _env_flag("RESTRICT_CONSTS", False),

        # Number of DataLoader worker processes for online data generation.
        # 0 = single-process (old behaviour). Keep at 0 on Mac (Python 3.14 spawn
        # overhead exceeds the gain).
        #
        # DEFAULT IS NOW DERIVED FROM THE CORES SLURM ACTUALLY GAVE US, not a
        # constant. The old constant 2 came with a note that "workers=4+ plateau";
        # that measurement was taken under --cpus-per-task=3, so the plateau was the
        # core limit, not the workers. Measured with 3 cores:
        # the H100 sat at ~40% duty cycle (util samples 0 0 97 98 98 96 0 0 0 0,
        # 122W of 700W between bursts) while a pt_data worker was pegged at 106% --
        # i.e. training was CPU-bound on formula generation, not GPU-bound.
        #
        # cpus-1 leaves one core for the main process (H2D copies + optimizer step).
        # SLURM_CPUS_PER_TASK is set by Slurm INSIDE the job, so unlike NUM_DATA_WORKERS
        # it survives an --export=NONE sbatch wrapper.
        # Falls back to the old 2 off-Slurm or when the var is unset/garbage.
        # WARNING: more workers = more RAM. At rest the main process uses ~2.4 GB and
        # each worker ~0.6 GB, so the cap of 23 needs ~16 GB -- fine on a node with a
        # 187.5 G per-GPU share, and still fine in a 32 G request
        # because those clusters only ask for 3 cores and so stay at 2 workers.
        # (Per-worker RSS used to be able to explode to 100s of GB and OOM-kill the
        # job; that was a constant power-tower bignum bomb in simplifyFormula.simplify
        # -- now guarded by _MAX_INT_BITS.)
        "num_data_workers": _env_int("NUM_DATA_WORKERS", _default_data_workers()),
        # Set to False to recycle workers after each epoch (clears accumulated heap
        # fragmentation). Slower to start each epoch but prevents gradual RSS growth.
        "persistent_workers": False,

        # Number of batches in the held-out online validation set (generated once
        # before training and kept fixed). Early stopping monitors this val loss.
        "val_steps": _env_int("VAL_STEPS", 200),
    }

    # Echo the settings that distinguish concurrent runs, so a .out file can always
    # be traced back to the arm that produced it.
    print(f"run_dir={configs['run_dir']}  use_prefactors={configs['use_prefactors']}  "
          f"restrict_consts={configs['restrict_consts']}  "
          f"simplify_targets={configs['simplify_targets']}  "
          f"simplify_backend={configs['simplify_backend']}  "
          f"fml_len_range={configs['fml_len_range']}  "
          f"batch={configs['grad_accum_chunk_size']}x{configs['accumulate_gradients']}"
          f"@{configs['steps_per_epoch']}steps  "
          f"cpus={os.environ.get('SLURM_CPUS_PER_TASK', '?')}/"
          f"workers={configs['num_data_workers']}  "
          f"resume={_resume_from or 'no (from scratch)'}", flush=True)

    # main function for training
    manager = TrainingManager(configs)
    manager.train_and_save()



