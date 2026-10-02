# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# linear operation super
class MultiheadProj(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, num_heads_super, use_bias = True, scale = True):
        # initialized with weights and biases
        super().__init__(super_in_dim, num_heads_super * super_out_dim, bias = use_bias)
        
        # notice that super_in_dim and super_out_dim are the largest values possible
        self.super_in_dim = super_in_dim
        self.num_heads_super = num_heads_super
        self.super_out_dim = super_out_dim

        # the current configuration
        self.sample_in_dim = None
        self.num_heads_sample = None
        self.sample_out_dim = None
        self.scale = scale

        # initialize the parameters
        self.register_buffer('_row_idx', None, persistent = False) # will hold 1D LongTensor or None
        self.use_bias = use_bias
        nn.init.kaiming_uniform_(self.weight)
        if (use_bias):
            nn.init.constant_(self.bias, 0.)


    def set_sample_config(self, sample_in_dim, sample_out_dim, num_heads_sample):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self.num_heads_sample = num_heads_sample
        self.sample_scale = float(self.super_out_dim) / float(self.sample_out_dim)

        # compute row indices (rows selected from weight and bias)
        device = self.weight.device
        head_offsets = torch.arange(self.num_heads_sample, device = device) * self.super_out_dim
        row_offsets = torch.arange(self.sample_out_dim, device = device)
        idx_flat = (head_offsets.unsqueeze(1) + row_offsets.unsqueeze(0)).reshape(-1)

        # store indices as buffer so they move with device/cuda
        self._row_idx = idx_flat.to(device)


    def forward(self, x):
        if self._row_idx is None:
            raise RuntimeError("Call set_sample_config(...) first")

        # always select current parameter values (no stale copies)
        base_weight = self.weight[:, :self.sample_in_dim] # (num_heads_super * super_out_dim, sample_in_dim)
        w = base_weight.index_select(0, self._row_idx) # (num_heads_sample * sample_out_dim, sample_in_dim)
        b = None
        if self.use_bias:
            b = self.bias.index_select(0, self._row_idx)

        out = F.linear(x, w, b)
        if self.scale:
            out = out * self.sample_scale
        return out


    def get_dims_str(self, indent = 0):
        # prints out dimensions of all components, for debug
        indent_str = "\t" * indent
        return f"{indent_str}[W: {list(self.samples['w'].shape)}, b: {list(self.samples['b'].shape)}]"



class MultiheadOut(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, num_heads_super, use_bias = True, scale = True):
        # initialized with weights and biases
        super().__init__(num_heads_super * super_in_dim, super_out_dim, bias = use_bias)
        
        # notice that super_in_dim and super_out_dim are the largest values possible
        self.super_in_dim = super_in_dim
        self.num_heads_super = num_heads_super
        self.super_out_dim = super_out_dim

        # the current configuration
        self.sample_in_dim = None
        self.num_heads_sample = None
        self.sample_out_dim = None
        self.scale = scale

        # initialize the parameters
        self.register_buffer('_col_idx', None, persistent = False) # not used here, but left for symmetry
        self.register_buffer('_row_idx', None, persistent = False) # not used here, but left for symmetry
        self.use_bias = use_bias
        nn.init.kaiming_uniform_(self.weight)
        if (use_bias):
            nn.init.constant_(self.bias, 0.)


    def set_sample_config(self, sample_in_dim, sample_out_dim, num_heads_sample):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self.num_heads_sample = num_heads_sample
        self.sample_scale = float(self.super_in_dim) / float(self.sample_in_dim)

        device = self.weight.device
        head_offsets = torch.arange(self.num_heads_sample, device = device) * self.super_in_dim
        col_offsets = torch.arange(self.sample_in_dim, device = device)
        cols = (head_offsets.unsqueeze(1) + col_offsets.unsqueeze(0)).reshape(-1)
        self._col_idx = cols.to(device)

        # rows for bias selection (first sample_out_dim rows)
        self._row_idx = torch.arange(self.sample_out_dim, device = device)


    def forward(self, x):
        if self._col_idx is None or self._row_idx is None:
            raise RuntimeError("Call set_sample_config(...) first")

        # select the correct output rows and input columns from current weight
        base_weight = self.weight[: self.sample_out_dim, :] # (sample_out_dim, num_heads_super * super_in_dim)
        w = base_weight.index_select(1, self._col_idx) # (sample_out_dim, num_heads_sample * sample_in_dim)

        b = None
        if self.use_bias:
            b = self.bias.index_select(0, self._row_idx) # (sample_out_dim,)

        out = F.linear(x, w, b)
        if self.scale:
            out = out * self.sample_scale
        return out


    def get_dims_str(self, indent = 0):
        # prints out dimensions of all components, for debug
        indent_str = "\t" * indent
        return f"{indent_str}[W: {list(self.samples['w'].shape)}, b: {list(self.samples['b'].shape)}]"


# class definition for the multihead cross attention super
class MultiHeadAttention(nn.Module):
    def __init__(self, super_x_dim, super_z_dim, super_attn_dim, super_mid_dim, super_num_heads, super_out_dim, use_bias = True):
        # if the primary sequence is the same as the context sequence, this is the multihead self attention
        # if they are difference, this is the multihead cross attention
        super(MultiHeadAttention, self).__init__()

        # the largest possible model
        self.super_x_dim = super_x_dim # primary sequence feature dimension
        self.super_z_dim = super_z_dim # context sequence feature dimension
        self.super_attn_dim = super_attn_dim # attention embedding dimension
        self.super_mid_dim = super_mid_dim # value embedding dimension
        self.super_num_heads = super_num_heads # Number of attention heads
        self.use_bias = use_bias
        self.super_out_dim = super_out_dim

        # Linear layers for transforming inputs
        self.W_q = MultiheadProj(super_x_dim, super_attn_dim, super_num_heads, use_bias) # Query weights
        self.W_k = MultiheadProj(super_z_dim, super_attn_dim, super_num_heads, use_bias) # Key and Value weights
        self.W_v = MultiheadProj(super_z_dim, super_mid_dim, super_num_heads, use_bias) # Key and Value weights

        # Output weights: the output dimension is the same as primary sequence dimension to enable multiple layers
        self.W_o = MultiheadOut(super_mid_dim, super_out_dim, super_num_heads, use_bias) 


        # the current configuration
        self.sample_x_dim = None
        self.sample_z_dim = None
        self.sample_attn_dim = None
        self.sample_mid_dim = None
        self.sample_num_heads = None
        self.sample_out_dim = None

        # initialize the parameters
        self.samples = {}


    def set_sample_config(self, sample_x_dim, sample_z_dim, sample_attn_dim, sample_mid_dim, sample_num_heads, sample_out_dim):
        # Dimension of each head's key, query, and value
        self.sample_x_dim = sample_x_dim
        self.sample_z_dim = sample_z_dim
        self.sample_num_heads = sample_num_heads
        self.sample_attn_dim = sample_attn_dim
        self.sample_mid_dim = sample_mid_dim
        self.sample_out_dim = sample_out_dim

        # set configurations for the Q, K, V and O matrices
        self.W_q.set_sample_config(sample_x_dim, sample_attn_dim, sample_num_heads)
        self.W_k.set_sample_config(sample_z_dim, sample_attn_dim, sample_num_heads)
        self.W_v.set_sample_config(sample_z_dim, sample_mid_dim, sample_num_heads)
        self.W_o.set_sample_config(sample_mid_dim, sample_out_dim, sample_num_heads)
        # if (self.attn_output_gate):
        #     self.W_g.set_sample_config(sample_x_dim, sample_mid_dim, sample_num_heads)


    def sdpa(self, Q, K, V, is_causal=False, padding_mask=None):
        """Scaled dot-product attention via torch.nn.functional.scaled_dot_product_attention.

        Uses Flash Attention on CUDA when available.  Falls back to the math
        kernel otherwise.  Handles causal masking and key-padding masks.
        """
        # Q shape: (B, H, S_q, d_k)
        # K, V shape: (B, H, S_k, d_k)
        if padding_mask is None:
            # Fast path: let PyTorch pick Flash Attention or math kernel.
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=is_causal)

        # Build a float additive mask from the bool padding mask so we can
        # combine it with an optional causal mask and pass it to SDPA.
        # padding_mask: (B, S_k) -- True = padded position.
        B, H, S_q, _ = Q.shape
        S_k = K.shape[-2]
        attn_mask = Q.new_zeros(B, 1, S_q, S_k)
        attn_mask.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        if is_causal:
            causal = torch.tril(torch.ones(S_q, S_k, dtype=torch.bool, device=Q.device))
            attn_mask = attn_mask.masked_fill(causal.logical_not(), float("-inf"))

        # nan_to_num guard (all-masked row -> uniform zero output) is provided
        # automatically by SDPA's softmax implementation.
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, is_causal=False)


    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, embed_dim = x.size()
        return x.view(batch_size, seq_length, self.sample_num_heads, int(embed_dim / self.sample_num_heads)).transpose(1, 2).contiguous()
        

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, num_heads, seq_length, feature_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, num_heads * feature_dim)
        

    def forward(self, X, Z, is_causal = False, padding_mask = None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(Z))
        V = self.split_heads(self.W_v(Z))

        # Perform scaled dot-product attention
        attn_output = self.sdpa(Q, K, V, is_causal = is_causal, padding_mask = padding_mask)

        # Attention Gate
        # Combine heads and apply output transformation
        combined_attn = self.combine_heads(attn_output)
        output = self.W_o(combined_attn)
        return output


    def precompute_kv(self, Z):
        """Compute and return (K, V) from Z for use as a static cross-attention cache."""
        K = self.split_heads(self.W_k(Z))   # (B, H, S, d_k)
        V = self.split_heads(self.W_v(Z))   # (B, H, S, d_v)
        return K, V

    def forward_step(self, X_new, K_cache, V_cache, padding_mask=None):
        """Incremental self-attention step: append new K/V and attend over full history.

        X_new:   (B, 1, d_x)
        K_cache: (B, H, T, d_k)  -- past keys (may be empty, T=0)
        V_cache: (B, H, T, d_v)  -- past values
        Returns: output (B, 1, d_out), K_new (B, H, T+1, d_k), V_new (B, H, T+1, d_v)
        """
        Q     = self.split_heads(self.W_q(X_new))   # (B, H, 1, d_k)
        K_new = self.split_heads(self.W_k(X_new))   # (B, H, 1, d_k)
        V_new = self.split_heads(self.W_v(X_new))   # (B, H, 1, d_v)
        K_full = torch.cat([K_cache, K_new], dim=2) # (B, H, T+1, d_k)
        V_full = torch.cat([V_cache, V_new], dim=2) # (B, H, T+1, d_v)
        # Q is the newest token -> it can attend to all T+1 positions; no causal mask needed.
        attn_out = F.scaled_dot_product_attention(Q, K_full, V_full, is_causal=False)
        return self.W_o(self.combine_heads(attn_out)), K_full, V_full

    def forward_step_cross(self, X_new, K_cross, V_cross, padding_mask=None):
        """Cross-attention step using a precomputed (K_cross, V_cross) from encoder output.

        X_new:   (B, 1, d_x)
        K_cross: (B, H, S_enc, d_k)  -- precomputed once from encoder output
        V_cross: (B, H, S_enc, d_v)
        Returns: output (B, 1, d_out)
        """
        Q = self.split_heads(self.W_q(X_new))  # (B, H, 1, d_k)
        # Delegate to sdpa so padding (IO mask) is handled identically to the full-sequence path.
        attn_out = self.sdpa(Q, K_cross, V_cross, is_causal=False, padding_mask=padding_mask)
        return self.W_o(self.combine_heads(attn_out))

    def get_dims_str(self, indent = 0):
        # get the dimension describing string
        indent_str = "\t" * indent
        dims_str = f"{indent_str}mha:\n"
        dims_str += f"{indent_str}\tW_q: {self.W_q.get_dims_str()}\n"
        dims_str += f"{indent_str}\tW_k: {self.W_k.get_dims_str()}\n"
        dims_str += f"{indent_str}\tW_v: {self.W_v.get_dims_str()}\n"
        dims_str += f"{indent_str}\tW_o: {self.W_o.get_dims_str()}\n"
        return dims_str



