## This file contains specification of the super transformer's configuration and some helping functions
# imports
import numpy as np
import random
from tfs.van import VanillaTransformer

def _lst_rg(a, b):
    return list(range(a, b + 1))

# define the super ViT space
# Upper bounds match ga_config so that get_supernet_config returns the ga_config architecture.
# num_emb_layers / emb_dims removed: the LPE path (use_lpe=True) is the only supported mode.
config_space = {
    # encoder
    # num_layers / num_heads: small integer counts, uniform sampling is fine.
    'num_enc_layers': _lst_rg(1, 4),               # 4 choices
    'num_enc_heads':  _lst_rg(1, 16),              # 16 choices
    # Embed / FFN dims: discretised to multiples of their natural granularity
    # so each choice is seen with ~10% probability instead of <0.2%.
    'in_dim_enc_embed': [64, 128, 192, 256, 320, 384, 448, 512],   # 8 choices (*64)
    'qk_dim_enc_embed': [8, 16, 24, 32],                           # 4 choices (*8)
    'v_dim_enc_embed':  [8, 16, 24, 32],                           # 4 choices (*8)
    'dim_enc_mlp':      [256, 512, 768, 1024, 1280, 1536, 1792, 2048],  # 8 choices (*256)

    # decoder
    'num_dec_layers': _lst_rg(1, 16),              # 16 choices
    'num_dec_heads':  _lst_rg(1, 16),              # 16 choices
    'in_dim_dec_embed': [64, 128, 192, 256, 320, 384, 448, 512],   # 8 choices (*64)
    'qk_dim_dec_embed': [8, 16, 24, 32],                           # 4 choices (*8)
    'v_dim_dec_embed':  [8, 16, 24, 32],                           # 4 choices (*8)
    'dim_dec_mlp':      [256, 512, 768, 1024, 1280, 1536, 1792, 2048],  # 8 choices (*256)
}

# the manually defined configuration
ga_config = {
    # encoder
    'num_enc_layers': 12,
    'num_enc_heads': [20] * 12,
    'in_dim_enc_embed': 512,
    'qk_dim_enc_embed': [32] * 12,
    'v_dim_enc_embed': [32] * 12,
    'dim_enc_mlp': [2048] * 12,

    # decoder
    'num_dec_layers': 20,
    'num_dec_heads': [20] * 20,
    'in_dim_dec_embed': 512,
    'qk_dim_dec_embed': [32] * 20,
    'v_dim_dec_embed': [32] * 20,
    'dim_dec_mlp': [2048] * 20,

    # LinearPointEmbedder (ignored when use_lpe=False)
    # Defaults match Kamienny et al. 2022 (Table 7): d_emb=8, expansion=4, n_layers=1.
    # mantissa_len=1: 3 tokens/scalar, flat_dim=10*3*8=240; hidden=240*4=960 → d_enc=512
    # mantissa_len=2: 4 tokens/scalar, flat_dim=10*4*8=320; hidden=320*4=1280 → d_enc=512
    'use_lpe': True,
    'lpe_d_emb': 64,
    'lpe_mantissa_len': 2,
    'lpe_n_mlp_layers': 1,
    'lpe_expansion_factor': 1.0,
}

'''
ga_config = {
    # encoder
    'num_enc_layers': 4,
    'num_enc_heads': [16] * 4,
    'in_dim_enc_embed': 512,
    'qk_dim_enc_embed': [32] * 4,
    'v_dim_enc_embed': [32] * 4,
    'dim_enc_mlp': [2048] * 4,

    # decoder
    'num_dec_layers': 16,
    'num_dec_heads': [16] * 16,
    'in_dim_dec_embed': 512,
    'qk_dim_dec_embed': [32] * 16,
    'v_dim_dec_embed': [32] * 16,
    'dim_dec_mlp': [2048] * 16,

    # LinearPointEmbedder (ignored when use_lpe=False)
    # Defaults match Kamienny et al. 2022 (Table 7): d_emb=8, expansion=4, n_layers=1.
    # mantissa_len=1: 3 tokens/scalar, flat_dim=10*3*8=240; hidden=240*4=960 -> d_enc=512
    # mantissa_len=2: 4 tokens/scalar, flat_dim=10*4*8=320; hidden=320*4=1280 -> d_enc=512
    'use_lpe': True,
    'lpe_d_emb': 64,
    'lpe_mantissa_len': 2,
    'lpe_n_mlp_layers': 1,
    'lpe_expansion_factor': 1.0,
}
'''

def get_supernet_config(config_spc, use_lpe=False, lpe_params=None):
    super_config = {}
    super_config['use_lpe'] = use_lpe

    if use_lpe:
        if lpe_params is not None:
            super_config.update(lpe_params)
        else:
            super_config['lpe_d_emb'] = 8
            super_config['lpe_mantissa_len'] = 2
            super_config['lpe_n_mlp_layers'] = 1
            super_config['lpe_expansion_factor'] = 4.0

    # inputs embedding (FNN path only -- LPE replaces this stack entirely)
    if not use_lpe:
        super_config['num_emb_layers'] = config_spc['num_emb_layers'][-1]
        super_config['emb_dims'] = [config_spc['emb_dims'][-1]] * super_config['num_emb_layers']

    # encoder
    super_config['num_enc_layers'] = config_spc['num_enc_layers'][-1]
    super_config['num_enc_heads'] = [config_spc['num_enc_heads'][-1]] * super_config['num_enc_layers']
    super_config['in_dim_enc_embed'] = config_spc['in_dim_enc_embed'][-1]
    super_config['qk_dim_enc_embed'] = [config_spc['qk_dim_enc_embed'][-1]] * super_config['num_enc_layers']
    super_config['v_dim_enc_embed'] = [config_spc['v_dim_enc_embed'][-1]] * super_config['num_enc_layers']
    super_config['dim_enc_mlp'] = [config_spc['dim_enc_mlp'][-1]] * super_config['num_enc_layers']

    # decoder
    super_config['num_dec_layers'] = config_spc['num_dec_layers'][-1]
    super_config['num_dec_heads'] = [config_spc['num_dec_heads'][-1]] * super_config['num_dec_layers']
    super_config['in_dim_dec_embed'] = config_spc['in_dim_dec_embed'][-1]
    super_config['qk_dim_dec_embed'] = [config_spc['qk_dim_dec_embed'][-1]] * super_config['num_dec_layers']
    super_config['v_dim_dec_embed'] = [config_spc['v_dim_dec_embed'][-1]] * super_config['num_dec_layers']
    super_config['dim_dec_mlp'] = [config_spc['dim_dec_mlp'][-1]] * super_config['num_dec_layers']
    return super_config


def get_infinet_config(config_spc, use_lpe=False, lpe_params=None):
    infi_config = {}
    infi_config['use_lpe'] = use_lpe

    if use_lpe:
        if lpe_params is not None:
            infi_config.update(lpe_params)
        else:
            infi_config['lpe_d_emb'] = 8
            infi_config['lpe_mantissa_len'] = 2
            infi_config['lpe_n_mlp_layers'] = 1
            infi_config['lpe_expansion_factor'] = 4.0

    # encoder
    infi_config['num_enc_layers'] = config_spc['num_enc_layers'][0]
    infi_config['num_enc_heads'] = [config_spc['num_enc_heads'][0]] * infi_config['num_enc_layers']
    infi_config['in_dim_enc_embed'] = config_spc['in_dim_enc_embed'][0]
    infi_config['qk_dim_enc_embed'] = [config_spc['qk_dim_enc_embed'][0]] * infi_config['num_enc_layers']
    infi_config['v_dim_enc_embed'] = [config_spc['v_dim_enc_embed'][0]] * infi_config['num_enc_layers']
    infi_config['dim_enc_mlp'] = [config_spc['dim_enc_mlp'][0]] * infi_config['num_enc_layers']

    # decoder
    infi_config['num_dec_layers'] = config_spc['num_dec_layers'][0]
    infi_config['num_dec_heads'] = [config_spc['num_dec_heads'][0]] * infi_config['num_dec_layers']
    infi_config['in_dim_dec_embed'] = config_spc['in_dim_dec_embed'][0]
    infi_config['qk_dim_dec_embed'] = [config_spc['qk_dim_dec_embed'][0]] * infi_config['num_dec_layers']
    infi_config['v_dim_dec_embed'] = [config_spc['v_dim_dec_embed'][0]] * infi_config['num_dec_layers']
    infi_config['dim_dec_mlp'] = [config_spc['dim_dec_mlp'][0]] * infi_config['num_dec_layers']
    return infi_config


def sample_config(config_spc, use_lpe=False, lpe_params=None):
    """Sample a random sub-network config from config_spc.

    lpe_params : dict of fixed LPE hyperparams to embed verbatim when use_lpe=True.
                 If None and use_lpe=True, sensible defaults matching Kamienny et al. 2022
                 are used.  These are NOT searched by NAS -- they are inherited from the
                 checkpoint that the supernet was trained with.
    """
    config = {}
    config['use_lpe'] = use_lpe

    if use_lpe:
        if lpe_params is not None:
            config.update(lpe_params)
        else:
            # defaults matching the ga_config / train_transformer.py defaults
            config['lpe_d_emb'] = 8
            config['lpe_mantissa_len'] = 2
            config['lpe_n_mlp_layers'] = 1
            config['lpe_expansion_factor'] = 4.0

    # inputs embedding (FNN path only -- LPE replaces this stack entirely)
    if not use_lpe:
        config['num_emb_layers'] = random.choice(config_spc['num_emb_layers'])
        config['emb_dims'] = [random.choice(config_spc['emb_dims']) for _ in range(config['num_emb_layers'])]

    # encoder
    config['num_enc_layers'] = random.choice(config_spc['num_enc_layers'])
    config['num_enc_heads'] = [random.choice(config_spc['num_enc_heads']) for _ in range(config['num_enc_layers'])]
    config['in_dim_enc_embed'] = random.choice(config_spc['in_dim_enc_embed'])
    config['qk_dim_enc_embed'] = [random.choice(config_spc['qk_dim_enc_embed']) for _ in range(config['num_enc_layers'])]
    config['v_dim_enc_embed'] = [random.choice(config_spc['v_dim_enc_embed']) for _ in range(config['num_enc_layers'])]
    config['dim_enc_mlp'] = [random.choice(config_spc['dim_enc_mlp']) for _ in range(config['num_enc_layers'])]

    # decoder
    config['num_dec_layers'] = random.choice(config_spc['num_dec_layers'])
    config['num_dec_heads'] = [random.choice(config_spc['num_dec_heads']) for _ in range(config['num_dec_layers'])]
    config['in_dim_dec_embed'] = random.choice(config_spc['in_dim_dec_embed'])
    config['qk_dim_dec_embed'] = [random.choice(config_spc['qk_dim_dec_embed']) for _ in range(config['num_dec_layers'])]
    config['v_dim_dec_embed'] = [random.choice(config_spc['v_dim_dec_embed']) for _ in range(config['num_dec_layers'])]
    config['dim_dec_mlp'] = [random.choice(config_spc['dim_dec_mlp']) for _ in range(config['num_dec_layers'])]
    return config


def crossover_config(config_enc, config_dec):
    # init
    new_config = {}
    use_lpe = config_enc.get('use_lpe', False)
    new_config['use_lpe'] = use_lpe

    # LPE params are fixed (not searched), so just inherit from either parent
    if use_lpe:
        for k in ('lpe_d_emb', 'lpe_mantissa_len', 'lpe_n_mlp_layers', 'lpe_expansion_factor'):
            if k in config_enc:
                new_config[k] = config_enc[k]

    # inputs embedding
    if not use_lpe:
        if (random.random() < 0.5):
            new_config['num_emb_layers'] = config_enc['num_emb_layers']
            new_config['emb_dims'] = config_enc['emb_dims']
        else:
            new_config['num_emb_layers'] = config_dec['num_emb_layers']
            new_config['emb_dims'] = config_dec['emb_dims']

    # encoder
    new_config['num_enc_layers'] = config_enc['num_enc_layers']
    new_config['num_enc_heads'] = config_enc['num_enc_heads']
    new_config['in_dim_enc_embed'] = config_enc['in_dim_enc_embed']
    new_config['qk_dim_enc_embed'] = config_enc['qk_dim_enc_embed']
    new_config['v_dim_enc_embed'] = config_enc['v_dim_enc_embed']
    new_config['dim_enc_mlp'] = config_enc['dim_enc_mlp']

    # decoder
    new_config['num_dec_layers'] = config_dec['num_dec_layers']
    new_config['num_dec_heads'] = config_dec['num_dec_heads']
    new_config['in_dim_dec_embed'] = config_dec['in_dim_dec_embed']
    new_config['qk_dim_dec_embed'] = config_dec['qk_dim_dec_embed']
    new_config['v_dim_dec_embed'] = config_dec['v_dim_dec_embed']
    new_config['dim_dec_mlp'] = config_dec['dim_dec_mlp']
    return new_config


def _sample_val(mean, choices, std):
    val = int(np.random.normal(mean, std))
    return min(choices, key=lambda c: abs(c - val))


def mutate_layers_dim(layer_dims, new_num_layers, choices, std):
    layers_dim = [_sample_val(layer_dims[i], choices, std) for i in range(min(new_num_layers, len(layer_dims)))]
    for i in range(new_num_layers - len(layer_dims)):
        layers_dim.append(_sample_val(layer_dims[-1], choices, std))
    return layers_dim


def mutate_config(config, config_spc, std = 2):
    # init
    new_config = {}
    use_lpe = config.get('use_lpe', False)
    new_config['use_lpe'] = use_lpe

    # LPE params are fixed (not searched); carry them forward unchanged
    if use_lpe:
        for k in ('lpe_d_emb', 'lpe_mantissa_len', 'lpe_n_mlp_layers', 'lpe_expansion_factor'):
            if k in config:
                new_config[k] = config[k]

    # inputs embedding
    if not use_lpe:
        new_config['num_emb_layers'] = _sample_val(config['num_emb_layers'], config_spc['num_emb_layers'], std)
        new_config['emb_dims'] = mutate_layers_dim(config['emb_dims'], new_config['num_emb_layers'], config_spc['emb_dims'], std)

    # encoder
    new_config['num_enc_layers'] = _sample_val(config['num_enc_layers'], config_spc['num_enc_layers'], std)
    new_config['num_enc_heads'] = mutate_layers_dim(config['num_enc_heads'], new_config['num_enc_layers'], config_spc['num_enc_heads'], std)
    new_config['in_dim_enc_embed'] = _sample_val(config['in_dim_enc_embed'], config_spc['in_dim_enc_embed'], std)
    new_config['qk_dim_enc_embed'] = mutate_layers_dim(config['qk_dim_enc_embed'], new_config['num_enc_layers'], config_spc['qk_dim_enc_embed'], std)
    new_config['v_dim_enc_embed'] = mutate_layers_dim(config['v_dim_enc_embed'], new_config['num_enc_layers'], config_spc['v_dim_enc_embed'], std)
    new_config['dim_enc_mlp'] = mutate_layers_dim(config['dim_enc_mlp'], new_config['num_enc_layers'], config_spc['dim_enc_mlp'], std)

    # decoder
    new_config['num_dec_layers'] = _sample_val(config['num_dec_layers'], config_spc['num_dec_layers'], std)
    new_config['num_dec_heads'] = mutate_layers_dim(config['num_dec_heads'], new_config['num_dec_layers'], config_spc['num_dec_heads'], std)
    new_config['in_dim_dec_embed'] = _sample_val(config['in_dim_dec_embed'], config_spc['in_dim_dec_embed'], std)
    new_config['qk_dim_dec_embed'] = mutate_layers_dim(config['qk_dim_dec_embed'], new_config['num_dec_layers'], config_spc['qk_dim_dec_embed'], std)
    new_config['v_dim_dec_embed'] = mutate_layers_dim(config['v_dim_dec_embed'], new_config['num_dec_layers'], config_spc['v_dim_dec_embed'], std)
    new_config['dim_dec_mlp'] = mutate_layers_dim(config['dim_dec_mlp'], new_config['num_dec_layers'], config_spc['dim_dec_mlp'], std)
    return new_config

def compute_config_size(config, io_dim, full_vocab_size, max_fml_len, pad_index):
    # initialize a temporary model
    tmp_model = VanillaTransformer(config, io_dim, full_vocab_size, max_fml_len, pad_index).to("cpu")

    # sum up the number of weights in the model
    model_size = sum(p.numel() for p in tmp_model.parameters())

    # return
    return model_size


if (__name__ == "__main__"):
    config = sample_config(config_space, use_lpe=True)
    print("original config: ", config, end = "\n\n")

    config_after = mutate_config(config)
    print("mutated config: ", config_after, end = "\n\n")

    crossed_config = crossover_config(config, config_after)
    print("crossed overd config: ", crossed_config, end = "\n\n")



