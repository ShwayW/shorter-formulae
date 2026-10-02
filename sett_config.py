## This file contains specification of the supet sett's configuration and some helping functions
# imports
import numpy as np
import random
from tfs.sett import SetTransformer

def _lst_rg(a, b):
    return list(range(a, b + 1))

# define the super ViT space
config_space = {
    # inputs embedding
    'num_emb_layers': _lst_rg(1, 2),
    'emb_dims': _lst_rg(16, 256),

    # encoder
    'num_enc_layers': _lst_rg(1, 10),
    'num_enc_heads': _lst_rg(1, 10),
    'in_dim_enc_embed': _lst_rg(64, 128),
    'num_induced_inds': _lst_rg(32, 128),

    # decoder
    'num_dec_layers': _lst_rg(1, 10),
    'num_dec_heads': _lst_rg(1, 10),
    'in_dim_dec_embed': _lst_rg(64, 128),
    'qk_dim_dec_embed': _lst_rg(64, 128),
    'v_dim_dec_embed': _lst_rg(64, 128),
    'dim_dec_mlp': _lst_rg(64, 512)
}

# the GA-found configuration
# this is the config used for 1M transformer training
ga_config = {
    # inputs embedding
    'num_emb_layers': 1,
    'emb_dims': [128],

    # encoder
    'num_enc_layers': 1,
    'num_enc_heads': [4],
    'in_dim_enc_embed': 64,
    'num_induced_inds': [128],

    # decoder
    'num_dec_layers': 5,
    'num_dec_heads': [4] * 5,
    'in_dim_dec_embed': 128,
    'qk_dim_dec_embed': [128] * 5,
    'v_dim_dec_embed': [64] * 5,
    'dim_dec_mlp': [512] * 5
}

# alternative config (unused)
alt1_ga_config = {
    # inputs embedding
    'num_emb_layers': 1,
    'emb_dims': [128],

    # encoder
    'num_enc_layers': 2,
    'num_enc_heads': [16] * 2,
    'in_dim_enc_embed': 128,
    'num_induced_inds': [512] * 2,

    # decoder
    'num_dec_layers': 16,
    'num_dec_heads': [16] * 16,
    'in_dim_dec_embed': 128,
    'qk_dim_dec_embed': [128] * 16,
    'v_dim_dec_embed': [128] * 16,
    'dim_dec_mlp': [512] * 16
}

alt2_ga_config = {
    # inputs embedding
    'num_emb_layers': 1,
    'emb_dims': [16],

    # encoder
    'num_enc_layers': 1,
    'num_enc_heads': [10],
    'in_dim_enc_embed': 64,
    'num_induced_inds': [88],

    # decoder
    'num_dec_layers': 1,
    'num_dec_heads': [7],
    'in_dim_dec_embed': 116,
    'qk_dim_dec_embed': [128],
    'v_dim_dec_embed': [64],
    'dim_dec_mlp': [436]
}

ga_config = {
    # inputs embedding
    'num_emb_layers': 1,
    'emb_dims': [128],

    # encoder
    'num_enc_layers': 2,
    'num_enc_heads': [16] * 2,
    'in_dim_enc_embed': 128,
    'num_induced_inds': [512] * 2,

    # decoder
    'num_dec_layers': 16,
    'num_dec_heads': [16] * 16,
    'in_dim_dec_embed': 128,
    'qk_dim_dec_embed': [128] * 16,
    'v_dim_dec_embed': [128] * 16,
    'dim_dec_mlp': [512] * 16
}

sonata_ga_config = {
    # inputs embedding
    'num_emb_layers': 1,
    'emb_dims': [16],

    # encoder
    'num_enc_layers': 1,
    'num_enc_heads': [10],
    'in_dim_enc_embed': 64,
    'num_induced_inds': [88],

    # decoder
    'num_dec_layers': 1,
    'num_dec_heads': [7],
    'in_dim_dec_embed': 116,
    'qk_dim_dec_embed': [128],
    'v_dim_dec_embed': [64],
    'dim_dec_mlp': [436]
}

def get_supernet_config(config_spc):
    super_config = {}

    # inputs embedding
    super_config['num_emb_layers'] = config_spc['num_emb_layers'][-1]
    super_config['emb_dims'] = [config_spc['emb_dims'][-1]] * super_config['num_emb_layers']

    # encoder
    super_config['num_enc_layers'] = config_spc['num_enc_layers'][-1]
    super_config['num_enc_heads'] = [config_spc['num_enc_heads'][-1]] * super_config['num_enc_layers']
    super_config['in_dim_enc_embed'] = config_spc['in_dim_enc_embed'][-1]
    super_config['num_induced_inds'] = [config_spc['num_induced_inds'][-1]] * super_config['num_enc_layers']
    
    # decocer
    super_config['num_dec_layers'] = config_spc['num_dec_layers'][-1]
    super_config['num_dec_heads'] = [config_spc['num_dec_heads'][-1]] * super_config['num_dec_layers']
    super_config['in_dim_dec_embed'] = config_spc['in_dim_dec_embed'][-1]
    super_config['qk_dim_dec_embed'] = [config_spc['qk_dim_dec_embed'][-1]] * super_config['num_dec_layers']
    super_config['v_dim_dec_embed'] = [config_spc['v_dim_dec_embed'][-1]] * super_config['num_dec_layers']
    super_config['dim_dec_mlp'] = [config_spc['dim_dec_mlp'][-1]] * super_config['num_dec_layers']
    return super_config


def get_infinet_config(config_spc):
    infi_config = {}

    # inputs embedding
    infi_config['num_emb_layers'] = config_spc['num_emb_layers'][0]
    infi_config['emb_dims'] = [config_spc['emb_dims'][0]] * infi_config['num_emb_layers']

    # encoder
    infi_config['num_enc_layers'] = config_spc['num_enc_layers'][0]
    infi_config['num_enc_heads'] = [config_spc['num_enc_heads'][0]] * infi_config['num_enc_layers']
    infi_config['in_dim_enc_embed'] = config_spc['in_dim_enc_embed'][0]
    infi_config['num_induced_inds'] = [config_spc['num_induced_inds'][0]] * infi_config['num_enc_layers']
    
    # decocer
    infi_config['num_dec_layers'] = config_spc['num_dec_layers'][0]
    infi_config['num_dec_heads'] = [config_spc['num_dec_heads'][0]] * infi_config['num_dec_layers']
    infi_config['in_dim_dec_embed'] = config_spc['in_dim_dec_embed'][0]
    infi_config['qk_dim_dec_embed'] = [config_spc['qk_dim_dec_embed'][0]] * infi_config['num_dec_layers']
    infi_config['v_dim_dec_embed'] = [config_spc['v_dim_dec_embed'][0]] * infi_config['num_dec_layers']
    infi_config['dim_dec_mlp'] = [config_spc['dim_dec_mlp'][0]] * infi_config['num_dec_layers']
    return infi_config


def sample_config(config_spc):
    config = {}

    # inputs embedding
    config['num_emb_layers'] = random.choice(config_spc['num_emb_layers'])
    config['emb_dims'] = [random.choice(config_spc['emb_dims']) for _ in range(config['num_emb_layers'])]

    # encoder
    config['num_enc_layers'] = random.choice(config_spc['num_enc_layers'])
    config['num_enc_heads'] = [random.choice(config_spc['num_enc_heads']) for _ in range(config['num_enc_layers'])]
    config['in_dim_enc_embed'] = random.choice(config_spc['in_dim_enc_embed'])
    config['num_induced_inds'] = [random.choice(config_spc['num_induced_inds']) for _ in range(config['num_enc_layers'])]
    
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

    # inputs embedding
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
    new_config['num_induced_inds'] = config_enc['num_induced_inds']

    # decoder
    new_config['num_dec_layers'] = config_dec['num_dec_layers']
    new_config['num_dec_heads'] = config_dec['num_dec_heads']
    new_config['in_dim_dec_embed'] = config_dec['in_dim_dec_embed']
    new_config['qk_dim_dec_embed'] = config_dec['qk_dim_dec_embed']
    new_config['v_dim_dec_embed'] = config_dec['v_dim_dec_embed']
    new_config['dim_dec_mlp'] = config_dec['dim_dec_mlp']
    return new_config


def _sample_val(mean, choices, std):
    return max(min(int(np.random.normal(mean, std)), choices[-1]), choices[0])


def mutate_layers_dim(layer_dims, new_num_layers, choices, std):
    layers_dim = [_sample_val(layer_dims[i], choices, std) for i in range(min(new_num_layers, len(layer_dims)))]
    for i in range(new_num_layers - len(layer_dims)):
        layers_dim.append(_sample_val(layer_dims[-1], choices, std))
    return layers_dim


def mutate_config(config, config_spc, std = 2):
    # init
    new_config = {}

    # inputs embedding
    new_config['num_emb_layers'] = _sample_val(config['num_emb_layers'], config_spc['num_emb_layers'], std)
    new_config['emb_dims'] = mutate_layers_dim(config['emb_dims'], new_config['num_emb_layers'], config_spc['emb_dims'], std)

    # encoder
    new_config['num_enc_layers'] = _sample_val(config['num_enc_layers'], config_spc['num_enc_layers'], std)
    new_config['num_enc_heads'] = mutate_layers_dim(config['num_enc_heads'], new_config['num_enc_layers'], config_spc['num_enc_heads'], std)
    new_config['in_dim_enc_embed'] = _sample_val(config['in_dim_enc_embed'], config_spc['in_dim_enc_embed'], std)
    new_config['num_induced_inds'] = mutate_layers_dim(config['num_induced_inds'], new_config['num_enc_layers'], config_spc['num_induced_inds'], std)

    # decoder
    new_config['num_dec_layers'] = _sample_val(config['num_dec_layers'], config_spc['num_dec_layers'], std)
    new_config['num_dec_heads'] = mutate_layers_dim(config['num_dec_heads'], new_config['num_dec_layers'], config_spc['num_dec_heads'], std)
    new_config['in_dim_dec_embed'] = _sample_val(config['in_dim_dec_embed'], config_spc['in_dim_dec_embed'], std)
    new_config['qk_dim_dec_embed'] = mutate_layers_dim(config['qk_dim_dec_embed'], new_config['num_dec_layers'], config_spc['qk_dim_dec_embed'], std)
    new_config['v_dim_dec_embed'] = mutate_layers_dim(config['v_dim_dec_embed'], new_config['num_dec_layers'], config_spc['v_dim_dec_embed'], std)
    new_config['dim_dec_mlp'] = mutate_layers_dim(config['dim_dec_mlp'], new_config['num_dec_layers'], config_spc['dim_dec_mlp'], std)
    return new_config

def compute_config_size(config, io_dim, full_vocab_size, max_fml_len, pad_index, num_patches):
    # initialize a temporary model
    tmp_model = SetTransformer(config, io_dim, full_vocab_size, max_fml_len, pad_index, num_patches).to("cpu")

    # sum up the number of weights in the model
    model_size = sum(p.numel() for p in tmp_model.parameters())

    # return
    return model_size


if (__name__ == "__main__"):
    super_config = get_supernet_config(config_space)
    print("super config: ", super_config, end = "\n\n")

    infi_config = get_infinet_config(config_space)
    print("infi config: ", infi_config, end = "\n\n")

    config = sample_config(config_space)
    print("random config: ", config, end = "\n\n")

    '''
    config_after = mutate_config(config, config_space)
    print("mutated config: ", config_after, end = "\n\n")

    crossed_config = crossover_config(config, config_after)
    print("crossed overd config: ", crossed_config, end = "\n\n")
    '''

    super_model_size = compute_config_size(super_config, io_dim = 32 * 8, full_vocab_size = 45, max_fml_len = 31, pad_index = 44, num_patches = 32)
    print("super model size: ", super_model_size, super_model_size)

    infi_model_size = compute_config_size(infi_config, io_dim = 32 * 8, full_vocab_size = 45, max_fml_len = 31, pad_index = 44, num_patches = 32)
    print("infi model size: ", infi_model_size, infi_model_size)

    model_size = compute_config_size(config, io_dim = 32 * 8, full_vocab_size = 45, max_fml_len = 31, pad_index = 44, num_patches = 32)
    print("model size: ", model_size, model_size)





