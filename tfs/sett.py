## Vision Transformer supernet
# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core.linear import Linear
from .core.set_encoderlayer import SetEncoderLayer
from .core.decoderlayer import DecoderLayer
from .core.embedding import Embedding
from .core.positional import PositionalEncoding
from .core.pooling_multihead_attn import PoolingMultiheadAttention

class SetTransformer(nn.Module):
    def __init__(self, config, patch_dim, vocab_size, max_seq_length, pad_idx, dropout = 0.00):
        super(SetTransformer, self).__init__()
        # stats
        self.pad_idx = pad_idx
        self.patch_dim = patch_dim
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.dropout = dropout

        ## The largest model possible
        # fnns
        self.super_num_emb_layers = 1 + config['num_emb_layers']
        self.super_emb_dims = [self.patch_dim] + config['emb_dims']

        # encoder part hyperparameters
        self.activation = config.get('activation', 'gelu')

        self.super_num_enc_layers = config['num_enc_layers'] # the L_enc in the paper
        self.super_num_enc_heads = config['num_enc_heads'] # number of head in the encoder's mhatn layer, the H in the paper
        self.super_in_dim_enc_embed = config['in_dim_enc_embed'] # d_e in the paper
        self.num_induced_inds_super = config['num_induced_inds']

        # decoder part hyperparameters
        self.super_num_dec_layers = config['num_dec_layers'] # the L_dec in the paper
        self.super_num_dec_heads = config['num_dec_heads'] # number of head in the decoder's mhatn layer, the H in the paper
        self.super_in_dim_dec_embed = config['in_dim_dec_embed'] # d_e in the paper
        self.super_qk_dim_dec_embed = config['qk_dim_dec_embed'] # d_qk in the paper
        self.super_v_dim_dec_embed = config['v_dim_dec_embed'] # d_v in the paper
        self.super_dim_dec_mlp = config['dim_dec_mlp'] # d_mlp in the paper

        # decoder dim sanity checks
        #assert(self.super_num_dec_layers == len(self.super_num_dec_heads) == len(self.super_in_dim_dec_embed) == len(self.super_qk_dim_dec_embed) == len(self.super_v_dim_dec_embed) == len(self.super_dim_dec_mlp))
        # fnns
        self.sample_num_emb_layers = None
        self.sample_emb_dims = None

        ## The current sampled model dimensions
        # encoder part hyperparameters
        self.sample_num_enc_layers = None
        self.sample_num_enc_heads = None
        self.sample_in_dim_enc_embed = None
        self.sample_qk_dim_enc_embed = None
        self.sample_v_dim_enc_embed = None
        self.sample_dim_enc_mlp = None

        # decoder part hyperparameters
        self.sample_num_dec_layers = None
        self.sample_num_dec_heads = None
        self.sample_in_dim_dec_embed = None
        self.sample_qk_dim_dec_embed = None
        self.sample_v_dim_dec_embed = None
        self.sample_dim_dec_mlp = None

        ## initialize the various ViT components
        # encoder embeddings
        self.emb_layers = nn.ModuleList([Linear(self.super_emb_dims[layerI],
                                                     self.super_emb_dims[layerI + 1]) for layerI in range(self.super_num_emb_layers - 1)])

        # encoder embeddings
        self.enc_mid_fc = Linear(self.super_emb_dims[-1], self.super_in_dim_enc_embed)

        # encder layers
        self.encoder_layers = nn.ModuleList([SetEncoderLayer(self.super_in_dim_enc_embed,
                                                               self.super_num_enc_heads[layerI],
                                                               self.num_induced_inds_super[layerI],
                                                               self.dropout) for layerI in range(self.super_num_enc_layers)])

        # intermediate layer
        self.interm = PoolingMultiheadAttention(self.super_in_dim_enc_embed, self.super_num_enc_heads[-1], self.patch_dim)

        # decoding embeddings
        self.decoder_embedding = Embedding(self.vocab_size, self.super_in_dim_dec_embed, padding_idx = self.pad_idx)
        self.dec_positional_encoding = PositionalEncoding(self.super_in_dim_dec_embed, self.max_seq_length)

        # decoder layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.super_in_dim_dec_embed,
                                                               self.super_in_dim_enc_embed,
                                                               self.super_qk_dim_dec_embed[layerI],
                                                               self.super_v_dim_dec_embed[layerI],
                                                               self.super_num_dec_heads[layerI],
                                                               self.super_dim_dec_mlp[layerI],
                                                               self.dropout,
                                                               activation=self.activation) for layerI in range(self.super_num_dec_layers)])

        # final layer
        self.fc = Linear(self.super_in_dim_dec_embed, self.vocab_size)


    def set_sample_config(self, config):
        ## FNNs
        self.sample_num_emb_layers = 1 + config['num_emb_layers']
        self.sample_emb_dims = [self.patch_dim] + config['emb_dims']

        ## Encoder configuration settings
        self.sample_num_enc_layers = config['num_enc_layers']
        self.sample_num_enc_heads = config['num_enc_heads']
        self.sample_in_dim_enc_embed = config['in_dim_enc_embed']
        self.num_induced_inds_sample = config['num_induced_inds']

        # set the embeder configurations
        # set the encoderlayers configurations
        for layI, layer in enumerate(self.emb_layers):
            if (layI < self.sample_num_emb_layers - 1):
                layer.set_sample_config(sample_in_dim = self.sample_emb_dims[layI],
                                        sample_out_dim = self.sample_emb_dims[layI + 1],
                                        is_identity_layer = False,)
            else:
                layer.set_sample_config(is_identity_layer = True)

        self.enc_mid_fc.set_sample_config(sample_in_dim = self.sample_emb_dims[self.sample_num_emb_layers - 1],
                                        sample_out_dim = self.sample_in_dim_enc_embed,
                                        is_identity_layer = False)
        self.enc_sample_dropout = self.compute_dropout(self.sample_in_dim_enc_embed, self.super_in_dim_enc_embed)

        # set the encoderlayers configurations
        for enclayI, layer in enumerate(self.encoder_layers):
            if (enclayI < self.sample_num_enc_layers):
                layer.set_sample_config(is_identity_layer = False,
                                        sample_in_embed_dim = self.sample_in_dim_enc_embed,
                                        sample_num_heads = self.sample_num_enc_heads[enclayI],
                                        num_induced_inds_sample = self.num_induced_inds_sample[enclayI],
                                        sample_dropout = self.compute_dropout(self.sample_in_dim_enc_embed, self.super_in_dim_enc_embed))
            else:
                layer.set_sample_config(is_identity_layer = True)

        # set the sample config for the intermediate layer
        self.interm.set_sample_config(self.sample_in_dim_enc_embed, self.sample_num_enc_heads[0], self.patch_dim)

        ## Decoder configuration settings
        self.sample_num_dec_layers = config['num_dec_layers']
        self.sample_num_dec_heads = config['num_dec_heads']
        self.sample_in_dim_dec_embed = config['in_dim_dec_embed']
        self.sample_qk_dim_dec_embed = config['qk_dim_dec_embed']
        self.sample_v_dim_dec_embed = config['v_dim_dec_embed']
        self.sample_dim_dec_mlp = config['dim_dec_mlp']

        # set the decoder configurations
        self.decoder_embedding.set_sample_config(self.sample_in_dim_dec_embed)
        self.dec_positional_encoding.set_sample_config(self.sample_in_dim_dec_embed)
        self.dec_sample_dropout = self.compute_dropout(self.sample_in_dim_dec_embed, self.super_in_dim_dec_embed)

        # set the encoderlayers configurations
        for declayI, layer in enumerate(self.decoder_layers):
            if (declayI < self.sample_num_dec_layers):
                layer.set_sample_config(is_identity_layer = False,
                                        sample_in_dec_dim = self.sample_in_dim_dec_embed,
                                        sample_in_enc_dim = self.sample_in_dim_enc_embed,
                                        sample_qk_embed_dim = self.sample_qk_dim_dec_embed[declayI],
                                        sample_v_embed_dim = self.sample_v_dim_dec_embed[declayI],
                                        sample_d_ff = self.sample_dim_dec_mlp[declayI],
                                        sample_num_heads = self.sample_num_dec_heads[declayI],
                                        sample_dropout = self.compute_dropout(self.sample_in_dim_dec_embed, self.super_in_dim_dec_embed))
            else:
                layer.set_sample_config(is_identity_layer = True)
        
        # the final layer
        self.fc.set_sample_config(self.sample_in_dim_dec_embed, self.vocab_size)


    def compute_dropout(self, sample_embed_dim, super_embed_dim):
        return self.dropout * sample_embed_dim / super_embed_dim


    def forward(self, ios, tgt):
        # mask and embed
        ios = ios.view(ios.shape[0], -1, self.patch_dim)

        # embeddings
        for layer in self.emb_layers:
            ios = torch.relu(layer(ios))

        # shape change
        ios = torch.relu(self.enc_mid_fc(ios))
        ios_embedded = F.dropout(ios, p = self.enc_sample_dropout, training = self.training)

        # apply pos embedding and token embedding to formula input
        tgt_embedded = F.dropout(self.dec_positional_encoding(self.decoder_embedding(tgt)), p = self.dec_sample_dropout, training = self.training)
        
        # the encoder layers
        enc_output = ios_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)

        # the intermediate layer
        enc_output = self.interm(enc_output)

        # the decoder layers
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output)

        # the unembedding layer
        output = self.fc(dec_output)
        return output


    def get_dims_str(self, indent = 0):
        # prints out dimensions of all components, for debug
        # encoder embeddings
        indent_str = "\t" * indent
        dims_str = f"{indent_str}enc fnn: {self.enc_mid_fc.get_dims_str()}\n"

        # encder layers
        for layerI, layer in enumerate(self.encoder_layers):
            layer_dims_str = layer.get_dims_str(indent)
            dims_str += f"{indent_str}enc layer {layerI + 1}:\n{layer_dims_str}\n"

        # intermediate layer
        dims_str += f"{indent_str}interm PMS:\n{self.interm.get_dims_str(indent + 1)}\n"

        # decoding embeddings
        dims_str += f"{indent_str}dec tok embed: {self.decoder_embedding.get_dims_str()}\n"
        dims_str += f"{indent_str}dec pos embed: {self.dec_positional_encoding.get_dims_str()}\n"

        # decoder layers
        for layerI, layer in enumerate(self.decoder_layers):
            dims_str += f"{indent_str}dec layer {layerI + 1}:\n{layer.get_dims_str(indent)}\n"

        # final layer
        dims_str += f"{indent_str}final fnn: {self.fc.get_dims_str(indent)}"
        return dims_str



