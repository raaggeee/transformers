import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from encoder.encoder import Encoder
from decoder.decoder import Decoder
from components.embeddings import InputEmbeddings
from components.positional_encoder import PositionalencodingV2
from components.projection import Projection

class Transformers(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_embeds: InputEmbeddings, pos_embeds: InputEmbeddings,
                 src_tgt: PositionalencodingV2, pos_tgt: PositionalencodingV2, proj: Projection):

        self.encoder = encoder
        self.decoder = decoder
        self.src_embeds = src_embeds
        self.pos_embeds = pos_embeds
        self.src_tgt = src_tgt
        self.pos_tgt = pos_tgt
        self.proj = proj

        

