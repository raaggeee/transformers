import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from encoder.encoder import Encoder
from encoder.encoder_block import EncoderBlock
from decoder.decoder import Decoder
from decoder.decoder_block import DecoderBlock
from components.embeddings import InputEmbeddings
from components.positional_encoder import PositionalencodingV2
from components.projection import Projection
from components.multi_head_attention import MultiHeadAttention
from components.feed_forward import FeedForwardNN

class Transformers(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_embeds: InputEmbeddings, tgt_embeds: InputEmbeddings,
                 src_pos: PositionalencodingV2, tgt_pos: PositionalencodingV2, proj: Projection):
        super().__init__()

        self.encoder = encoder  
        self.decoder = decoder
        self.src_embeds = src_embeds
        self.tgt_embeds = tgt_embeds
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj

    def encode(self, src, src_mask):
        print(src.dtype)
        src = self.src_embeds(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embeds(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(encoder_output, src_mask, tgt, tgt_mask)

    def project(self, x):
        return self.proj(x)
    
def build_transformers(src_vocab_size: int, tgt_vocab_size: int, src_seq: int, 
                       tgt_seq: int, d_model: int = 512, h: int = 8, 
                       dropout: float=0.1, d_ff: int = 2048, N:int = 6) -> Transformers:
    src_embeds = InputEmbeddings(d_model, src_vocab_size)
    tgt_embeds = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalencodingV2(d_model, src_seq)
    tgt_pos = PositionalencodingV2(d_model, tgt_seq)

    encoder_blocks = []
    for _ in range(N):
        encoder_block = EncoderBlock(src_vocab_size, d_model, h)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_block = DecoderBlock(tgt_vocab_size, d_model, h)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(d_model, encoder_blocks)
    decoder = Decoder(d_model, decoder_blocks)

    proj_layer = Projection(d_model, tgt_vocab_size)

    transformers = Transformers(encoder, decoder, src_embeds, tgt_embeds, src_pos, tgt_pos, proj_layer)

    #for weight initialization of Weights layer
    for p in transformers.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformers
    
        

