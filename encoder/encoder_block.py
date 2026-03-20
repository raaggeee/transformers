# Encoder block flow
# input - Input sequences, can be multiple also | dim (batch, seq, feats)
# from input we generate embeddings which are futher added with positional encodings | dim (batch, seq, feats)
# then the output of previous layer passes through 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from components.multi_head_attention import MultiHeadAttention
from components.feed_forward import FeedForwardNN
from components.residual_connection import Residual

class EncoderBlock(nn.Module):
    def __init__(self, vocab_size, d_model=256, heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.heads = heads
        self.residual1 = Residual(self.d_model)
        self.residual2 = Residual(self.d_model)
        self.self_attention = MultiHeadAttention(self.d_model, self.heads)
        self.feed_forward = FeedForwardNN(self.d_model)

    def forward(self, x, mask):
        "x: it can be either input+pos embeddings or previous encoding layer output"
        add_and_norm = self.residual1(x, lambda x: self.self_attention(x, x, x, mask))

        feed_forward_output = self.feed_forward(add_and_norm)
        add_and_norm = self.residual2(add_and_norm, lambda x: self.feed_forward(x))

        return add_and_norm

