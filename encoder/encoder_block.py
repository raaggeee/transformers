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
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.heads = heads

    def forward(self, x):
        "x: it can be either input+pos embeddings or previous encoding layer output"
        attention_output = MultiHeadAttention(self.d_model, self.heads).forward(x, x, x, mask=False)
        add_and_norm = Residual(self.d_model).forward(x, attention_output)

        feed_forward_output = FeedForwardNN(self.d_model).forward(add_and_norm)
        add_and_norm = Residual(self.d_model).forward(add_and_norm, feed_forward_output)

        return add_and_norm
    


