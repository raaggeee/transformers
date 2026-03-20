import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from components.multi_head_attention import MultiHeadAttention
from components.feed_forward import FeedForwardNN
from components.residual_connection import Residual

class DecoderBlock(nn.Module):
    def __init__(self, vocab_size, d_model=256, heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.heads = heads
        self.residual1 = Residual(self.d_model)
        self.residual2 = Residual(self.d_model)
        self.residual3 = Residual(self.d_model)
        self.self_attention = MultiHeadAttention(self.d_model, self.heads)
        self.cross_attention = MultiHeadAttention(self.d_model, self.heads)
        self.feed_forward = FeedForwardNN(self.d_model)

    def forward(self, encoder_output, x, mask, tgt_mask):
        """
        encoder_output: Used for K and V values for cross attention
        input_embeds: It is the predicted output.
        """
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual2(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, mask))
        x = self.residual3(x, self.feed_forward)
        return x

