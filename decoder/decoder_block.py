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

    def forward(self, encoder_output, input_embeds, mask, tgt_mask):
        """
        encoder_output: Used for K and V values for cross attention
        input_embeds: It is the predicted output.
        """
        attention_output = self.self_attention.forward(input_embeds,
                                                        input_embeds,
                                                        input_embeds,
                                                        mask)
        attention_add_and_norm = self.residual1.forward(input_embeds, attention_output)

        cross_attention_output = self.cross_attention\
        .forward(attention_add_and_norm, encoder_output, encoder_output, tgt_mask)
        cross_attention_add_and_norm =self.residual2.forward(attention_add_and_norm, 
                                                            cross_attention_output)

        feed_forward = self.feed_forward.forward(cross_attention_add_and_norm)
        add_and_norm = self.residual3.forward(cross_attention_add_and_norm, feed_forward)

        return add_and_norm

