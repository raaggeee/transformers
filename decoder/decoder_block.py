import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from components.multi_head_attention import MultiHeadAttention
from components.feed_forward import FeedForwardNN
from components.residual_connection import Residual


class DecoderBlock(nn.Module):
    def __init__(self, vocab_size, d_model=256, heads=4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.heads = heads
        self.residual = Residual(self.d_model)

    def forward(self, encoder_output, input_embeds):
        """
        encoder_output: Used for K and V values for cross attention
        input_embeds: It is the predicted output.
        """
        attention_output = MultiHeadAttention(self.d_model, self.heads).forward(input_embeds,
                                                                                input_embeds,
                                                                                input_embeds,
                                                                                mask=True)
        
        attention_add_and_norm = self.residual.forward(input_embeds, attention_output)

        cross_attention_output = MultiHeadAttention(self.d_model, self.heads)\
        .forward(attention_add_and_norm, encoder_output, encoder_output)

        cross_attention_add_and_norm =self.residual.forward(attention_add_and_norm, 
                                                            cross_attention_output)

        feed_forward = FeedForwardNN(self.d_model).forward(cross_attention_add_and_norm)

        add_and_norm = self.residual.forward(cross_attention_add_and_norm, feed_forward)

        return add_and_norm

