import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from components.layer_normalization import LayerNormalization

class Residual(nn.Module):
    def __init__(self, d_model):
        self.dropout = nn.Dropout(p=0.1)
        self.norm = LayerNormalization(d_model)

    def forward(self, x:torch.Tensor, sublayer: nn.Module):
        """
        sublayer: Output of layer like 
                    - Multi-Head Attention
                    - Feed Forward Network
                    - Cross Attention etc.
        """

        sublayer_output = sublayer(x)

        return self.norm(x + self.dropout(sublayer_output))