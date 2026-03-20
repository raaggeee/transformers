import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from components.layer_normalization import LayerNormalization

class Residual(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        """
        sublayer: Output of layer like 
                    - Multi-Head Attention
                    - Feed Forward Network
                    - Cross Attention etc.
        """

        sublayer_output = sublayer(self.norm(x))

        return x + self.dropout(sublayer_output)