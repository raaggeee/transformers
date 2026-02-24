import torch.nn as nn
import torch.nn.functional as F
from decoder.decoder_block import EncoderBlock
from components.layer_normalization import LayerNormalization

class Encoder(nn.Module):
    def __init__(self, features, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

        