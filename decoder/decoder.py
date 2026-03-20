import torch.nn as nn
import torch.nn.functional as F
from decoder.decoder_block import DecoderBlock
from components.layer_normalization import LayerNormalization

class Decoder(nn.Module):
    def __init__(self, features, layers: nn.ModuleList):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.layers = layers

    def forward(self, encoder_output, x, mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, mask, tgt_mask)
        return self.norm(x)
    