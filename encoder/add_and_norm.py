import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class AddAndNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        add = torch.add(x, y)

        return nn.LayerNorm(add)
    
    