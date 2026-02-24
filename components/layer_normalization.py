import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps:float=10**-6):
        super().__init__()
        self.thetha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True) #mean
        var = x.std(dim=-1, keepdim=True) #var

        layer_norm = ((x - mu)/(var + self.eps))

        return self.alpha(layer_norm) + self.bias

       
    
    