import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class FeedForwardNN(nn.Module):
    def __init__(self, d_model=512, hidden=2048):
        super().__init__()
        self.input_layer = nn.Linear(d_model, hidden)
        self.output_layer = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        hidden = F.relu(self.input_layer(x))
        output = self.output_layer(self.dropout(hidden))
        return output
