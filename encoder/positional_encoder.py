import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np

#v1
class PositionalEncodingV1(nn.Module):
    """
    the most inefficient approach
    """
    def __init__(self, d_model):
        self.d_model = d_model

    def calculate_positions(self, embeds):
        batch = embeds.shape[0]
        dims = embeds.shape[1]
        feats = embeds.shape[2]
        
        pos_encodes = []

        for i in range(dims):
            pos = []
            for j in range(feats//2):
                pos_i = np.sin(i/10000**((2*j)/self.d_model))
                pos.append(pos_i)
                pos_i1 = np.cos(i/10000**((2*j)/self.d_model))
                pos.append(pos_i1)

            pos_encodes.append(pos)
        return torch.tensor(pos_encodes)

    def combine_encodings(self, embeddings):
        pos_encodings = self.calculate_positions(embeddings)
        print(pos_encodings)
        print(pos_encodings.shape)
        return torch.add(embeddings, pos_encodings)


#v2
class PositionalencodingV2(nn.Module):
    def __init__(self, d_model, max_length=5000):
        #init 0s matrix
        pe = torch.zeros(max_length, d_model)

        #init k
        k = torch.arange(0, max_length).unsqueeze(1)

        #init div_term
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000)/d_model) 
        )

        pe[:, 0::2] = torch.sin(k * div_term) #broadcast and then apply sin to even rows
        pe[:, 1::2] = torch.cos(k * div_term) #broadcast and then apply cos to odd rows

        pe = pe.unsqueeze(0)

        self.pe = pe

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad(False)
        return x

        
