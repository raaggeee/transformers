import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model = 6, heads=3):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        assert d_model % heads == 0
        self.hidden_dims = d_model
        self.heads = heads
        self.d_k = d_model//heads

    def scaled_dot_product_attention(self, q, k, v, mask):
        dims_k = k.shape[-1]

        # attention = torch.matmul(q, k.transpose(-2, -1))
        attention = (q @ k.transpose(-2, -1)/math.sqrt(dims_k))

        if mask is not None:
            fill_val = -float("inf")
            attention.masked_fill_(mask==0, fill_val)
            
        scaled_qk = F.softmax(attention)

        scaled_attention = torch.matmul(scaled_qk, v)
        
        return scaled_attention
    
    def forward(self, q, k, v, mask=False): #this is also done for cross attention
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        print(q.shape)

        q = q.view(q.shape[0], q.shape[1], self.heads, self.d_k).transpose(1, 2) 
        k = k.view(k.shape[0], k.shape[1], self.heads, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.heads, self.d_k).transpose(1, 2)
        
        x = self.scaled_dot_product_attention(q, k, v, mask)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads*self.d_k)

        return self.Wo(x)