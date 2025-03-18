import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderBlock()
        self.decoder = DecoderBlock()

    def forward(self, x):
        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, num_of_layers=6):
        super().__init__()
        layers = []
        for _ in range(num_of_layers):
            layers.append(Encoder())

    def forward(self, x):
        return x
    

class Encoder(nn.Moudle):
    def __init__(self):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention()
        self.ffn = FFN()

    def forward(self, x):
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_embed, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_embed = d_embed
        self.num_heads = num_heads 
        self.d_k = d_embed // num_heads

        self.W_q = nn.Linear(d_embed, d_embed, bias=False)
        self.W_k = nn.Linear(d_embed, d_embed, bias=False)
        self.W_v = nn.Linear(d_embed, d_embed, bias=False)

        self.W_o = nn.Linear(d_embed, d_embed, bias=False)

    def forward(self, x):
        B, N, _ = x.shape

         # (B, h, N, d_k)
        Q = self.W_q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2) 
        V = self.W_v(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        # (B, h, N, N)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)
        concat_output = attention_output.transpose(1, 2).reshape(B, N, self.d_embed)
        output = self.W_o(concat_output)

        return output



class FFN(nn.Moudle):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        return x


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        return x