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


class Encoder(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_embed)
        self.ffn = FFN(d_embed)
        self.add_norm1 = AddNorm(d_embed)
        self.add_norm2 = AddNorm(d_embed)

    def forward(self, x):
        residual = x
        x = self.multi_head_attention(x)
        x = self.add_norm1(x, residual)
        
        residual = x
        x = self.ffn(x)
        x = self.add_norm2(x, residual)
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

        attention_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, h, N, N)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)  # (B, h, N, d_k)
        concat_output = attention_output.transpose(
            1, 2).reshape(B, N, self.d_embed)  # (B, N, d_embed)
        output = self.W_o(concat_output)

        return output


class FFN(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(d_embed, d_embed*4)
        self.fc2 = nn.Linear(d_embed*4, d_embed)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(self.relu(x))
        return x


class AddNorm(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_embed)

    def forward(self, x, residual):
        x = self.layer_norm(x+residual)
        return x


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
