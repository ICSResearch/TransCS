import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.ModuleList()
        for i in range(3):
            proj = nn.Sequential(
                nn.Linear(dim, dim, bias=True))
            self.projection.append(proj)

    def forward(self, x, extra=None):
        if extra is None:
            q, k, v = self.projection[0](x),\
                      self.projection[1](x),\
                      self.projection[2](x)
        else:
            q, k, v = self.projection[0](x),\
                      self.projection[1](extra),\
                      self.projection[2](extra)
        return q, k, v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, head=None):
        super().__init__()
        self.dim = dim
        self.head = head or 8
        if self.dim % self.head != 0:
            raise NotImplementedError("Dimensions cannot be divisible by head, get dim {}, but head {}"
                                      .format(dim, head))
        self.attention = Attention(dim)

    def forward(self, x, extra=None):
        q, k, v = self.attention(x, extra)
        _temp = torch.chunk(q, self.head, dim=2)
        q = torch.cat(_temp, dim=0)
        _temp = torch.chunk(k, self.head, dim=2)
        k = torch.cat(_temp, dim=0)
        _temp = torch.chunk(v, self.head, dim=2)
        v = torch.cat(_temp, dim=0)

        weights = torch.softmax(torch.matmul(q, k.permute(0, 2, 1)), dim=2)
        y = torch.matmul(weights, v)

        _temp = torch.chunk(y, self.head, dim=0)
        y = torch.cat(_temp, dim=2)
        return y


class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden_dim, n_head, drop):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MHA = MultiHeadSelfAttention(dim=d_model, head=n_head)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.ELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, d_model, bias=True)
        )

    def forward(self, x):
        y = x + self.norm1(self.MHA(x))
        y = y + self.norm2(self.MLP(y))
        return y


class DecoderLayer(nn.Module):
    def __init__(self, d_model, hidden_dim, n_head, drop):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.MHA1 = MultiHeadSelfAttention(dim=d_model, head=n_head)
        self.MHA2 = MultiHeadSelfAttention(dim=d_model, head=n_head)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.ELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, d_model, bias=True)
        )

    def forward(self, x, extra_x):
        y = x + self.norm1(self.MHA1(x))
        y = y + self.norm2(self.MHA2(y, extra_x))
        y = y + self.norm3(self.MLP(y))
        return y


class Encoder(nn.Module):
    def __init__(self, dim, hidden_dim=None, num_head=8, factor=4, dropout=0.):
        super(Encoder, self).__init__()
        hidden_dim = hidden_dim or dim * factor
        assert dim % num_head == 0, f"dim {dim} should be divided by num_heads {num_head}."
        self.dim = dim
        self.size = int(dim ** 0.5)
        self.pos_embed = PositionalEmbedding(d_model=dim)
        self.layer = EncoderLayer(d_model=dim, hidden_dim=hidden_dim, n_head=num_head, drop=dropout)

    def forward(self, x):
        x = Rearrange('b c h w -> b c (h w)')(x)
        x = x * math.sqrt(self.dim)
        x = self.pos_embed(x)
        y = self.layer(x)
        y = Rearrange('b c (h w)-> b c h w', h=self.size, w=self.size)(y)
        return y


class Decoder(nn.Module):
    def __init__(self, dim, hidden_dim=None, num_head=8, factor=4, dropout=0.):
        super(Decoder, self).__init__()
        hidden_dim = hidden_dim or dim * factor
        assert dim % num_head == 0, f"dim {dim} should be divided by num_heads {num_head}."
        self.dim = dim
        self.size = int(dim ** 0.5)
        self.pos_embed = PositionalEmbedding(d_model=dim)
        self.layer = DecoderLayer(d_model=dim, hidden_dim=hidden_dim, n_head=num_head, drop=dropout)

    def forward(self, x, x_extra):
        x_extra = Rearrange('b c h w -> b c (h w)')(x_extra)
        x = Rearrange('b c h w -> b c (h w)')(x)
        y = x * math.sqrt(self.dim)
        y = self.pos_embed(y)
        y = self.layer(y, x_extra)
        y = Rearrange('b c (h w)-> b c h w', h=self.size, w=self.size)(y)
        return y
