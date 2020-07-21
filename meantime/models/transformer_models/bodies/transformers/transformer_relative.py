from meantime.models.transformer_models.utils import PositionwiseFeedForward, SublayerConnection

import torch
from torch import nn as nn
from torch.nn import functional as F


class TransformerRelativeBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        attn_heads = args.num_heads
        hidden = args.hidden_units
        feed_forward_hidden = 4 * hidden
        dropout = args.dropout
        self.attention = RelAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, act='gelu', middle_drop=False)
        self.input_sublayer = SublayerConnection(args=args, size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(args=args, size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.args = args

    def forward(self, x, r, mask, layer, info):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, r, mask=mask, layer=layer, info=info))
        x = self.output_sublayer(x, self.feed_forward)
        return x


class RelAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.scale = 1 / (self.d_k ** 0.5)

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.r_layer = nn.Linear(d_model, d_model)
        self.r_bias = nn.Parameter(torch.FloatTensor(1, self.h, 1, self.d_k))
        self.output_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, r, mask=None, layer=None, info=None):
        batch_size, T = query.size(0), query.size(1)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]  # B x n x T x d

        r = self.r_layer(r)  # B x T x T x H
        r = r.view(batch_size, T, T, self.h, self.d_k)  # B x T x T x n x d
        r = r.permute(0, 3, 1, 2, 4)  # B x n x T x T x d

        x, attn = self.attention(query, key, value, r, mask=mask, layer=layer, info=info)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        x =  self.output_linear(x)
        return x

    def attention(self, query, key, value, r, mask=None, layer=None, info=None):
        scores = torch.matmul(query, key.transpose(-2, -1))  # B x n x T x T
        scores += torch.einsum('bnid,bnijd->bnij', query + self.r_bias, r)
        scores = scores * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn