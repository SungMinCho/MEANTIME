from meantime.models.transformer_models.utils import PositionwiseFeedForward, SublayerConnection

import torch
from torch import nn as nn
from torch.nn import functional as F


class SasTransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        attn_heads = args.num_heads
        hidden = args.hidden_units
        # feed_forward_hidden = 4 * hidden
        feed_forward_hidden = hidden  # H->H->H instead of H->4H->H in PFF
        dropout = args.dropout
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, act='relu')
        self.input_sublayer = SublayerConnection(args=args, size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(args=args, size=hidden, dropout=dropout)

    def forward(self, x, mask, layer, info):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask, layer=layer, info=info))
        x = self.output_sublayer(x, self.feed_forward)
        return x


class MultiHeadedAttention(nn.Module):
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
        # self.output_linear = nn.Linear(d_model, d_model)
        # no output linear in SASRec

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer=None, info=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        if info is not None:
            info['input_seq' + str(layer)] = value
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        if info is not None:
            info['value_seq' + str(layer)] = value

        x, attn = self.attention(query, key, value, mask=mask, layer=layer, info=info)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        if info is not None:
            info['attn_seq' + str(layer)] = x

        # x =  self.output_linear(x)
        # no output linear in SASRec
        # if info is not None:
        #     info['output_seq' + str(layer)] = x
        return x

    def attention(self, query, key, value, mask=None, layer=None, info=None):
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if info is not None:
            info['attn_scores' + str(layer)] = p_attn

        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn