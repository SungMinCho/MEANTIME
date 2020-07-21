from meantime.models.transformer_models.utils import PositionwiseFeedForward

import torch
from torch import nn as nn
from torch.nn import functional as F


class ExactTisasTransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        attn_heads = args.num_heads
        hidden = args.hidden_units
        # feed_forward_hidden = hidden * 4
        feed_forward_hidden = hidden  # H->H->H instead of H->4H->H in TiSASRec
        dropout = args.dropout
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, act='relu')
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, pos_k, pos_v, r_k, r_v, layer, info):
        q, k, v = x, x, x
        q = self.norm1(q)  # only LN query in original code
        x = self.attention(q, k, v, mask, pos_k, pos_v, r_k, r_v, layer, info)
        x = q + x  # no dropout, LN(x) + g(LN(x))
        z = self.norm2(x)
        x = self.feed_forward(z)
        x = z + self.dropout(x)  # LN(x) + Dropout(g(LN(x)))
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
        # self.output_linear = nn.Linear(d_model, d_model)  # no output linear in TiSASRec

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask, pos_k, pos_v, r_k, r_v, layer, info):
        batch_size = query.size(0)
        T = query.size(1)

        if info is not None:
            info['input_seq' + str(layer)] = value
        # B x n x T x d
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        if info is not None:
            info['value_seq' + str(layer)] = value

        # B x n x T x d
        pos_k, pos_v = [x.view(batch_size, T, self.h, self.d_k).transpose(1, 2)
                        for x in [pos_k, pos_v]]

        # B x n x T x T x d
        r_k, r_v = [x.view(batch_size, T, T, self.h, self.d_k).permute(0, 3, 1, 2, 4)
                    for x in [r_k, r_v]]

        # x : B x n x T x d
        x, attn = self.attention(query, key, value, mask, pos_k, pos_v, r_k, r_v, layer, info)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        if info is not None:
            info['attn_seq' + str(layer)] = x

        # x = self.output_linear(x)
        # if info is not None:
        #     info['output_seq' + str(layer)] = x
        # no output linear in TiSASRec
        return x

    def attention(self, query, key, value, mask, pos_k, pos_v, r_k, r_v, layer, info):
        # q, k, v : B x n x T x d
        # pos_k, pos_v : B x n x T x d
        # r_k, r_v : B x n x T x T x d
        # scores = torch.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size(-1))

        scores = torch.einsum('bnid,bnjd->bnij', query, key)  # B x n x T x T
        scores += torch.einsum('bnid,bnjd->bnij', query, pos_k)  # B x n x T x T
        scores += torch.einsum('bnid,bnijd->bnij', query, r_k)  # B x n x T x T

        scores = scores * self.scale

        if mask is not None:  # B x n x T x T
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if info is not None:
            info['attn_scores' + str(layer)] = p_attn

        p_attn = self.dropout(p_attn)

        out = torch.einsum('bnij,bnjd->bnid', p_attn, value)
        out += torch.einsum('bnij,bnjd->bnid', p_attn, pos_v)
        out += torch.einsum('bnij,bnijd->bnid', p_attn, r_v)  # B x n x T x d
        return out, p_attn