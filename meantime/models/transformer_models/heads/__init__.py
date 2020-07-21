from ..utils import GELU

import torch
import torch.nn as nn


class BertLinearPredictionHead(nn.Module):
    def __init__(self, args, input_size=None):
        super().__init__()
        self.vocab_size = args.num_items + 1
        hidden = input_size if input_size is not None else args.hidden_units
        if args.head_use_ln:
            self.out = nn.Sequential(
                nn.Linear(hidden, hidden),
                GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, self.vocab_size)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(hidden, hidden),
                GELU(),
                nn.Linear(hidden, self.vocab_size)
            )

    def forward(self, x, candidates=None):
        x = self.out(x)  # B x V or M x V
        if candidates is not None:
            x = x.gather(1, candidates)  # B x C or M x C
        return x


class BertDotProductPredictionHead(nn.Module):
    def __init__(self, args, token_embeddings, input_size=None):
        super().__init__()
        self.token_embeddings = token_embeddings
        hidden = args.hidden_units
        if input_size is None:
            input_size = hidden
        self.vocab_size = args.num_items + 1
        if args.head_use_ln:
            self.out = nn.Sequential(
                nn.Linear(input_size, hidden),
                GELU(),
                nn.LayerNorm(hidden)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(input_size, hidden),
                GELU(),
            )
        self.bias = nn.Parameter(torch.zeros(1, self.vocab_size))

    def forward(self, x, candidates=None):
        x = self.out(x)  # B x H or M x H
        if candidates is not None:  # x : B x H
            emb = self.token_embeddings(candidates)  # B x C x H
            logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
            bias = self.bias.expand(logits.size(0), -1).gather(1, candidates)  # B x C
            logits += bias
        else:  # x : M x H
            emb = self.token_embeddings.weight[:self.vocab_size]  # V x H
            logits = torch.matmul(x, emb.transpose(0, 1))  # M x V
            logits += self.bias
        return logits


class BertL2PredictionHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden = args.hidden_units
        self.vocab_size = args.num_items + 1
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            GELU(),
            nn.LayerNorm(hidden)
        )
        self.bias = nn.Parameter(torch.zeros(1, 1, self.vocab_size))

    def forward(self, x, token_embeddings):
        # x = self.out(x).unsqueeze(2)  # B x T x 1 x H
        x = x.unsqueeze(2)  # B x T x 1 x H
        emb = token_embeddings.weight[:self.vocab_size].unsqueeze(0).unsqueeze(1)  # 1 x 1 x V x H
        diff = x - emb  # B x T x V x H
        dist = (diff ** 2).sum(-1).sqrt()  # B x T x V
        return (-dist) + self.bias


class BertDiscriminatorHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden = args.hidden_units
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            GELU(),
            nn.LayerNorm(hidden)
        )
        self.w = nn.Parameter(torch.zeros(1, 1, hidden))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : B x T x H
        x = self.out(x)
        x = (x * self.w).sum(-1)  # B x T
        return self.sigmoid(x)
