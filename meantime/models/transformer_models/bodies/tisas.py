from torch import nn as nn

from .transformers.transformer_tisas import TisasTransformerBlock


class TisasBody(nn.Module):
    def __init__(self, args):
        super().__init__()

        n_layers = args.num_blocks

        self.transformer_blocks = nn.ModuleList(
            [TisasTransformerBlock(args) for _ in range(n_layers)])

    def forward(self, x, attn_mask, pos_k, pos_v, r_k, r_v, info=None):
        # x : B x T x H
        # r : B x T x T x H
        for layer, transformer in enumerate(self.transformer_blocks):
            x = transformer.forward(x, attn_mask, pos_k, pos_v, r_k, r_v, layer, info)

        return x
