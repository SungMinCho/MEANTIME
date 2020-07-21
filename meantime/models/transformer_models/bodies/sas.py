from torch import nn as nn

from .transformers.transformer_sas import SasTransformerBlock


class SasBody(nn.Module):
    def __init__(self, args):
        super().__init__()

        n_layers = args.num_blocks

        self.transformer_blocks = nn.ModuleList(
            [SasTransformerBlock(args) for _ in range(n_layers)])

    def forward(self, x, attn_mask, info=None):
        for layer, transformer in enumerate(self.transformer_blocks):
            x = transformer.forward(x, attn_mask, layer, info)

        return x
