from torch import nn as nn

from .transformers.transformer_relative import TransformerRelativeBlock


class BertRelativeBody(nn.Module):
    def __init__(self, args):
        super().__init__()

        n_layers = args.num_blocks

        self.transformer_blocks = nn.ModuleList(
            [TransformerRelativeBlock(args) for _ in range(n_layers)])

    def forward(self, x, r, attn_mask, info=None):
        for layer, transformer in enumerate(self.transformer_blocks):
            x = transformer.forward(x, r, attn_mask, layer, info)
        return x
