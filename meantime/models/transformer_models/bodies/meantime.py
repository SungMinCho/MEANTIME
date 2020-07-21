from torch import nn as nn

from .transformers.transformer_meantime import TransformerMeantimeBlock


class MeantimeBody(nn.Module):
    def __init__(self, args, La, Lr):
        super().__init__()

        n_layers = args.num_blocks

        self.transformer_blocks = nn.ModuleList(
            [TransformerMeantimeBlock(args, La, Lr) for _ in range(n_layers)])

    def forward(self, x, attn_mask, abs_kernel, rel_kernel, info):
        # x : B x T x H
        # abs_kernel : La of [B x T x H]
        # rel_kernel : Lr of [B x T x T x H]
        for layer, transformer in enumerate(self.transformer_blocks):
            x = transformer.forward(x, attn_mask, abs_kernel, rel_kernel, layer=layer, info=info)
        return x
