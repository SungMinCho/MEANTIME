from torch import nn as nn

from .transformers.transformer_sas_exact import ExactSasTransformerBlock


class ExactSasBody(nn.Module):
    def __init__(self, args):
        super().__init__()

        n_layers = args.num_blocks

        self.transformer_blocks = nn.ModuleList(
            [ExactSasTransformerBlock(args) for _ in range(n_layers)])

        self.ln = nn.LayerNorm(args.hidden_units)

    def forward(self, x, attn_mask, info=None):
        for layer, transformer in enumerate(self.transformer_blocks):
            x = transformer.forward(x, attn_mask, layer, info)
        x = self.ln(x)  # original code does this at the end of body
        return x
