import torch.nn as nn
# from .layer_norm import LayerNorm


def SublayerConnection(args, size, dropout):
    if args.residual_ln_type == 'pre':
        return SublayerConnectionPreLN(size, dropout)
    elif args.residual_ln_type == 'post':
        return SublayerConnectionPostLN(size, dropout)
    else:
        raise ValueError


class SublayerConnectionPreLN(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # return self.norm(x + self.dropout(sublayer(x)))
        sub_output = sublayer(self.norm(x))
        if isinstance(sub_output, tuple):
            sub_output, rest = sub_output[0], sub_output[1:]
            output = x + self.dropout(sub_output)
            return (output, *rest)
        else:
            return x + self.dropout(sub_output)


class SublayerConnectionPostLN(SublayerConnectionPreLN):
    def forward(self, x, sublayer):
        sub_output = sublayer(x)
        if isinstance(sub_output, tuple):
            sub_output, rest = sub_output[0], sub_output[1:]
            output = x + self.dropout(sub_output)
            output = self.norm(output)
            return (output, *rest)
        else:
            return self.norm(x + self.dropout(sub_output))
