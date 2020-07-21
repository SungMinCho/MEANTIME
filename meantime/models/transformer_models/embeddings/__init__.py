import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 2
        hidden = args.hidden_units
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)

    def forward(self, d):
        x = d['tokens']  # B x T
        return self.emb(x)  # B x T x H


class TokenEmbeddingDirect(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 2
        hidden = args.hidden_units
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)

    def forward(self, x):
        return self.emb(x)  # B x T x H


class ConstantEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden = args.hidden_units
        self.emb = nn.Embedding(1, hidden)

    def forward(self, d):
        batch_size, T = d['tokens'].shape
        return self.emb.weight.unsqueeze(0).repeat(batch_size, T, 1)  # B x T x H


class PositionalEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        max_len = args.max_len
        hidden = args.hidden_units
        self.emb = nn.Embedding(max_len, hidden)

    def forward(self, d):
        x = d['tokens']
        batch_size = x.size(0)
        return self.emb.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # B x T x H


class PositionalEmbeddingDirect(nn.Module):
    def __init__(self, args):
        super().__init__()
        max_len = args.max_len
        hidden = args.hidden_units
        self.emb = nn.Embedding(max_len, hidden)

    def forward(self, x):
        batch_size = x.size(0)
        return self.emb.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # B x T x H


class DayEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_days = args.num_days
        hidden = args.hidden_units
        self.emb = nn.Embedding(num_days, hidden)

    def forward(self, d):
        days = d['days']  # B x T
        return self.emb(days)  # B x T x H


class TiSasRelativeTimeEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        time_difference_clip = args.tisas_max_time_intervals
        hidden = args.hidden_units
        self.time_emb = nn.Embedding(time_difference_clip + 1, hidden)
        self.time_difference_clip = time_difference_clip

    def forward(self, d):
        # t : B x T
        # time_diff : B x T x T  (value range: -time_range ~ time_range)
        t = d['timestamps']
        query_time, key_time = t, t
        time_diff = query_time.unsqueeze(2) - key_time.unsqueeze(1)
        time_diff.abs_().clamp_(max=self.time_difference_clip)  # B x T x T
        return self.time_emb(time_diff)  # B x T x T x H


class SinusoidTimeDiffEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.time_unit_divide = args.time_unit_divide
        self.hidden = args.hidden_units
        self.freq = args.freq

    def forward(self, d):
        # t : B x T
        # time_diff : B x T x T  (value range: -time_range ~ time_range)
        t = d['timestamps']
        time_diff = t.unsqueeze(2) - t.unsqueeze(1)
        time_diff = time_diff.to(torch.float)
        time_diff = time_diff / self.time_unit_divide

        freq_seq = torch.arange(0, self.hidden, 2.0, dtype=torch.float)  # [0, 2, ..., H-2]
        freq_seq = freq_seq.to(time_diff)  # device
        inv_freq = 1 / torch.pow(self.freq, (freq_seq / self.hidden))  # 1 / 10^(4 * [0, 2/H, 4/H, (H-2)/H])

        sinusoid_inp = torch.einsum('bij,d->bijd', time_diff, inv_freq)  # B x T x T x (H/2)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)  # B x T x T x H

        return pos_emb


class ExponentialTimeDiffEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.time_unit_divide = args.time_unit_divide
        self.hidden = args.hidden_units
        self.freq = args.freq

    def forward(self, d):
        # t : B x T
        # time_diff : B x T x T  (value range: -time_range ~ time_range)
        t = d['timestamps']
        time_diff = t.unsqueeze(2) - t.unsqueeze(1)
        time_diff = time_diff.to(torch.float)
        time_diff = time_diff / self.time_unit_divide
        time_diff.abs_()  # absolute to only use the positive part of exp(-x)

        freq_seq = torch.arange(0, self.hidden, 1.0, dtype=torch.float)  # [0, 1, ..., H-1]
        freq_seq = freq_seq.to(time_diff)  # device
        inv_freq = 1 / torch.pow(self.freq, (freq_seq / self.hidden))  # 1 / 10^(4 * [0, 1/H, 2/H, (H-1)/H])

        exponential_inp = torch.einsum('bij,d->bijd', time_diff, inv_freq)  # B x T x T x H
        pos_emb = (-exponential_inp).exp()  # B x T x T x H
        return pos_emb


class Log1pTimeDiffEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.time_unit_divide = args.time_unit_divide
        self.hidden = args.hidden_units
        self.freq = args.freq

    def forward(self, d):
        # t : B x T
        # time_diff : B x T x T  (value range: -time_range ~ time_range)
        t = d['timestamps']
        time_diff = t.unsqueeze(2) - t.unsqueeze(1)
        time_diff = time_diff.to(torch.float)
        time_diff = time_diff / self.time_unit_divide
        time_diff.abs_()  # absolute to only use the positive part of log(1+x)

        freq_seq = torch.arange(0, self.hidden, 1.0, dtype=torch.float)  # [0, 1, ..., H-1]
        freq_seq = freq_seq.to(time_diff)  # device
        inv_freq = 1 / torch.pow(self.freq, (freq_seq / self.hidden))  # 1 / 10^(4 * [0, 1/H, 2/H, (H-1)/H])

        log1p_inp = torch.einsum('bij,d->bijd', time_diff, inv_freq)  # B x T x T x H
        pos_emb = log1p_inp.log1p()  # B x T x T x H
        return pos_emb
