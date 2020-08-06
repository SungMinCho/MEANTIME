from ..base import BaseModel
from .embeddings import *
from .bodies import ExactTisasBody

import torch
import torch.nn as nn


class TiSasModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.token_embedding = TokenEmbedding(args)
        self.position_key_embedding = PositionalEmbedding(args)
        self.position_value_embedding = PositionalEmbedding(args)
        self.relative_time_key_embedding = TiSasRelativeTimeEmbedding(args)
        self.relative_time_value_embedding = TiSasRelativeTimeEmbedding(args)
        self.body = ExactTisasBody(args)
        self.dropout = nn.Dropout(p=args.dropout)
        self.init_weights()

        self.sigmoid = nn.Sigmoid()
        self.vocab_size = args.num_items + 1  # vocab size for prediction

    @classmethod
    def code(cls):
        return 'tisas'

    def forward(self, d):
        logits, info = self.get_logits(d)
        ret = {'logits':logits, 'info':info}
        if self.training:
            labels = d['labels']
            negative_labels = d['negative_labels']
            loss, loss_cnt = self.get_loss(logits, labels, negative_labels)
            ret['loss'] = loss
            ret['loss_cnt'] = loss_cnt
        else:
            # get scores (B x C) for validation
            last_logits = logits[:, -1, :].unsqueeze(1)  # B x 1 x H
            candidate_embeddings = self.token_embedding.emb(d['candidates'])  # B x C x H
            scores = (last_logits * candidate_embeddings).sum(-1)  # B x C
            ret['scores'] = scores
        return ret

    def get_logits(self, d):
        x = d['tokens']
        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # B x 1 x T x T
        attn_mask.tril_()  # causal attention for sasrec
        token_embeddings = self.dropout(self.token_embedding(d))  # B x T x H
        position_key = self.dropout(self.position_key_embedding(d))  # B x T x H
        position_value = self.dropout(self.position_value_embedding(d))  # B x T x H
        relative_time_key = self.dropout(self.relative_time_key_embedding(d))  # B x T x H
        relative_time_value = self.dropout(self.relative_time_value_embedding(d))  # B x T x H

        # relative_time_embeddings = None
        b = self.body(token_embeddings, attn_mask,
                      pos_k=position_key,
                      pos_v=position_value,
                      r_k=relative_time_key,
                      r_v=relative_time_value)
        info = None
        return b, info

    def get_loss(self, logits, labels, negative_labels):
        _logits = logits.view(-1, logits.size(-1))  # BT x H
        _labels = labels.view(-1)  # BT
        _negative_labels = negative_labels.view(-1)  # BT

        valid = _labels > 0
        loss_cnt = valid.sum()  # = M
        valid_index = valid.nonzero().squeeze()  # M

        valid_logits = _logits[valid_index]  # M x H
        valid_labels = _labels[valid_index]  # M
        valid_negative_labels = _negative_labels[valid_index]  # M

        valid_labels_emb = self.token_embedding.emb(valid_labels)  # M x H
        valid_negative_labels_emb = self.token_embedding.emb(valid_negative_labels)  # M x H

        valid_labels_prob = self.sigmoid((valid_logits * valid_labels_emb).sum(-1))  # M
        valid_negative_labels_prob = self.sigmoid((valid_logits * valid_negative_labels_emb).sum(-1))  # M

        loss = -torch.log(valid_labels_prob + 1e-24) - torch.log((1-valid_negative_labels_prob) + 1e-24)
        loss = loss.mean()
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        return loss, loss_cnt
