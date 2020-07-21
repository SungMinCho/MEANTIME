from meantime.models.base import BaseModel

import torch.nn as nn

from abc import *


class BertBaseModel(BaseModel, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__(args)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, d):
        logits, info = self.get_logits(d)
        ret = {'logits':logits, 'info':info}
        if self.training:
            labels = d['labels']
            loss, loss_cnt = self.get_loss(d, logits, labels)
            ret['loss'] = loss
            ret['loss_cnt'] = loss_cnt
        else:
            # get scores (B x V) for validation
            last_logits = logits[:, -1, :]  # B x H
            ret['scores'] = self.get_scores(d, last_logits)  # B x C
        return ret

    @abstractmethod
    def get_logits(self, d):
        pass

    @abstractmethod
    def get_scores(self, d, logits):  # logits : B x H or M x H, returns B x C or M x V
        pass

    def get_loss(self, d, logits, labels):
        _logits = logits.view(-1, logits.size(-1))  # BT x H
        _labels = labels.view(-1)  # BT

        valid = _labels > 0
        loss_cnt = valid.sum()  # = M
        valid_index = valid.nonzero().squeeze()  # M

        valid_logits = _logits[valid_index]  # M x H
        valid_scores = self.get_scores(d, valid_logits)  # M x V
        valid_labels = _labels[valid_index]  # M

        loss = self.ce(valid_scores, valid_labels)
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        return loss, loss_cnt
