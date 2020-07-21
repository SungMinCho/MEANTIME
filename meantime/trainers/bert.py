from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        # loss = self.model(batch, loss=True)
        # loss = loss.mean()
        # return loss
        d = self.model(batch)
        loss, loss_cnt = d['loss'], d['loss_cnt']
        loss = (loss * loss_cnt).sum() / loss_cnt.sum()
        return loss

    def calculate_metrics(self, batch):
        labels = batch['labels']
        scores = self.model(batch)['scores']  # B x C
        # scores = scores.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics
