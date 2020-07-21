from meantime.models.base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class MARankModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        hidden = args.hidden_units
        dropout = args.dropout
        num_att_layers = args.marank_num_att_layers
        num_linear_layers = args.marank_num_linear_layers

        self.user_latent_embeddings = nn.Embedding(args.num_users+1, hidden)
        self.item_latent_embeddings = nn.Embedding(args.num_items+1, hidden)
        self.item_context_embeddings = nn.Embedding(args.num_items+1, hidden)

        self.num_att_layers = num_att_layers
        self.num_linear_layers = num_linear_layers

        self.ures_linear_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(self.num_att_layers)])
        self.ires_linear_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(self.num_att_layers)])
        self.res_linear_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(self.num_linear_layers)])

        self.i_att_item = nn.Linear(hidden, hidden)
        self.i_att_user = nn.Linear(hidden, hidden)
        self.i_att_out = nn.Linear(hidden, 1)

        self.rel_att_item = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(self.num_att_layers)])
        self.rel_att_user = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(self.num_att_layers)])
        self.rel_att_out = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(self.num_att_layers)])

        self.hid_att = nn.Linear(hidden, hidden)
        self.hid_att_out = nn.Linear(hidden, 1)

        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()  # TODO: add inits for MARankModel's components

    @classmethod
    def code(cls):
        return 'marank'

    def forward(self, d):
        logits = self.get_logits(d)  # B x H
        ret = {}
        if self.training:
            loss, loss_cnt = self.get_loss(logits, d)
            ret['loss'] = loss
            ret['loss_cnt'] = loss_cnt
        else:
            scores = self.get_scores(logits, d)  # B x V
            ret['scores'] = scores  # B x V
        return ret

    def get_logits(self, d):
        users = d['users']  # B x 1
        users = users.squeeze()  # B
        local_contexts = d['tokens']  # B x T

        user_latent = self.user_latent_embeddings(users)  # B x H

        iloc_embs = self.item_context_embeddings(local_contexts)  # B x T x H
        iloc_att_embs = self.itemContAtt(iloc_embs, user_latent)  # B x H

        uhid_units = self.res_nn(user_latent, self.ures_linear_layers)  # L of B x H
        ihid_units = self.res_nn(iloc_embs, self.ires_linear_layers)  # L of B x T x H
        uihid_atts = self.multRelAtt(uhid_units, ihid_units)  # L of B x 1 x H

        iloc_att_embs = iloc_att_embs.unsqueeze(1)  # B x 1 x H
        uihid_atts.append(iloc_att_embs)  # (L+1) of (B x 1 x H)
        uihid_att_embs = torch.cat(uihid_atts, 1)  # B x (L+1) x H
        uihid_agg_emb = self.hidAtt(uihid_att_embs)  # B x H
        aggre_embs = self.res_nn(uihid_agg_emb, self.res_linear_layers)[-1]  # B x H
        neu_embs = aggre_embs + uihid_agg_emb  # B x H

        return neu_embs

    def get_loss(self, logits, d):
        # logits : B x H
        users = d['users']  # B x 1
        users = users.squeeze()  # B
        positive_items, negative_items = d['labels'], d['negative_labels']  # B x 1
        positive_items = positive_items.squeeze()  # B
        negative_items = negative_items.squeeze()  # B

        user_latent = self.user_latent_embeddings(users)  # B x H
        pi_latent = self.item_latent_embeddings(positive_items)  # B x H
        ni_latent = self.item_latent_embeddings(negative_items)  # B x H
        p_latent = self.item_context_embeddings(positive_items)  # B x H
        n_latent = self.item_context_embeddings(negative_items)  # B x H

        diff1 = (logits * (p_latent - n_latent)).sum(1)  # B  # short term loss
        diff2 = (user_latent * (pi_latent - ni_latent)).sum(1)  # B  # long term loss
        diff = diff1 + diff2  # B

        loss = -torch.log(torch.sigmoid(diff))
        loss = loss.mean()
        loss_cnt = torch.ones_like(users).sum()  # B
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        return loss, loss_cnt

    def get_scores(self, logits, d):
        # logits : B x H
        candidates_context = self.item_context_embeddings(d['candidates'])  # B x C x H
        scores1 = (logits.unsqueeze(1) * candidates_context).sum(-1)  # B x C

        users = d['users']
        users = users.squeeze()  # B
        user_latent = self.user_latent_embeddings(users)  # B x H
        candidates_latent = self.item_latent_embeddings(d['candidates'])  # B x C x H
        scores2 = (user_latent.unsqueeze(1) * candidates_latent).sum(-1)  # B x C

        scores = scores1 + scores2
        return scores  # B x C

    def res_nn(self, aggre_embs, linear_layers):
        out = aggre_embs
        hid_units = []
        for lin in linear_layers:
            pre = out
            out = lin(self.dropout(out))
            out = F.relu(out + pre)
            hid_units.append(out)
        return hid_units

    def itemContAtt(self, item_embs, user_embs):
        # item_embs : B x T x H
        # user_embs : B x H
        # returns : B x H

        drop_item_embs = self.dropout(item_embs)  # B x T x H
        wi = self.i_att_item(drop_item_embs)  # B x T x H

        drop_user_embs = self.dropout(user_embs)
        wu = self.i_att_user(drop_user_embs).unsqueeze(1)  # B x 1 x H

        w = torch.tanh(wi + wu)  # B x T x H
        w = self.i_att_out(w)  # B x T x 1
        outs = torch.sigmoid(w)  # B x T x 1

        outs = outs.transpose(1, 2).softmax(-1)  # B x 1 x T
        outs = torch.matmul(outs, item_embs)  # B x 1 x H
        outs = outs + item_embs[:,-1,:].unsqueeze(1)  # B x 1 x H
        return outs.squeeze()  # B x H

    def multRelAtt(self, uhid_units, ihid_units):
        # uhid_units : L of B x H
        # ihid_units : L of B x T x H
        hid_atts = [None for _ in range(self.num_att_layers)]

        for i in range(self.num_att_layers):
            drop_item_embs = self.dropout(ihid_units[i])  # B x T x H
            wi = self.rel_att_item[i](drop_item_embs)  # B x T x H

            drop_user_embs = self.dropout(uhid_units[i])  # B x H
            wu = self.rel_att_user[i](drop_user_embs).unsqueeze(1)  # B x 1 x H

            w = torch.tanh(wi + wu)  # B x T x H
            w = self.rel_att_out[i](w)   # B x T x 1
            outs = torch.sigmoid(w)  # B x T x 1

            outs = outs.transpose(1, 2).softmax(-1)  # B x 1 x T
            outs = torch.matmul(outs, ihid_units[i])  # B x 1 x H

            hid_atts[i] = outs
        return hid_atts

    def hidAtt(self, hidden_embs):
        # hidden_embs : B x L x H  (L is actually L+1 but we write L for the sake of convenience)
        drop_hid_embs = self.dropout(hidden_embs)  # B x L x H

        w = torch.tanh(self.hid_att(drop_hid_embs))  # B x L x H
        w = self.hid_att_out(w)  # B x L x 1
        outs = torch.sigmoid(w)  # B x L x 1

        outs = outs.transpose(1, 2).softmax(-1)  # B x 1 x L
        outs = torch.matmul(outs, hidden_embs)  # B x 1 x H
        return outs.squeeze()  # B x H