from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils


class BertDataloader(AbstractDataloader):
    @classmethod
    def code(cls):
        return 'bert'

    def _get_dataset(self, mode):
        if mode == 'train':
            return self._get_train_dataset()
        elif mode == 'val':
            return self._get_eval_dataset('val')
        else:
            return self._get_eval_dataset('test')

    def _get_train_dataset(self):
        train_ranges = self.train_targets
        dataset = BertTrainDataset(self.args, self.dataset, self.train_negative_samples, self.rng, train_ranges)
        return dataset

    def _get_eval_dataset(self, mode):
        positions = self.validation_targets if mode=='val' else self.test_targets
        dataset = BertEvalDataset(self.args, self.dataset, self.test_negative_samples, positions)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, args, dataset, negative_samples, rng, train_ranges):
        self.args = args
        self.user2dict = dataset['user2dict']
        self.users = sorted(self.user2dict.keys())
        self.train_window = args.train_window
        self.max_len = args.max_len
        self.mask_prob = args.mask_prob
        self.special_tokens = dataset['special_tokens']
        self.num_users = len(dataset['umap'])
        self.num_items = len(dataset['smap'])
        self.rng = rng
        self.train_ranges = train_ranges

        self.index2user_and_offsets = self.populate_indices()

        self.output_timestamps = args.dataloader_output_timestamp
        self.output_days = args.dataloader_output_days
        self.output_user = args.dataloader_output_user

        self.negative_samples = negative_samples

    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        return self.rng.setstate(state)

    def populate_indices(self):
        index2user_and_offsets = {}
        i = 0
        T = self.max_len
        W = self.train_window

        # offset is exclusive
        for user, pos in self.train_ranges:
            if W is None or W == 0:
                offsets = [pos]
            else:
                offsets = list(range(pos, T-1, -W))  # pos ~ T
                if len(offsets) == 0:
                    offsets = [pos]
            for offset in offsets:
                index2user_and_offsets[i] = (user, offset)
                i += 1
        return index2user_and_offsets

    def __len__(self):
        return len(self.index2user_and_offsets)

    def __getitem__(self, index):
        user, offset = self.index2user_and_offsets[index]
        seq = self.user2dict[user]['items']
        beg = max(0, offset-self.max_len)
        end = offset  # exclude offset (meant to be)
        seq = seq[beg:end]

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.special_tokens.mask)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        padding_len = self.max_len - len(tokens)
        valid_len = len(tokens)

        tokens = [0] * padding_len + tokens
        labels = [0] * padding_len + labels

        d = {}
        d['tokens'] = torch.LongTensor(tokens)
        d['labels'] = torch.LongTensor(labels)

        if self.output_timestamps:
            timestamps = self.user2dict[user]['timestamps']
            timestamps = timestamps[beg:end]
            timestamps = [0] * padding_len + timestamps
            d['timestamps'] = torch.LongTensor(timestamps)

        if self.output_days:
            days = self.user2dict[user]['days']
            days = days[beg:end]
            days = [0] * padding_len + days
            d['days'] = torch.LongTensor(days)

        if self.output_user:
            d['users'] = torch.LongTensor([user])
        return d


class BertEvalDataset(data_utils.Dataset):
    def __init__(self, args, dataset, negative_samples, positions):
        self.user2dict = dataset['user2dict']
        self.positions = positions
        self.max_len = args.max_len
        self.num_items = len(dataset['smap'])
        self.special_tokens = dataset['special_tokens']
        self.negative_samples = negative_samples

        self.output_timestamps = args.dataloader_output_timestamp
        self.output_days = args.dataloader_output_days
        self.output_user = args.dataloader_output_user

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        user, pos = self.positions[index]
        seq = self.user2dict[user]['items']

        beg = max(0, pos + 1 - self.max_len)
        end = pos + 1
        seq = seq[beg:end]

        negs = self.negative_samples[user]
        answer = [seq[-1]]
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq[-1] = self.special_tokens.mask
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        tokens = torch.LongTensor(seq)
        candidates = torch.LongTensor(candidates)
        labels = torch.LongTensor(labels)
        d = {'tokens':tokens, 'candidates':candidates, 'labels':labels}

        if self.output_timestamps:
            timestamps = self.user2dict[user]['timestamps']
            timestamps = timestamps[beg:end]
            timestamps = [0] * padding_len + timestamps
            d['timestamps'] = torch.LongTensor(timestamps)

        if self.output_days:
            days = self.user2dict[user]['days']
            days = days[beg:end]
            days = [0] * padding_len + days
            d['days'] = torch.LongTensor(days)

        if self.output_user:
            d['users'] = torch.LongTensor([user])
        return d
