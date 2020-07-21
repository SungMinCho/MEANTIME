from meantime.loggers import *
# from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY, TRAIN_LOADER_RNG_STATE_DICT_KEY
from meantime.config import *
from meantime.utils import AverageMeterSet
from meantime.utils import fix_random_seed_as
from meantime.analyze_table import find_saturation_point

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from abc import *
from pathlib import Path
import os


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, local_export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.use_parallel = args.use_parallel
        if self.use_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)
        self.clip_grad_norm = args.clip_grad_norm
        self.epoch_start = 0
        self.best_epoch = self.epoch_start - 1
        self.best_metric_at_best_epoch = -1
        self.accum_iter_start = 0

        self.num_epochs = args.num_epochs
        if self.num_epochs == -1:
            self.num_epochs = 987654321  # Technically Infinite
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.saturation_wait_epochs = args.saturation_wait_epochs

        self.pilot = args.pilot
        if self.pilot:
            self.num_epochs = 1
            self.pilot_batch_cnt = 1

        self.local_export_root = local_export_root
        self.train_loggers, self.val_loggers, self.test_loggers = self._create_loggers() if not self.pilot else (None, None, None)
        self.add_extra_loggers()

        self.logger_service = LoggerService(args, self.train_loggers, self.val_loggers, self.test_loggers)
        self.log_period_as_iter = args.log_period_as_iter

        self.resume_training = args.resume_training
        if self.resume_training:
            print('Restoring previous training state')
            self._restore_training_state()
            print('Finished restoring')

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    def train(self):
        epoch = self.epoch_start
        best_epoch = self.best_epoch
        accum_iter = self.accum_iter_start
        # self.validate(epoch-1, accum_iter, self.val_loader)
        best_metric = self.best_metric_at_best_epoch
        stop_training = False
        for epoch in range(self.epoch_start, self.num_epochs):
            if self.pilot:
                print('epoch', epoch)
            fix_random_seed_as(epoch)  # fix random seed at every epoch to make it perfectly resumable
            accum_iter = self.train_one_epoch(epoch, accum_iter, self.train_loader)
            self.lr_scheduler.step()  # step before val because state_dict is saved at val. it doesn't affect val result

            val_log_data = self.validate(epoch, accum_iter, mode='val')
            metric = val_log_data[self.best_metric]
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
            elif (self.saturation_wait_epochs is not None) and\
                    (epoch - best_epoch >= self.saturation_wait_epochs):
                stop_training = True  # stop training if val perf doesn't improve for saturation_wait_epochs

            if stop_training:
                # load best model
                best_model_logger = self.val_loggers[-1]
                assert isinstance(best_model_logger, BestModelLogger)
                weight_path = best_model_logger.filepath()
                if self.use_parallel:
                    self.model.module.load(weight_path)
                else:
                    self.model.load(weight_path)
                # self.validate(epoch, accum_iter, mode='test')  # test result at best model
                self.validate(best_epoch, accum_iter, mode='test')  # test result at best model
                break

        self.logger_service.complete({
            'state_dict': (self._create_state_dict(epoch, accum_iter)),
        })

    def just_validate(self, mode):
        dummy_epoch, dummy_accum_iter = 0, 0
        self.validate(dummy_epoch, dummy_accum_iter, mode)

    def train_one_epoch(self, epoch, accum_iter, train_loader, **kwargs):
        self.model.train()

        average_meter_set = AverageMeterSet()
        num_instance = 0
        tqdm_dataloader = tqdm(train_loader) if not self.pilot else train_loader

        for batch_idx, batch in enumerate(tqdm_dataloader):
            if self.pilot and batch_idx >= self.pilot_batch_cnt:
                # print('Break training due to pilot mode')
                break
            batch_size = next(iter(batch.values())).size(0)
            batch = {k:v.to(self.device) for k, v in batch.items()}
            num_instance += batch_size

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            if isinstance(loss, tuple):
                loss, extra_info = loss
                for k, v in extra_info.items():
                    average_meter_set.update(k, v)
            loss.backward()

            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            average_meter_set.update('loss', loss.item())
            if not self.pilot:
                tqdm_dataloader.set_description(
                    'Epoch {}, loss {:.3f} '.format(epoch, average_meter_set['loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                if not self.pilot:
                    tqdm_dataloader.set_description('Logging')
                log_data = {
                    # 'state_dict': (self._create_state_dict()),
                    'epoch': epoch,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                log_data.update(kwargs)
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        log_data = {
            # 'state_dict': (self._create_state_dict()),
            'epoch': epoch,
            'accum_iter': accum_iter,
            'num_train_instance': num_instance,
        }
        log_data.update(average_meter_set.averages())
        log_data.update(kwargs)
        self.log_extra_train_info(log_data)
        self.logger_service.log_train(log_data)
        return accum_iter

    def validate(self, epoch, accum_iter, mode, doLog=True, **kwargs):
        if mode == 'val':
            loader = self.val_loader
        elif mode == 'test':
            loader = self.test_loader
        else:
            raise ValueError

        self.model.eval()

        average_meter_set = AverageMeterSet()
        num_instance = 0

        with torch.no_grad():
            tqdm_dataloader = tqdm(loader) if not self.pilot else loader
            for batch_idx, batch in enumerate(tqdm_dataloader):
                if self.pilot and batch_idx >= self.pilot_batch_cnt:
                    # print('Break validation due to pilot mode')
                    break
                batch = {k:v.to(self.device) for k, v in batch.items()}
                batch_size = next(iter(batch.values())).size(0)
                num_instance += batch_size

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                if not self.pilot:
                    description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                          ['Recall@%d' % k for k in self.metric_ks[:3]]
                    description = '{}: '.format(mode.capitalize()) + ', '.join(s + ' {:.3f}' for s in description_metrics)
                    description = description.replace('NDCG', 'N').replace('Recall', 'R')
                    description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                    tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict(epoch, accum_iter)),
                'epoch': epoch,
                'accum_iter': accum_iter,
                'num_eval_instance': num_instance,
            }
            log_data.update(average_meter_set.averages())
            log_data.update(kwargs)
            if doLog:
                if mode == 'val':
                    self.logger_service.log_val(log_data)
                elif mode == 'test':
                    self.logger_service.log_test(log_data)
                else:
                    raise ValueError
        return log_data

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def _create_loggers(self):
        train_table_definitions = [
            ('train_log', ['epoch', 'loss'])
        ]
        val_table_definitions = [
            ('val_log', ['epoch'] + \
             ['NDCG@%d' % k for k in self.metric_ks] +
             ['Recall@%d' % k for k in self.metric_ks]),
        ]
        test_table_definitions = [
            ('test_log', ['epoch'] + \
             ['NDCG@%d' % k for k in self.metric_ks] +
             ['Recall@%d' % k for k in self.metric_ks]),
        ]

        train_loggers = [TableLoggersManager(args=self.args, export_root=self.local_export_root, table_definitions=train_table_definitions)]
        val_loggers = [TableLoggersManager(args=self.args, export_root=self.local_export_root, table_definitions=val_table_definitions)]
        test_loggers = [TableLoggersManager(args=self.args, export_root=self.local_export_root, table_definitions=test_table_definitions)]

        if self.local_export_root is not None:
            root = Path(self.local_export_root)
            model_checkpoint = root.joinpath('models')
            val_loggers.append(RecentModelLogger(model_checkpoint))
            val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))

        if USE_WANDB:
            train_loggers.append(WandbLogger(table_definitions=train_table_definitions))
            val_loggers.append(WandbLogger(table_definitions=val_table_definitions, prefix='val_'))
            test_loggers.append(WandbLogger(table_definitions=test_table_definitions, prefix='test_'))

        return train_loggers, val_loggers, test_loggers

    def _create_state_dict(self, epoch, accum_iter):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.use_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
            SCHEDULER_STATE_DICT_KEY: self.lr_scheduler.state_dict(),
            TRAIN_LOADER_DATASET_RNG_STATE_DICT_KEY: self.train_loader.dataset.get_rng_state(),
            TRAIN_LOADER_SAMPLER_RNG_STATE_DICT_KEY: self.train_loader.sampler.get_rng_state(),
            STEPS_DICT_KEY: (epoch, accum_iter),
        }

    def _restore_best_state(self):
        ### restore best epoch
        df_path = os.path.join(self.local_export_root, 'tables', 'val_log.csv')
        df = pd.read_csv(df_path)
        sat, reached_end = find_saturation_point(df, self.saturation_wait_epochs, display=False)
        e = sat['epoch'].iloc[0]
        self.best_epoch = e
        print('Restored best epoch:', self.best_epoch)

        ###
        state_dict_path = os.path.join(self.local_export_root, 'models', BEST_STATE_DICT_FILENAME)
        chk_dict = torch.load(os.path.abspath(state_dict_path))

        ### sanity check
        _e, _ = chk_dict[STEPS_DICT_KEY]
        assert e == _e

        ### load weights
        d = chk_dict[STATE_DICT_KEY]
        model_state_dict = {}
        # this is for stupid reason
        for k, v in d.items():
            if k.startswith('model.'):
                model_state_dict[k[6:]] = v
            else:
                model_state_dict[k] = v
        if self.use_parallel:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        ### restore best metric
        val_log_data = self.validate(0, 0, mode='val', doLog=False)
        metric = val_log_data[self.best_metric]
        self.best_metric_at_best_epoch = metric
        print('Restored best metric:', self.best_metric_at_best_epoch)

    def _restore_training_state(self):
        self._restore_best_state()

        ###
        state_dict_path = os.path.join(self.local_export_root, 'models', RECENT_STATE_DICT_FILENAME)
        chk_dict = torch.load(os.path.abspath(state_dict_path))

        ### restore epoch, accum_iter
        epoch, accum_iter = chk_dict[STEPS_DICT_KEY]
        self.epoch_start = epoch + 1
        self.accum_iter_start = accum_iter

        ### restore train dataloader rngs
        train_loader_dataset_rng_state = chk_dict[TRAIN_LOADER_DATASET_RNG_STATE_DICT_KEY]
        self.train_loader.dataset.set_rng_state(train_loader_dataset_rng_state)
        train_loader_sampler_rng_state = chk_dict[TRAIN_LOADER_SAMPLER_RNG_STATE_DICT_KEY]
        self.train_loader.sampler.set_rng_state(train_loader_sampler_rng_state)

        ### restore model
        d = chk_dict[STATE_DICT_KEY]
        model_state_dict = {}
        # this is for stupid reason
        for k, v in d.items():
            if k.startswith('model.'):
                model_state_dict[k[6:]] = v
            else:
                model_state_dict[k] = v
        if self.use_parallel:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        ### restore optimizer
        self.optimizer.load_state_dict(chk_dict[OPTIMIZER_STATE_DICT_KEY])

        ### restore lr_scheduler
        self.lr_scheduler.load_state_dict(chk_dict[SCHEDULER_STATE_DICT_KEY])

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
