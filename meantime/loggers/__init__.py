from meantime.config import RECENT_STATE_DICT_FILENAME, BEST_STATE_DICT_FILENAME, USE_WANDB

import torch
import wandb
import pandas as pd

import os
from abc import ABCMeta, abstractmethod
from pathlib import Path


def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))


class LoggerService(object):
    def __init__(self, args, train_loggers=None, val_loggers=None, test_loggers=None):
        self.train_loggers = train_loggers if train_loggers else []
        self.val_loggers = val_loggers if val_loggers else []
        self.test_loggers = test_loggers if test_loggers else []

        if args.pilot:
            return

        if USE_WANDB:
            project_name = args.wandb_project_name
            run_name = args.wandb_run_name
            run_id = args.wandb_run_id
            resume_training = args.resume_training

            assert project_name is not None and run_name is not None and run_id is not None
            wandb.init(project=project_name, name=run_name, config=args, id=run_id, resume=resume_training)

    def complete(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.val_loggers:
            logger.complete(**log_data)
        for logger in self.test_loggers:
            logger.complete(**log_data)

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_val(self, log_data):
        for logger in self.val_loggers:
            logger.log(**log_data)

    def log_test(self, log_data):
        for logger in self.test_loggers:
            logger.log(**log_data)


class AbstractBaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass


class RecentModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, filename=RECENT_STATE_DICT_FILENAME):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs['state_dict']
            state_dict['epoch'] = kwargs['epoch']
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename + '.final')


class BestModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, metric_key='NDCG@10', filename=BEST_STATE_DICT_FILENAME):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.best_metric = 0.
        self.metric_key = metric_key
        self.filename = filename

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:
            print("Update Best {} Model at {}".format(self.metric_key, kwargs['epoch']))
            self.best_metric = current_metric
            save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename)

    def filepath(self):
        return os.path.join(self.checkpoint_path, self.filename)


class WandbLogger(AbstractBaseLogger):
    def __init__(self, table_definitions=[], prefix=''):
        self.table_loggers = []
        for table_name, table_columns in table_definitions:
            self.table_loggers.append(WandbTableLogger(table_name, table_columns))
        self.prefix = prefix

    def log(self, *args, **kwargs):
        for table_logger in self.table_loggers:
            table_logger.log(**kwargs)

        step = kwargs['accum_iter']
        log_dict = {}
        for k, v in kwargs.items():
            if k == 'accum_iter' or 'state' in k:
                continue
            elif k == 'epoch':
                log_dict[k] = v
            else:
                log_dict[self.prefix + k] = v
        wandb.log(log_dict, step=step)

    def complete(self, *args, **kwargs):
        for table_logger in self.table_loggers:
            table_logger.complete(**kwargs)
        wandb.log({})  # so that the last log is not missing


class WandbTableLogger(AbstractBaseLogger):
    def __init__(self, table_name, table_columns):
        self.table_name = table_name
        self.table_columns = table_columns
        self.table_rows = []

    def log(self, *args, **kwargs):
        row = [kwargs[col] for col in self.table_columns]
        self.table_rows.append(row)
        table = wandb.Table(columns=self.table_columns,
                            data=self.table_rows)
        wandb.log({self.table_name: table}, commit=False)  # final commit is done at WandbLogger


class TableLoggersManager(AbstractBaseLogger):
    def __init__(self, args=None, export_root=None, table_definitions=[]):
        self.table_loggers = []
        for table_name, table_columns in table_definitions:
            self.table_loggers.append(TableLogger(args, export_root, table_name, table_columns))

    def log(self, *args, **kwargs):
        for table_logger in self.table_loggers:
            table_logger.log(**kwargs)

    def complete(self, *args, **kwargs):
        for table_logger in self.table_loggers:
            table_logger.complete(**kwargs)


class TableLogger(AbstractBaseLogger):
    def __init__(self, args, export_root, table_name, table_columns):
        self.args = args
        self.filepath = Path(export_root).joinpath('tables').joinpath(table_name + '.csv')
        self.table_name = table_name
        self.table_columns = table_columns
        self.table_rows = []
        if self.args.resume_training:
            self.recover()

    def recover(self):
        if os.path.exists(self.filepath):
            print('Recovering', self.filepath)
            df = pd.read_csv(self.filepath)
            rows = df.values.tolist()
            print('last 2 rows')
            print(rows[-2:])
            self.table_rows = rows

    def log(self, *args, **kwargs):
        row = [kwargs[col] for col in self.table_columns]
        self.table_rows.append(row)
        # save table offline
        if not self.filepath.parent.is_dir():
            self.filepath.parent.mkdir(parents=True)
        df = pd.DataFrame(self.table_rows, columns=self.table_columns)
        df.to_csv(self.filepath, index=False)

    def complete(self, *args, **kwargs):
        # save table offline
        if not self.filepath.parent.is_dir():
            self.filepath.parent.mkdir(parents=True)
        print('saving table to', self.filepath)
        df = pd.DataFrame(self.table_rows, columns=self.table_columns)
        df.to_csv(self.filepath, index=False)
