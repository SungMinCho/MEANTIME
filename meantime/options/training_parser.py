from meantime.datasets import DATASETS
from meantime.dataloaders import DATALOADERS
from meantime.models import MODELS
from meantime.trainers import TRAINERS
from meantime.dataloaders.negative_samplers import NEGATIVE_SAMPLERS
from meantime.options.set_template import set_template
from meantime.utils import str2bool
from meantime.config import LOCAL_DATA_FOLDER

import argparse


class TrainingParser:
    def __init__(self, sys_argv):
        self.sys_argv = sys_argv

    def parse(self):
        conf = {}
        conf.update(self.parse_top())
        conf.update(self.parse_dataset())
        conf.update(self.parse_dataloader())
        conf.update(self.parse_negative_sampler())
        conf.update(self.parse_trainer())
        conf.update(self.parse_model())
        conf.update(self.parse_experiment())
        conf.update(self.parse_wandb())

        set_template(conf)
        self.post_process(conf)
        return conf

    @staticmethod
    def post_process(conf):
        if conf['model_code'] == 'meantime':
            conf['dataloader_output_timestamp'] = len(conf['relative_kernel_types']) > 0
            conf['time_unit_divide'] = 1
            conf['dataloader_output_days'] = 'd' in conf['absolute_kernel_types']

    def parse_top(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--templates', nargs='+', type=str, default=[])
        parser.add_argument('--mode', type=str, choices=['train', 'validate', 'test'], help='Determines whether to train/validate/test the model')
        parser.add_argument('--resume_training', type=str2bool, default=False, help='Whether to resume training from the last checkpoint')
        parser.add_argument('--pilot', type=str2bool, default=False, help='If true, run the program in minimal amount to check for errors')
        parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained weights')
        parser.add_argument('--num_users', type=int, help='Number of users in the dataset. Its value is dynamically determined in dataloader')
        parser.add_argument('--num_items', type=int, help='Number of items in the dataset. Its value is dynamically determined in dataloader')
        parser.add_argument('--num_ratings', type=int, help='Number of possible ratings in the dataset. Its value is dynamically determined in dataloader')
        parser.add_argument('--num_days', type=int, help='Number of possible days in the dataset. Its value is dynamically determined in dataloader')
        parser.add_argument('--local_data_folder', type=str, help='Folder that contains raw/preprocessed data', default=LOCAL_DATA_FOLDER)

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_dataset(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--dataset_code', type=str, choices=DATASETS.keys(), help='Selects the dataset to use for the experiment')
        parser.add_argument('--min_rating', type=int, help='Minimum rating to regard as implicit rating. Interactions whose rating is below this value will be discarded')
        parser.add_argument('--min_uc', type=int, help='Discard users whose number of ratings is below this value')
        parser.add_argument('--min_sc', type=int, help='Discard items whose number of ratings is below this value')
        parser.add_argument('--split', type=str, choices=['leave_one_out'], help='How to split the dataset')

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_dataloader(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--dataloader_code', type=str, choices=DATALOADERS.keys(), help='Selects the dataloader to use for the experiment')
        parser.add_argument('--dataloader_random_seed', type=float, help='Random seed to initialize the random state of the dataloader')
        parser.add_argument('--train_batch_size', type=int, help='Batch size for training')
        parser.add_argument('--val_batch_size', type=int, help='Batch size for validation')
        parser.add_argument('--test_batch_size', type=int, help='Batch size for test')
        parser.add_argument('--train_window', type=int, help="How much to slide the training window to obtain subsequences from the user's entire item sequence")
        parser.add_argument('--dataloader_output_timestamp', type=str2bool, help='If true, the dataloader outputs timestamp information')
        parser.add_argument('--dataloader_output_days', type=str2bool, help='If true, the dataloader outputs day information')
        parser.add_argument('--dataloader_output_user', type=str2bool, help='If true, the dataloader outputs user information')

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_negative_sampler(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--train_negative_sampler_code', type=str, choices=NEGATIVE_SAMPLERS.keys(), help='Selects negative sampler for training')
        parser.add_argument('--train_negative_sample_size', type=int, help='Negative sample size for training')
        parser.add_argument('--train_negative_sampling_seed', type=int, help='Seed to fix the random state of negative sampler for training')
        parser.add_argument('--test_negative_sampler_code', type=str, choices=NEGATIVE_SAMPLERS.keys(), help='Selects negative sampler for testing')
        parser.add_argument('--test_negative_sample_size', type=int, help='Negative sample size for testing')
        parser.add_argument('--test_negative_sampling_seed', type=int, help='Seed to fix the random state of negative sampler for testing')

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_trainer(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--trainer_code', type=str, choices=TRAINERS.keys(), help='Selects the trainer for the experiment')
        parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])
        parser.add_argument('--use_parallel', type=str2bool, help='If true, the program uses all visible cuda devices with DataParallel')
        parser.add_argument('--num_workers', type=int)
        # optimizer #
        parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'])
        parser.add_argument('--lr', type=float, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, help='l2 regularization')
        parser.add_argument('--momentum', type=float, help='SGD momentum')
        # lr scheduler #
        parser.add_argument('--decay_step', type=int, help='Decay step for StepLR')
        parser.add_argument('--gamma', type=float, help='Gamma for StepLR')
        # clip grad norm #
        parser.add_argument('--clip_grad_norm', type=float)
        # epochs #
        parser.add_argument('--num_epochs', type=int, help='Maximum number of epochs to run. Training will terminate early if saturation point is found. If you want to never stop until saturation, give num_epochs=-1')
        # logger #
        parser.add_argument('--log_period_as_iter', type=int, help='Will log every log_period_as_iter')
        # evaluation #
        parser.add_argument('--metric_ks', nargs='+', type=int, help='list of k for NDCG@k and Recall@k')
        parser.add_argument('--best_metric', type=str, help='This metric will be used to compare and determine the best model')
        # saturation wait epochs
        parser.add_argument('--saturation_wait_epochs', type=int, help="If validation performance doesn't improve for this number of epochs, the training will stop")

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_model(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--model_code', type=str, choices=MODELS.keys(), help='Selects the model for the experiment')
        parser.add_argument('--model_init_seed', type=int, help='Seed used to initialize the model parameters')
        parser.add_argument('--model_init_range', type=float, help='Range used to initialize the model parameters')
        # BERT #
        parser.add_argument('--max_len', type=int, help='Length of the transformer model')
        parser.add_argument('--hidden_units', type=int, help='Hidden dimension size')
        parser.add_argument('--num_blocks', type=int, help='Number of transformer layers')
        parser.add_argument('--num_heads', type=int, help='Number of attention heads')
        parser.add_argument('--dropout', type=float, help='Dropout probability')
        parser.add_argument('--mask_prob', type=float, help='Mask probability for bert training sequences')
        parser.add_argument('--output_info', type=str2bool, help='If true, the transformer also returns extra information such as attention scores')
        parser.add_argument('--residual_ln_type', type=str, choices=['post', 'pre'], help='Two variations exist regarding the order of layer normalization in residual connections in transformers')
        parser.add_argument('--headtype', type=str, choices=['linear', 'dot'], help='Two types of prediction heads on top of transformers')
        parser.add_argument('--head_use_ln', type=str2bool, help='If true, the prediction head also uses layer normalization')
        # Bert timestamp
        parser.add_argument('--time_unit_divide', type=int, help='The timestamp difference is divided by this value')
        parser.add_argument('--freq', type=int, default=10000, help='freq hyperparameter used in temporal embeddings')
        # TiSas
        parser.add_argument('--tisas_max_time_intervals', type=int, help='Maximum time interval to consider in tisas')
        # MARank
        parser.add_argument('--marank_max_len', type=int, help='Needs separate max_len for marank because original max_len is used for dataloader. (this is kind of hackish and needs a fix i know)')
        parser.add_argument('--marank_num_att_layers', type=int, help='Number of attention layers for MARank')
        parser.add_argument('--marank_num_linear_layers', type=int, help='Number of linear layers for MARank')
        # MEANTIME
        parser.add_argument('--absolute_kernel_types', type=str, help="Absolute kernel types separated by'-'(e.g. d-c). p=Pos, d=Day, c=Con")
        parser.add_argument('--relative_kernel_types', type=str, help="Relative kernel types separated by'-'(e.g. e-l). s=Sin, e=Exp, l=Log")

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_experiment(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--experiment_root', type=str, default='experiments', help='Root folder of all experiments')
        parser.add_argument('--experiment_group', type=str, default='test', help='Group folder inside Root folder')
        parser.add_argument('--experiment_name', type=str, help='Name for this particular experiment inside Group folder')

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_wandb(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--wandb_project_name', type=str, help='Project name for wandb, if wandb is used')
        parser.add_argument('--wandb_run_name', type=str, help='Run name for wandb, if wandb is used')
        parser.add_argument('--wandb_run_id', type=str, help='Run id for wandb, if wandb is used')

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)