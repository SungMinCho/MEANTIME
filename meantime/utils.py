from meantime.config import *
from meantime.communicator import Communicator

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim as optim

import json
import os
import shutil
import random
import pkgutil
from importlib import import_module
import inspect
import sys
import argparse
import filecmp


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def import_all_subclasses(_file, _name, _class):
    modules = get_all_submodules(_file, _name)
    for m in modules:
        for i in dir(m):
            attribute = getattr(m, i)
            if inspect.isclass(attribute) and issubclass(attribute, _class):
                setattr(sys.modules[_name], i, attribute)


def get_all_submodules(_file, _name):
    modules = []
    _dir = os.path.dirname(_file)
    for _, name, ispkg in pkgutil.iter_modules([_dir]):
        module = import_module('.' + name, package=_name)
        modules.append(module)
        if ispkg:
            modules.extend(get_all_submodules(module.__file__, module.__name__))
    return modules


def setup_train(args, machine_is_host=False):
    # set_up_gpu(args)
    exp_root, exp_group, exp_name = args.experiment_root, args.experiment_group, args.experiment_name
    assert exp_name is not None
    local_export_root = os.path.join(exp_root, exp_group, exp_name)

    if machine_is_host:
        remote_export_root = None
        communicator = None
        if os.path.exists(local_export_root):
            if exp_group == 'test':
                print('Removing local test export root {}'.format(local_export_root))
                shutil.rmtree(local_export_root)
            else:
                print('Local export root exists {}'.format(local_export_root))
                exit(0)
        create_local_export_root(args, local_export_root)
        export_config(args, local_export_root)
        print('Export root', local_export_root)
    else:
        remote_export_root = os.path.join(REMOTE_ROOT, local_export_root)
        communicator = Communicator(HOST, PORT, USERNAME, PASSWORD)
        created = communicator.create_dir(remote_dir_path=remote_export_root)
        if not created:
            if os.path.exists(local_export_root):
                print('Local export root exists while checking status')
                exit(0)
            os.makedirs(local_export_root)
            local_status = os.path.join(local_export_root, 'status.txt')
            remote_status = os.path.join(remote_export_root, 'status.txt')
            try:
                communicator.sftp.get(remote_status, local_status)
                status = open(local_status).readline()
            except Exception:
                status = 'failed to download status'
            print('Checking status')
            if status == STATUS_RECOVERY:
                print('Status is recovery')
                with open(local_status, 'w') as f:
                    f.write('running\n')
                print("Write 'running' on remote")
                communicator.sftp.put(local_status, remote_status)
                print('Downloading remote export root')
                communicator.download_dir(remote_export_root, local_export_root)
                args.resume_training = True
            else:
                print('Status is not recovery')
                shutil.rmtree(local_export_root)
                print('Remote export root {} exists. Existing'.format(remote_export_root))
                exit(0)
        else:
            print('Created export_root={} in remote'.format(remote_export_root))
        create_local_export_root(args, local_export_root)
        export_config(args, local_export_root)
        print('Export root', local_export_root)
    return local_export_root, remote_export_root, communicator


def create_local_export_root(args, local_export_root):
    experiment_dir = os.path.join(args.experiment_root, args.experiment_group)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    if args.resume_training:
        assert os.path.exists(local_export_root)
    else:
        if os.path.exists(local_export_root):
            if args.experiment_group != 'test':
                print('Local export root exists. Existing')
                exit(0)
            else:
                print('Removing already existing test export root')
                shutil.rmtree(local_export_root)
        os.makedirs(local_export_root)


def export_config(args, local_export_root):
    with open(os.path.join(local_export_root, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=2)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def load_pretrained_weights(model, path):
    chk_dict = torch.load(os.path.abspath(path))
    model_state_dict = chk_dict[STATE_DICT_KEY] if STATE_DICT_KEY in chk_dict else chk_dict['state_dict']
    d = {}
    # this is for stupid reason
    for k, v in model_state_dict.items():
        if k.startswith('model.'):
            d[k[6:]] = v
        else:
            d[k] = v
    model_state_dict = d
    model.load_state_dict(model_state_dict)


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def sameDirs(dir1, dir2):
    cmp = filecmp.dircmp(dir1, dir2)
    return sameDirsAux(cmp)


def sameDirsAux(cmp):
    return len(cmp.diff_files) == 0 and all(sameDirsAux(c) for c in cmp.subdirs.values())