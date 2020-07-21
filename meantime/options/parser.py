from meantime.options.training_parser import TrainingParser

import argparse


class Parser:
    def __init__(self, sys_argv):
        self.sys_argv = sys_argv

    def parse(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--meta', type=str, choices=['training'], default='training')

        args = parser.parse_known_args(self.sys_argv)[0]
        meta = args.meta
        if meta == 'training':
            conf = TrainingParser(self.sys_argv).parse()
        else:
            raise ValueError
        conf['meta'] = meta
        return conf