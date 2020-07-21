from meantime.options import parse_args
from meantime.top.training import main as training_main

from dotmap import DotMap

import sys
from typing import List


def main(sys_argv: List[str] = None):
    if sys_argv is None:
        sys_argv = sys.argv[1:]
    conf = parse_args(sys_argv)
    args = DotMap(conf, _dynamic=False)
    if args.meta == 'training':
        training_main(args)
    else:
        raise ValueError
