from meantime.options.parser import Parser


def parse_args(sys_argv):
    return Parser(sys_argv).parse()
