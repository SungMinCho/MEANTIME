import yaml

import os
from copy import deepcopy


def set_template(conf):
    given = deepcopy(conf)
    for template_name in conf['templates']:
        set_single_template(conf, template_name)  # overwrite by templates (ascending priority)
    overwrite_with_nonnones(conf, given)  # apply given(non-nones) last (highest priority)


def set_single_template(conf, template_name):
    template = load_template(template_name)
    overwrite(conf, template)


def load_template(template_name):
    return yaml.safe_load(open(os.path.join('templates', f'{template_name}.yaml')))


def overwrite(this_dict, other_dict):
    for k, v in other_dict.items():
        if isinstance(v, dict):
            overwrite(this_dict[k], v)
        else:
            this_dict[k] = v


def overwrite_with_nonnones(this_dict, other_dict):
    for k, v in other_dict.items():
        if isinstance(v, dict):
            overwrite_with_nonnones(this_dict[k], v)
        elif v is not None:
            this_dict[k] = v