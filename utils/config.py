# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import argparse

import yaml


def _val_to_argparse_kwargs(val):
    if isinstance(val, str):
        return {"type": strings_with_none, "default": val}
    elif isinstance(val, bool):
        return {"type": bool_string_to_bool, "default": val}
    else:
        return {"type": eval, "default": val}


def strings_with_none(arg_str):
    if arg_str.lower() in ["null", "none"]:
        return None
    else:
        return str(arg_str)


def bool_string_to_bool(bool_str):
    if str(bool_str).lower() == "false":
        return False
    elif str(bool_str).lower() == "true":
        return True
    else:
        raise argparse.ArgumentTypeError('For boolean args, use "True" or "False" strings, not {}'.format(bool_str))


def yaml_to_parser(yaml_path):
    default_hyperparams = yaml.safe_load(open(yaml_path))
    parser = argparse.ArgumentParser()

    for k, v in default_hyperparams.items():
        argparse_kwargs = _val_to_argparse_kwargs(v)
        parser.add_argument("--{}".format(k.replace("_", "-")), **argparse_kwargs)
    return parser
