import argparse

def add_arg(parser: argparse.ArgumentParser, name, fixed_values, **kwargs):
    if name not in fixed_values:
        parser.add_argument(f"--{name}", **kwargs)
    else:
        kwargs["default"] = fixed_values[name]
        parser.add_argument(f"--{name}", **kwargs)

def get_fixed_values(fixed_values):
    return  fixed_values or {}

