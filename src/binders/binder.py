import types
from src.utils.util_parse import add_arg, get_fixed_values


class Binder:
    @staticmethod
    def bind_dict_leaves_as_var(data: dict):
        def decorator(cls_to_bind):
            def recurse(value):
                if isinstance(value, dict):
                    for v in value.values():
                        recurse(v)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            setattr(cls_to_bind, item, item)
            return recurse(data)
        return decorator
    
    @staticmethod
    def bind_list_as_vars(data: list):
        def decorator(cls_to_bind):
            def recurse(data):
                for value in data:    
                    if isinstance(value, list):
                        recurse(value)
                    else:
                        attr = getattr(cls_to_bind, value, None)
                        if attr is None:
                            setattr(cls_to_bind, value, value)
                return cls_to_bind
            return recurse(data)
        return decorator

    @staticmethod
    def create_parse_from_config(config):
        def decorator(target):
            def apply_arguments(parser, fixed_values):
                fixed_values = get_fixed_values(fixed_values)
                for var in config.var_names:
                    values = getattr(config,var)
                    add_arg(
                        parser, values.name, fixed_values,
                        type=values.type, default=values.default_value,
                        help=values.description, choices=values.choices,
                        nargs = values.nargs
                    )
            if isinstance(target, type):
                def __init__(self, parser, fixed_values):
                    apply_arguments(parser, fixed_values)
                setattr(target, "__init__", __init__)
                return target
            elif isinstance(target, types.FunctionType): 
                def wrapper(parser, fixed_values):
                    apply_arguments(parser, fixed_values)
                wrapper.__name__ = target.__name__
                return wrapper
            else:
                raise TypeError("Unsupported type")
        return decorator

        