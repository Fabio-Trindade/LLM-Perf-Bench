
def get_cls_from_str(factory: object, var_name: str, str_value: str):
        attr = getattr(factory, var_name)
        cls  = attr.get(str_value)
        if cls is None:
            raise ValueError(f"Unknown registry '{str_value}' in {factory}")
        return cls