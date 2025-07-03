class ParserRegistry:
    _registry = set()

    @classmethod
    def registry(cls):
        def decorator(cls_or_func):
            name = cls_or_func.__name__
            if name in cls._registry:
                raise RuntimeError(f"Duplicated {name} in ParserRegistry")
            cls._registry.add(cls_or_func)
            return cls_or_func
        return decorator