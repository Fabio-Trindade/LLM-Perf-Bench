from enum import Enum
import inspect

class EnumRegistryBinder:
    _create_prefix = "create"
    @staticmethod
    def format_func_name(prefix: str, component_name:str):
        return f"{prefix}_{component_name.lower()}"
    @staticmethod
    def format_create_func_name(component_name:str):
        return f"{EnumRegistryBinder._create_prefix}_{component_name.lower()}"
    
    @staticmethod
    def bind_method_by_enum_cls(cls, enum_cls:Enum, make_func, prefix:str):
         for component in enum_cls:
            component_name = component.value
            setattr(cls, EnumRegistryBinder.format_func_name(prefix, component_name) , make_func(component_name))

    @staticmethod
    def dyn_bind_register_method(cls, enum_cls: Enum):
        def make_register_func(c_name):
            @classmethod
            def register(cls, typename):
                return cls.register_cls(c_name, typename)
            return register
        EnumRegistryBinder.bind_method_by_enum_cls(cls,enum_cls, make_register_func, "register")
    
    @staticmethod
    def dyn_bind_create_method(cls, enum_cls: Enum):        
        def make_create_func(comp_name):
            @classmethod
            def create(cls, typename, *args, **kwargs):
                return cls.create_cls(comp_name, typename, *args, **kwargs)
            return create
        
        EnumRegistryBinder.bind_method_by_enum_cls(cls,enum_cls, make_create_func, EnumRegistryBinder._create_prefix)
    
    @staticmethod
    def bind_all_methods_to_class_from_enum(cls):
        def decorator(enum_cls: Enum):
            for name, meth in inspect.getmembers(EnumRegistryBinder,inspect.isfunction):
                if name.startswith("dyn"):
                    meth(cls,enum_cls)
            return enum_cls
        return decorator

        

        
        