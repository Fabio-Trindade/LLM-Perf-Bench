from enum import Enum
import inspect
from binders.enum_registry_binder import EnumRegistryBinder


class FactoryRegistryBinder:

    @staticmethod
    def get_component(class_registry,config, component_name):
        func_name = EnumRegistryBinder.format_create_func_name(component_name)
        func = getattr(class_registry, func_name)
        component = func(config)
        return component
    
    @staticmethod
    def dyn_bind_get_method(factory_cls, class_registry, enum_cls):
        def make_get_func(component_name: str):
            def get(config, force_rebuild = False):
                attr = getattr(factory_cls, component_name, None)
                if attr is None or force_rebuild:
                    component_cls = FactoryRegistryBinder.get_component(class_registry, config, component_name)
                    setattr(factory_cls, component_name, component_cls)
                return getattr(factory_cls, component_name)
            return get
        EnumRegistryBinder.bind_method_by_enum_cls(factory_cls, enum_cls, make_get_func, "get")
    
    def dyn_bind_init_method(factory_cls, class_registry, enum_cls):
        def make_init_func(class_registry, enum_cls):
            def init(config):
                for component in enum_cls:
                    component_name = component.value
                    component_cls = FactoryRegistryBinder.get_component(class_registry, config, component_name)
                    setattr(factory_cls, component_name, component_cls)
            return init
        setattr(factory_cls, "__init__",make_init_func(class_registry, enum_cls))
            

    @staticmethod
    def bind_all_methods_to_class_with_registry(factory_cls, class_registry):
        def decorator(enum_cls: Enum):
            for name, meth in inspect.getmembers(FactoryRegistryBinder,inspect.isfunction):
                if name.startswith("dyn"):
                    meth(factory_cls, class_registry, enum_cls)
            return enum_cls
        return decorator
