
from src.catalogs.component_catalog import ComponentCatalog
from src.binders.component_instance_binder import ComponentInstanceBinder
from src.data_structures.my_obj import MyObj

@ComponentInstanceBinder.create_registration_functions_from_catalog(ComponentCatalog)
class ComponentClassRegistry:
    _registry = {}

    @classmethod
    def register(cls, component_type: str, component_name: str, instance_name: str, instance):
        cls._registry.setdefault(component_type, {})
        cls._registry[component_type].setdefault(component_name, {})
       
        if cls._registry[component_type][component_name].get(instance_name,None) is not None:
        
            raise RuntimeError(f"Found duplicated registry for {instance_name} {component_name} {component_type}") 
        cls._registry[component_type][component_name][instance_name] = instance

    @classmethod
    def get_instance(cls, component_type: str, component_name: str, instance_name: str):
        return cls._registry.get(component_type, {}).get(component_name, {}).get(instance_name)

    @classmethod
    def exists(cls, component_type: str, component_name: str, instance_name: str):
        return cls.get_instance(component_type, component_name, instance_name) is not None
