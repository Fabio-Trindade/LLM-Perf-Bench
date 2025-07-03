from types import SimpleNamespace
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.catalogs.component_catalog import ComponentCatalog


class WorkloadComponentFactory:
    @staticmethod
    def build_components_from_config(config):
        component_typename = ComponentRegistry.workload
        comp_names = ComponentCatalog.get_comp_names()
        kwargs = {}
        for comp_name in comp_names:
            comp_cls_name = getattr(config,comp_name)
            comp_cls = ComponentClassRegistry.get_instance(component_typename,
                                                 comp_name,
                                                 comp_cls_name)
            component_inst = comp_cls(config)
            kwargs [comp_name] = component_inst
        return SimpleNamespace(**kwargs)

