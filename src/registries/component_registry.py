from src.catalogs.component_catalog import ComponentCatalog
from src.binders.binder import Binder


@Binder.bind_list_as_vars(ComponentCatalog._types + 
                          ComponentCatalog._component_names +
                          ComponentCatalog._comp_instances)
class ComponentRegistry: pass