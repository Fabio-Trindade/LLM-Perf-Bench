import argparse

from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.utils.util_parse import add_arg, get_fixed_values

@Binder.create_parse_from_config(ConfigCatalog._dummy_config)
@ComponentClassRegistry.register_requester_parser(ComponentRegistry.dummy)
class DummyArgParser: pass