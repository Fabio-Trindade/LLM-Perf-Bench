import argparse

from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry

@Binder.create_parse_from_config(ConfigCatalog._openai_config)
@ComponentClassRegistry.register_requester_parser(ComponentRegistry.openai)
class OpenAIArgParser: pass