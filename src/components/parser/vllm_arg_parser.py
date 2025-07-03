import argparse

from src.catalogs.config_catalog import ConfigCatalog
from src.binders.binder import Binder
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry

@Binder.create_parse_from_config(ConfigCatalog._vllm_config)
@ComponentClassRegistry.register_requester_parser(ComponentRegistry.vllm)
class vLLMArgParser: pass

    
