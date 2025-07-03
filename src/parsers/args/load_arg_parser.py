import argparse
from src.registries.parser_registry import ParserRegistry
from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog
from src.utils.util_parse import add_arg, get_fixed_values

@ParserRegistry.registry()
@Binder.create_parse_from_config(ConfigCatalog._load_config)
def add_load_config_args(): pass