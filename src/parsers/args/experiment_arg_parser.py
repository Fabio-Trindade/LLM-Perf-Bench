from src.binders.binder import Binder
from src.registries.parser_registry import ParserRegistry
from src.catalogs.config_catalog import ConfigCatalog

@ParserRegistry.registry()
@Binder.create_parse_from_config(ConfigCatalog._experiment_config)
def add_experiment_args(): pass