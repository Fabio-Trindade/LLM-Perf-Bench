from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog
from src.registries.parser_registry import ParserRegistry


@ParserRegistry.registry()
@Binder.create_parse_from_config(ConfigCatalog._path_config)
def add_path_config_args(): pass