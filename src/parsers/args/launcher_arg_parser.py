from src.registries.parser_registry import ParserRegistry
from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog


@ParserRegistry.registry()
@Binder.create_parse_from_config(ConfigCatalog._launcher_config)
def add_launcher_args(): pass