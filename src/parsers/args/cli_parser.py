import argparse
import logging
from src.registries.parser_registry import ParserRegistry
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.catalogs.component_catalog import ComponentCatalog

class CLIParser:
    @staticmethod
    def parse_all(parser: argparse.ArgumentParser, fixed_values: dict = None):
        for parse in ParserRegistry._registry:
            parse(parser, fixed_values)
            
        component_typename = ComponentRegistry.parser
        comp_names = ComponentCatalog.get_comp_names()
        args, _ = parser.parse_known_args()
        for comp_name in comp_names:
            comp_instance_name = getattr(args, comp_name)
            parser_cls = ComponentClassRegistry.get_instance(component_typename,
                                                 comp_name,
                                                 comp_instance_name)
            if parser_cls is None:
                logging.warning(f"Parser wasn't found. Skipping parsing for {comp_instance_name} {comp_name} {component_typename}")
                continue
            parser_cls(parser, fixed_values)



        