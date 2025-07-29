import importlib
import pkgutil
import os
from src import components
from src import parsers

def import_all_modules(package):
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module_name)

def initialize_modules():
    import_all_modules(components)
    import_all_modules(parsers)