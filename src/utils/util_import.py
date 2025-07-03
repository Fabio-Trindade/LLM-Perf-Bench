import importlib
import pkgutil
import os

def import_all_modules(package):

    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module_name)