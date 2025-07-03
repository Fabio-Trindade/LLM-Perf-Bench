
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.servers.server_interface import ServerI

@ComponentClassRegistry.register_server_workload(ComponentRegistry.dummy)
class DummyServer(ServerI):
    def __init__(self, config):
        pass
    def init(self,):
        print("Initializing dummy server")
    
    def shutdown(self,):
        print("Shutting down dummy server")