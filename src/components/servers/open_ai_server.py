from openai import OpenAI
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.servers.server_interface import ServerI

@ComponentClassRegistry.register_server_workload(ComponentRegistry.openai)
class OpenAIServer(ServerI):
    def __init__(self, config):
        self.requester_config = config
        self.client = None
    
    def init(self):
        self.client = OpenAI(api_key = self.requester_config.api_key)

    def shutdown(self):
        self.client.close()

    
        