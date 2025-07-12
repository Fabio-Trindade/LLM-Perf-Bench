from openai import AsyncOpenAI
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.servers.server_interface import ServerI

@ComponentClassRegistry.register_server_workload(ComponentRegistry.openai)
class OpenAIServer(ServerI):
    def __init__(self, config):
        self.requester_config = config
        self.client = None
    
    def init(self):
        self.client = AsyncOpenAI(api_key=self.requester_config.api_key)

    async def shutdown(self):
        await self.client.close()

    
        