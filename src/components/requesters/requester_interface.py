from abc import ABC, abstractmethod
from src.data_structures.prompt import Prompt
from src.components.servers.server_interface import ServerI
from src.buffers.performance.performance_metrics_buffer import PerformanceMetricsBuffer

class RequesterI(ABC):
    def __init__(
            self,             
            config,
            ):
        self.config = config
    
    @abstractmethod
    async def async_request(self, req_id: int, prompts: list[str] | str, buffer: PerformanceMetricsBuffer ,server: ServerI):
        raise RuntimeError("Must be implemented")
    
    def get_str_prompts(self, prompts: list[Prompt] | str) -> list[str]:
        if isinstance(prompts, str):
            return [prompts]
        return [prompt.prompt for prompt in prompts]
    
    