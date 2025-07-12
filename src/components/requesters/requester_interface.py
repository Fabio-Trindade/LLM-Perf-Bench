from abc import ABC, abstractmethod
from src.components.servers.server_interface import ServerI
from src.queue.queue_interface import QueueI
from src.buffers.performance.performance_metrics_buffer import PerformanceMetricsBuffer
import threading

class RequesterI(ABC):
    def __init__(
            self,             
            config,
            ):
        self.config = config
        self._global_request_id = 0
        self._lock = threading.Lock()
    
    @abstractmethod
    async def async_request(self, queue: QueueI, buffer: PerformanceMetricsBuffer, server: ServerI):
        raise RuntimeError("Must be implemented")
    
    def get_request_id(self) -> int:
        with self._lock:
            self._global_request_id += 1
            return self._global_request_id
    
    def reset(self):
        with self._lock:
            self._global_request_id = 0
    

    