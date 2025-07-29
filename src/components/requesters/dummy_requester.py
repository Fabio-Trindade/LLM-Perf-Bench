
from src.components.servers.server_interface import ServerI
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.requesters.requester_interface import RequesterI
from src.queue.queue_interface import QueueI
from src.buffers.performance.performance_metrics_buffer import PerformanceMetricsBuffer
import asyncio
import time

@ComponentClassRegistry.register_requester_workload(ComponentRegistry.dummy)
class DummyRequester(RequesterI):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.prompts_per_request = config.prompts_per_request
        self.infer_sleep_time = config.infer_sleep_time
        self.max_out_tokens = config.max_out_tokens
        
    async def async_request(self,  req_id, prompts, buffer, server):
        
        async def infer():
            await asyncio.sleep(self.infer_sleep_time)

        prompts_per_request = self.prompts_per_request
        predicted_tokens = 0
        total_target_pred_tokens = self.max_out_tokens * prompts_per_request 

        while predicted_tokens < total_target_pred_tokens:
             for i,idx in range(prompts_per_request):
                await infer()
                last_time = time.time()
                key = (req_id, i)
                buffer.add_decode_data(key, last_time,f"token_{int(predicted_tokens/prompts_per_request)}")
                predicted_tokens += 1
                   
        

    