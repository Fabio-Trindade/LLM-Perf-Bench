
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
    async def async_request(self, queue: QueueI, buffer: PerformanceMetricsBuffer, server: ServerI):
        dummy_config = self.config
        load_config = self.config
        assert(self.config.prompts_per_request == 1), "We do not allow more than 1 prompt per request. We have to to assert all prompts are the same max_out_len first."

        
        async def infer():
            await asyncio.sleep(dummy_config.infer_sleep_time)

        prompts_per_request = load_config.prompts_per_request
        req_id = self.get_request_id()
        prompt_idx_list : list[int] = []
        first_prompt = None
        for i in range(prompts_per_request):
            prompt, prompt_idx = await queue.get_prompt_and_idx_async()
            buffer.initialize_metrics(prompt, (req_id, i), req_id, True)
            prompt_idx_list.append(prompt_idx)
            if first_prompt is None:
                first_prompt = prompt

    
        predicted_tokens = 0
        total_target_pred_tokens = first_prompt.max_out_tokens * prompts_per_request 

        while predicted_tokens < total_target_pred_tokens:
             for i,idx in enumerate(prompt_idx_list):
                await infer()
                last_time = time.time()
                key = (req_id, i)
                buffer.add_decode_data(key, last_time,f"token_{predicted_tokens}")
                predicted_tokens += 1
                   
        

    