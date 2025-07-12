from abc import ABC, abstractmethod
from src.components.servers.server_interface import ServerI
from src.components.requesters.requester_interface import RequesterI
from src.prompt_sampler.prompt_sampler_interface import PromptSamplerI
from src.queue.queue_interface import QueueI
from src.buffers.performance.performance_metrics_buffer import PerformanceMetricsBuffer
import asyncio
import time

class LoadGenerator:
    def __init__(
                self, 
                queue: QueueI,
                config
            ):
        self.buffer: PerformanceMetricsBuffer = PerformanceMetricsBuffer()
        self.queue: QueueI = queue
        self.load_config = config
    
    async def async_continuous_prompt_generation_task(self, prompt_sampler: PromptSamplerI):
        init_time = time.time()
        end_time = time.time()
        while end_time - init_time < self.load_config.run_time:
            await self.generate_prompt_async(prompt_sampler)
            await asyncio.sleep(self.load_config.prompt_gen_sleep_time)
            end_time = time.time()

    async def generate_prompt_async(self, prompt_sampler: PromptSamplerI):
        prompt_idx, prompt = prompt_sampler.get_prompt_with_idx()
        await self.queue.add_prompt_and_idx_async(prompt, prompt_idx)

    def create_async_continuous_prompt_generation_tasks(self,prompt_sampler):
        return [asyncio.create_task(self.async_continuous_prompt_generation_task(prompt_sampler)) for _ in range(self.load_config.num_requester_threads)]

    async def async_continuous_request_task(self, requester: RequesterI, server: ServerI):
        
        request_count = 0

        while self.queue.empty():
            await asyncio.sleep(0.01)
            continue

        await requester.async_request(self.queue, self.buffer, server)
        request_count += 1


    async def create_async_continuous_request_tasks(self, requester: RequesterI, server:ServerI):
        init_time = time.time()
        end_time = time.time()
        tasks = []
        while end_time - init_time <= self.load_config.run_time:
            for _ in range(self.load_config.num_requester_threads):
                tasks.append(asyncio.create_task(self.async_continuous_request_task(requester, server)))
            await asyncio.sleep(self.load_config.requester_sleep_time)
            end_time = time.time()
        return tasks




    


