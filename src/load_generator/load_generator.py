from abc import ABC, abstractmethod

import tqdm
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
        self._stop_load = False

    async def async_initialize_queue(self, prompt_sampler:PromptSamplerI, size):
        for _ in tqdm.tqdm(range(size), desc = "Filling queue with prompts"):
            prompt_idx, prompt = prompt_sampler.get_prompt_with_idx()
            await self.queue.add_prompt_and_idx_async(prompt, prompt_idx)
    
    async def async_continuous_prompt_generation_task(self, prompt_sampler: PromptSamplerI):
        # init_time = time.time()
        # end_time = time.time()
        while not self._stop_load: 
            await self.generate_prompt_async(prompt_sampler)
            await asyncio.sleep(self.load_config.prompt_gen_sleep_time)
            # end_time = time.time()

    async def generate_prompt_async(self, prompt_sampler: PromptSamplerI):
        prompt_idx, prompt = prompt_sampler.get_prompt_with_idx()
        await self.queue.add_prompt_and_idx_async(prompt, prompt_idx)
    
    def stop_load(self):
        self._stop_load = True

    def create_async_continuous_prompt_generation_tasks(self,prompt_sampler):
        return [asyncio.create_task(self.async_continuous_prompt_generation_task(prompt_sampler)) for _ in range(self.load_config.num_prompt_gen_threads)]
    
    def get_total_prompts(self):
        return int(self.load_config.run_time*self.load_config.prompts_per_request * self.load_config.num_requester_threads/self.load_config.requester_sleep_time + 1e-10)

    def get_total_requests(self):
        return int(self.load_config.run_time* self.load_config.num_requester_threads/self.load_config.requester_sleep_time + 1e-10)

    async def run(self, requester: RequesterI, server: ServerI):
        tasks = []
        req_id = 0
        total_requests = self.get_total_requests()

        pbar_sent = tqdm.tqdm(total=total_requests, desc="Requests sent", position=0)
        pbar_done = tqdm.tqdm(total=total_requests, desc="Requests done", position=1)

        async def request_task(req_id):
            prompts = []
            for i in range(self.load_config.prompts_per_request):
                prompt, __ = await self.queue.get_prompt_and_idx_async()
                self.buffer.initialize_metrics(prompt, (req_id, i), req_id, True)
                prompts.append(prompt.prompt)

            result = await requester.async_request(req_id, prompts, self.buffer, server)
            pbar_done.update(1)  
            return result

        while req_id < total_requests:
            cur_num_req = min(self.load_config.num_requester_threads, total_requests - req_id)
            for _ in range(cur_num_req):
                tasks.append(asyncio.create_task(request_task(req_id)))
                req_id += 1
                pbar_sent.update(1)  

            await asyncio.sleep(self.load_config.requester_sleep_time)

        await asyncio.gather(*tasks, return_exceptions=True)

        pbar_sent.close()
        pbar_done.close()




        


