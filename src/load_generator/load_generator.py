from abc import ABC, abstractmethod
import math
from src.components.requesters.requester_interface import RequesterI
from src.prompt_sampler.prompt_sampler_interface import PromptSamplerI
from src.queue.queue_interface import QueueI
from src.buffers.performance.performance_metrics_buffer import PerformanceMetricsBuffer
import asyncio
import time
import tqdm

class LoadGenerator:
    def __init__(self, queue: QueueI, config):
        self.buffer: PerformanceMetricsBuffer = PerformanceMetricsBuffer()
        self.queue: QueueI = queue
        self.load_config = config
        self._stop_load = False

    async def async_initialize_queue(self, prompt_sampler: PromptSamplerI, size):
        for _ in tqdm.tqdm(range(size), desc="Filling queue with prompts"):
            prompt_idx, prompt = prompt_sampler.get_prompt_with_idx()
            await self.queue.add_prompt_and_idx_async(prompt, prompt_idx)
    
    async def async_continuous_prompt_generation_task(self, prompt_sampler: PromptSamplerI):
        while not self._stop_load: 
            await self.generate_prompt_async(prompt_sampler)
            await asyncio.sleep(self.load_config.prompt_gen_sleep_time)

    async def generate_prompt_async(self, prompt_sampler: PromptSamplerI):
        prompt_idx, prompt = prompt_sampler.get_prompt_with_idx()
        await self.queue.add_prompt_and_idx_async(prompt, prompt_idx)
    
    def stop_load(self):
        self._stop_load = True

    def create_async_continuous_prompt_generation_tasks(self, prompt_sampler):
        return [
            asyncio.create_task(self.async_continuous_prompt_generation_task(prompt_sampler))
            for _ in range(self.load_config.num_prompt_gen_threads)
        ]
    
    def get_total_prompts(self):
        return math.ceil(self.load_config.request_rate_per_requester * self.load_config.load_time)

    async def run(self, requester: RequesterI):
        tasks = []
        req_id = 0
        req_per_sec = self.load_config.request_rate_per_requester
        sleep_time = 1 / req_per_sec
        initial_time = time.time()

        # total_prompts = self.get_total_prompts()
        # print(f"[LoadGenerator] Total prompts to send: {total_prompts}")

        async def request_task(req_id, remaining_time):
            # start_time = time.time()
            # print(f"[Req {req_id}] Started at {start_time:.3f}")
            prompt, __ = await self.queue.get_prompt_and_idx_async()
            self.buffer.initialize_metrics(prompt, req_id, req_id, True)
            await requester.async_request(req_id, [prompt.prompt], self.buffer, timeout=remaining_time)
            # end_time = time.time()
            # print(f"[Req {req_id}] Finished at {end_time:.3f} (duration: {end_time - start_time:.3f}s)")

        remaining_time = self.load_config.load_time - (time.time() - initial_time)

        while remaining_time >= 0:
            tasks.append(
                asyncio.create_task(
                    request_task(
                        req_id, 
                        remaining_time if self.load_config.dont_wait_requests_finish else None
                    )
                )
            )
            req_id += 1
            await asyncio.sleep(sleep_time)
            remaining_time = self.load_config.load_time - (time.time() - initial_time)

        # print(f"[LoadGenerator] Awaiting {len(tasks)} tasks...")
        await asyncio.gather(*tasks, return_exceptions=True)
        # print("[LoadGenerator] All tasks completed.")
