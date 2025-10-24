import asyncio
import time
from src.data_structures.device.host_data import HostData
from src.monitors.accelerator_monitor import AcceleratorMonitor
from src.monitors.host_monitor import HostMonitor
from src.monitors.device_monitor_interface import DeviceMonitorI
from src.queue.async_queue import AsyncQueue
from src.data_structures.prompt import Prompt
from src.prompt_sampler.random_prompt_sampler import RandomPromptSampler
from src.data_structures.perf_results import PerfResults
from src.load_generator.load_generator import LoadGenerator
from src.components.requesters.requester_interface import RequesterI
from src.components.servers.server_interface import ServerI

class WorkloadLauncher():
    def __init__(self, config, server: ServerI, requester: RequesterI, prompts: list[Prompt]):
        self.load_generator: LoadGenerator = LoadGenerator(AsyncQueue(), config)
        self.prompt_sampler = RandomPromptSampler(prompts, config.seed)
        self.monitors: list[DeviceMonitorI] = [HostMonitor(), AcceleratorMonitor()]
        self.server = server
        self.requester = requester
    
    def get_host_data(self) -> HostData:
        return self.monitors[0].get_data()

    def get_accelerator_data(self) -> HostData:
        return self.monitors[1].get_data()
    
    async def async_run(self) -> PerfResults:

        await self.load_generator.async_initialize_queue(self.prompt_sampler,self.load_generator.get_total_prompts())
        
        initial_time = time.time()

        request_tasks = await self.load_generator.create_async_continuous_request_tasks(self.requester, self.server)

        await asyncio.gather(*request_tasks)

        final_time = time.time()

        
        exp_time = final_time - initial_time

        buffer = self.load_generator.buffer
        buffer.calc_perf_times()

        all_request_prompts = buffer.get_prompts_grouped_by_request()

        return PerfResults(all_request_prompts, exp_time)



