from typing import List

class HostData:
    def __init__(
        self,
        total_cpus: int,
        cpu_name: str,
        ram_mem_name: str,
        ram_capacity: float, 
        ram_consumption_over_time: List[float],
        cpu_consumption_over_time: List[float]
    ):
        self.total_cpus: int = total_cpus
        self.cpu_name: str = cpu_name
        self.ram_mem_name: str = ram_mem_name
        self.ram_capacity: float = ram_capacity
        self.ram_consumption_over_time: List[float] = ram_consumption_over_time
        self.cpu_consumption_over_time: List[float] = cpu_consumption_over_time
