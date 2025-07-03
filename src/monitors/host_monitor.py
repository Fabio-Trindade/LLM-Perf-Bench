import time 
import psutil
from threading import Thread
from src.data_structures.device.host_data import HostData
from src.monitors.device_monitor_interface import DeviceMonitorI
from src.utils.util_cpu import get_cpu_name

class HostMonitor(DeviceMonitorI):
    def __init__(self):
        self.cpu_thread = None
        self.ram_thread = None
        self.cpu_usage = []
        self.ram_usage = []
        self._stop_flag = False

    def init(self):
        def monitor_cpu():
            while not self._stop_flag:
                cpu = psutil.cpu_percent(interval=1)
                self.cpu_usage.append(cpu)
        
        def monitor_ram():
            while not self._stop_flag:
                ram = psutil.virtual_memory().percent
                self.ram_usage.append(ram)
                time.sleep(1)  
        
        self.cpu_thread = Thread(target=monitor_cpu, daemon=True)
        self.ram_thread = Thread(target=monitor_ram, daemon=True)
        
        self.cpu_thread.start()
        self.ram_thread.start()

    def shutdown(self):
        self._stop_flag = True
        if self.cpu_thread:
            self.cpu_thread.join()
        if self.ram_thread:
            self.ram_thread.join()
    
    def get_data(self):
        total_cpus = psutil.cpu_count()
        ram = psutil.virtual_memory()
        ram_capacity = ram.total / (1024 ** 3) 

        return HostData(
            total_cpus= total_cpus,
            cpu_name= get_cpu_name(),
            ram_mem_name= "TODO",
            ram_capacity=ram_capacity,
            cpu_consumption_over_time= self.cpu_usage,
            ram_consumption_over_time= self.ram_usage
        )
