import logging
import time
from threading import Thread
from pynvml import (
    nvmlInit,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage,
    nvmlShutdown,
)
from src.monitors.device_monitor_interface import DeviceMonitorI
from src.data_structures.device.accelerator_data import AcceleratorData 


class AcceleratorMonitor(DeviceMonitorI):
    def __init__(self):
        nvmlInit()
        self.gpu_count = nvmlDeviceGetCount()
        self.gpu_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
        self.gpu_names = [f"{nvmlDeviceGetName(handle)}:{i}" for i,handle in enumerate(self.gpu_handles)]

        self.watt_per_device_distribution: dict[str, list] = {name: [] for name in self.gpu_names}
        self.vram_per_device_distribution: dict[str, list] = {name: [] for name in self.gpu_names}

        self.vram_per_device: list[int] = [
            (nvmlDeviceGetMemoryInfo(handle).total) for handle in self.gpu_handles
        ]
      

        self.vram_per_node: list[int] = [sum(self.vram_per_device)]

        self.monitor_thread = None
        self._stop_flag = False

    def _monitor_loop(self):
        while not self._stop_flag:
            for i, handle in enumerate(self.gpu_handles):
                name = self.gpu_names[i]

                mem_info = nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used  

                try:
                    power_mw = nvmlDeviceGetPowerUsage(handle)
                    power_w = power_mw / 1000.0
                except:
                    power_w = 0.0

                self.vram_per_device_distribution[name].append(mem_used)
                self.watt_per_device_distribution[name].append(power_w)

            time.sleep(1)

    def init(self):
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def shutdown(self):
        self._stop_flag = True
        if self.monitor_thread:
            self.monitor_thread.join()
        nvmlShutdown()

    def get_data(self) -> AcceleratorData:
        logging.warning("Accelerator monitor doesn't collect data for more than one node")
        return AcceleratorData(
            name=", ".join(set(self.gpu_names)),
            num_devices=self.gpu_count,
            num_nodes=1,
            accelerators_per_node=self.gpu_count,
            vram_per_device=self.vram_per_device,
            vram_per_node=self.vram_per_node,
            watt_per_device_distribution=self.watt_per_device_distribution,
            vram_per_device_distribution=self.vram_per_device_distribution,
        )
