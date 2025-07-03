from abc import ABC, abstractmethod
from typing import Union
from src.data_structures.device.accelerator_data import AcceleratorData
from src.data_structures.device.host_data import HostData

class DeviceMonitorI(ABC):
    @abstractmethod
    def init(self):
        raise RuntimeError("Not implemented.")
    
    @abstractmethod
    def shutdown(self):
        raise RuntimeError("Not implemented.")
    
    @abstractmethod
    def get_data(self) -> Union[HostData, AcceleratorData]:
        raise RuntimeError("Not implemented.")