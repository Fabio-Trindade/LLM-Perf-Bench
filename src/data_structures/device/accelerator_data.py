from typing import List, Dict, Union

class AcceleratorData:
    def __init__(
        self,
        name: str,
        num_devices: int,
        num_nodes: int,
        accelerators_per_node: int,
        vram_per_device,
        vram_per_node,
        watt_per_device_distribution:dict[str,list],
        vram_per_device_distribution:dict[str,list]
    ):
        self.name: str = name
        self.num_devices: int = num_devices
        self.num_nodes: int = num_nodes
        self.vram_per_device = vram_per_device
        self.vram_per_node = vram_per_node
        self.accelerators_per_node: int = accelerators_per_node
        self.watt_per_device_distribution: dict[str,list] = watt_per_device_distribution
        self.vram_per_device_distribution: dict[str,list] = vram_per_device_distribution
