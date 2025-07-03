
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.dataset_generator.dataset_generator_interface import DatasetGenI
import random


@ComponentClassRegistry.register_dataset_gen_workload(ComponentRegistry.replay)
class ReplayDatasetGenerator(DatasetGenI):
    def __init__(self, config):
        super().__init__()
        self.count = 0
        self.generator = random.Random(config.seed)
        self.distribution = None
        
    def gen_inp_out_len_stats(self):
        ret =  self.distribution[self.count]
        self.count += 1
        return ret
