
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.dataset_generator.dataset_generator_interface import DatasetGenI
import random

@ComponentClassRegistry.register_dataset_gen_workload(ComponentRegistry.synthetic)
class SynDatasetGenerator(DatasetGenI):
    def __init__(self,  config):
        super().__init__(config)
        input_range_values = config.prompt_size_range 
        max_out_size = config.max_out_tokens 
        seed = config.seed
        self.inp_range = range(input_range_values[0], input_range_values[1] + 1)
        self.max_out_size = max_out_size
        self.generator = random.Random(seed)
    
    def gen_inp_out_len_stats(self):
        inp_len = self.generator.choice(self.inp_range)
        out_len = self.max_out_size
        return inp_len, out_len
