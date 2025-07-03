from src.data_structures.prompt import Prompt
from src.prompt_sampler.prompt_sampler_interface import PromptSamplerI
import random

class RandomPromptSampler(PromptSamplerI):
    def __init__(self, prompts, seed):
        super().__init__(prompts,seed)
        self.generator = random.Random(seed)

    def get_prompt_with_idx(self) -> tuple[int, Prompt]:
        idx = self.generator.randint(0,len(self.prompts) - 1)
        prompt = self.prompts[idx]
        return idx, prompt
    
    def reset_generator(self):
        self.generator.seed(self.seed)
 
    

                