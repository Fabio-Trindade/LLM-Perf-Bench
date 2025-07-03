import logging
from src.data_structures.prompt import Prompt
from src.prompt_generator.prompt_generator import PromptGeneratorBase
from abc import ABC, abstractmethod
import tqdm

class DatasetGenI(ABC):
    def __init__(self, config ):
        self.num_prompts = config.num_prompts
    
    def gen_dataset(self, tokenizer, prompt_generator: PromptGeneratorBase) -> list[Prompt]:  
        prompts :list[Prompt] = []
        for i in tqdm.tqdm(range(self.num_prompts), desc = "Generating prompts..."):
            inp_len , out_len = self.gen_inp_out_len_stats()
            prompt = prompt_generator.gen_prompt(tokenizer, inp_len, out_len)
            prompts.append(prompt)
        return prompts
    
    @abstractmethod
    def gen_inp_out_len_stats(self) -> tuple[int,int]:
        raise RuntimeError("Must be implemented.")