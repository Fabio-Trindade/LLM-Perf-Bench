from abc import ABC, abstractmethod
import random

from src.data_structures.prompt import Prompt

class PromptSamplerI:
    def __init__(self, prompts, seed: int):
        self.prompts: list[Prompt] = prompts
        self.seed = seed
    @abstractmethod
    def get_prompt_with_idx() -> tuple[Prompt,int]:
        raise RuntimeError("Must be implemented")
    
    @abstractmethod
    def reset_generator(self):
        raise RuntimeError("Must be implemented")