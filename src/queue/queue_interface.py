from abc import ABC, abstractmethod

from src.data_structures.prompt import Prompt

class QueueI:

    async def add_prompt_and_idx_async(self, prompt: str, idx: int) -> None:
        raise RuntimeError("Not implemented.")
    
    async def get_prompt_and_idx_async(self) -> tuple[Prompt, int]:
        raise RuntimeError("Not implemented.")    
    
    def empty(self) -> bool:
        raise RuntimeError("Not implemented.")