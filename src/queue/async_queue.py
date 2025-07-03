from src.queue.queue_interface import QueueI
import asyncio

class AsyncQueue(QueueI):
    def __init__(self,**kwargs):
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        
    async def add_prompt_and_idx_async(self, prompt, idx) -> None:
        await self.queue.put((prompt, idx))

    async def get_prompt_and_idx_async(self) -> tuple[str,int]:
        prompt, idx = await self.queue.get()
        return prompt, idx

    def empty(self) -> bool:
        return self.queue.empty()
        
    

