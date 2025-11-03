import logging
import time
from openai import AsyncOpenAI
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.requesters.requester_interface import RequesterI

@ComponentClassRegistry.register_requester_workload(ComponentRegistry.openai)
class OpenAIRequester(RequesterI):
    def __init__(self, config):
        super().__init__(config)
        self.requester_config = config
        self.client: AsyncOpenAI = None

    async def _get_client(self):
        if self.client is None:
            self.client = AsyncOpenAI(api_key=self.requester_config.api_key)
        return self.client

    async def close(self):
        if self.client is not None:
            self.client = None

    async def __aenter__(self):
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def async_request(self, req_id, prompts, buffer, timeout=None):
        client = await self._get_client()
        config = self.requester_config

        def get_template(prompt: str):
            return {"role": "user", "content": prompt}

        prompt = prompts[0]
        messages = [get_template(prompt)]

        try:
            stream = await client.chat.completions.create(
                model=config.model,
                messages=messages,
                stream=True,
                max_completion_tokens=config.max_out_tokens,
            )
        except Exception as e:
            logging.error(f"Error during OpenAI request: {e}")
            raise

        async for event in stream:
            content = event.choices[0].delta.content
            if content:
                now = time.time()
                buffer.add_decode_data((req_id, 0), now, content)
