import logging
import time
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.servers.server_interface import ServerI
from src.components.servers.open_ai_server import OpenAIServer
from src.components.requesters.requester_interface import RequesterI
from src.queue.queue_interface import QueueI
from src.buffers.performance.performance_metrics_buffer import PerformanceMetricsBuffer

@ComponentClassRegistry.register_requester_workload(ComponentRegistry.openai)
class OpenAIRequester(RequesterI):
    def __init__(self, config):
        super().__init__(config)
        assert config.prompts_per_request == 1, "Only 1 prompt per request is supported with OpenAI."

    async def async_request(self, req_id, prompts, buffer, server):
        config = self.config

        def get_template(prompt: str):
            return {
                "role": "user",
                "content": prompt
            }

        server: OpenAIServer = server
        client = server.client 
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
