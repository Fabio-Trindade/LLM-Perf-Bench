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
    async def async_request(self, queue: QueueI, buffer: PerformanceMetricsBuffer, server: ServerI):
        config = self.config

        def get_template(prompt: str):
            return {
                "role": "user",
                "content": prompt
            }

        server: OpenAIServer = server
        prompts_per_request = config.prompts_per_request

        assert prompts_per_request == 1, "Only 1 prompt per request is supported with OpenAI."

        req_id = self.get_request_id()
        i = 0

        prompt, prompt_idx = await queue.get_prompt_and_idx_async()
        client = server.client 

        messages = [get_template(prompt.prompt)]

        buffer.initialize_metrics(prompt, (req_id, i), req_id, True)

        try:
            stream = await client.chat.completions.create(
                model=config.model,
                messages=messages,
                stream=True,
                max_completion_tokens=prompt.max_out_tokens,
            )
        except Exception as e:
            logging.error(f"Error during OpenAI request: {e}")
            raise

        async for event in stream:
            content = event.choices[0].delta.content
            if content:
                now = time.time()
                buffer.add_decode_data((req_id, i), now, content)
