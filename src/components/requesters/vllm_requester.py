import json
import time
import httpx
from src.components.requesters.requester_interface import RequesterI
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.utils.vllm_utils import get_url_from_config

@ComponentClassRegistry.register_requester_workload(ComponentRegistry.vllm)
class VLLMRequester(RequesterI):
    def __init__(self, config):
        super().__init__(config)
        self.model = config.model
        self.max_tokens = config.max_out_tokens
        self.requester_config  = config
        self.prompts_per_request = config.prompts_per_request
        self.vllm_server_url = get_url_from_config(config)
        self.timeout = self.requester_config.vllm_request_timeout
        self.client = None
    
    async def _get_client(self):
        if self.client is None:
            self.client = httpx.AsyncClient(timeout = self.timeout)
        return self.client

    async def close(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def __aenter__(self):
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def async_request(self, req_id, prompts, buffer, server):
        request_template = {
            "model": self.model,
            "prompt": prompts,
            "stream": True,
            "temperature": 0,
            "ignore_eos": self.config.ignore_eos,
            "n": 1,
            "skip_special_tokens": False,
            "max_tokens": self.max_tokens,
            "min_tokens": self.max_tokens
            }

        try:
            async with self.client.stream(
                "POST",
                self.vllm_server_url,
                json=request_template,
                timeout=self.timeout,
                
            ) as response:
                if response.status_code != 200:
                    error_msg = await response.aread()
                    raise Exception(f"Request failed: {response.status_code} - {error_msg}")

                async for chunk in response.aiter_lines():
                    chunk = chunk.strip()
                    if not chunk or "[DONE]" in chunk:
                        continue

                    if chunk.startswith("data:"):
                        chunk = chunk[5:].strip()

                    try:
                        data = json.loads(chunk)
                        for choice in data.get("choices", []):
                            idx = choice.get("index", -1)
                            text = choice.get("text", "")
                            
                            if text:
                                key = (req_id, idx)
                                current_time = time.time()
                                buffer.add_out_token_time(key, current_time)
                                buffer.add_decoded_token(key, text)
                            
                    except json.JSONDecodeError as e:
                        print(f"Failed to decode chunk: {chunk}. Error: {e}")
                        continue

        except Exception as e:
            raise RuntimeError(f"Request error: {str(e)}")
            
        