from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.tokenizers.tokenizer_interface import TokenizerI
from vllm.tokenizers import get_tokenizer


@ComponentClassRegistry.register_tokenizer_workload(ComponentRegistry.vllm)
class vLLMTokenizer(TokenizerI):
    def __init__(self, config):
        self.tokenizer = get_tokenizer(config.model)
        self.is_chat_model = "chat" in config.endpoint
    
    def tokenize(self, prompt):
        return self.tokenizer.encode(prompt, add_special_tokens=False)
    
    def calc_num_additional_tokens(self, prompt):
        wo_special_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if self.is_chat_model:
            conversation = [{"role": "user", "content": prompt}]
            ids = self.tokenizer.apply_chat_template(conversation = conversation)
            return len(ids) - len(wo_special_tokens)
        w_special_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        return len(w_special_tokens) - (len(wo_special_tokens))

    def decode_ids(self, ids):
        return self.tokenizer.decode(ids,skip_special_tokens=True)
    