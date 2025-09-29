from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.tokenizers.tokenizer_interface import TokenizerI
from vllm.transformers_utils.tokenizer import encode_tokens, decode_tokens, get_tokenizer


@ComponentClassRegistry.register_tokenizer_workload(ComponentRegistry.vLLM)
class vLLMTokenizer(TokenizerI):
    def __init__(self, config):
        self.tokenizer = get_tokenizer(config.model)
    
    def tokenize(self, prompt):
        return encode_tokens(self.tokenizer, prompt, add_special_tokens=False)
    
    def calc_num_additional_tokens(self, prompt):
        wo_special_tokens = encode_tokens(self.tokenizer, prompt, add_special_tokens=False)
        w_special_tokens = encode_tokens(self.tokenizer, prompt, add_special_tokens=True)
        return len(w_special_tokens) - len(wo_special_tokens)

    def decode_ids(self, ids):
        return decode_tokens(self.tokenizer,ids,skip_special_tokens=True)
    