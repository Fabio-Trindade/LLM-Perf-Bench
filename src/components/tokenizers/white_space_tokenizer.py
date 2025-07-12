from src.components.tokenizers.tokenizer_interface import TokenizerI
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry

@ComponentClassRegistry.register_tokenizer_workload(ComponentRegistry.whitespace)
class WSTokenizer(TokenizerI):
    def __init__(self, config):
        pass
    
    def tokenize(self, prompt: str):
        return prompt.split(" ")

    def decode_ids(self, ids):
        return " ".join(ids)

    def calc_num_additional_tokens(self, prompt: str):
        return 0