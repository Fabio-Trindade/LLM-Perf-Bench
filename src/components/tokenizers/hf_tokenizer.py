from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.tokenizers.tokenizer_interface import TokenizerI
from transformers import AutoTokenizer

@ComponentClassRegistry.register_tokenizer_workload(ComponentRegistry.HF)
class HFTokenizer(TokenizerI):
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
    
    def tokenize(self, prompt):
        return self.tokenizer(prompt, add_special_tokens=False)['input_ids']
    
    def calc_num_additional_tokens(self, prompt):
        wo_special_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        w_special_tokens = self.tokenizer(prompt, add_special_tokens=True)['input_ids']
        return len(w_special_tokens) - len(wo_special_tokens)

    def should_decode(self):
        return True
    
    def decode_ids(self, ids):
        return self.tokenizer.decode(ids,skip_special_tokens=True, clean_up_tokenization_spaces=True)
    