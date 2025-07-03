import logging
from src.components.tokenizers.tokenizer_interface import TokenizerI

from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
import tiktoken

@ComponentClassRegistry.register_tokenizer_workload(ComponentRegistry.openai)
class OpenAITokenizer(TokenizerI):
    def __init__(self, config):
        self.model = config.model
        self.encoding =  tiktoken.encoding_for_model(config.model)
        # raise RuntimeError("We must consider the tokens in chat template during tokenize")
    
    def tokenize(self, prompt: str):
         return self.encoding.encode(prompt)

    def should_decode(self):
        return True
    
    def decode_ids(self, ids):
        return self.encoding.decode(ids)
    
    def calc_num_additional_tokens(self, prompt):
        logging.warning("We assume that the messages don't have a name key.")
        model = self.model
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            
        }:
            tokens_per_message = 3 
        elif model in {"gpt-3.5-turbo-0301", 
                       "gpt-3.5-turbo-0125"}:
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        else:
            raise NotImplementedError(
                f"""calc_num_additional_tokens() is not implemented for model {model}."""
            )
        num_tokens = 0
        num_tokens += tokens_per_message
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
        
    
