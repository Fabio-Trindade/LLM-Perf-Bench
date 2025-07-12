from abc import ABC, abstractmethod
# from src.enums.enum_tokenizer import TokenizerType

class TokenizerI(ABC):    
    @abstractmethod
    def tokenize(self,prompt: str) -> list[str | int]:
        raise RuntimeError("Not Implemented")
    
    @abstractmethod
    def calc_num_additional_tokens(self,prompt: str)-> int:
        raise RuntimeError("Not Implemented")
    
    @abstractmethod 
    def decode_ids(self,ids: list[int]) -> list[str]:
        raise RuntimeError("Not Implemented")

