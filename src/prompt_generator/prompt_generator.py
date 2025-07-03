from math import ceil
from src.data_structures.prompt import Prompt
from itertools import chain
from src.components.tokenizers.tokenizer_interface import TokenizerI

class PromptGeneratorBase():
    def gen_prompt(self, tokenizer: TokenizerI, num_input_tokens, num_out_tokens) -> Prompt:
        
        def create_number_seq(length):
            seq = []
            for i in range(length):
                seq.append(str(i))
            return seq  
        
        initial_prompt = f"Continue the following sequence with the next {num_out_tokens} values: "
        seq = " ".join(create_number_seq(num_input_tokens))

        prompt = initial_prompt + seq
        tokenized_prompt = tokenizer.tokenize(prompt)

        additional_tokens = tokenizer.calc_num_additional_tokens(prompt)
        inp_len = len(tokenized_prompt) + additional_tokens
        diff = inp_len - num_input_tokens

        tokenized_seq = tokenizer.tokenize(seq)
        if len(tokenized_seq) <= diff:
            raise RuntimeError("Increase the prompt size")

        tokenized_prompt = tokenized_prompt[:len(tokenized_prompt) - diff]

        if tokenizer.should_decode():
            prompt = tokenizer.decode_ids(tokenized_prompt)
        else:
            prompt = "".join(tokenized_prompt)
        
        inp_len = len(tokenizer.tokenize(prompt)) + additional_tokens
        assert( inp_len  == num_input_tokens )

        return Prompt(prompt, inp_len)
