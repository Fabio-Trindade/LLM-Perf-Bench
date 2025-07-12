class Prompt():
    def __init__(self, prompt: str, prompt_len: int, max_out_tokens: int):
        self.prompt = prompt
        self.prompt_len =  prompt_len
        self.max_out_tokens = max_out_tokens
        self.decoded_tokens = []