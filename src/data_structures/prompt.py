class Prompt():
    def __init__(self, prompt: str, num_tokens: int):
        self.prompt = prompt
        self.token_len =  num_tokens
        self.decoded_tokens = []