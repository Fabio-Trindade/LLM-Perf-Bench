from types import SimpleNamespace

class DataFormat:
    def __init__(self, 
                    prompt_distributions: SimpleNamespace,
                    request_distributions: SimpleNamespace,
                    total_prefill_tokens: float,
                    total_decode_tokens: float,
                    total_prefill_time : float,
                    total_decode_time : float
                 ):
        self.total_prefill_tokens = total_prefill_tokens
        self.total_decode_tokens = total_decode_tokens
        self.total_prefill_time = total_prefill_time
        self.total_decode_time = total_decode_time

        total_time = self.total_prefill_time + self.total_decode_time
        self.total_time = total_time
        self.total_throughput = (self.total_prefill_tokens + self.total_decode_tokens) / total_time 
        self.total_prefill_throughput = self.total_prefill_tokens / self.total_prefill_time 
        self.total_decode_throughput = self.total_decode_tokens / self.total_decode_time  

        self.request_distributions: SimpleNamespace = request_distributions
        self.prompt_distributions: SimpleNamespace = prompt_distributions