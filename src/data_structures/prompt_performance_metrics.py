import time
from src.data_structures.prompt import Prompt
from typing import Final 
class PromptPerformanceMetrics:
    TTFT: Final = "TTFT_time"
    E2ET: Final = "e2e_time"
    TBT: Final = "tbt_times"
    THP: Final = "throughput"
    PTHP: Final = "prefill_throughput"
    DTHP: Final = "decode_throughput"
    PLEN: Final = "prompt_len"
    DLEN: Final = "decode_len"

    def __init__(self, prompt: Prompt, req_id: int):
        self._out_token_times = []
        self._initial_req_time = None
        self._success = False
        self.prompt: Prompt = prompt
        self._req_id = req_id
        
        self.TTFT_time = None
        self.e2e_time = None
        self.tbt_times = []
        self.throughput = None
        self.prefill_throughput = None
        self.decode_throughput = None
        self.prompt_len = None
        self.decode_len = None
        self.decode_time = None 

    def get_req_id(self):
        return self._req_id

    def get_total_processed_tokens(self):
        return self.prompt.token_len + len(self._out_token_times)

    def start_req_time(self):
        if self._initial_req_time is not None:
            raise RuntimeError("Initial request time has already been set.")
        self._initial_req_time = time.time()
    
    def set_success(self, success: bool):
        self._success = success
    
    def add_out_token_time(self, time):
        self._out_token_times.append(time)

    def add_decoded_token(self,token):
        self.prompt.decoded_tokens.append(token)
        
    def calc_results(self):
        self.e2e_time = self._out_token_times[-1] - self._initial_req_time

        self.decode_len =  len(self._out_token_times)

        self.prompt_len = self.prompt.token_len

        self.throughput = (self.prompt.token_len + self.decode_len)/self.e2e_time

        self.TTFT_time = self._out_token_times[0] - self._initial_req_time
        
        self.prefill_throughput = (self.prompt.token_len)/self.TTFT_time
        self.decode_time = self._out_token_times[-1] - self._out_token_times[0]
        self.decode_throughput = self.decode_len/self.decode_time

        for idx in range(len(self._out_token_times) - 1):
            self.tbt_times.append(
                self._out_token_times[idx + 1] - self._out_token_times[idx]
                )
            
    def get_metrics(self, var_name: str) -> list:
        attr = getattr(self, var_name)
        data_list = attr if isinstance(attr, list) else [attr]
        return data_list

PPM = PromptPerformanceMetrics




