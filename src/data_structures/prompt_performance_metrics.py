import logging
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
        self._success = None
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
    
    def get_success(self):
        if self._success is None:
            raise RuntimeError("Success status has not been set.")
        return self._success

    def get_initial_req_time(self):
        if self._initial_req_time is None:
            raise RuntimeError("Initial request time has not been set.")
        return self._initial_req_time
    
    def get_final_time(self):
        if not self._out_token_times:
            raise RuntimeError("No output token times recorded.")
        return self._out_token_times[-1]

    def get_total_processed_tokens(self):
        return self.prompt.prompt_len + len(self._out_token_times)

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
        if self.prompt.max_out_tokens < 2:
            # logging.warning("Remove condition after test. [file=data_structures/prompt_performance_metrics.py, line=64]")
            raise RuntimeError("Not enough output token times to calculate performance metrics.")
        self.e2e_time = self._out_token_times[-1] - self._initial_req_time

        self.decode_len =  len(self._out_token_times)

        self.prompt_len = self.prompt.prompt_len

        self.throughput = (self.prompt.prompt_len + self.decode_len)/self.e2e_time

        self.TTFT_time = self._out_token_times[0] - self._initial_req_time
        
        self.prefill_throughput = (self.prompt.prompt_len)/self.TTFT_time
        self.decode_time = self._out_token_times[-1] - self._out_token_times[0]
        self.decode_throughput = self.decode_len/self.decode_time

        logging.warning("Remove after testing. file=data_structures/prompt_performance_metrics.py, line=83")
        assert(self.decode_time + self.TTFT_time == self.e2e_time), \
            "The sum of decode time and TTFT time should equal the e2e time."

        self._success = self.prompt.max_out_tokens == len(self._out_token_times)

        for idx in range(len(self._out_token_times) - 1):
            self.tbt_times.append(
                self._out_token_times[idx + 1] - self._out_token_times[idx]
                )
            
    def get_metrics(self, var_name: str) -> list:
        attr = getattr(self, var_name)
        data_list = attr if isinstance(attr, list) else [attr]
        return data_list

PPM = PromptPerformanceMetrics




