from src.data_structures.prompt_performance_metrics import PromptPerformanceMetrics
import json

class CSVDataFormat:
    def __init__(self, exp_key, 
                 model_name,
                 req_per_sec,
                 prompt_metrics: PromptPerformanceMetrics
                 
                 ):
        self.key = exp_key
        self.model_name = model_name
        self.req_per_sec = req_per_sec
        var_names = ["total_throughput",
                     "prefill_throughput",
                     "decode_throughput",
                     "e2e_latency",
                     "ttft",
                     "tbt_times",
                     "total_decode_time",
                     "prompt_size",
                     "num_decoded_tokens",
                     "prompt",
                     "decoded_tokens",
                     "initial_time",
                     "final_time",
                     "request_id",
                     "success",
                  ]
        
        for var_name in var_names:
            setattr(self, var_name, None)
            
        if prompt_metrics:
            self.total_throughput = prompt_metrics.throughput
            self.prefill_throughput = prompt_metrics.prefill_throughput  
            self.decode_throughput = prompt_metrics.decode_throughput

            self.e2e_latency = prompt_metrics.e2e_time
            self.ttft = prompt_metrics.TTFT_time
            self.tbt_times = json.dumps(prompt_metrics.tbt_times)
            self.total_decode_time = prompt_metrics.decode_time

            self.prompt_size = prompt_metrics.prompt_len
            self.num_decoded_tokens = prompt_metrics.decode_len

            self.prompt = prompt_metrics.prompt.prompt
            self.decoded_tokens = json.dumps(prompt_metrics.prompt.decoded_tokens)

            self.initial_time = prompt_metrics.get_initial_req_time()
            self.final_time = prompt_metrics.get_final_time()

            self.request_id = prompt_metrics.get_req_id()
            self.success = prompt_metrics.get_success()
        