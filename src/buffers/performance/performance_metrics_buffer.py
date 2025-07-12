from copy import deepcopy
from src.data_structures.prompt_performance_metrics import PromptPerformanceMetrics
from typing import Any
import threading

class PerformanceMetricsBuffer():
    def __init__(self):
        self.metrics: dict[Any, PromptPerformanceMetrics] = {}
        # self._lock = threading.Lock()

    def empty(self) -> bool:
        return len(self.metrics) == 0
        
    def initialize_metrics(self, prompt, key: Any, req_id:int , start_request_time: bool) -> None:
        assert key not in self.metrics, f"Metrics for key {key} already exists."
        prompt_metrics = PromptPerformanceMetrics(deepcopy(prompt), req_id)
        if start_request_time:
            prompt_metrics.start_req_time()
        # with self._lock:
        self.metrics[key] = prompt_metrics
        
    def add_decode_data(self, key, time, token):
        self.metrics[key].add_out_token_time(time)
        self.metrics[key].add_decoded_token(token)
        
            
    def add_out_token_time(self, key, time):
        self.metrics[key].add_out_token_time(time)
    
    def add_decoded_token(self, key, token):
        self.metrics[key].add_decoded_token(token)

    def calc_perf_times(self):
        for metric in self.metrics.values():
            metric.calc_results()
    
    def get_prompts_grouped_by_request(self) -> list[list[PromptPerformanceMetrics]]:
        req_to_prompts: dict[int, list[PromptPerformanceMetrics]] = {}
        for prompt in self.metrics.values():
            req_id = prompt.get_req_id()
            if req_id not in req_to_prompts:
                req_to_prompts[req_id] = []
            req_to_prompts[req_id].append(prompt)
        return list(req_to_prompts.values())
    
    def get_all_prompts(self):
        return [prompt for prompt in self.metrics.values()]
    
            