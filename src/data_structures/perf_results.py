from src.data_structures.request_results import RequestResults
from src.data_structures.prompt_performance_metrics import PromptPerformanceMetrics

class PerfResults():
    def __init__(self, all_request_prompts: list[list[PromptPerformanceMetrics]], total_time):
        self.requests_metrics = [RequestResults(request)for request in all_request_prompts]
        self.total_time = total_time
        self.total_throughput = None
        self.cache = {}
    
    # def calc_throughputs(self):
    #     if self.throughput:
    #         return
        
    #     if self.throughput:
    #         return self.throughput
    #     processed_tokens = 0
    #     for req in self.requests_metrics:
    #         processed_tokens += req.get_total_processed_tokens()
    #     self.throughput = processed_tokens/self.total_time
    #     return self.throughput
    
    def get_metrics(self, var_name: str):
        if var_name in self.cache:
            return self.cache[var_name]
        
        metrics = []
        for req in self.requests_metrics:
            metrics.append(req.get_metrics(var_name))
        
        self.cache[var_name] = metrics
        return self.cache[var_name]

    def get_all_prompts (self):
        all_prompts = []
        for req in self.requests_metrics:
            all_prompts.extend(req.get_all_prompts())
        return all_prompts
