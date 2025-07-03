import itertools
from src.data_structures.prompt_performance_metrics import PromptPerformanceMetrics
from itertools import chain


class RequestResults():
    def __init__(self, prompts: list[PromptPerformanceMetrics]):
        self.req_id = prompts[0].get_req_id()
        self.prompts = prompts

    def get_total_processed_tokens(self):
        processed_tokens = 0
        for prompt in self.prompts:
            processed_tokens += prompt.get_total_processed_tokens()
        
        return processed_tokens
    
    def get_metrics(self, var_name: str):
        all_req_data = []
        for prompt in self.prompts:
            all_req_data.append(prompt.get_metrics(var_name))
        return all_req_data
    
    def get_all_prompts(self):
        return [prompt for prompt in self.prompts]
    
