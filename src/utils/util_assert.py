from src.data_structures.prompt_performance_metrics import PromptPerformanceMetrics

def assert_all_decode_size_equal(prompts: list[PromptPerformanceMetrics], target_out_size: int):
    for prompt_data in prompts:
        prompt = prompt_data.prompt
        assert len(prompt.decoded_tokens) == target_out_size