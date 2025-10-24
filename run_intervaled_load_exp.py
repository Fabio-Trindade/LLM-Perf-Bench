import argparse
import logging
from src.prompt_generator.prompt_generator import PromptGeneratorBase
from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog
from src.utils.util_import import initialize_modules
from src.utils.util_experiment import LoadResults, finish_experiment, get_args_from_parser, get_components_from_config, run_workload_loop
from src.utils.util_logging import config_logging

initialize_modules()

@Binder.create_parse_from_config([ConfigCatalog._intervaled_load_config])
def add_intervaled_load_exp_args(): pass

def run_intervaled_load_exp(config):
    config_logging(config.logging)

    logging.info(f"----------- Running Intervaled Load Experiment -----------")

    prompt_size = config.prompt_size
    
    config.prompt_size_range = (prompt_size, prompt_size)
    config.max_out_tokens_range = (config.max_out_tokens, config.max_out_tokens)
    prompt_generator = PromptGeneratorBase()
    
    tokenizer, dataset_gen, server, requester = get_components_from_config(config)
    server.init()

    results = LoadResults()
    total_intervals = int(100/config.interval_percentage) 
    max_out_tokens = config.max_out_tokens

    tokens_per_sec_frac = (config.max_tokens_per_sec)/total_intervals
    targets_thp = [tokens_per_sec_frac*i for i in range(1, total_intervals + 1)] 
    num_requesters = config.num_requester_threads
    num = num_requesters*(prompt_size + max_out_tokens) * config.prompts_per_request
    requester_sleep_times = [num/thp for thp in targets_thp]

    results = run_workload_loop(config, tokenizer, prompt_generator, server, requester, dataset_gen, requester_sleep_times, "requester_sleep_time")
    server.shutdown()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    fixed_args = {
        "prompts_per_request": 1,
        "num_prompts": 1,
        "prompt_sizes": None
    }

    args = get_args_from_parser(parser, fixed_args, add_intervaled_load_exp_args)
    results = run_intervaled_load_exp(config = args)
    all_results, all_host_data, all_accelerator_data = results.get_all()
    finish_experiment(all_results, args)

