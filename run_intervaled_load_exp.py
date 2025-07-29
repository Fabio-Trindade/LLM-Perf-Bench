import argparse
import logging
from src.prompt_generator.prompt_generator import PromptGeneratorBase
from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog
from src.utils.util_import import initialize_modules
from src.utils.util_experiment import LoadResults, finish_experiment, get_args_from_parser, get_components_from_config, run_workload
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
    
    results = LoadResults()
    total_intervals = int(100/config.interval_percentage) 
    prompt_size = config.prompt_size 
    max_out_tokens = config.max_out_tokens
    tokens_per_sec_frac = config.max_tokens_per_sec/total_intervals
    targets_thp = [tokens_per_sec_frac*i for i in range(1, total_intervals + 1)] 
    num_requesters = config.num_requester_threads
    num = num_requesters*(prompt_size + max_out_tokens) * config.prompts_per_request
    requester_sleep_times = [num/thp for thp in targets_thp]

    # num_prompt_gen_threads = config.num_prompt_gen_threads
    setattr(config, "requester_sleep_times", requester_sleep_times )
    for i,value in enumerate(requester_sleep_times):
        config.requester_sleep_time = value
        # for async implentation:
        # req_per_sec = num_requesters/value
        # config.prompt_gen_sleep_time = num_prompt_gen_threads/req_per_sec
        all_results = run_workload(config, tokenizer, prompt_generator, server, requester, dataset_gen,  i == 0, i == len(requester_sleep_times) - 1)
        results.add_data(*all_results)

    all_results, all_host_data, all_accelerator_data =  results.get_all()
 
    finish_experiment(all_results, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    fixed_args = {
        "prompts_per_request": 1,
        "num_prompts": 1,
        "prompt_sizes": None
    }

    args = get_args_from_parser(parser, fixed_args, add_intervaled_load_exp_args)
    run_intervaled_load_exp(config = args)
