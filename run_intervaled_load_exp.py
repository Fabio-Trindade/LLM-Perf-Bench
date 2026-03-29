import argparse
import logging
import math
from src.distributed_workload import DistributedWorkload
from src.prompt_generator.prompt_generator import PromptGeneratorBase
from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog
from src.utils.util_import import initialize_modules
from src.utils.util_experiment import  finish_experiment, get_args_from_parser, get_components_from_config
from src.load_results import LoadResults
from src.utils.util_logging import config_logging

initialize_modules()

@Binder.create_parse_from_config([ConfigCatalog._intervaled_load_config])
def add_intervaled_load_exp_args(): pass

def run_intervaled_load_exp(config, requester, server, prompts):
    total_intervals = int(100/config.interval_percentage) 
    print(f"----------- Running Intervaled Load Experiment with {total_intervals} Intervals -----------")

    req_rate_per_interval = config.request_rate_per_requester/total_intervals
    request_rate_per_requester_list = [req_rate_per_interval*i for i in range(1, total_intervals + 1)] 
    workload_executor = DistributedWorkload(config, requester, prompts) 
    setattr(config, "request_rate_per_requester_list", request_rate_per_requester_list)
    results = workload_executor.run_param_loop(config, requester, prompts, request_rate_per_requester_list, "request_rate_per_requester")
    return results

def run():
    parser = argparse.ArgumentParser()
    
    fixed_args = {
        "prompts_per_request": 1,
        "num_prompts": 1
        }

    config = get_args_from_parser(parser, fixed_args, add_intervaled_load_exp_args)
    config_logging(config.logging)
    tokenizer, dataset_gen, server, requester = get_components_from_config(config)
    server.init()
    prompt_generator = PromptGeneratorBase()
    prompts = dataset_gen.gen_dataset(tokenizer, prompt_generator)
    results = run_intervaled_load_exp(config, requester, server, prompts)
    all_results, all_host_data, all_accelerator_data = results.get_all()
    server.shutdown()

    finish_experiment(all_results, config)
if __name__ == "__main__":
   run()

