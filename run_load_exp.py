import argparse
import logging
from src.prompt_generator.prompt_generator import PromptGeneratorBase
from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog
from src.utils.util_import import initialize_modules
from src.utils.util_experiment import LoadResults, finish_experiment, get_args_from_parser, get_components_from_config, run_workload
from src.utils.util_logging import config_logging

initialize_modules()

@Binder.create_parse_from_config(ConfigCatalog._load_exp_config)
def add_load_exp_args(): pass

def run_load_exp(config):
    config_logging(config.logging)

    logging.info(f"----------- Running Load Experiment -----------")

    prompt_size = config.prompt_size
    config.prompt_size_range = (prompt_size, prompt_size)
    config.max_out_tokens_range = (config.max_out_tokens, config.max_out_tokens)
    prompt_generator = PromptGeneratorBase()
    
    tokenizer, dataset_gen, server, requester = get_components_from_config(config)
    
    results = LoadResults()
    num_requesters = config.num_requesters

    for i,value in enumerate(num_requesters):
        config.num_requester_threads = value
        all_results = run_workload(config, tokenizer, prompt_generator, server, requester, dataset_gen,  i == 0, i == len(num_requesters) - 1)
        results.add_data(*all_results)

    all_results, all_host_data, all_accelerator_data = results.get_all()
    logging.warning("Continue the implementation [file=run_single_prompt.py]. Missing host and GPU usage")
    finish_experiment(all_results,config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    fixed_args = {
        "num_prompts": 1
    }

    args = get_args_from_parser(parser, fixed_args, add_load_exp_args)
    run_load_exp(config = args)
