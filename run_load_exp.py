import argparse
import logging
from src.prompt_generator.prompt_generator import PromptGeneratorBase
from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog
from src.utils.util_import import initialize_modules
from src.utils.util_experiment import finish_experiment, get_args_from_parser, get_components_from_config, run_workload_loop
from src.utils.util_logging import config_logging

initialize_modules()

@Binder.create_parse_from_config(ConfigCatalog._load_exp_config)
def add_load_exp_args(): pass

def config_load_exp(config):
    config_logging(config.logging)
    prompt_size = config.prompt_size
    config.prompt_size_range = (prompt_size, prompt_size)
    config.max_out_tokens_range = (config.max_out_tokens, config.max_out_tokens)
    prompt_generator = PromptGeneratorBase()
    
    tokenizer, dataset_gen, server, requester = get_components_from_config(config)
    return tokenizer, dataset_gen, server, requester, prompt_generator
    
def run_load_exp(config):
    tokenizer, dataset_gen, server, requester, prompt_generator = config_load_exp(config)
    server.init()
    logging.info(f"----------- Running Load Experiment -----------")

    num_requesters = config.num_requesters
    results = run_workload_loop(config, tokenizer, prompt_generator, server, requester, dataset_gen, num_requesters, "num_requester_threads")
    server.shutdown()
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    fixed_args = {
        "num_prompts": 1
    }

    args = get_args_from_parser(parser, fixed_args, add_load_exp_args)
    results = run_load_exp(config = args)
    all_results, all_host_data, all_accelerator_data = results.get_all()
    finish_experiment(all_results,args)
