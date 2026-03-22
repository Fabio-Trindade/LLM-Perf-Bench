import argparse
import logging
from src.distributed_workload import DistributedWorkload
from src.prompt_generator.prompt_generator import PromptGeneratorBase
from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog
from src.utils.util_import import initialize_modules
from src.utils.util_experiment import finish_experiment, get_args_from_parser, get_components_from_config
from src.utils.util_logging import config_logging

initialize_modules()

@Binder.create_parse_from_config(ConfigCatalog._load_exp_config)
def add_load_exp_args(): pass

def config_load_exp(config):
    config_logging(config.logging)
    prompt_generator = PromptGeneratorBase()
    
    tokenizer, dataset_gen, server, requester = get_components_from_config(config)
    return tokenizer, dataset_gen, server, requester, prompt_generator
    
def run_load_exp(config, requester, prompts):
    logging.info(f"----------- Running Load Experiment -----------")
    request_rates_per_requester = config.request_rates_per_requester
    results = DistributedWorkload.run_param_loop(config, requester, prompts, request_rates_per_requester, "request_rate_per_requester")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args_from_parser(parser, {}, add_load_exp_args)
    tokenizer, dataset_gen, server, requester, prompt_generator = config_load_exp(args)
    prompts = dataset_gen.gen_dataset(tokenizer,prompt_generator)
    server.init()
    try:
        results = run_load_exp(args, requester, prompts)
        server.shutdown()
    except Exception as e:
        server.shutdown()
        raise RuntimeError(e)
    all_results, all_host_data, all_accelerator_data = results.get_all()
    finish_experiment(all_results,args)
            