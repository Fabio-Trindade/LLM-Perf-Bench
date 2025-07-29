import logging
import argparse
from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog
from src.utils.util_import import initialize_modules
from src.utils.util_experiment import *
from src.utils.util_assert import assert_all_decode_size_equal

initialize_modules()

@Binder.create_parse_from_config([ConfigCatalog._single_prompt_exp_config,
                                  ConfigCatalog._prompt_variation_config])
def add_single_prompt_args(): pass

def run_prompt_variation_exp_by_config(title, config):
    config_logging(config.logging)

    logging.info(f"----------- {title} -----------")
    
    config.max_out_tokens_range = (config.max_out_tokens, config.max_out_tokens)

    prompt_sizes = config.prompt_sizes
    prompt_generator = PromptGeneratorBase()
    tokenizer, dataset_gen, server, requester = get_components_from_config(config)
    results = LoadResults()

    for i,prompt_size in enumerate(prompt_sizes):
        config.prompt_size_range = (prompt_size,prompt_size)
        all_results = run_workload(config, tokenizer, prompt_generator, server, requester, dataset_gen, i == 0, i == len(prompt_sizes) - 1)
        results.add_data(*all_results)

    return  results.get_all()

def run(args):
    config = args
    config.max_out_tokens_range = (config.max_out_tokens, config.max_out_tokens)
    repeat_times = config.repeat_times
    initialize_writer(config)
    for i in range(repeat_times + 1):
        logging.info(f"Running repetition {i}")
        all_results, all_host_data, all_accelerator_data = run_prompt_variation_exp_by_config("Running Single Prompt Experiment", config)
    
        for result in all_results:
            all_prompts = result.get_all_prompts()
            assert_all_decode_size_equal(all_prompts, config.max_out_tokens)
        write_results(config, all_results)
    save_results(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    time = 0.1
    logging.warning("Remove infer_sleep_time from fixed args [file=run_single_prompt_exp.py]")
    fixed_args = {
        "num_prompts": 1,
        "prompts_per_request": 1,
        "prompt_gen_sleep_time": time,
        "requester_sleep_time": time,
        "num_prompt_gen_threads": 1,
        "num_requester_threads": 1,
        "run_time": time
    }

    args = get_args_from_parser(parser, fixed_args, add_single_prompt_args)
    run(args)

