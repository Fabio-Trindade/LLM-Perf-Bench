import argparse
import asyncio
import json
import os
from pathlib import Path
from src.parsers.args.cli_parser import CLIParser
from src.registries.parser_registry import ParserRegistry
from src.data_structures.csv_data_format import CSVDataFormat
from src.components.servers.server_interface import ServerI
from src.components.tokenizers.tokenizer_interface import TokenizerI
from src.factories.workload_components_factory import WorkloadComponentFactory
from src.prompt_generator.prompt_generator import PromptGeneratorBase
from src.components.dataset_generator.dataset_generator_interface import DatasetGenI
from src.components.requesters.requester_interface import RequesterI
from src.workload_launcher import WorkloadLauncher
from src.utils.util_logging import config_logging
import logging
from src.utils.single_csv_writer import SingleCSVWriter
from src.utils.util_assert import assert_all_decode_size_equal
from fractions import Fraction
class LoadResults:
    def __init__(self,):
        self.all_results = []
        self.all_host_data = []
        self.all_accelerator_data = []
    
    def add_data(self, results, host_data, acc_data):
        self.all_results.append(results)
        self.all_host_data.append(host_data)
        self.all_accelerator_data.append(acc_data)

    def get_all(self):
        return self.all_results, self.all_host_data, self.all_accelerator_data

def run_workload(config, tokenizer, prompt_generator, server, requester, dataset_gen, is_first_iter, is_last_iter):
    dataset_gen: DatasetGenI = type(dataset_gen)(config)
        
    prompts = dataset_gen.gen_dataset(tokenizer, prompt_generator)
    
    launcher = WorkloadLauncher(
        config, server, requester, prompts
    )

    result = asyncio.run(launcher.async_run(is_first_iter, is_last_iter))
    
    host_data = launcher.get_host_data()
    acc_data = launcher.get_accelerator_data()
    return result, host_data, acc_data

def get_components_from_config(config):
    workload_components = WorkloadComponentFactory.build_components_from_config(config)

    tokenizer: TokenizerI = workload_components.tokenizer
    dataset_gen: DatasetGenI = workload_components.dataset_gen
    server: ServerI = workload_components.server
    requester: RequesterI = workload_components.requester

    return tokenizer, dataset_gen, server, requester 

def write_markdown(lines,config):
    output_dir = config.path_to_save_results + config.experiment_key
    filename = "report.md"
    os.makedirs(output_dir, exist_ok=True)
    markdown_path = Path(output_dir) / filename

    SingleCSVWriter.save(config.path_to_csv_filename)
    with open(markdown_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ Markdown report saved to: {markdown_path}")

def write_config(config):
    output_dir = config.path_to_save_results + config.experiment_key
    filename = "config.json"
    os.makedirs(output_dir, exist_ok=True)
    config_path = Path(output_dir) / filename
    json_data = {var_name: getattr(config, var_name) for var_name in vars(config)}
    if "api_key" in json_data:
        del json_data["api_key"]  
    with open(config_path, "w") as f:
        f.write(json.dumps(json_data))

    print(f"✅ Configuration saved to: {config_path}")

def initialize_writer(config):
    output_dir = Path(config.path_to_csv_filename).parent
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(config.path_to_csv_filename) 

    SingleCSVWriter.initialize(CSVDataFormat(None,None,None,None), csv_path)
def write_results(config, results):
    num_requesters = config.num_requesters if hasattr(config, "num_requesters") else None  
    requester_sleep_times = config.requester_sleep_times if hasattr(config, "requester_sleep_times") else None  
    for i, result in enumerate(results):
        if num_requesters:
            req_per_sec = num_requesters[i]
        elif requester_sleep_times:
            req_per_sec = config.num_requester_threads / requester_sleep_times[i]
        else:
            req_per_sec = config.num_requester_threads / config.requester_sleep_time
            
        for prompt in result.get_all_prompts():
            csv_data = CSVDataFormat(config.experiment_key,config.model, req_per_sec, prompt)
            SingleCSVWriter.write(csv_data)

def write_csv_from_results(results, config):
    initialize_writer(config)
    write_results(config,results)
    save_results(config)

def save_results(config):
    csv_path = Path(config.path_to_csv_filename)
    SingleCSVWriter.save(csv_path)
    print(f"✅ Results saved to: {csv_path}")

def finish_experiment(all_results, config):
    write_config(config)
    write_csv_from_results(all_results, config)

def get_args_from_parser(parser: argparse.ArgumentParser, fixed_args, *function_to_registry):
    for func in function_to_registry:
        ParserRegistry.registry()(func)
    CLIParser.parse_all(parser, fixed_args)
    return parser.parse_args()