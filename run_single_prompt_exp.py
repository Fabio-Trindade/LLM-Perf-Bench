import asyncio
import argparse
import os
from pathlib import Path
from src.data_structures.csv_data_format import CSVDataFormat
from src.data_structures.prompt_performance_metrics import PromptPerformanceMetrics
from src.data_structures.prompt import Prompt
from src.binders.binder import Binder
from src.catalogs.config_catalog import ConfigCatalog
from src.registries.parser_registry import ParserRegistry
from src.components import *
from src.components.servers.server_interface import ServerI
from src.components.tokenizers.tokenizer_interface import TokenizerI
from src.factories.workload_components_factory import WorkloadComponentFactory
from src.parsers.args.cli_parser import CLIParser
from src.reporters.single_prompt_reporter import SinglePromptReporter
from src.prompt_generator.prompt_generator import PromptGeneratorBase
from src.components.dataset_generator.dataset_generator_interface import DatasetGenI
from src.components.requesters.requester_interface import RequesterI
from src.workload_launcher import WorkloadLauncher
from src.utils.util_logging import config_logging
import logging
from src.stats_generator.stats_generator import StatsGenerator
from src import components
from src import parsers
from src.utils.util_import import import_all_modules
from src.utils.single_csv_writer import SingleCSVWriter
import_all_modules(components)
import_all_modules(parsers)

@Binder.create_parse_from_config(ConfigCatalog._single_prompt_exp_config)
def add_single_prompt_args(): pass   

def assert_all_decode_size_equal(prompts: list[PromptPerformanceMetrics], target_out_size: int):
    for prompt_data in prompts:
        prompt = prompt_data.prompt
        assert len(prompt.decoded_tokens) == target_out_size

def run(args):
    config = args
    config_logging(config.logging)

    logging.info("----------- Running Single Prompt Experiment -----------")
    logging.info(f"Tensorboard plots and configuration were saved in ... {'path_to_tensorboard'}\n")
    SingleCSVWriter.initialize(CSVDataFormat(None, None, None), config.path_to_csv_filename)

    prompt_sizes = args.prompt_sizes

    prompt_generator = PromptGeneratorBase()
    all_results = []
    all_host_data = []
    all_accelerator_data = []

    workload_components = WorkloadComponentFactory.build_components_from_config(config)

    tokenizer: TokenizerI = workload_components.tokenizer
    dataset_gen: DatasetGenI= workload_components.dataset_gen
    server: ServerI = workload_components.server
    requester: RequesterI = workload_components.requester

    for i,prompt_size in enumerate(prompt_sizes):
        config.prompt_size_range = (prompt_size,prompt_size)
    
        dataset_gen: DatasetGenI = type(dataset_gen)(config)
        
        prompts = dataset_gen.gen_dataset(tokenizer, prompt_generator)
        
        launcher = WorkloadLauncher(
            config, server, requester, prompts
        )

        result = asyncio.run(launcher.async_run(i == 0, i == len(prompt_sizes) - 1))
        all_prompts = result.get_all_prompts()
        assert_all_decode_size_equal(all_prompts, config.max_out_tokens)

        for prompt in all_prompts:
            data = CSVDataFormat(config.experiment_key, args.model, prompt)
            SingleCSVWriter.write(data)

        data = StatsGenerator.gen_single_prompt_data_from_metrics(result)
        host_data = launcher.get_host_data()
        acc_data = launcher.get_accelerator_data()

        all_results.append(data)
        all_host_data.append(host_data)
        all_accelerator_data.append(acc_data)
    lines = []
    SinglePromptReporter.report_single_prompt_perf_md(lines, config, all_results, prompt_sizes, args.model_name_alias)
    SinglePromptReporter.report_host_usage_md(lines, all_host_data, prompt_sizes)
    SinglePromptReporter.report_accelerator_usage_md(lines, all_accelerator_data, prompt_sizes)
    # Save csv; save tensorboard (maybe)
    logging.warning("Continue the implementation [file=run_single_prompt.py]")
    output_dir = config.path_to_save_results + config.experiment_key

    filename = "report.md"
    os.makedirs(output_dir, exist_ok=True)
    markdown_path = Path(output_dir) / filename

    SingleCSVWriter.save(config.path_to_csv_filename)
    with open(markdown_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ Markdown report saved to: {markdown_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    time = 0.1

    fixed_args = {
        "num_prompts": 1,
        "infer_sleep_time": 0.11,
        "prompts_per_request": 1,
        "prompt_gen_sleep_time": time,
        "requester_sleep_time": time,
        "num_prompt_gen_threads": 1,
        "num_requester_threads": 1,
        "run_time": time
    }

    ParserRegistry.registry()(add_single_prompt_args)
    CLIParser.parse_all(parser, fixed_args)
    args = parser.parse_args()
    run(args)

