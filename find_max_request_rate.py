import argparse
import asyncio
import math
from src.prompt_generator.prompt_generator import PromptGeneratorBase
from src.distributed_workload import DistributedWorkload
from run_load_exp import add_load_exp_args, config_load_exp
from src.utils.util_import import initialize_modules
from src.utils.util_experiment import get_args_from_parser
from src.utils.util_parse import add_arg

initialize_modules()
    
def parse_best_req_rate_args(parser, fixed_args):
    add_arg(parser, "initial_request_rate", fixed_args, type=float, default=1,
            help="Initial run time in seconds."             )
    add_arg(parser, "increment_request_rate_factor", fixed_args, type=float, default=60,
            help="Increment time in seconds.")
    add_arg(parser, "epsilon_request_rate", fixed_args, type=float, default=0.05,
            help="Epsilon for throughput convergence.")


def find_max_request_rate(config):
    tokenizer, dataset_gen, server, requester, prompt_generator = config_load_exp(config)
    prompt_generator = PromptGeneratorBase()
    prompts = dataset_gen.gen_dataset(tokenizer, prompt_generator)
    server.init()
    epsilon = config.epsilon_request_rate
    increment_value = config.increment_request_rate_factor
    cur_req_rate = config.initial_request_rate
    last_thp = None
    max_thp = -math.inf

    while True:
        config.request_rate_per_requester = cur_req_rate
        load_results =  DistributedWorkload.run_multiprocess_workload(config,  type(requester), prompts)
        # load_results =  asyncio.run(DistributedWorkload.run_single_async_workload(config, requester, prompts))
        perf_results = load_results[0]
        if last_thp is None:
            last_thp = perf_results.calc_total_throughput()
        else:
            cur_thp = perf_results.calc_total_throughput()
            max_thp = max(max_thp, cur_thp)
            if 1 - epsilon <= cur_thp/last_thp <= 1 + epsilon:
                print(f"Last thp: {last_thp} - Current thp: {cur_thp} - var = {cur_thp/last_thp}")
                break
            else:
                print(f"Last thp: {last_thp} - Current thp: {cur_thp} - var = {cur_thp/last_thp}")
                last_thp = cur_thp
        cur_req_rate = increment_value * cur_req_rate

    server.shutdown()
    return cur_req_rate, last_thp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    fixed_args = {
      
    }

    args = get_args_from_parser(parser, fixed_args, add_load_exp_args, parse_best_req_rate_args)

    args = parser.parse_args()
    best_num_req, thp = find_max_request_rate(config = args)
    print(f"Best num requester: {best_num_req} with throughput {thp}") 



