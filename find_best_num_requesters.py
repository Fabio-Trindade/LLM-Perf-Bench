import argparse
import copy
import math
from run_load_exp import add_load_exp_args, config_load_exp
from src.utils.util_import import initialize_modules
from src.utils.util_experiment import get_args_from_parser, run_workload_loop
from src.utils.util_parse import add_arg

initialize_modules()
    
def parse_best_num_requester_args(parser, fixed_args):
    add_arg(parser, "initial_num_requester", fixed_args, type=int, default=60,
            help="Initial run time in seconds.")
    add_arg(parser, "max_num_requesters", fixed_args, type=int, default=600,
            help="Maximum run time in seconds.")
    add_arg(parser, "increment_requester_value", fixed_args, type=float, default=60,
            help="Increment time in seconds.")
    add_arg(parser, "epsilon_requester", fixed_args, type=float, default=0.05,
            help="Epsilon for throughput convergence.")


def find_best_num_requesters(config):
    tokenizer, dataset_gen, server, requester, prompt_generator = config_load_exp(config)
    server.init()
    epsilon = config.epsilon_requester
    max_num_requester = config.max_num_requesters
    increment_value = config.increment_requester_value
    cur_num_requester = config.initial_num_requester
    last_thp = None
    max_thp = -math.inf

    while cur_num_requester <= max_num_requester:
        config.num_requester_threads = cur_num_requester
        load_results = run_workload_loop(config, tokenizer, prompt_generator, server, requester, dataset_gen)
        perf_results = load_results.all_results[0]
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
        cur_num_requester = math.ceil(increment_value * cur_num_requester)

    server.shutdown()
    return cur_num_requester, max_thp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    fixed_args = {
        "num_prompts": 1,
        "num_requesters": [None]
    }

    args = get_args_from_parser(parser, fixed_args, add_load_exp_args, parse_best_num_requester_args)

    args = parser.parse_args()
    best_num_req, thp = find_best_num_requesters(config = args)
    print(f"Best num requester: {best_num_req} with throughput {thp}") 



