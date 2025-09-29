import argparse
import copy
from run_load_exp import add_load_exp_args, config_load_exp
from src.utils.util_import import initialize_modules
from src.utils.util_experiment import get_args_from_parser, run_workload
from src.utils.util_parse import add_arg

initialize_modules()
    
def parse_best_run_time_args(parser, fixed_args):
    add_arg(parser, "initial_run_time", fixed_args, type=int, default=60,
            help="Initial run time in seconds.")
    add_arg(parser, "max_run_time", fixed_args, type=int, default=600,
            help="Maximum run time in seconds.")
    add_arg(parser, "increment_time", fixed_args, type=int, default=60,
            help="Increment time in seconds.")
    add_arg(parser, "epsilon_run_time", fixed_args, type=float, default=0.05,
            help="Epsilon for throughput convergence.")


def find_best_run_time(config):
    tokenizer, dataset_gen, server, requester, prompt_generator = config_load_exp(config)
    epsilon = config.epsilon_run_time
    max_run_time = config.max_run_time
    increment_time = config.increment_time
    cur_time = config.initial_run_time
    last_thp = None
    
    while cur_time <= max_run_time:
        config.run_time = cur_time
        perf_results = run_workload(config, tokenizer, prompt_generator, server, requester, dataset_gen, last_thp == None, cur_time == max_run_time)[0]
        
        if last_thp is None:
            last_thp = perf_results.calc_total_throughput()
        else:
            cur_thp = perf_results.calc_total_throughput()
            if 1 - epsilon <= cur_thp/last_thp <= 1 + epsilon:
                print(f"Last thp: {last_thp} - Current thp: {cur_thp} - var = {cur_thp}/{last_thp}")
                break
            else:
                print(f"Last thp: {last_thp} - Current thp: {cur_thp} - var = {cur_thp}/{last_thp}")
                last_thp = cur_thp
        cur_time += increment_time

    server.shutdown()
    return cur_time - increment_time, last_thp
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    fixed_args = {
        "num_prompts": 1
    }

    args = get_args_from_parser(parser, fixed_args, add_load_exp_args, parse_best_run_time_args)
    assert(len(args.num_requesters) == 1, "Use only one value for num_requesters.")

    args = parser.parse_args()
    best_time, thp = find_best_run_time(config = args)
    print(f"Best run_time: {best_time} with throughput {thp}") 



