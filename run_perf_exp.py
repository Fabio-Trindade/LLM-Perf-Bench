import argparse
import logging
import math
from find_best_num_requesters import find_best_num_requesters, parse_best_num_requester_args
from find_best_run_time import find_best_run_time, parse_best_run_time_args
from src.utils.util_experiment import LoadResults, get_args_from_parser
from find_best_num_parallel_batches_vllm import find_best_num_parallel_batches_vllm, parse_best_batch_args
from src.catalogs.config_catalog import ConfigCatalog
from run_intervaled_load_exp import add_intervaled_load_exp_args, run_intervaled_load_exp
from src.utils.vllm_utils import kill_vllm_server_process, start_and_wait_vllm_server
import multiprocessing as mp

def find_argmax_thp(results: LoadResults):
    max_thp = -math.inf
    arg_idx = -1
    for i, result in enumerate(results.all_results):
        thp = result.calc_total_throughput()
        if thp > max_thp:
            max_thp = thp
            arg_idx = i
    return arg_idx, max_thp


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parse_best_batch_args(parser,{})
    args = parser.parse_known_args()[0]
    
    print("\n\n=========== Finding max number of parallel sequences ===========\n", flush=True)

    # max_num_seqs = find_best_num_parallel_batches_vllm(
    #     args.model, args.prompt_size, args.max_out_tokens, args.initial_batch_size,
    #     args.dtype, args.gpu_memory_utilization, args.seed, args.port
    # )

    max_num_seqs = 224

    print(f"\n =========== Max number of parallel sequences found: {max_num_seqs} =========== \n\n", flush=True)

    fixed_args = {
        ConfigCatalog._load_exp_config.prompt_size.name: args.prompt_size,
        ConfigCatalog._load_exp_config.max_out_tokens.name: args.max_out_tokens,
        ConfigCatalog._load_exp_config.num_requesters.name: [None],
        ConfigCatalog._load_config.seed.name: args.seed,
        "port": args.port,
        ConfigCatalog._load_config.prompts_per_request.name: 1,
        ConfigCatalog._load_config.num_prompts.name: 1,
        ConfigCatalog._intervaled_load_config.max_tokens_per_sec.name: None,  # will update later
        ConfigCatalog._experiment_config.model.name: args.model
    }

    args = get_args_from_parser(
        parser, 
        fixed_args, 
        parse_best_run_time_args, 
        parse_best_num_requester_args, 
        add_intervaled_load_exp_args
    )

    print("\n=========== Initializing vLLM server ===========\n")
    server_process = start_and_wait_vllm_server(
        args.model,
        args.port,
        args.prompt_size + args.max_out_tokens,
        max_num_seqs,
        args.dtype,
        args.gpu_memory_utilization,
        args.seed
    )

    print("\n\n=========== Running load simulations to find best number of requesters ===========\n")
    best_num_requesters, max_thp_req = find_best_num_requesters(config=args)
    print(f"Best number of requesters from load simulations: {best_num_requesters} with throughput {max_thp_req}.\n")
    args.num_requester_threads = best_num_requesters

    print("\n\n=========== Running runtime test to find best run time ===========\n")
    best_time, thp_best_time = find_best_run_time(config=args)
    print(f"Best run time from run_time test: {best_time} - with throughput {thp_best_time}.\n")
    args.max_tokens_per_sec = thp_best_time
    args.run_time = best_time

    print("\\nn=========== Running intervaled load experiment ===========\n")
    run_intervaled_load_exp(config=args)
    print("\\nn=========== All experiments completed ===========\n")
    kill_vllm_server_process(server_process)

