import argparse
import math
import os
import json
from pathlib import Path
import multiprocessing as mp
import pandas as pd

from src.components.servers.vllm_server import vLLMServer
from find_best_num_requesters import find_best_num_requesters, parse_best_num_requester_args
from find_best_run_time import find_best_run_time, parse_best_run_time_args
from src.registries.component_registry import ComponentRegistry
from src.utils.util_experiment import LoadResults, finish_experiment, get_args_from_parser
from optimize_vllm_param import optimize_vllm_parameter, parse_best_batch_args, find_idx_of_arg
from src.catalogs.config_catalog import ConfigCatalog
from run_intervaled_load_exp import add_intervaled_load_exp_args, run_intervaled_load_exp
from copy import deepcopy

def find_argmax_thp(results: LoadResults):
    max_thp = -math.inf
    arg_idx = -1
    for i, result in enumerate(results.all_results):
        thp = result.calc_total_throughput()
        if thp > max_thp:
            max_thp = thp
            arg_idx = i
    return arg_idx, max_thp


def save_exp_state(state, state_filepath):
    os.makedirs(os.path.dirname(state_filepath), exist_ok=True)
    with open(state_filepath, "w") as file:
        json.dump(state, file)


def load_exp_state(state_filepath):
    if os.path.exists(state_filepath):
        with open(state_filepath, "r") as file:
            return json.load(file)
    os.makedirs(os.path.dirname(state_filepath), exist_ok=True)
    return {
        "finished": None,
        "best_num_requesters": None,
    }


def run_step(state, key, state_filepath, func):
    if state.get(key, None) is not None:
        print(f"[INFO] Loaded value '{state[key]}' for key '{key}'")
        return state[key]
    value = func()
    state[key] = value
    save_exp_state(state, state_filepath)
    return value


if __name__ == "__main__":
    parser_verification = argparse.ArgumentParser()
    parser_verification.add_argument('--experiment_key', type=str, required=True)
    parser_verification.add_argument('--path_to_save_results', type=str, required=True)
    verification_args, _ = parser_verification.parse_known_args()

    state_filepath = os.path.join(os.path.dirname(verification_args.path_to_save_results), "state.json")
    exp_state = load_exp_state(state_filepath)

    if exp_state.get("finished") is True:
        print(f"Experiment with key {verification_args.experiment_key} already completed. Exiting.")
        exit(0)

    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parse_best_batch_args(parser, {})
    args, _ = parser.parse_known_args()
    vllm_param_to_optimize = args.vllm_param_to_optimize

    def optimize_vllm():
        if vllm_param_to_optimize:
            print(f"\n\n=========== Optimizing {vllm_param_to_optimize} ===========\n", flush=True)
            value = optimize_vllm_parameter(
                args.port, args.model, args.seed, args.vllm_serve_args, vllm_param_to_optimize
            )
            print(f"\n=========== Optimized value: {value} ===========\n\n", flush=True)
            return value
        return None

    optimized_vllm_param = run_step(exp_state, vllm_param_to_optimize, state_filepath, optimize_vllm)

    fixed_args = {
        ConfigCatalog._load_exp_config.num_requesters.name: [None],
        ConfigCatalog._load_config.seed.name: args.seed,
        ConfigCatalog._vllm_config.port.name: args.port,
        ConfigCatalog._load_config.prompts_per_request.name: 1,
        ConfigCatalog._load_config.num_prompts.name: 1,
        ConfigCatalog._intervaled_load_config.max_tokens_per_sec.name: None,
        ConfigCatalog._experiment_config.model.name: args.model,
        ComponentRegistry.server: ComponentRegistry.dummy
    }

    args = get_args_from_parser(
        parser,
        fixed_args,
        parse_best_num_requester_args,
        add_intervaled_load_exp_args
    )

    try:
        if optimized_vllm_param:
            param_idx = find_idx_of_arg(args.vllm_serve_args, vllm_param_to_optimize)
            args.vllm_serve_args[param_idx + 1] = str(optimized_vllm_param)
        print("\n=========== Initializing vLLM server ===========\n")
        server = vLLMServer(args)
        server.init()
        print("=========== Server initialized successfully ===========")
    except Exception as e:
        print("Failed to start vLLM server:")
        raise e

    try:
        temp_config = deepcopy(args)
        temp_config.run_time = 1
        def find_best_num_requesters_func():
            print("\n\n=========== Running load simulations to find best number of requesters ===========\n")
            best_num_requesters, max_thp_req = find_best_num_requesters(config=temp_config)
            print(f"Best number of requesters from load simulations: {best_num_requesters} with max throughput {max_thp_req}.\n")
            return best_num_requesters
        
        args.num_requester_threads = run_step(exp_state, "best_num_requesters", state_filepath, find_best_num_requesters_func)

        args.max_tokens_per_sec = args.prompt_size * args.num_requester_threads * args.prompts_per_request/args.requester_sleep_time

        print(f"\n=========== Running intervaled load experiment using max {args.max_tokens_per_sec} tok/s ===========\n")
        results = run_intervaled_load_exp(config=args)
        print("\n=========== All experiments completed ===========\n")
        all_results, all_host_data, all_accelerator_data = results.get_all()
        finish_experiment(all_results, args)

        exp_state["finished"] = True
        save_exp_state(exp_state, state_filepath)
        server.shutdown()

    except KeyboardInterrupt:
        server.shutdown()
        print("Experiment interrupted. Server shut down.")
    except Exception as e:
        server.shutdown()
        raise e
