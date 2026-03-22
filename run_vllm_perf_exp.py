import argparse
import math
import multiprocessing
import os
import json
from src.load_results import LoadResults
from src.prompt_generator.prompt_generator import PromptGeneratorBase
from src.components.servers.vllm_server import vLLMServer
from find_max_request_rate import find_max_request_rate, parse_best_req_rate_args
from src.registries.component_registry import ComponentRegistry
from src.utils.util_experiment import finish_experiment, get_args_from_parser, get_components_from_config
from optimize_vllm_param import optimize_vllm_parameter, parse_best_batch_args, find_idx_of_arg
from src.catalogs.config_catalog import ConfigCatalog
from run_intervaled_load_exp import add_intervaled_load_exp_args, run_intervaled_load_exp
from src.utils.util_logging import config_logging

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
        "best_request_rate": None,
    }

def run_step(state, key, state_filepath, func):
    if state.get(key, None) is not None:
        print(f"[INFO] Loaded value '{state[key]}' for key '{key}'", flush=True)
        return state[key]
    value = func()
    state[key] = value
    save_exp_state(state, state_filepath)
    return value

if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn", force=True)

    parser_verification = argparse.ArgumentParser()
    parser_verification.add_argument('--experiment_key', type=str, required=True)
    parser_verification.add_argument('--path_to_save_results', type=str, required=True)
    verification_args, _ = parser_verification.parse_known_args()

    state_filepath = os.path.join(os.path.dirname(verification_args.path_to_save_results), "state.json")
    exp_state = load_exp_state(state_filepath)

    if exp_state.get("finished") is True:
        print(f"Experiment with key {verification_args.experiment_key} already completed. Exiting.", flush=True)
        exit(0)

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
        print("[INFO] No vLLM parameter to optimize.", flush=True)
        return None

    optimized_vllm_param = run_step(exp_state, vllm_param_to_optimize, state_filepath, optimize_vllm)

    fixed_args = {
        ConfigCatalog._load_config.seed.name: args.seed,
        ConfigCatalog._vllm_config.port.name: args.port,
        ConfigCatalog._load_config.prompts_per_request.name: 1,
        ConfigCatalog._experiment_config.model.name: args.model,
        ComponentRegistry.server: ComponentRegistry.dummy
    }

    args = get_args_from_parser(
        parser,
        fixed_args,
        parse_best_req_rate_args,
        add_intervaled_load_exp_args
    )

    config_logging(args.logging)

    try:
        if optimized_vllm_param:
            param_idx = find_idx_of_arg(args.vllm_serve_args, vllm_param_to_optimize)
            args.vllm_serve_args[param_idx + 1] = str(optimized_vllm_param)

        print("\n=========== Initializing vLLM server ===========\n", flush=True)
        server = vLLMServer(args)
        server.init()
        print("=========== Server initialized successfully ===========\n", flush=True)
    except Exception as e:
        print("Failed to start vLLM server:", flush=True)
        raise e

    try:
        def find_best_num_request_rate_func():
            print("\n\n=========== Running load simulations to find max request rate ===========\n", flush=True)
            best_request_rate, max_thp_req = find_max_request_rate(config=args)
            print(f"[RESULT] Max request rate: {best_request_rate}, throughput: {max_thp_req:.2f} tokens/s\n", flush=True)
            return best_request_rate
        
        args.request_rate_per_requester = run_step(exp_state, "best_request_rate", state_filepath, find_best_num_request_rate_func)

        print(f"\n=========== Running intervaled load experiment using max request rate {args.request_rate_per_requester} tok/s ===========\n", flush=True)
        tokenizer, dataset_gen, _, requester = get_components_from_config(args)
        prompt_generator = PromptGeneratorBase()
        prompts = dataset_gen.gen_dataset(tokenizer, prompt_generator)
        results = run_intervaled_load_exp(args, requester, server, prompts)
        print("\n=========== Intervaled load experiment completed successfully ===========\n", flush=True)

        all_results, all_host_data, all_accelerator_data = results.get_all()
        finish_experiment(all_results, args)

        print("\n=========== Experiment Summary ===========", flush=True)
        print(f"Optimized vLLM param: {optimized_vllm_param}", flush=True)
        print(f"Max request rate: {args.request_rate_per_requester}", flush=True)

        exp_state["finished"] = True
        save_exp_state(exp_state, state_filepath)
        server.shutdown()
        print("Server shut down. Experiment completed successfully.\n", flush=True)

    except KeyboardInterrupt:
        server.shutdown()
        print("Experiment interrupted. Server shut down.", flush=True)
    except Exception as e:
        server.shutdown()
        raise e
