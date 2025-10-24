import argparse
import math
from pathlib import Path
from src.components.servers.vllm_server import vLLMServer
from find_best_num_requesters import find_best_num_requesters, parse_best_num_requester_args
from find_best_run_time import find_best_run_time, parse_best_run_time_args
from src.registries.component_registry import ComponentRegistry
from src.utils.util_experiment import LoadResults, finish_experiment, get_args_from_parser
from find_best_num_parallel_batches_vllm import find_best_num_parallel_batches_vllm, parse_best_batch_args
from src.catalogs.config_catalog import ConfigCatalog
from run_intervaled_load_exp import add_intervaled_load_exp_args, run_intervaled_load_exp
import multiprocessing as mp
import pandas as pd

def find_argmax_thp(results: LoadResults):
    max_thp = -math.inf
    arg_idx = -1
    for i, result in enumerate(results.all_results):
        thp = result.calc_total_throughput()
        if thp > max_thp:
            max_thp = thp
            arg_idx = i
    return arg_idx, max_thp

def verify_experiment_completed(config):
    csv_path = Path(config.path_to_csv_filename)
    experiment_key = config.experiment_key
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        completed_experiments = df['experiment_key'].unique().tolist()
        if experiment_key in completed_experiments:
            return True
    return False

if __name__ == "__main__":  
    # ---------------------------------------------
    parser_verification = argparse.ArgumentParser()
    parser_verification.add_argument('--experiment_key', type=str, required=True)
    parser_verification.add_argument('--path_to_csv_filename', type=str, required=True)
    verification_args = parser_verification.parse_known_args()[0]
    if verify_experiment_completed(verification_args):
        print(f"Experiment with key {verification_args.experiment_key} already completed. Exiting.")
        exit(0)
    #----------------------------------------------

    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parse_best_batch_args(parser,{})
    args= parser.parse_known_args()[0]
    print("\n\n=========== Finding max number of parallel sequences ===========\n", flush=True)

    max_num_seqs = find_best_num_parallel_batches_vllm(
        args.port,  args.model, args.seed, args.vllm_serve_args
    )
    
    # max_num_seqs = 222
    print(f"\n =========== Max number of parallel sequences found: {max_num_seqs} =========== \n\n", flush=True)

    fixed_args = {
        ConfigCatalog._load_exp_config.num_requesters.name: [None],
        ConfigCatalog._load_config.seed.name: args.seed,
        ConfigCatalog._vllm_config.port.name: args.port,
        ConfigCatalog._load_config.prompts_per_request.name: 1,
        ConfigCatalog._load_config.num_prompts.name: 1,
        ConfigCatalog._intervaled_load_config.max_tokens_per_sec.name: None,  # will update later
        ConfigCatalog._experiment_config.model.name: args.model,
        ComponentRegistry.server: ComponentRegistry.dummy
    }

    args = get_args_from_parser(
        parser, 
        fixed_args, 
        parse_best_run_time_args, 
        parse_best_num_requester_args, 
        add_intervaled_load_exp_args
    )
    
    try:
        print("\n=========== Initializing vLLM server ===========\n")
        server = vLLMServer(args)
        server.init()
        print("=========== Server initialized successfully =========== ")
    except Exception as e:
        print(f"Failed to start vLLM server:")
        raise e

    try:
        print("\n\n=========== Running load simulations to find best number of requesters ===========\n")
        best_num_requesters, max_thp_req = find_best_num_requesters(config=args)
        # best_num_requesters, max_thp_req = 7, 22000

        print(f"Best number of requesters from load simulations: {best_num_requesters} with max throughput {max_thp_req}.\n")
        args.num_requester_threads = best_num_requesters

        print("\n\n=========== Running runtime test to find best run time ===========\n")
        best_time, thp_best_time = find_best_run_time(config=args)
        # best_time, thp_best_time = 1, 21500
        print(f"Best run time from run_time test: {best_time} - with throughput {thp_best_time}.\n")
        args.max_tokens_per_sec = (args.prompt_size * best_num_requesters * args.prompts_per_request * args.requester_sleep_time)
        args.run_time = best_time

        print("\\nn=========== Running intervaled load experiment ===========\n")
        results = run_intervaled_load_exp(config=args)
        print("\\nn=========== All experiments completed ===========\n")
        all_results, all_host_data, all_accelerator_data = results.get_all()
        finish_experiment(all_results, args)

        server.shutdown()
    except Exception as e:
        server.shutdown()
        raise e
    except KeyboardInterrupt:
        server.shutdown()
        print("Experiment interrupted. Server shut down.")

