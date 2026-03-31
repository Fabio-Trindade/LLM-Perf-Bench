import os
import sys
import argparse

# def resolve_root(levels):
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"*levels))
#     sys.path.insert(0, project_root)

# def remove_root():
#     sys.path.pop(0)

# def get_models():
#     original_root = sys.path.pop(0)
#     keys_to_remove = [key for key in sys.modules if key == 'src' or key.startswith('src.')]
#     for key in keys_to_remove:
#         del sys.modules[key]
        
#     resolve_root(7)
#     from experiments.pareto_v0.compression.quantization.sym_asym.quantize_weight_sym_asym_llama_3_1 import register_all
#     register_all()
#     from src.utils.hf import HFModelRegistry
#     remove_root()
#     keys_to_remove = [key for key in sys.modules if key == 'src' or key.startswith('src.')]
#     for key in keys_to_remove:
#         del sys.modules[key]
    
#     sys.path.append(original_root)
#     MODELS = [model.get_hf_path() for model in HFModelRegistry.get_models()]
#     print(MODELS)

#     return MODELS

    

def create_argvs(args, prompt_size, decode_size):
    # MODELS = sorted(get_models())
    # MODELS = sorted(get_models())
    MODELS = [["FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-sym-GS-32",1],
              ["FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-asym-GS-32",1],
              
              ["FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-sym-GS-64",1],
              ["FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-asym-GS-64",1],
              
              ["FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-sym-GS-128",1],
              ["FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-asym-GS-128",1],
              
              ["FabioTrindade/Llama-3.1-70B-Instruct-W4A16KV16-sym-GS-32",2],
              ["FabioTrindade/Llama-3.1-70B-Instruct-W4A16KV16-asym-GS-32",2],
              
              ["FabioTrindade/Llama-3.1-70B-Instruct-W4A16KV16-sym-GS-64",2],
              ["FabioTrindade/Llama-3.1-70B-Instruct-W4A16KV16-asym-GS-64",2],
              
              ["FabioTrindade/Llama-3.1-70B-Instruct-W4A16KV16-sym-GS-128",2],
              ["FabioTrindade/Llama-3.1-70B-Instruct-W4A16KV16-asym-GS-128",2],
              ]
    
    if args.print_models_only:
        for i,model in (enumerate(MODELS)):
            print(f"model: {model} | id: {i}")
        exit(0)
    task_index = args.task_id

    if task_index >= len(MODELS):
        print(f"[ERROR] Task ID ({task_index}) is larger than the model list ({len(MODELS)})")
        sys.exit(1)

    MODEL,GPUs = MODELS[task_index]
    MODEL_ALIAS = MODEL.split("/")[-1]
    MAX_MODEL_LEN = prompt_size + decode_size
    if task_index >= len(MODELS):
        print(f"[ERROR] SLURM_ARRAY_TASK_ID ({task_index}) is larger than the model list ({len(MODELS)})")
        sys.exit(1)
    ENDPOINT = "completions"
    is_big_model = "70" in MODEL or "405" in MODEL
    args = {
        "model": MODEL,
        "model_name_alias": MODEL_ALIAS,
        "experiment_key": MODEL_ALIAS,
        "experiment_group": "pareto_v0",
        "path_to_save_results": f"results/sym_vs_asym/{MODEL_ALIAS}/",
        "path_to_csv_filename": f"results/sym_vs_asym/{MODEL_ALIAS}/perf_data.csv",
        "concurrent_requesters": 1,
        "prompt_size_range": (prompt_size, prompt_size),
        "decode_size_range": (decode_size, decode_size),
        # "initial_request_rate": 1,
        # "increment_request_rate_factor": 1.5,
        # "epsilon_request_rate": 0.03,
        "load_time": 30,
        "request_rate_per_requester": 4 if not is_big_model else 1.5,
        "interval_percentage": 5,
        "seed": 1234,
        "logging": "info",
        "dataset_gen": "synthetic",
        "requester": "vllm",
        "tokenizer": "vllm",
        "server": "dummy",
        "endpoint": ENDPOINT,
        "host": "localhost",
        "port": 7000 + task_index,
        "ignore_eos": "True",
        "vllm_serve_args": f"--max-num-seqs 64 --max-model-len {MAX_MODEL_LEN} --dtype auto "
                        "--gpu_memory_utilization 0.95 --no-enable-prefix-caching "
                        "--max-num-batched-tokens 8192 --stream-interval 1 "
                        f"--pipeline-parallel-size {GPUs}"
    }

    sys.argv = ["run_vllm_experiment.py"] 

    for k, v in args.items():
        flag = f"--{k}"
        if isinstance(v, (tuple, list)):
            sys.argv.append(flag)
            sys.argv.extend(map(str, v)) 
        else:
            sys.argv.append(flag)
            sys.argv.append(str(v))
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type = int)
    parser.add_argument("--print-models-only", action="store_true")
    args = parser.parse_args()
    return args

def run_experiment(prompt_size, decode_size):
    from run_intervaled_load_exp import run
    from run_vllm_perf_exp import run_vllm_experiment

    args = get_args()
    create_argvs(args, prompt_size, decode_size)
    run_vllm_experiment()
