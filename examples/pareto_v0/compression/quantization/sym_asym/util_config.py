import os
import sys
import argparse

def resolve_root(levels):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"*levels))
    sys.path.insert(0, project_root)

def remove_root():
    sys.path.pop(0)

def get_models():
    original_root = sys.path.pop(0)
    keys_to_remove = [key for key in sys.modules if key == 'src' or key.startswith('src.')]
    for key in keys_to_remove:
        del sys.modules[key]
        
    resolve_root(7)
    from experiments.pareto_v0.compression.quantization.sym_asym.quantize_weight_sym_asym_llama_3_1 import register_all
    register_all()
    from src.utils.hf import HFModelRegistry
    remove_root()
    keys_to_remove = [key for key in sys.modules if key == 'src' or key.startswith('src.')]
    for key in keys_to_remove:
        del sys.modules[key]
    
    sys.path.append(original_root)
    MODELS = [model.get_hf_path() for model in HFModelRegistry.get_models()]

    return MODELS

    

def create_argvs(task_id, prompt_size, decode_size):
    MODELS = get_models()
    if task_id is None:
        print("[WARN] SLURM_ARRAY_TASK_ID not found, using index 0")
        task_index = 0
    else:
        task_index = int(task_id)

    if task_index >= len(MODELS):
        print(f"[ERROR] SLURM_ARRAY_TASK_ID ({task_index}) is larger than the model list ({len(MODELS)})")
        sys.exit(1)

    MODEL = MODELS[task_index]
    MODEL_ALIAS = MODEL.split("/")[-1]
    MAX_MODEL_LEN = prompt_size + decode_size
    if task_index >= len(MODELS):
        print(f"[ERROR] SLURM_ARRAY_TASK_ID ({task_index}) is larger than the model list ({len(MODELS)})")
        sys.exit(1)
    ENDPOINT = "completions"
    args = {
        "model": MODEL,
        "model_name_alias": MODEL_ALIAS,
        "experiment_key": MODEL_ALIAS,
        "experiment_group": "pareto_v0",
        "path_to_save_results": f"results/Thp/{MODEL_ALIAS}/",
        "path_to_csv_filename": f"results/Thp/{MODEL_ALIAS}/perf_data.csv",
        "concurrent_requesters": 16,
        "prompt_size_range": (prompt_size, prompt_size),
        "decode_size_range": (decode_size, decode_size),
        "initial_request_rate": 1,
        "increment_request_rate_factor": 1.5,
        "epsilon_request_rate": 0.03,
        "load_time": 30,
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
        "vllm_serve_args": f"--max-num-seqs 512 --max-model-len {MAX_MODEL_LEN} --dtype auto "
                        "--gpu_memory_utilization 0.95 "
                        "--max-num-batched-tokens 8192 --stream-interval 1"
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

def get_task_id():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default= -1, type = int)
    args = parser.parse_args()
    return args.task_id

def run_experiment(prompt_size, decode_size):
    from run_vllm_perf_exp import run_vllm_experiment
    task_id = get_task_id()
    create_argvs(task_id, prompt_size, decode_size)
    run_vllm_experiment()
