import os
import sys

PROMPT_SIZE = 4096
DECODE_SIZE = 8192
MAX_MODEL_LEN = PROMPT_SIZE + DECODE_SIZE

from experiments.pareto_v0.compression.quantization.sym_asym.quantize_weight_sym_asym_llama_3_1 import register_all
register_all()
from src.utils.hf import HFModelRegistry
from ...run_vllm_perf_exp import run_vllm_experiment

MODELS = [model.get_hf_path() for model in HFModelRegistry.get_models()]

task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
if task_id is None:
    print("[WARN] SLURM_ARRAY_TASK_ID not found, using index 0")
    task_index = 0
else:
    task_index = int(task_id)

if task_index >= len(MODELS):
    print(f"[ERROR] SLURM_ARRAY_TASK_ID ({task_index}) is larger than the model list ({len(MODELS)})")
    sys.exit(1)

MODEL = MODELS[task_index]
ENDPOINT = "completions"
MODEL_ALIAS = MODEL.split("/")[-1]

if task_index >= len(MODELS):
    print(f"[ERROR] SLURM_ARRAY_TASK_ID ({task_index}) is larger than the model list ({len(MODELS)})")
    sys.exit(1)

MODEL = MODELS[task_index]
MODEL_ALIAS = MODEL.split("/")[-1]

args = {
    "model": MODEL,
    "model_name_alias": MODEL_ALIAS,
    "experiment_key": MODEL_ALIAS,
    "experiment_group": "paretov0",
    "path_to_save_results": f"results/Thp/{MODEL_ALIAS}/",
    "path_to_csv_filename": f"results/Thp/{MODEL_ALIAS}/perf_data.csv",
    "concurrent_requesters": 16,
    "prompt_size_range": (PROMPT_SIZE, PROMPT_SIZE),
    "decode_size_range": (DECODE_SIZE, DECODE_SIZE),
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
    "ignore_eos": True,
    "vllm_serve_args": f"--max-num-seqs 512 --max-model-len {MAX_MODEL_LEN} --dtype auto "
                       "--gpu_memory_utilization 0.95"
                       "--max-num-batched-tokens 8192 --stream-interval 1"
}

run_vllm_experiment(args)