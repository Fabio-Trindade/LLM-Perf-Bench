#!/bin/bash

set -e

PROMPT_SIZE=1200
DECODE_SIZE=64
MAX_MODEL_LEN=$((PROMPT_SIZE + DECODE_SIZE))
VARS=(
    "facebook/opt-125m:completions"
    "huggyllama/llama-7b:completions"
    "meta-llama/Meta-Llama-3-8B:completions"
)

for ENTRY in "${VARS[@]}"; do
    MODEL=$(echo "$ENTRY" | cut -d':' -f1)
    ENDPOINT=$(echo "$ENTRY" | cut -d':' -f2)

    MODEL_ALIAS="${MODEL#*/}"

    echo "Running benchmark for model: $MODEL (endpoint: $ENDPOINT, alias: $MODEL_ALIAS)"

    python3 run_vllm_perf_exp.py \
        --concurrent_requesters 1 \
        --prompt_size_range $PROMPT_SIZE $PROMPT_SIZE \
        --decode_size_range $DECODE_SIZE $DECODE_SIZE \
        --initial_request_rate 0.1 \
        --increment_request_rate_factor 1.5 \
        --epsilon_request_rate 0.03 \
        --load_time 30 \
        --interval_percentage 5 \
        --seed 1234 \
        --logging info \
        --dataset_gen synthetic \
        --model "$MODEL" \
        --model_name_alias "$MODEL_ALIAS" \
        --experiment_key "$MODEL_ALIAS" \
        --experiment_group "test" \
        --requester vllm \
        --tokenizer vllm \
        --server dummy \
        --endpoint "$ENDPOINT" \
        --host localhost \
        --port 8000 \
        --ignore_eos True \
        --path_to_csv_filename results/Llama/llama_bench.csv \
        --path_to_save_results "results/Llama/$MODEL_ALIAS/" \
        --vllm_serve_args "--max-num-seqs 4096 --max-model-len $MAX_MODEL_LEN --dtype float16 --gpu_memory_utilization 0.9 --no-enable-prefix-caching --max-num-batched-tokens 8192" \
    || break  
done


python gen_report.py \
    --csv_path results/Llama/llama_bench.csv \
    --experiments llama-7b  Meta-Llama-3-8B\
    --type load \
    --output_path results/reports/Llama/ \
    --e2e_goal 2

echo "All benchmarks completed."
