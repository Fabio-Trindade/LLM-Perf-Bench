#!/bin/bash

PROMPT_SIZE=1200
DECODE_SIZE=64
MAX_MODEL_LEN=$((PROMPT_SIZE + DECODE_SIZE))
VARS=(
    "huggyllama/llama-7b:completions"
    "meta-llama/Meta-Llama-3-8B:completions"
)

for ENTRY in "${VARS[@]}"; do
    MODEL=$(echo "$ENTRY" | cut -d':' -f1)
    ENDPOINT=$(echo "$ENTRY" | cut -d':' -f2)

    echo "Running benchmark for model: $MODEL (endpoint: $ENDPOINT)"

    python3 run_vllm_perf_exp.py \
        --prompt_size $PROMPT_SIZE \
        --max_out_tokens $DECODE_SIZE \
        --run_time 30 \
        --requester_sleep_time 0.1 \
        --initial_num_requester 1 \
        --max_num_requesters 128 \
        --increment_requester_value 1.1 \
        --epsilon_requester 0.03 \
        --interval_percentage 0.1 \
        --seed 1234 \
        --logging info \
        --dataset_gen synthetic \
        --model "$MODEL" \
        --experiment_key "$MODEL" \
        --requester vllm \
        --tokenizer vllm \
        --server dummy \
        --endpoint "$ENDPOINT" \
        --host localhost \
        --port 8000 \
        --ignore_eos True \
        --path_to_csv_filename results/Llama/llama_bench.csv \
        --path_to_save_results "results/Llama/${MODEL//\//_}/"\
        --vllm_param_to_optimize "max-num-batched-tokens" \
        --vllm_serve_args "--max-num-seqs 4096 --max-model-len $MAX_MODEL_LEN --dtype float16  --gpu_memory_utilization 0.9 --no-enable-prefix-caching --max-num-batched-tokens 8192" 
done
echo "All benchmarks completed."