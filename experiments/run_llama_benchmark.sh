MODELS=(
    "huggyllama/llama-7b"
    "meta-llama/Meta-Llama-3-8B"
)

for MODEL in "${MODELS[@]}"; do
    echo "Running benchmark for model: $MODEL"
    python3 run_perf_exp.py \
    --prompt_size 1200 \
    --max_out_tokens 64 \
    --run_time 1 \
    --gpu_memory_utilization 0.9 \
    --dtype float16 \
    --initial_batch_size 64 \
    --requester_sleep_time 0.1 \
    --initial_num_requester 1 \
    --max_num_requesters 128 \
    --increment_requester_value 1.1 \
    --epsilon_requester 0.03 \
    --initial_run_time 30 \
    --max_run_time 300 \
    --increment_time 30 \
    --epsilon_run_time 0.03 \
    --seed 1234 \
    --logging info \
    --dataset_gen synthetic \
    --model $MODEL \
    --experiment_key $MODEL \
    --requester vllm \
    --tokenizer vllm \
    --server dummy \
    --endpoint completions \
    --host localhost \
    --port 8000 \
    --vllm_request_timeout 3600 \
    --ignore_eos True \
    --path_to_csv_filename results/Llama/llama_bench.csv \
    --path_to_save_results results/Llama/$MODEL/
done
