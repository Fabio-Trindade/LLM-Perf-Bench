#!/bin/bash

python run_single_prompt_exp.py \
    --repeat_times 3 \
    --prompt_sizes 32 32 64 128 256 512 1024 1984 \
    --max_out_tokens 64 \
    --model facebook/opt-125m \
    --model_name_alias opt-125m \
    --experiment_key opt-125m_single_prompt_exp \
    --requester vllm \
    --server dummy \
    --tokenizer HF \
    --dataset_gen synthetic \
    --logging info \
    --seed 1234 \
    --host localhost \
    --port 8000 \
    --endpoint completions \
    --vllm_server_init_timeout 100 \
    --vllm_request_timeout 100 \
    --path_to_csv_filename results/single_prompt_data.csv \
    --path_to_save_results results/single_prompt_exp/