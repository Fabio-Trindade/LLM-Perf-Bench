#! /bin/bash

if [ $# -ne 1 ]; then
  echo "Provide the OpenAI API key"
  exit 1
fi

OPENAI_API_KEY=$1
python run_single_prompt_exp.py \
  --repeat_times 3 \
  --prompt_sizes 32 32 64 128 256 512 1024 1984 \
  --max_out_tokens 64 \
  --model gpt-3.5-turbo-0125 \
  --model_name_alias gpt-3.5-turbo-0125 \
  --experiment_key gpt_3.5_turbo_0125_single_prompt \
  --seed 1234 \
  --requester openai \
  --tokenizer openai \
  --server openai \
  --dataset_gen synthetic \
  --logging info \
  --api_key $OPENAI_API_KEY \
  --path_to_csv_filename results/single_prompt_data.csv \
  --path_to_save_results results/single_prompt_exp/
