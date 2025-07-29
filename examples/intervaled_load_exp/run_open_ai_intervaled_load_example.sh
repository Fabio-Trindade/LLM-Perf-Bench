if [ $# -ne 1 ]; then
  echo "Provide the OpenAI API key"
  exit 1
fi

OPENAI_API_KEY=$1

python run_intervaled_load_exp.py \
  --interval_percentage 5 \
  --max_tokens_per_sec 17696 \
  --prompt_size 1200 \
  --max_out_tokens 64 \
  --num_requester_threads 14 \
  --num_prompt_gen_threads 1 \
  --prompts_per_request 1 \
  --run_time 60 \
  --seed 1234 \
  --dataset_gen synthetic \
  --path_to_csv_filename results/intervaled_load_data.csv \
  --path_to_save_results results/ \
  --logging info \
  --model_name_alias gpt-3.5-turbo-0125 \
  --experiment_key gpt_3.5_turbo_0125_intervaled_load \
  --requester openai \
  --tokenizer openai \
  --server openai \
  --api_key $OPENAI_API_KEY
