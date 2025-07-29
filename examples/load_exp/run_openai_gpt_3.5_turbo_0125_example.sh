if [ $# -ne 1 ]; then
  echo "Provide the OpenAI API key"
  exit 1
fi

OPENAI_API_KEY=$1
source examples/load_exp/load_commom_args.sh
python run_load_exp.py \
  $LOAD_COMMOM_ARGS \
  --model gpt-3.5-turbo-0125 \
  --model_name_alias gpt-3.5-turbo-0125 \
  --experiment_key gpt_3.5_turbo_0125_load \
  --requester openai \
  --tokenizer openai \
  --server openai \
  --api_key $OPENAI_API_KEY
