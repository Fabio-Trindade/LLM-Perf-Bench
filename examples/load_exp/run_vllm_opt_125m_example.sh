source examples/load_exp/load_commom_args.sh

python run_load_exp.py \
  $LOAD_COMMOM_ARGS \
  --vllm_request_timeout 2000 \
  --model facebook/opt-125m \
  --model_name_alias opt_125m \
  --experiment_key opt_125m_load_exp \
  --requester vllm \
  --tokenizer HF \
  --server dummy \
  --endpoint completions \
  --host localhost \
  --port 8000 \
  --ignore_eos True