LOAD_COMMOM_ARGS="\
  --prompt_size 1200 \
  --max_out_tokens 64 \
  --num_requesters 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --requester_sleep_time 1 \
  --prompts_per_request 1 \
  --run_time 60 \
  --seed 1234 \
  --logging info \
  --path_to_csv_filename results/load_data.csv \
  --path_to_save_results results/  
  --dataset_gen synthetic "
  