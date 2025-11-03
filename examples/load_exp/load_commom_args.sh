LOAD_COMMOM_ARGS="\
  --concurrent_requesters 16 \
  --prompt_size_range 1200 1200 \
  --decode_size_range 64 64 \
  --request_rates_per_requester 10 50 90 100 \
  --load_time 30 \
  --seed 1234 \
  --logging info \
  --path_to_csv_filename results/load_data.csv \
  --path_to_save_results results/  
  --dataset_gen synthetic "
  