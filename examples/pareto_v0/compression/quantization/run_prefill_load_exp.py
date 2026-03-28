from examples.pareto_v0.compression.quantization.sym_asym.util_config import run_experiment
PROMPT_SIZE = 8192*2
DECODE_SIZE = 1

run_experiment(prompt_size= PROMPT_SIZE, 
               decode_size = DECODE_SIZE)

