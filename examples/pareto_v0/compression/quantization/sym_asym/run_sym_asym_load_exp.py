from examples.pareto_v0.compression.quantization.sym_asym.util_config import run_experiment

PROMPT_SIZE = 4096
DECODE_SIZE = 4096

run_experiment(prompt_size= PROMPT_SIZE, 
               decode_size = DECODE_SIZE)

