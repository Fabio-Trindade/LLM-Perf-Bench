import argparse
from types import SimpleNamespace
from src.components.servers.vllm_server import vLLMServer
from src.utils.util_parse import add_arg

def parse_best_batch_args(argparser: argparse.ArgumentParser, fixed_args):
    add_arg(argparser, "model", fixed_args, type=str, default="")
    add_arg(argparser, "seed", fixed_args, type=int, default = 1234)
    add_arg(argparser, "port", fixed_args, type=str, default = "8000")
    add_arg(argparser, "vllm_param_to_optimize", fixed_args, type=str, default = None, required = False)
    add_arg(argparser, "vllm_serve_args", fixed_args, type=str.split, default = [])

    
def infer(port, model_name, seed, vllm_serve_args):
    server_config = SimpleNamespace(model=model_name, port = port, seed = seed, vllm_serve_args = vllm_serve_args)
    server = vLLMServer(server_config)
    server.init()
    server.shutdown()

def find_idx_of_arg(arg_list, arg_name):
    for i, arg in enumerate(arg_list):
        if arg.startswith(f"--{arg_name}"):
            return i
    return -1

def optimize_vllm_parameter(port, model_name, seed, vllm_serve_args, vllm_param_to_optimize):
    if model_name is None or model_name == "":
        raise ValueError("model_name is None or empty string")

    # tokenizer = vLLMTokenizer(config)
    # prompt_generator = PromptGeneratorBase()
    # prompt = prompt_generator.gen_prompt(tokenizer, prompt_size, max_out_tokens)
    key_idx = find_idx_of_arg(vllm_serve_args,vllm_param_to_optimize)
    assert (key_idx != -1)
    last_valid_batch = 0
    last_worst_batch_size = float("inf")
    cur_value = int(vllm_serve_args[key_idx + 1])
    num_out_of_memory = 0

    while cur_value != last_valid_batch:
        # prompts = [prompt.prompt] * cur_value
        vllm_serve_args[key_idx + 1] = str(cur_value)
        try:
            infer(port,model_name, seed,vllm_serve_args)
            last_valid_batch = cur_value
            print(f"\nSucceeded at value {last_valid_batch}")
            if last_worst_batch_size == float("inf"):
                print(f"Trying value {cur_value * 2}")
                cur_value *= 2
            else:
                print(f"Trying value {(last_valid_batch + last_worst_batch_size) // 2}")
                cur_value = (last_valid_batch + last_worst_batch_size) // 2

        except Exception as e:
            # raise e
            last_worst_batch_size = cur_value
            print(f"\nFailed at value {cur_value}")
            cur_value = (last_valid_batch + cur_value) // 2
            print(f"Trying value {cur_value}")
            num_out_of_memory += 1

    if last_valid_batch == 0:
        raise ValueError("Something went wrong during vLLM initialization")
        
    return last_valid_batch

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    parse_best_batch_args(argparser, {})
    args = argparser.parse_args()
    
    max_num_seqs = optimize_vllm_parameter(
        args.port, args.model_name, args.seed, args.vllm_serve_args, args.vllm_param_to_optimize
    )

    print(f"\n\nBest value (vLLM): {max_num_seqs}\n\n")



    

