import argparse
from types import SimpleNamespace
from src.components.servers.vllm_server import vLLMServer
from src.utils.util_parse import add_arg

def parse_best_batch_args(argparser: argparse.ArgumentParser, fixed_args):
    add_arg(argparser, "model", fixed_args, type=str, default="")
    add_arg(argparser, "seed", fixed_args, type=int, default = 1234)
    add_arg(argparser, "port", fixed_args, type=str, default = "8000")
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

def find_best_num_parallel_batches_vllm(port, model_name, seed, vllm_serve_args):
    if model_name is None or model_name == "":
        raise ValueError("model_name is None or empty string")

    # tokenizer = vLLMTokenizer(config)
    # prompt_generator = PromptGeneratorBase()
    # prompt = prompt_generator.gen_prompt(tokenizer, prompt_size, max_out_tokens)
    max_parallel_seq_idx = find_idx_of_arg(vllm_serve_args, "max-num-seqs")
    assert (max_parallel_seq_idx != -1)
    last_valid_batch_size = 0
    last_worst_batch_size = float("inf")
    cur_batch_size = int(vllm_serve_args[max_parallel_seq_idx + 1])
    num_out_of_memory = 0

    while cur_batch_size != last_valid_batch_size:
        # prompts = [prompt.prompt] * cur_batch_size
        vllm_serve_args[max_parallel_seq_idx + 1] = str(cur_batch_size)
        try:
            infer(port,model_name, seed,vllm_serve_args)
            last_valid_batch_size = cur_batch_size
            print(f"\nSucceeded at batch size {last_valid_batch_size}")
            if last_worst_batch_size == float("inf"):
                print(f"Trying batch size {cur_batch_size * 2}")
                cur_batch_size *= 2
            else:
                print(f"Trying batch size {(last_valid_batch_size + last_worst_batch_size) // 2}")
                cur_batch_size = (last_valid_batch_size + last_worst_batch_size) // 2

        except Exception as e:
            # raise e
            last_worst_batch_size = cur_batch_size
            print(f"\nFailed at batch size {cur_batch_size}")
            cur_batch_size = (last_valid_batch_size + cur_batch_size) // 2
            print(f"Trying batch size {cur_batch_size}")
            num_out_of_memory += 1
    return last_valid_batch_size

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    parse_best_batch_args(argparser, {})
    args = argparser.parse_args()
    
    max_num_seqs = find_best_num_parallel_batches_vllm(
        args.port, args.model_name, args.seed, args.vllm_serve_args
    )

    print(f"\n\nBest batch size (vLLM): {max_num_seqs}\n\n")



    

