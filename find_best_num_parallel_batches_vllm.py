import argparse
from src.utils.vllm_utils import kill_vllm_server_process, start_and_wait_vllm_server
from src.utils.util_parse import add_arg

def parse_best_batch_args(argparser: argparse.ArgumentParser, fixed_args):
    add_arg(argparser, "model", fixed_args, type=str, default="")
    add_arg(argparser, "prompt_size", fixed_args, type=int)
    add_arg(argparser, "max_out_tokens", fixed_args, type=int, default=64)
    add_arg(argparser, "initial_batch_size", fixed_args, type=int, default=1)
    add_arg(argparser, "dtype", fixed_args, type=str, default="float16")
    add_arg(argparser, "gpu_memory_utilization", fixed_args, type=float, default=0.9)
    add_arg(argparser, "seed", fixed_args, type=int, default = 1234)
    add_arg(argparser, "port", fixed_args, type=str, default = "8000")

    
def infer(port, model_name, prompt_size, max_out_tokens, num_seqs, dtype,gpu_memory_utilization, seed):
    server_process = start_and_wait_vllm_server(model_name,port,prompt_size + max_out_tokens, num_seqs,dtype, gpu_memory_utilization,seed)
    kill_vllm_server_process(server_process)

def find_best_num_parallel_batches_vllm(model_name, prompt_size, max_out_tokens, initial_batch_size, dtype,
                                        gpu_memory_utilization, seed, port):
    if model_name is None or model_name == "":
        raise ValueError("model_name is None or empty string")

    # config = SimpleNamespace(model=model_name)
    # tokenizer = vLLMTokenizer(config)
    # prompt_generator = PromptGeneratorBase()
    # prompt = prompt_generator.gen_prompt(tokenizer, prompt_size, max_out_tokens)

    last_valid_batch_size = 0
    last_worst_batch_size = float("inf")
    cur_batch_size = initial_batch_size
    num_out_of_memory = 0

    while cur_batch_size != last_valid_batch_size:
        # prompts = [prompt.prompt] * cur_batch_size
        try:
            infer(port,model_name, prompt_size, max_out_tokens, cur_batch_size, dtype,gpu_memory_utilization,seed)
            last_valid_batch_size = cur_batch_size
            print(f"\nSucceeded at batch size {last_valid_batch_size}")
            if last_worst_batch_size == float("inf"):
                print(f"Trying batch size {cur_batch_size * 2}")
                cur_batch_size *= 2
            else:
                print(f"Trying batch size {(last_valid_batch_size + last_worst_batch_size) // 2}")
                cur_batch_size = (last_valid_batch_size + last_worst_batch_size) // 2

        except Exception as e:
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
        args.model, args.prompt_size, args.max_out_tokens, args.initial_batch_size,
        args.dtype, args.gpu_memory_utilization, args.seed, args.port
    )

    print(f"\n\nBest batch size (vLLM): {max_num_seqs}\n\n")



    

