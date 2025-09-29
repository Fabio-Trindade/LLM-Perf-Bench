import asyncio
import itertools
import os
from pathlib import Path
import signal
import sys
from time import sleep
import time
import traceback
import requests
from vllm.entrypoints.openai.api_server import run_server
from vllm.utils import FlexibleArgumentParser
from multiprocessing import Process, Queue
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.entrypoints.utils import cli_env_setup
import logging

def find_available_terminal():
    pts_path = Path("/dev/pts")
    for term in pts_path.iterdir():
        if term.name.isdigit():
            try:
                with open(term, "w") as f:
                    pass
                return str(term)
            except PermissionError:
                continue
    raise RuntimeError("No terminal available")


def get_url(host, port, endpoint):
    return f"http://{host}:{port}/v1/{endpoint}"

def get_url_from_config(config):
    return get_url(config.host, config.port, config.endpoint)


def run_vllm_server(model, port, max_model_len, max_num_seqs, dtype, gpu_utilization, seed, excpt_queue,terminal_path = None):
    try:
        if terminal_path:
            fd = os.open(terminal_path, os.O_WRONLY)
            os.dup2(fd, sys.stdout.fileno())
            os.dup2(fd, sys.stderr.fileno())

        async def async_server():
            try:
                cli_env_setup()
                parser = FlexibleArgumentParser(
                    description="vLLM OpenAI-Compatible RESTful API server."
                )
                parser = make_arg_parser(parser)

                args = parser.parse_args([
                    "--model", model,
                    "--port", str(port),
                    "--max-model-len", str(max_model_len),
                    "--max-seq-len-to-capture", str(max_model_len),
                    "--max-num-seqs", str(max_num_seqs),
                    "--gpu-memory-utilization", str(gpu_utilization),
                    "--dtype", dtype,
                    "--no-enable-prefix-caching",
                    "--seed", str(seed)
                ])
                validate_parsed_serve_args(args)

                await run_server(args)

            except Exception:
                if excpt_queue is not None:
                    excpt_queue.put(traceback.format_exc())
                raise

        asyncio.run(async_server())

    except Exception:
        if excpt_queue is not None and excpt_queue.empty():
            excpt_queue.put(traceback.format_exc())
        raise

def get_vllm_server_process(model, port, max_model_len, max_num_seqs, dtype, gpu_utilization, seed,terminal_path):
    excp_queue = Queue()
    server_process = Process(
        target=run_vllm_server,
        args=(model, port, max_model_len, max_num_seqs, dtype, gpu_utilization, seed, excp_queue,terminal_path),
    )
    return server_process, excp_queue

def wait_vllm_server_ready(port,excp_queue):
    symbols = itertools.cycle("|/-\\")
    while True:
        try:
            r = requests.get(f"http://localhost:{port}/health/", timeout=1)
            if r.status_code == 200:
                break
        except requests.exceptions.RequestException:
            if not excp_queue.empty():
                error_trace = excp_queue.get()
                raise RuntimeError(error_trace)

            print(f"\rWaiting for vLLM server to initialize... {next(symbols)}", end="")
            time.sleep(0.2) 
            
    logging.info(f"\nvLLM server is up.")

def kill_vllm_server_process(process:Process):
    os.kill(process.pid, signal.SIGINT)
    process.join()


def start_and_wait_vllm_server(model, port, max_model_len, max_num_seqs, dtype,gpu_utilization,seed, terminal_path = "auto"):
    if terminal_path == "auto":
        terminal_path = find_available_terminal()
    
    if terminal_path:
        logging.info(f"vLLM server output redirected to terminal: {terminal_path}")
    # logging.info(f"Use 'cat {terminal_path}' in another terminal to see logs.")
    server_process, excp_queue = get_vllm_server_process(model, port, max_model_len, max_num_seqs, dtype,gpu_utilization, seed, terminal_path)
    server_process.start()
    try:
        wait_vllm_server_ready(port,excp_queue)
        return server_process
    except Exception as e:
        kill_vllm_server_process(server_process)
        raise e
