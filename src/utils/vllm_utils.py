import subprocess
import requests
import itertools
import time
import os
import signal
import logging


def get_url(host, port, endpoint):
    return f"http://{host}:{port}/v1/{endpoint}"


def get_url_from_config(config):
    return get_url(config.host, config.port, config.endpoint)


def start_vllm_server(model, port, max_model_len, max_num_seqs, dtype, gpu_utilization, seed):
    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--max-seq-len-to-capture", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", str(gpu_utilization),
        "--dtype", dtype,
        "--no-enable-prefix-caching",
        "--seed", str(seed),
    ]

    logging.info(f"Starting vLLM server: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True
    )
    return process


def wait_vllm_server_ready(port, process, timeout=60):
    symbols = itertools.cycle("|/-\\")
    start = time.time()
    while True:
        if process.poll() is not None:  
            stderr_output = process.stderr.read().decode() if process.stderr else ""
            raise RuntimeError(
                f"vLLM server crashed with code {process.returncode}\nLogs:\n{stderr_output}"
            )
        try:
            r = requests.get(f"http://localhost:{port}/health/", timeout=1)
            if r.status_code == 200:
                break
        except requests.exceptions.RequestException:
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for vLLM server to start")
            print(f"\rWaiting for vLLM server to initialize... {next(symbols)}", end="")
            time.sleep(0.2)
    logging.info(f"\nvLLM server is up.")


def kill_vllm_server_process(process: subprocess.Popen):
    try:
        os.kill(process.pid, signal.SIGINT)
        process.wait(timeout=10)
    except Exception:
        process.kill()
        process.wait()


def start_and_wait_vllm_server(model, port, max_model_len, max_num_seqs, dtype, gpu_utilization, seed):
    process = start_vllm_server(
        model=model,
        port=port,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        gpu_utilization=gpu_utilization,
        seed=seed,
    )

    try:
        wait_vllm_server_ready(port, process)
        return process
    except Exception as e:
        kill_vllm_server_process(process)
        raise e
