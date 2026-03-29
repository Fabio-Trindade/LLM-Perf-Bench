import itertools
import logging
import signal
import subprocess
import sys
import requests
import time
import os
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.servers.server_interface import ServerI

@ComponentClassRegistry.register_server_workload(ComponentRegistry.vllm)
class vLLMServer(ServerI):
    def __init__(self, config):
        self.vllm_config = config 
        self.model = config.model
        self.server_process = None
        self.process = None
        self.port = config.port
        self.seed = config.seed
        
    def start_vllm_server(self):
        cmd = [
            "vllm", "serve", self.model, "--port", self.port,
            "--seed", str(self.seed)
        ] + self.vllm_config.vllm_serve_args

        logging.info(f"Starting vLLM server: \n{' '.join(cmd)}")

        # process = subprocess.Popen(
        #     cmd,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     close_fds=True
        # )

        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            close_fds=True
        )


        
        return process
    
    def wait_vllm_server_ready(self,port, process, timeout=1800):
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


    def kill_vllm_server_process(self,process: subprocess.Popen):
        try:
            os.kill(process.pid, signal.SIGINT)
            process.wait(timeout=1800)
        except Exception:
            process.kill()
            process.wait()


    def init(self):
        try:
            self.process = self.start_vllm_server()
            self.wait_vllm_server_ready(self.port, self.process)
        except Exception as e:
            self.kill_vllm_server_process(self.process)
            raise e

    def shutdown(self):
        self.kill_vllm_server_process(self.process)

    