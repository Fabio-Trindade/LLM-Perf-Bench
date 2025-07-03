import atexit
import logging
import platform
import subprocess
import requests
import time
import os
from src.registries.component_class_registry import ComponentClassRegistry
from src.registries.component_registry import ComponentRegistry
from src.components.servers.server_interface import ServerI
from src.utils.vllm_utils import get_url, get_url_from_config 

@ComponentClassRegistry.register_server_workload(ComponentRegistry.vllm)
class vLLMServer(ServerI):
    def __init__(self, config):
        self.vllm_config = config 
        self.init_timeout = self.vllm_config.vllm_server_init_timeout
        self.model = config.model
        self.server_process = None
        self.url = get_url(config.host, config.port, "health")
        self.is_windows = platform.system() == "Windows"
        atexit.register(self.cleanup)
        self.log_path = "logs/"
        raise ValueError("Initialize the vLLM server using the terminal and select the dummy server instead of the vLLM server.")

    def _wait_server_initialize(self):
        logging.info("Waiting for the vLLM server to start...")
        start_time = time.time()
        while time.time() - start_time < self.init_timeout:
            logging.info(f"Time: {time.time() - start_time}/{self.init_timeout}")
            try:
                response = requests.get(self.url)
                if response.status_code == 200:
                    logging.info("vLLM server initialized successfully!")
                    return True
            except requests.ConnectionError:
                pass
            time.sleep(1)
        raise TimeoutError(f"Server didn't initialize within {self.init_timeout} seconds.")

    def init(self):
        command = [
            "vllm", "serve", self.model
        ]
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        with open(self.log_path+"server_log.log", "w") as logfile:
            if self.is_windows:
                self.server_process = subprocess.Popen(
                    command,
                    stdout=logfile,
                    stderr=logfile
                )
            else:
                self.server_process = subprocess.Popen(
                    command,
                    preexec_fn=os.setsid,
                    stdout=logfile,
                    stderr=logfile
                )

        self._wait_server_initialize()

    def shutdown(self):
        return self.cleanup()

    def cleanup(self):
        print("Shutting down vLLM server...")
        if self.server_process:
            try:
                if self.server_process:
                    self.server_process.terminate()
                    self.server_process = None

            except Exception as e:
                print(f"[Warning] Failed to terminate the server process: {e}")
