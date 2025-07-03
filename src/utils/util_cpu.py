import os
import platform
import subprocess

def get_cpu_name():
    system = platform.system()

    if system == "Linux":
        return _get_cpu_name_linux()
    elif system == "Windows":
        return _get_cpu_name_windows()
    elif system == "Darwin": 
        return _get_cpu_name_macos()
    else:
        return "Unknown OS"

def _get_cpu_name_linux():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.strip().split(":")[1].strip()
    except Exception:
        pass
    return "Unknown CPU"

def _get_cpu_name_windows():
    try:
        output = subprocess.check_output("wmic cpu get name", shell=True)
        lines = output.decode().split("\n")
        for line in lines:
            if line.strip() and "Name" not in line:
                return line.strip()
    except Exception:
        pass
    return "Unknown CPU"

def _get_cpu_name_macos():
    try:
        output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        return output.decode().strip()
    except Exception:
        pass
    return "Unknown CPU"
