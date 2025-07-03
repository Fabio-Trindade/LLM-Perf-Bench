import platform
import subprocess
import sys

def get_ram_info_windows():
    try:
        output = subprocess.check_output(
            "wmic memorychip get Manufacturer, Capacity, PartNumber, Speed",
            shell=True,
            stderr=subprocess.DEVNULL
        )
        return output.decode(errors='ignore').strip()
    except Exception as e:
        return f"Error fetching RAM info on Windows: {e}"

def get_ram_info_linux():
    try:
        output = subprocess.check_output(
            "sudo dmidecode --type memory",
            shell=True,
            stderr=subprocess.DEVNULL
        )
        return output.decode(errors='ignore').strip()
    except Exception as e:
        return f"Error fetching RAM info on Linux: {e}\nNote: This command usually requires sudo privileges."

def get_ram_info_macos():
    try:
        output = subprocess.check_output(
            ["system_profiler", "SPMemoryDataType"],
            stderr=subprocess.DEVNULL
        )
        return output.decode(errors='ignore').strip()
    except Exception as e:
        return f"Error fetching RAM info on macOS: {e}"

def get_ram_info():
    system = platform.system()
    if system == "Windows":
        return get_ram_info_windows()
    elif system == "Linux":
        return get_ram_info_linux()
    elif system == "Darwin":
        return get_ram_info_macos()
    else:
        return f"Unsupported OS: {system}"

