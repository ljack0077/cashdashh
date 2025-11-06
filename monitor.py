import psutil
import pynvml
import time
import logging
import sys
from datetime import datetime

## --- CONFIGURATION ---

# --- Thresholds ---
# Set the percentage (1-100) at which to trigger an alert.
CPU_THRESHOLD = 80.0    # %
RAM_THRESHOLD = 80.0    # %
GPU_MEM_THRESHOLD = 80.0 # % (GPU Memory)

# --- Monitoring ---
# How many seconds to wait between checks.
CHECK_INTERVAL = 10     # seconds

# --- Logging ---
LOG_FILE = "server_monitor.log"
LOG_LEVEL = logging.INFO

# How many top processes to log when a threshold is breached.
TOP_PROCESS_COUNT = 5

## --- END CONFIGURATION ---


def setup_logging():
    """Configures the logger to output to both console and a file."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_process_details(pid):
    """Gets details for a specific process PID using psutil."""
    try:
        proc = psutil.Process(pid)
        # Get process info, handling potential 'zombie' processes
        with proc.oneshot():
            name = proc.name()
            cpu = proc.cpu_percent()
        return f"[PID: {pid}, Name: {name}, CPU: {cpu:.2f}%]"
    except psutil.NoSuchProcess:
        return f"[PID: {pid}, Name: Not Found (Terminated)]"
    except psutil.AccessDenied:
        return f"[PID: {pid}, Name: Access Denied]"


def check_cpu_and_ram():
    """Checks CPU and RAM usage and logs top offenders if thresholds are hit."""
    
    # 1. Check Overall CPU
    # interval=None makes it non-blocking, returns usage since last call
    cpu_usage = psutil.cpu_percent(interval=None) 
    if cpu_usage > CPU_THRESHOLD:
        logging.warning(f"--- CPU USAGE ALERT: {cpu_usage:.2f}% ---")
        
        # Get top processes
        procs = sorted(
            psutil.process_iter(['pid', 'name', 'cpu_percent']),
            key=lambda p: p.info['cpu_percent'],
            reverse=True
        )
        
        log_top_processes(procs[:TOP_PROCESS_COUNT], 'cpu_percent')

    # 2. Check Overall RAM
    ram = psutil.virtual_memory()
    ram_usage = ram.percent
    
    if ram_usage > RAM_THRESHOLD:
        logging.warning(f"--- RAM USAGE ALERT: {ram_usage:.2f}% ---")
        
        # Get top processes
        procs = sorted(
            psutil.process_iter(['pid', 'name', 'memory_percent']),
            key=lambda p: p.info['memory_percent'],
            reverse=True
        )

        log_top_processes(procs[:TOP_PROCESS_COUNT], 'memory_percent')

def log_top_processes(proc_list, metric_name):
    """Helper function to log a list of processes."""
    logging.info(f"Top {len(proc_list)} processes by {metric_name}:")
    for proc in proc_list:
        try:
            logging.info(f"  > {proc.info['name']} (PID: {proc.info['pid']}) - {proc.info[metric_name]:.2f}%")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass # Process might have terminated

def check_gpu():
    """Checks GPU memory usage and logs processes on the GPU if thresholds are hit."""
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get GPU memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_percent = (mem_info.used / mem_info.total) * 100
            
            if gpu_mem_percent > GPU_MEM_THRESHOLD:
                logging.warning(f"--- GPU {i} MEMORY ALERT: {gpu_mem_percent:.2f}% ---")
                
                # Get list of processes running on the GPU
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                
                if not procs:
                    logging.info("  > No compute processes found on GPU.")
                    continue

                logging.info(f"  Processes on GPU {i} (by memory usage):")
                
                # Sort processes by memory usage
                procs.sort(key=lambda p: p.usedGpuMemory, reverse=True)

                for proc in procs[:TOP_PROCESS_COUNT]:
                    mem_mb = proc.usedGpuMemory / (1024**2) # Convert bytes to MiB
                    # Get more process details using the PID
                    proc_details = get_process_details(proc.pid)
                    logging.info(f"  > {proc_details} - GPU Memory: {mem_mb:.2f} MiB")

    except pynvml.NVMLError as e:
        # Don't log an error if no NVIDIA driver is found (e.g., no GPU)
        if str(e) != "Driver Not Loaded":
            logging.error(f"Failed to query GPU: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass # Ignore shutdown error if init failed

def main():
    setup_logging()
    logging.info("--- Server Monitor Started ---")
    
    # Call cpu_percent once before the loop to establish a baseline
    psutil.cpu_percent(interval=None)
    
    try:
        while True:
            check_cpu_and_ram()
            check_gpu()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        logging.info("--- Server Monitor Stopped ---")
        print("\nMonitor stopped.")

if __name__ == "__main__":
    main()
