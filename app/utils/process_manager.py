import os, sys, subprocess, shlex, time, psutil, subprocess, multiprocessing
from datetime import datetime
from ib_insync import *

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils.constants import CONSTANTS, PATHS
from utils import helpers
import live_scans_fetcher, live_L2_fetcher, live_worker, live_queue_manager


# def run_generic_fetcher(fetcher_class, fetcher_name, log_path, *fetcher_args, **fetcher_kwargs):
#     """
#     Generic runner for any fetcher class.

#     :param fetcher_class: The class to instantiate (e.g. LiveScansFetcher, LiveL2Fetcher).
#     :param fetcher_name: Name used in logging (for header only).
#     :param log_path: Path ending with `.log` (timestamp will be added before .log).
#     :param fetcher_args: Positional args for the fetcher class.
#     :param fetcher_kwargs: Keyword args for the fetcher class.
#     """
#     ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     base, ext = os.path.splitext(log_path)
#     log_path = f"{base}_{ts}{ext}"
#     os.makedirs(os.path.dirname(log_path), exist_ok=True)

#     with logs.LogContext(log_path, overwrite=True):
#         print(f"🚀 Starting {fetcher_name} | {datetime.datetime.now().isoformat()}")
#         print(f"Args: {fetcher_args}, Kwargs: {fetcher_kwargs}")

#         fetcher = fetcher_class(*fetcher_args, **fetcher_kwargs)
#         fetcher.run()


# def run_live_scans_fetcher(paper_trading, wait_minutes, continuous, log_path):
#     run_generic_fetcher(fetcher_class=live_scans_fetcher.LiveScansFetcher, fetcher_name="Scans Fetcher", log_path=log_path, paper_trading=paper_trading,
#                         wait_minutes=wait_minutes, continuous=continuous, ib_disconnect=False)

# def run_live_L2_fetcher(paper_trading, wait_minutes, continuous, single_symbol, log_path):
#     run_generic_fetcher(fetcher_class=live_L2_fetcher.LiveL2Fetcher, fetcher_name="L2 Fetcher", log_path=log_path, paper_trading=paper_trading,
#                         wait_minutes=wait_minutes, continuous=continuous, single_symbol=single_symbol, ib_disconnect=True)


# ✅ Define your allowlist of functions that are safe to call
ALLOWED_WORKERS = {
    "run_live_scans_fetcher",
    "run_live_L2_fetcher",
    "run_live_queue_manager",
    "run_live_worker"
}


def run_live_scans_fetcher(wait_seconds:int, live_mode:str, ib_client_id:int, paper_trading:str, remote_ib:bool, seed:int=None):
    scans_fetcher = live_scans_fetcher.LiveScansFetcher(wait_seconds=wait_seconds, live_mode=live_mode, ib_client_id=ib_client_id,
                                                        paper_trading=paper_trading, remote_ib=remote_ib)
    scans_fetcher.run()


def run_live_L2_fetcher(wait_seconds:int, live_mode:str, ib_client_id:int, paper_trading:str, remote_ib:bool, seed:int=None):
    L2_fetcher = live_L2_fetcher.LiveL2Fetcher(wait_seconds=wait_seconds, live_mode=live_mode, ib_client_id=ib_client_id,
                                               paper_trading=paper_trading, remote_ib=remote_ib, ib_disconnect=True)
    L2_fetcher.run()

def run_live_queue_manager(wait_seconds:int, live_mode:str, ib_client_id:int, paper_trading:str, remote_ib:bool, seed:int=None):
    qmanager = live_queue_manager.LiveQueueManager(wait_seconds=wait_seconds, live_mode=live_mode, ib_client_id=ib_client_id, seed=seed,
                                             paper_trading=paper_trading, remote_ib=remote_ib)
    qmanager.run()

def run_live_worker(action:str, wait_seconds:int, look_backward:str, live_mode:str, paper_trading:bool, remote_ib:bool, initialize:bool=True):
    worker = live_worker.LiveWorker(action=action, wait_seconds=wait_seconds, initialize=initialize, look_backward=look_backward,
                                                live_mode=live_mode, paper_trading=paper_trading, remote_ib=remote_ib)
    worker.run()


class ProcessManager:
    def __init__(self, processes_params:dict, processes:list=[], new_terminal:bool=True, tail_log:bool=True, timezone=None):

        self.ib = IB()
        self.processes_params = processes_params
        self.new_terminal = new_terminal
        self.tail_log = tail_log
        self.timezone = timezone or CONSTANTS.TZ_WORK
        self.processes = processes

    def launch_process(self, target, wtype:str):#args=(), log_path:str=None, pname:str=None):
        args = self.processes_params[wtype]['args']
        pname = self.processes_params[wtype]['pname']
        log_folder = helpers.get_path_daily_logs_folder()
        log_path =  os.path.join(log_folder, f"process_{pname}.log")

        processes, pids = [], []
        if self.new_terminal:
            command = self._get_terminal_command(target, args)
            if command:
                print(f"▶️ Starting process {pname}...")
                process = subprocess.Popen(command, shell=True)
                self.ib.sleep(5 * CONSTANTS.PROCESS_TIME['long'])
                processes = ProcessUtils.get_process_by('cmdline', pname)
                pids = [proc['pid'] for proc in processes if isinstance(proc['cmdline'], list)]
            else:
                print("❌ Could not determine terminal command for your OS.")
        else:
            # Add timestamp to log path if provided
            if log_path:
                ts = datetime.now(self.timezone).strftime("%Y%m%d_%H%M%S_%z")
                base, ext = os.path.splitext(log_path)
                log_path = f"{base}_{ts}{ext}"
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
            else:
                log_path = os.devnull  # silence output if no log specified

            # Start a wrapper process that sets up logging
            p = multiprocessing.Process(target=self._run_with_logging, args=(target, args, log_path))
            p.start()
            processes.append(p)
            pids.append(p.pid)

            if self.tail_log and log_path != os.devnull:
                self._tail_log_in_new_terminal(log_path)
                self.ib.sleep(5 * CONSTANTS.PROCESS_TIME['long'])
                processes_log = ProcessUtils.get_process_by('cmdline', pname)
                pids.extend([proc['pid'] for proc in processes_log if isinstance(proc['cmdline'], list)])
                processes.extend(processes_log)

        self.processes.append({'name': pname, 'pids': pids, 'processes': processes})

    @staticmethod
    def _run_with_logging(target, args, log_path):
        try:
            with open(log_path, "w", buffering=1, encoding="utf-8") as log_file:
                sys.stdout = log_file
                sys.stderr = log_file
                target(*args)
        except Exception as e:
            with open(log_path, "a", buffering=1, encoding="utf-8") as log_file:
                log_file.write(f"❌ Error during process execution: {e}\n")

    @staticmethod
    def _get_terminal_command(fetcher_func, args=()):
        """
        Builds a command to run fetcher_manager.py in a new terminal with the specified function and arguments.
        """
        system = CONSTANTS.SYS
        sys_list = CONSTANTS.SYS_LIST
        current_path = os.path.dirname(os.path.realpath(__file__))
        script_path = os.path.join(current_path, "process_manager.py")

        # Serialize the arguments into a JSON string (this keeps it flexible)
        # args_json = json.dumps(args)

        # # Inspect the fetcher function's signature
        # signature = inspect.signature(fetcher_func)

        # Command to call the fetcher via the launcher
        command = f"{PATHS.python_path} {script_path} {fetcher_func.__name__} " + " ".join(map(str, args))
        # command = f"python3 {script_path} {fetcher_func.__name__} {args_json}"

        # Ensure the command is properly quoted and escaped for PowerShell
        # In PowerShell, we should escape the command properly using shlex.quote() to avoid extra parameters
        # escaped_command = shlex.quote(command)

        if system == sys_list['linux']:
            return f'tmux new-session -d "{command}"'
        elif system == sys_list['macos']:
            apple_script = f'''
            tell application "Terminal"
                do script "{command}"
                activate
            end tell
            '''
            return f"osascript -e {shlex.quote(apple_script)}"
        elif system == sys_list['windows']:
            return f'start cmd /k "{command}"'
            # return f'start powershell -NoExit -Command "{escaped_command}"'
        else:
            return None

    @staticmethod
    def _tail_log_in_new_terminal(log_path, timeout=60):
        """
        Launch a new terminal to tail the given log file.
        Waits until the log file is created and non-empty.
        """
        start_time = time.time()

        print(f"Waiting to load log file {log_path}")
        while (not os.path.exists(log_path) or os.path.getsize(log_path) == 0):
            if time.time() - start_time > timeout:
                print(f"⚠️ Timeout waiting for log file: {log_path}")
                return
            time.sleep(CONSTANTS.PROCESS_TIME['medium'])  # Wait a bit and try again

        system = CONSTANTS.SYS
        if system == "Windows":
            # Windows: Use 'type' to dump existing content, then tail
            command = f'start cmd /k "type {log_path} & powershell -Command Get-Content -Path {log_path} -Wait"'
        elif system == "Linux":
            command =  f'tmux new-session -d "tail -f {log_path}"'
            # command = f'gnome-terminal -- bash -c "tail -f {log_path}; exec bash"'
        elif system == "Darwin":
            command = f'osascript -e \'tell application "Terminal" to do script "tail -f {log_path}"\''
        else:
            print(f"⚠️ Cannot tail logs: unsupported system '{system}'")
            return

        subprocess.Popen(command, shell=True)

    def get_process_ids(self):
        """Retrieve the PIDs of all running processes."""
        return [p.pid for p in self.processes]

    def check_process_alive(self, pid:int):
        """Check if a process with a given PID is still running."""
        # Check the process status using the poll method of the Popen object
        for p in self.processes:
            if p.pid == pid:
                return p.poll() is None  # If the process hasn't terminated, poll() will return None
        return False

    def stop_process(self, pname):
        """Function to restart a specific process."""
        for process in self.processes:
            if process['name'] == pname:
                # Restart the process
                for pid in process['pids']:
                    try:
                        psutil.Process(pid).terminate()
                        # os.kill(pid, signal.SIGTERM)  # Kill the process
                        print(f"Terminated process {pname} (PID: {pid})")
                    except Exception as e:
                        print(f"Error terminating process {pname}: {e}")
                return True
        return False


class ProcessUtils:

    @staticmethod
    def get_process_by(attr:str, value:str):
        attr_list = ['cmdline', 'pid', 'name', 'status', 'create_time']
        # 'exe', 'cpu_times', 'cpu_percent', 'memory_percent', 'memory_info', 'io_counters', 'num_threads', 'threads', 'open_files', 'environ', 'nice', 'cpu_affinity'
        if not value:
            print(f"⚠️ Value for 'value' is needed")
            return[]
        if not attr or attr not in attr_list:
            print(f"⚠️ 'attr' must be within {attr_list}")
            return[]

        pall = psutil.process_iter(attr_list)
        matched_processes = []

        for proc in pall:
            proc_info = proc.info.get(attr, None)
            if not proc_info:
                continue

            condition_num = isinstance(proc_info, (int, float)) and proc_info == value
            condition_str = isinstance(proc_info, str) and value in proc_info
            condition_list = isinstance(proc_info, list) and all(isinstance(p, str) for p in proc_info) and any([value in p for p in proc_info])
            if condition_num or condition_str or condition_list:
                matched_processes.append({**proc.info,
                                            'create_time': datetime.fromtimestamp(proc.info['create_time']).strftime('%Y-%m-%d %H:%M:%S')
                                            if 'create_time' in proc.info else None})
        return matched_processes
        # matched_processes = [{
        #     **proc.info, # Copy all info of the process
        #     'create_time': datetime.fromtimestamp(proc.info['create_time']).strftime('%Y-%m-%d %H:%M:%S') if 'create_time' in proc.info else None}
        #     for proc in pall if proc.info.get(attr, '') and \
        #         (any([value in p for p in proc.info.get(attr, '')]) \
        #          if not isinstance(proc.info.get(attr, ''), (int, float)) else proc.info.get(attr, '') == value)  # Filter based on the value in the specified attribute
        #     ]
        # return matched_processes

    @staticmethod
    def list_all_processes():
        """List all running processes with PID, name, and status."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'status', 'username', 'cmdline']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Ignore processes that no longer exist or can't be accessed
                continue
        return processes

    @staticmethod
    def parse_arg(arg):
        if arg.lower() == 'true':
            return True
        elif arg.lower() == 'false':
            return False
        elif arg.lower() == 'none':
            return None
        elif arg.isdigit():
            return int(arg)
        try:
            return float(arg)
        except ValueError:
            return arg


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetcher_manager.py <fetcher_name> [args...]")
        sys.exit(1)

    worker_name = sys.argv[1]
    args = list(map(ProcessUtils.parse_arg, sys.argv[2:]))

    # # Deserialize the arguments
    # args = json.loads(args_json)

    # ✅ Check if the requested function is in the allowlist
    if worker_name not in ALLOWED_WORKERS:
        print(f"❌ Not allowed to run: {worker_name}")
        print(f"✅ Allowed fetchers: {', '.join(ALLOWED_WORKERS)}")
        sys.exit(1)

    # ✅ Get function from the module dynamically
    # fetcher_func = getattr(live_orchestrator, fetcher_name, None)
    worker_func = globals().get(worker_name)

    if not callable(worker_func):
        print(f"❌ Function '{worker_name}' not found or not callable.")
        sys.exit(1)

    # ✅ Call the function with parsed arguments
    worker_func(*args)

    # def main():
    #     # ✅ Get function from the current module dynamically using globals()
    #     fetcher_func = globals().get(fetcher_name)

    #     if not callable(fetcher_func):
    #         print(f"❌ Function '{fetcher_name}' not found or not callable.")
    #         sys.exit(1)

    #     # ✅ Call the function with parsed arguments
    #     fetcher_func(*args)


if __name__ == "__main__":
    main()













# def run_with_output_redirect(target, args, log_file):
#     sys.stdout = log_file
#     sys.stderr = log_file
#     try:
#         target(*args)
#     finally:
#         log_file.close()



    # def escape_backslashes(value):
#     """Escape backslashes in Windows paths for PowerShell compatibility."""
#     if isinstance(value, str):
#         return value.replace("\\", "\\\\")
#     return value

# def build_command(script_path, function_name, *args):
#     """Builds the PowerShell command dynamically with proper escaping."""
#     args_serialized = []

#     for arg in args:
#         if isinstance(arg, dict):
#             arg = {key: escape_backslashes(val) if isinstance(val, str) else val for key, val in arg.items()}
#             args_serialized.append(json.dumps(arg))
#         elif isinstance(arg, list) or isinstance(arg, tuple):
#             args_serialized.append(json.dumps([escape_backslashes(val) if isinstance(val, str) else val for val in arg]))
#         else:
#             args_serialized.append(escape_backslashes(str(arg)))

#     args_str = " ".join(args_serialized)

#     command = f'start powershell -NoExit -Command "python3 {script_path} {function_name} {args_str}"'

#     return command