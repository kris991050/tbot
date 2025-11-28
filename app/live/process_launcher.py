import os, sys, subprocess, shlex, time

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import constants
import live_scans_fetcher, live_L2_fetcher, live_worker


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
ALLOWED_FETCHERS = {
    "run_live_scans_fetcher",
    "run_live_L2_fetcher",
    "run_live_worker"
}


def run_live_scans_fetcher(wait_seconds:int, live_mode:str, ib_client_id:int, paper_trading:str, remote_ib:bool):
    fetcher = live_scans_fetcher.LiveScansFetcher(wait_seconds=wait_seconds, live_mode=live_mode, ib_client_id=ib_client_id, paper_trading=paper_trading, 
                                                  remote_ib=remote_ib)
    fetcher.run()


def run_live_L2_fetcher(wait_seconds:int, live_mode:str, ib_client_id:int, paper_trading:str, remote_ib:bool):
    fetcher = live_L2_fetcher.LiveL2Fetcher(wait_seconds=wait_seconds, live_mode=live_mode, ib_client_id=ib_client_id, paper_trading=paper_trading, 
                                                  remote_ib=remote_ib, ib_disconnect=True)
    fetcher.run()


# def run_live_data_fetcher(paper_trading, wait_seconds, args_trade_manager_json, args_logger_json, tickers_list, timezone, initialize=True):
#     fetcher = live_data_fetcher.LiveDataFetcher(args_tmanager=args_trade_manager_json, args_logger=args_logger_json, tickers_list=tickers_list, initialize=initialize, 
#                                                 paper_trading=paper_trading, wait_seconds=wait_seconds, timezone=timezone)
#     fetcher.run()


# def run_live_data_enricher(paper_trading, wait_seconds, args_trade_manager_json, args_logger_json, tickers_list, timezone, initialize=True):
#     fetcher = live_data_enricher.LiveDataEnricher(args_tmanager=args_trade_manager_json, args_logger=args_logger_json, tickers_list=tickers_list, initialize=initialize, 
#                                                   paper_trading=paper_trading, wait_seconds=wait_seconds, timezone=timezone)
#     fetcher.run()

def run_live_worker(handler_type, strategy_name, stop, revised, look_backward, step_duration, mode, sim_offset_seconds, paper_trading, wait_seconds, 
                          initialize=True):
    worker = live_worker.LiveWorker(handler_type=handler_type, strategy_name=strategy_name, stop=stop, revised=revised, look_backward=look_backward, 
                                                step_duration=step_duration, mode=mode, sim_offset_seconds=sim_offset_seconds, 
                                                initialize=initialize, paper_trading=paper_trading, wait_seconds=wait_seconds)
    worker.run()


# def run_with_output_redirect(target, args, log_file):
#     sys.stdout = log_file
#     sys.stderr = log_file
#     try:
#         target(*args)
#     finally:
#         log_file.close()


def run_with_logging(target, args, log_path):
    try:
        with open(log_path, "w", buffering=1, encoding="utf-8") as log_file:
            sys.stdout = log_file
            sys.stderr = log_file
            target(*args)
    except Exception as e:
        with open(log_path, "a", buffering=1, encoding="utf-8") as log_file:
            log_file.write(f"❌ Error during process execution: {e}\n")


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

def get_terminal_command(fetcher_func, args=()):
    """
    Builds a command to run fetcher_launcher.py in a new terminal with the specified function and arguments.
    """
    system = constants.CONSTANTS.SYS
    sys_list = constants.CONSTANTS.SYS_LIST
    current_path = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(current_path, "process_launcher.py")

    # Serialize the arguments into a JSON string (this keeps it flexible)
    # args_json = json.dumps(args)

    # # Inspect the fetcher function's signature
    # signature = inspect.signature(fetcher_func)

    # Command to call the fetcher via the launcher
    command = f"python3 {script_path} {fetcher_func.__name__} " + " ".join(map(str, args))
    # command = f"python3 {script_path} {fetcher_func.__name__} {args_json}"

    # Ensure the command is properly quoted and escaped for PowerShell
    # In PowerShell, we should escape the command properly using shlex.quote() to avoid extra parameters
    escaped_command = shlex.quote(command)

    if system == sys_list['linux']:
        return f'gnome-terminal -- bash -c "{escaped_command}; exec bash"'
    elif system == sys_list['macos']:
        apple_script = f'''
        tell application "Terminal"
            do script "{escaped_command}"
            activate
        end tell
        '''
        return f"osascript -e {shlex.quote(apple_script)}"
    elif system == sys_list['windows']:
        return f'start cmd /k "{command}"'
        # return f'start powershell -NoExit -Command "{escaped_command}"'
    else:
        return None
    

def tail_log_in_new_terminal(log_path, timeout=60):
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
        time.sleep(constants.CONSTANTS.PROCESS_TIME['medium'])  # Wait a bit and try again

    system = constants.CONSTANTS.SYS
    if system == "Windows":
        # Windows: Use 'type' to dump existing content, then tail
        command = f'start cmd /k "type {log_path} & powershell -Command Get-Content -Path {log_path} -Wait"'
    elif system == "Linux":
        command = f'gnome-terminal -- bash -c "tail -f {log_path}; exec bash"'
    elif system == "Darwin":
        command = f'osascript -e \'tell application "Terminal" to do script "tail -f {log_path}"\''
    else:
        print(f"⚠️ Cannot tail logs: unsupported system '{system}'")
        return

    subprocess.Popen(command, shell=True)


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
        print("Usage: python fetcher_launcher.py <fetcher_name> [args...]")
        sys.exit(1)

    fetcher_name = sys.argv[1]
    args = list(map(parse_arg, sys.argv[2:]))
    
    # # Deserialize the arguments
    # args = json.loads(args_json)

    # ✅ Check if the requested function is in the allowlist
    if fetcher_name not in ALLOWED_FETCHERS:
        print(f"❌ Not allowed to run: {fetcher_name}")
        print(f"✅ Allowed fetchers: {', '.join(ALLOWED_FETCHERS)}")
        sys.exit(1)

    # ✅ Get function from the module dynamically
    # fetcher_func = getattr(live_orchestrator, fetcher_name, None)
    fetcher_func = globals().get(fetcher_name)

    if not callable(fetcher_func):
        print(f"❌ Function '{fetcher_name}' not found or not callable.")
        sys.exit(1)

    # ✅ Call the function with parsed arguments
    fetcher_func(*args)

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