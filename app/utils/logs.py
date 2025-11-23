import sys, os


class DualLogger:
    def __init__(self, log_path:str, overwrite:bool=False, no_log:bool=False):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.terminal = sys.stdout
        mode = 'w' if overwrite else 'a'
        self.no_log = no_log
        if not self.no_log: self.log_file = open(log_path, mode, buffering=1, encoding='utf-8')  # Line-buffered
        sys.stdout = self  # redirect stdout

    def write(self, message, no_log=False):
        no_log = no_log or self.no_log # Option to prevent log to logfile both locally and globally (from LogContext)
        self.terminal.write(message)
        if not no_log:
            self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.no_log: self.log_file.flush()

    def close(self):
        sys.stdout = self.terminal
        if not self.no_log: self.log_file.close()


class LogContext:
    """
    Usage:
    with LogContext("outputs/my_log.txt"):
        print("This will go to both terminal and file")
    """
    def __init__(self, log_path:str, overwrite:bool=False, no_log:bool=False):
        self.log_path = log_path
        self.overwrite = overwrite
        self.logger = None
        self.no_log = no_log

    def __enter__(self):
        self.logger = DualLogger(log_path=self.log_path, overwrite=self.overwrite, no_log=self.no_log)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.close()
