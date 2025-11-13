import sys, os


class DualLogger:
    def __init__(self, log_path, overwrite=False):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        mode = 'w' if overwrite else 'a'
        self.log_file = open(log_path, mode, buffering=1, encoding='utf-8')  # Line-buffered
        sys.stdout = self  # redirect stdout

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self.terminal
        self.log_file.close()


class LogContext:
    """
    Usage:
    with LogContext("outputs/my_log.txt"):
        print("This will go to both terminal and file")
    """
    def __init__(self, log_path, overwrite=False):
        self.log_path = log_path
        self.overwrite = overwrite
        self.logger = None

    def __enter__(self):
        self.logger = DualLogger(self.log_path, self.overwrite)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.close()
