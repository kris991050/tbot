import sys, os, pandas as pd

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

class TickerEvent:
    def __init__(self, symbol, recorded_time, metadata=None):
        self.symbol = symbol
        self.recorded_time = pd.to_datetime(recorded_time)
        self.metadata = metadata or {}
