import sys, os, abc
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from strategies import bb_rsi_reversal_strategy


# from abc import ABC, abstractmethod

class BaseStrategy(abc.ABC):
    """
    Abstract base class for all strategies.
    """

    def __init__(self, name: str, description=None):
        self.name = name
        # self.strategy_subcategories = ["bull", "bear"]
        # self.trigger_columns = [f"{self.name.lower()}_{st_sub}" for st_sub in self.strategy_subcategories]
        self.trigger_columns = [f"{self.name.lower()}"]
        self.description = description or ""
        self.timeframe = None
        self.params = {}  # Optional: strategy-specific parameters (for optimization)

    @abc.abstractmethod
    def apply(self, df):
        """
        Apply the strategy logic to the given DataFrame.

        :param df: pandas DataFrame containing market data.
        :return: Tuple of (modified DataFrame, list of trigger column names)
        """
        pass

    def get_parameters(self):
        """
        Optional: Return parameters for optimization / reporting.
        """
        return self.params

    def __repr__(self):
        return f"<Strategy: {self.name}>"

