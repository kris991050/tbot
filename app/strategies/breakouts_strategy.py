import sys, os, abc
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from strategies import base_strategy


class BreakoutsStrategyBull(base_strategy.BaseStrategy):

    def __init__(self):

        super().__init__(name='Breakouts_Bull', description='Bullish Breakouts signal using consolidation and ranges')
        # self.trigger_columns = ['breakout_up', 'breakout_down']
        self.trigger_columns = ['breakout_up']
        self.params = {}

    def apply(self, df):
        if df.empty:
            return df, []

        tc_list = []
        for tc in self.trigger_columns:
            if tc in df.columns:
                tc_list.append(tc)

        return df, tc_list
    

class BreakoutsStrategy(base_strategy.BaseStrategy):

    def __init__(self, side):

        super().__init__(name=f'Breakouts_{side}', description=f'{side}ish Breakouts signal using consolidation and ranges')
        # self.trigger_columns = ['breakout_up', 'breakout_down']
        self.trigger_columns = ['breakout_up'] if side == "Bull" else ['breakout_down'] if side == "Bear" else []
        self.params = {}

    def apply(self, df):
        if df.empty:
            return df, []

        tc_list = []
        for tc in self.trigger_columns:
            if tc in df.columns:
                tc_list.append(tc)

        return df, tc_list
    

class BreakoutsStrategyBull(BreakoutsStrategy):

    def __init__(self):
        super().__init__(side='Bull')

class BreakoutsStrategyBear(BreakoutsStrategy):

    def __init__(self):
        super().__init__(side='Bear')
