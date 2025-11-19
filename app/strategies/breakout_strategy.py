import sys, os
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from strategies import base_strategy


import sys, os
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.constants import CONSTANTS
from utils.timeframe import Timeframe
from strategies import base_strategy, target_handler
from features import indicators


class BreakoutStrategy(base_strategy.BaseStrategy):

    def __init__(self, direction:str, timeframe:Timeframe, sr_timeframes:list=None, revised:bool=False, 
                 target_factor:float=5, max_time_factor:int=50):

        super().__init__(name=f'breakout_{timeframe}_{direction}', description=f'{direction}ish Breakout signal using S/R levels')
        self.direction = helpers.set_var_with_constraints(direction, CONSTANTS.DIRECTIONS)
        self.timeframe = timeframe
        self.sr_timeframes = indicators.IndicatorsUtils.resolve_sr_timeframes(self.timeframe, sr_timeframes)
        self.revised = revised
        rev = '' if not revised else '_R' if revised else None
        self.trigger_columns = [self.name]
        self.name += rev
        self.trigger_columns = [c + rev for c in self.trigger_columns]
        self.params = {
            'direction': self.direction,
            'revised': self.revised,
        }
        # self.target_handler = target_handler.TimeDeltaTargetHandler(time_target_factor * self.timeframe.to_timedelta)
        level_types = [f'sr_{sr_tf}' for sr_tf in self.sr_timeframes]
        max_time = max_time_factor * self.timeframe.to_timedelta
        # self.target_handler = target_handler.NextLevelTargetHandler(level_types=level_types, timeframe=self.timeframe, direction=self.direction, 
        #                                                             max_time=max_time)
        self.target_handler = target_handler.PercGainTargetHandler(perc_gain=target_factor, direction=self.direction, max_time=max_time)
        self.stop_handler = None
        self.required_columns = self._build_required_columns()
        self.required_features = {'indicators': [], 'levels': [], 'patterns': ['range', 'breakout'], 'sr': self.sr_timeframes}

    def _build_required_columns(self):
        common_req_cols = [f'atr_{self.timeframe}']
        olhc_req_cols = ['open', 'high', 'low', 'close', 'volume']
        direction_req_cols = {'bear': [f'sr_{sr_tf}_dist_to_next_up' for sr_tf in self.sr_timeframes] + [f'breakout_down_{self.timeframe}'], 
                              'bull': [f'sr_{sr_tf}_dist_to_next_down' for sr_tf in self.sr_timeframes] + [f'breakout_up_{self.timeframe}']}
        revised_req_cols = {'bear': [], 'bull': []}

        if self.revised:
            revised_req_cols = {'bear': [], 'bull': []}

        return list(set(common_req_cols + olhc_req_cols + direction_req_cols[self.direction] + revised_req_cols[self.direction]))

    def evaluate_trigger(self, row):
        if self.direction.lower() == 'bull':
            trigger =  breakout_trigger_bull(row, self.timeframe, sr_timeframes=self.sr_timeframes, revised=self.revised)
        elif self.direction.lower() == 'bear':
            trigger = breakout_trigger_bear(row, self.timeframe, sr_timeframes=self.sr_timeframes, revised=self.revised)
        else:
            trigger = None
        if trigger is None:
            print()
        return trigger

    def evaluate_discard(self, row):
        if self.direction.lower() == 'bull':
            return breakout_discard_bull()
        elif self.direction.lower() == 'bear':
            return breakout_discard_bear()
        else:
            return None

    def apply(self, df):
        if df.empty: return df, []
        # Check df timeframe
        if helpers.get_df_timeframe(df).pandas not in [self.timeframe.pandas]:
            print(f"Timeframe must be {self.timeframe}")
            return df, []

        df[self.trigger_columns[0]] = helpers.apply_df_in_chunks(df, evaluate_func=self.evaluate_trigger, memory_threshold_MB=1500)

        return df

def breakout_trigger_bull(row, timeframe, sr_timeframes, revised=False):
    sr_conditions = [row[f'sr_{sr_tf}_dist_to_next_down'] <= row[f'atr_{timeframe}'] for sr_tf in sr_timeframes]
    breakout_condition = row[f'breakout_up_{timeframe}']

    base_trigger = (any(sr_conditions) and breakout_condition)

    if not revised:
        return base_trigger

    if len(row) == 1: row = row.iloc[0]
    revised_trigger = False
    return base_trigger and revised_trigger

def breakout_discard_bull():
    discard_conditions = False
    return all(discard_conditions)

def breakout_trigger_bear(row, timeframe, sr_timeframes, revised=False):
    sr_conditions = [row[f'sr_{sr_tf}_dist_to_next_up'] <= row[f'atr_{timeframe}'] for sr_tf in sr_timeframes]
    breakout_condition = row[f'breakout_down_{timeframe}']

    base_trigger = (any(sr_conditions) and breakout_condition)

    if not revised:
        return base_trigger

    if len(row) == 1: row = row.iloc[0]
    revised_trigger = False
    return base_trigger and revised_trigger


def breakout_discard_bear():
    discard_conditions = False
    return all(discard_conditions)








# class BreakoutsStrategyBull(base_strategy.BaseStrategy):

#     def __init__(self):

#         super().__init__(name='Breakouts_Bull', description='Bullish Breakouts signal using consolidation and ranges')
#         # self.trigger_columns = ['breakout_up', 'breakout_down']
#         self.trigger_columns = ['breakout_up']
#         self.params = {}

#     def apply(self, df):
#         if df.empty:
#             return df, []

#         tc_list = []
#         for tc in self.trigger_columns:
#             if tc in df.columns:
#                 tc_list.append(tc)

#         return df, tc_list
    

# class BreakoutsStrategy(base_strategy.BaseStrategy):

#     def __init__(self, side):

#         super().__init__(name=f'Breakouts_{side}', description=f'{side}ish Breakouts signal using consolidation and ranges')
#         # self.trigger_columns = ['breakout_up', 'breakout_down']
#         self.trigger_columns = ['breakout_up'] if side == "Bull" else ['breakout_down'] if side == "Bear" else []
#         self.params = {}

#     def apply(self, df):
#         if df.empty:
#             return df, []

#         tc_list = []
#         for tc in self.trigger_columns:
#             if tc in df.columns:
#                 tc_list.append(tc)

#         return df, tc_list
    

# class BreakoutsStrategyBull(BreakoutsStrategy):

#     def __init__(self):
#         super().__init__(side='Bull')

# class BreakoutsStrategyBear(BreakoutsStrategy):

#     def __init__(self):
#         super().__init__(side='Bear')
