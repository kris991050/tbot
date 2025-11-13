import sys, os
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.timeframe import Timeframe
from strategies import base_strategy
from strategies import target_handler


class SRBounceStrategy(base_strategy.BaseStrategy):

    def __init__(self, direction:str, timeframe:Timeframe, sr_timeframes:list=['1D', '1W'], cam_M_threshold:int=3, time_target_factor:int=10, revised:bool=False):

        super().__init__(name=f'SR_bounce_{timeframe}_{direction}', description=f'{direction}ish SR_Bounce signal using S/R levels and RSI')
        self.direction = direction
        self.timeframe = timeframe
        self.sr_timeframes = sr_timeframes
        self.cam_M_threshold = cam_M_threshold
        self.revised = revised
        rev = '' if not revised else '_R' if revised else None
        self.trigger_columns = [self.name.lower()]
        self.name += rev
        self.trigger_columns = [c + rev for c in self.trigger_columns]
        self.params = {
            'direction': self.direction,
            'cam_M_threshold': self.cam_M_threshold,
            'revised': self.revised,
        }
        self.target_handler = target_handler.TimeDeltaTargetHandler(time_target_factor * self.timeframe.to_timedelta)
        self.stop_handler = None
        self.required_columns = self._build_required_columns()
        self.required_features = {'indicators': [], 'levels': ['monthly'], 'patterns': [], 'sr': self.sr_timeframes}

    def _build_required_columns(self):
        common_req_cols = ['cam_M_position']#, 'market_cap_cat']
        olhc_req_cols = ['open', 'high', 'low', 'close', 'volume']
        direction_req_cols = {'bear': [f'sr_{sr_tf}_dist_to_next_up' for sr_tf in self.sr_timeframes], 
                              'bull': [f'sr_{sr_tf}_dist_to_next_down' for sr_tf in self.sr_timeframes]}
        revised_req_cols = {'bear': [], 'bull': []}

        if self.revised:
            revised_req_cols = {'bear': [], 'bull': []}

        return list(set(common_req_cols + olhc_req_cols + direction_req_cols[self.direction] + revised_req_cols[self.direction]))

    def evaluate_trigger(self, row):
        if self.direction.lower() == 'bull':
            trigger =  bb_rsi_trigger_bull(row, self.timeframe, sr_timeframes=self.sr_timeframes, cam_M_threshold=self.cam_M_threshold, revised=self.revised)
        elif self.direction.lower() == 'bear':
            trigger = bb_rsi_trigger_bear(row, self.timeframe, sr_timeframes=self.sr_timeframes, cam_M_threshold=self.cam_M_threshold, revised=self.revised)
        else:
            trigger = None
        if trigger is None:
            print()
        return trigger

    def evaluate_discard(self, row):
        if self.direction.lower() == 'bull':
            return bb_rsi_discard_bull()
        elif self.direction.lower() == 'bear':
            return bb_rsi_discard_bear()
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

def bb_rsi_trigger_bull(row, timeframe, sr_timeframes, cam_M_threshold, revised=False):
    sr_conditions = [row[f'sr_{sr_tf}_dist_to_next_down'] <= row[f'atr_{timeframe}'] for sr_tf in sr_timeframes]
    cam_condition = row['cam_M_position'] < -cam_M_threshold

    base_trigger = (any(sr_conditions) and cam_condition)

    if not revised:
        return base_trigger

    if len(row) == 1: row = row.iloc[0]
    revised_trigger = False
    return base_trigger and revised_trigger

def bb_rsi_discard_bull():
    discard_conditions = False
    return all(discard_conditions)

def bb_rsi_trigger_bear(row, timeframe, sr_timeframes, cam_M_threshold, revised=False):
    sr_conditions = [row[f'sr_{sr_tf}_dist_to_next_up'] <= row[f'atr_{timeframe}'] for sr_tf in sr_timeframes]
    cam_condition = row['cam_M_position'] > cam_M_threshold

    base_trigger = (any(sr_conditions) and cam_condition)

    if not revised:
        return base_trigger

    if len(row) == 1: row = row.iloc[0]
    revised_trigger = False
    return base_trigger and revised_trigger


def bb_rsi_discard_bear():
    discard_conditions = False
    return all(discard_conditions)