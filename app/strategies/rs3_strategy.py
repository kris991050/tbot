import sys, os, abc
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from strategies import base_strategy


class RS3StrategyBull(base_strategy.BaseStrategy):

    def __init__(self, daily_pivots=False):

        super().__init__(name='RS3_Bull', description='Bullish RS3 signal using Camarilla pivot level -3')
        self.daily_pivots = daily_pivots
        # self.strategy_subcategories = ['bull']
        # self.trigger_columns = ['rs3']
        self.trigger_columns = [self.name.lower()]
        self.params = {
            'daily_pivots': self.daily_pivots
        }

    def apply(self, df):
        if df.empty:
            return df, []

        tf_seconds = helpers.timeframe_to_seconds(helpers.get_df_timeframe(df))

        if tf_seconds < helpers.timeframe_to_seconds('1 day'):
            cam_type = '_D' if self.daily_pivots else ''
        else:
            cam_type = '_M'

        cam_position_str = f'cam{cam_type}_position'
        df[self.trigger_columns[0]] = (df[cam_position_str] == -3)

        return df, self.trigger_columns


