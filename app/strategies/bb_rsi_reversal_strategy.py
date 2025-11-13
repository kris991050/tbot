import sys, os
from numba import njit
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.timeframe import Timeframe
from strategies import base_strategy
from strategies import target_handler


class BBRSIReversalStrategy(base_strategy.BaseStrategy):

    def __init__(self, direction:str, timeframe:Timeframe, rsi_threshold:int=75, cam_M_threshold:int=4, revised:bool=False):

        super().__init__(name=f'bb_rsi_reversal_{timeframe}_{direction}', description=f'{direction}ish BB_RSI_Reversal signal using RSI and BB from higher timeframes')
        self.direction = direction
        # self.timeframe_min = timeframe_min
        # self.timeframe = f"{timeframe_min}min" if timeframe_min == 1 else f"{timeframe_min}mins" if timeframe_min > 1 else None
        self.timeframe = timeframe
        self.timeframes = [self.timeframe] + [self.timeframe.next_timeframe(i) for i in [1, 2, 3]]
        # if self.timeframe.pandas == '1min': self.timeframes = [Timeframe('1min'), Timeframe('5min'), Timeframe('15min'), Timeframe('1h')]
        self.rsi_threshold = rsi_threshold
        self.cam_M_threshold = cam_M_threshold
        self.revised = revised
        rev = '' if not revised else '_R' if revised else None
        self.trigger_columns = [self.name.lower()]
        self.name += rev
        self.trigger_columns = [c + rev for c in self.trigger_columns]
        self.params = {
            'direction': self.direction,
            'rsi_threshold': self.rsi_threshold,
            'cam_M_threshold': self.cam_M_threshold,
            'revised': self.revised,
            # 'timeframe': self.timeframe,
            # # 'target': 'vwap_cross',
            # # 'target': f'vwap_{self.timeframe}',
            # 'trigger_features': {'indicators': ['momentum', 'volatility'], 'levels': True, 'patterns': False, 'sr': False},
            # 'trigger_timeframes': self.timeframes#]['2min', '5min', '15min', '1h'] if timeframe_min == 1 else ['5min', '15min', '1h', '4h'] if timeframe_min == 2 else ['1min', '5min', '15min', '1h'] if timeframe_min == 5 else []
        }
        self.target_handler = target_handler.VWAPCrossTargetHandler(self.timeframe, max_time='eod')
        self.stop_handler = None
        self.required_columns = self._build_required_columns()
        self.required_features = {'indicators': ['momentum', 'volatility'], 'levels': ['monthly'], 'patterns': [], 'sr': False}

    def _build_required_columns(self):
        common_req_cols = [f'rsi_{tf}' for tf in self.timeframes] + ['cam_M_position']#, 'market_cap_cat']
        olhc_req_cols = ['open', 'high', 'low', 'close', 'volume']
        direction_req_cols = {'bear': [f'bband_h_{tf}' for tf in self.timeframes[1:]],
                              'bull': [f'bband_l_{tf}' for tf in self.timeframes[1:]]}
        revised_req_cols = {'bear': [], 'bull': []}

        # if self.timeframe_min == 1: timeframe_specific_cols = ['rsi_1min']
        # elif self.timeframe_min == 2: timeframe_specific_cols = ['rsi_2min']
        # elif self.timeframe_min == 5: timeframe_specific_cols = ['rsi_4h', 'bband_h_4h', 'bband_l_4h']

        # common_req_cols = ['rsi_5min', 'rsi_15min', 'rsi_1h', 'cam_M_position']#, 'market_cap_cat']
        # olhc_req_cols = ['open', 'high', 'low', 'close', 'volume']
        # direction_req_cols = {'bear': ['bband_h_5min', 'bband_h_15min', 'bband_h_1h'],
        #                               'bull': ['bband_l_5min', 'bband_l_15min', 'bband_l_1h']}
        # revised_req_cols = {'bear': [], 'bull': []}
        if self.revised:
            revised_req_cols = {'bear': ['rsi_slope_1D', 'bband_h_1h_pct_diff', 'bband_h_1D_pct_diff', 'breakout_down_since_last'],
                                'bull': ['rsi_slope_1D', 'rsi_1D', 'rsi_slope_1h', 'bband_width_ratio_1D', 'bband_l_1D_pct_diff']}
                                    # 'bull': ['rsi_slope_1D', 'levels_dist_to_next_up_pct', 'bband_h_1D_dist_pct_atr', 'bband_l_1D_dist_pct_atr']}

        return list(set(common_req_cols + olhc_req_cols + direction_req_cols[self.direction] + revised_req_cols[self.direction])) # + timeframe_specific_cols

    # def _build_required_columns(self):
    #     if self.timeframe_min == 1: timeframe_specific_cols = ['rsi_1min']
    #     elif self.timeframe_min == 2: timeframe_specific_cols = ['rsi_2min']
    #     elif self.timeframe_min == 5: timeframe_specific_cols = ['rsi_4h', 'bband_h_4h', 'bband_l_4h']

    #     common_req_cols = ['rsi_5min', 'rsi_15min', 'rsi_1h', 'cam_M_position']#, 'market_cap_cat']
    #     olhc_req_cols = ['open', 'high', 'low', 'close', 'volume']
    #     direction_req_cols = {'bear': ['bband_h_5min', 'bband_h_15min', 'bband_h_1h'],
    #                                   'bull': ['bband_l_5min', 'bband_l_15min', 'bband_l_1h']}
    #     revised_req_cols = {'bear': [], 'bull': []}
    #     if self.revised:
    #         revised_req_cols = {'bear': ['rsi_slope_1D', 'bband_h_1h_pct_diff', 'bband_h_1D_pct_diff', 'breakout_down_since_last'],
    #                             'bull': ['rsi_slope_1D', 'rsi_1D', 'rsi_slope_1h', 'bband_width_ratio_1D', 'bband_l_1D_pct_diff']}
    #                                 # 'bull': ['rsi_slope_1D', 'levels_dist_to_next_up_pct', 'bband_h_1D_dist_pct_atr', 'bband_l_1D_dist_pct_atr']}

    #     return list(set(common_req_cols + timeframe_specific_cols + olhc_req_cols + direction_req_cols[self.direction] + revised_req_cols[self.direction]))

    def evaluate_trigger(self, row):
        if self.direction.lower() == 'bull':
            trigger =  bb_rsi_trigger_bull(row, self.timeframes, rsi_threshold=self.rsi_threshold, cam_M_threshold=self.cam_M_threshold, revised=self.revised)
        elif self.direction.lower() == 'bear':
            trigger = bb_rsi_trigger_bear(row, self.timeframes, rsi_threshold=self.rsi_threshold, cam_M_threshold=self.cam_M_threshold, revised=self.revised)
        else:
            trigger = None
        if trigger is None:
            print()
        return trigger

    def evaluate_discard(self, row):
        if self.direction.lower() == 'bull':
            return bb_rsi_discard_bull(row, self.timeframes, rsi_threshold=self.rsi_threshold)
        elif self.direction.lower() == 'bear':
            return bb_rsi_discard_bear(row, self.timeframes, rsi_threshold=self.rsi_threshold)
        else:
            return None

    def apply(self, df):
        if df.empty: return df, []
        # Check df timeframe
        if helpers.get_df_timeframe(df).pandas not in [self.timeframe.pandas]:
            print(f"Timeframe must be {self.timeframe}")
            return df, []

        # df[self.trigger_columns[0]] = df.apply(self.evaluate_trigger, axis=1)
        df[self.trigger_columns[0]] = helpers.apply_df_in_chunks(df, evaluate_func=self.evaluate_trigger, memory_threshold_MB=1500)
        # df[self.trigger_columns[0]] = df.apply(lambda row: self.evaluate_trigger(row, row_tr), axis=1)

        return df#.copy()

def bb_rsi_trigger_bull(row, timeframes, rsi_threshold, cam_M_threshold, revised=False):
    rsi_conditions = [row[f'rsi_{tf}'] < (100 - rsi_threshold) for tf in timeframes]
    bband_conditions = [row[f'bband_l_{tf}'] > row['close'] for tf in timeframes[1:]]
    cam_condition = row['cam_M_position'] < -cam_M_threshold

    base_trigger = (all(rsi_conditions) and all(bband_conditions) and cam_condition)

    # if timeframe_min in [1, 2]: timeframe_specific_cond = (row['rsi_5min'] < (100 - rsi_threshold) and row['bband_l_5min'] > row['close'])
    # elif timeframe_min == 5: timeframe_specific_cond = (row['rsi_4h'] < (100 - rsi_threshold) and row['bband_l_4h'] > row['close'])
    # base_trigger = (
    #     timeframe_specific_cond and
    #     row['rsi'] < (100 - rsi_threshold) and
    #     row['rsi_15min'] < (100 - rsi_threshold) and
    #     row['rsi_1h'] < (100 - rsi_threshold) and
    #     row['bband_l_15min'] > row['close'] and
    #     row['bband_l_1h'] > row['close'] and
    #     row['cam_M_position'] < -cam_M_threshold
    #     # row.get('cam_M_position', row.get('cam_M', 0)) < -cam_M_threshold
    # )

    if not revised:
        return base_trigger

    # row_tr, _ = features_processor.apply_feature_transformations(row)
    if len(row) == 1: row = row.iloc[0]
    revised_trigger = (
        (row['rsi_slope_1D'] <= -4.085 and row['rsi_1D'] > 35.715 and row['rsi_slope_1h'] <= -0.21) or
        (row['rsi_slope_1D'] > -4.085 and row['bband_width_ratio_1D'] <= 0.98 and row['bband_l_1D_pct_diff'] <= 0.0276)
    )
    # revised_trigger = (
    #     (row['rsi_slope_1D'] < 2 and row['levels_dist_to_next_up_pct'] < 0.008) or
    #     (row['rsi_slope_1D'] < 2 and row['levels_dist_to_next_up_pct'] > 0.008 and row['market_cap_cat'] < 4.5) or
    #     (row['rsi_slope_1D'] > 2 and row['bband_h_1D_dist_pct_atr'] > 2.8 and row['bband_l_1D_dist_pct_atr'] < 42)

        # (row['rsi_1D'] < 28 and 5 < row['pivots_M_dist_to_next_up'] < 7.5) or
        # (row['rsi_1D'] > 28 and row['pivots_M_dist_to_next_up'] < 2.5 and row['atr_D'] < 4) or
        # (row['rsi_1D'] > 28 and row['pivots_M_dist_to_next_up'] > 2.5 and row['avg_volume'] < 240000)
    # )

    return base_trigger and revised_trigger


def bb_rsi_discard_bull(row, timeframes, rsi_threshold):
    discard_conditions = [row[f'rsi_{tf}'] > (100 - rsi_threshold) for tf in timeframes[2:]]
    return all(discard_conditions)
    # if timeframe_min == 1 or timeframe_min == 2:
    #     return (row['rsi_15min'] > (100 - rsi_threshold) or row['rsi_1h'] > (100 - rsi_threshold))
    # elif timeframe_min == 5:
    #     return (row['rsi_1h'] > (100 - rsi_threshold) or row['rsi_4h'] > (100 - rsi_threshold))
    # else: return True


def bb_rsi_trigger_bear(row, timeframes, rsi_threshold, cam_M_threshold, revised=False):
    rsi_conditions = [row[f'rsi_{tf}'] > rsi_threshold for tf in timeframes]
    bband_conditions = [row[f'bband_h_{tf}'] < row['close'] for tf in timeframes[1:]]
    cam_condition = row['cam_M_position'] > cam_M_threshold

    base_trigger = (all(rsi_conditions) and all(bband_conditions) and cam_condition)

    # if timeframe_min in [1, 2]: timeframe_specific_cond = (row['rsi_5min'] > (rsi_threshold) and row['bband_h_5min'] < row['close'])
    # elif timeframe_min == 5: timeframe_specific_cond = (row['rsi_4h'] > (rsi_threshold) and row['bband_h_4h'] > row['close'])
    # base_trigger = (
    #     timeframe_specific_cond and
    #     row['rsi'] > (rsi_threshold) and
    #     row['rsi_15min'] > (rsi_threshold) and
    #     row['rsi_1h'] > (rsi_threshold) and
    #     row['bband_h_15min'] < row['close'] and
    #     row['bband_h_1h'] < row['close'] and
    #     row['cam_M_position'] > cam_M_threshold
    #     # row.get('cam_M_position', row.get('cam_M', 0)) > cam_M_threshold
    # )

    if not revised:
        return base_trigger

    # row_tr, _ = features_processor.apply_feature_transformations(row)
    if len(row) == 1: row = row.iloc[0]
    # revised_trigger = (
    #     (row['rsi_1D'] < 71.5 and row['hammer_up_since_last'] > 33.5) or
    #     (row['rsi_1D'] > 71.5 and row['sr_1h_dist_to_next_down'] < 0.0065) or
    #     (row['rsi_1D'] > 71.5 and row['sr_1h_dist_to_next_down'] > 0.0065 and row['avg_volume'] < 243)
    # )

    # revised_trigger = row_tr.apply(evaluate_trigger, axis=1)
    revised_trigger = (
        (row['rsi_slope_1D'] <= 4.625 and row['rsi_slope_1D'] <= 1.92 and row['levels_dist_to_next_up_pct'] <= 0.0087) or
        (row['rsi_slope_1D'] <= 4.625 and row['rsi_slope_1D'] > 1.92 and row['bband_h_1h_pct_diff'] > 0.0069) or
        (row['rsi_slope_1D'] > 4.625 and row['bband_h_1D_pct_diff'] > 0.0227 and row['breakout_down_since_last_1min'] > 871.0)
    )

    return base_trigger and revised_trigger


def bb_rsi_discard_bear(row, timeframes, rsi_threshold):
    discard_conditions = [row[f'rsi_{tf}'] < rsi_threshold for tf in timeframes[2:]]
    return all(discard_conditions)
    # if timeframe_min == 1 or timeframe_min == 2:
    #     return (row['rsi_15min'] < rsi_threshold or row['rsi_1h'] < rsi_threshold)
    # elif timeframe_min == 5:
    #     return (row['rsi_1h'] < rsi_threshold or row['rsi_4h'] < rsi_threshold)
    # else: return True


class BBRSIReversalStrategy_1(BBRSIReversalStrategy):
    def __init__(self, direction, rsi_threshold=75, cam_M_threshold=4, revised=False):
        super().__init__(direction=direction, timeframe_min = 1, rsi_threshold=rsi_threshold, cam_M_threshold=cam_M_threshold, revised=revised)


class BBRSIReversalStrategy_2(BBRSIReversalStrategy):
    def __init__(self, direction, rsi_threshold=75, cam_M_threshold=4, revised=False):
        super().__init__(direction=direction, timeframe_min = 2, rsi_threshold=rsi_threshold, cam_M_threshold=cam_M_threshold, revised=revised)


class BBRSIReversalStrategy_5(BBRSIReversalStrategy):
    def __init__(self, direction, rsi_threshold=75, cam_M_threshold=4, revised=False):
        super().__init__(direction=direction, timeframe_min = 5, rsi_threshold=rsi_threshold, cam_M_threshold=cam_M_threshold, revised=revised)



# ======================= #
#   Fast Ngit functions   #
# ======================= #

@njit
def bb_rsi_trigger_bull_fast(rsi_5, rsi_15, rsi_60, rsi, bband_l_5, bband_l_15, bband_l_60, close, cam_M_position, rsi_slope_1D, levels_dist_to_next_up_pct, market_cap_cat,
                             bband_h_1D_dist_pct_atr, bband_l_1D_dist_pct_atr, rsi_threshold=75, cam_M_threshold=4, revised=False) -> bool:
    base_trigger = (
        rsi_5 < (100 - rsi_threshold) and
        rsi_15 < (100 - rsi_threshold) and
        rsi_60 < (100 - rsi_threshold) and
        rsi < (100 - rsi_threshold) and
        bband_l_5 > close and
        bband_l_15 > close and
        bband_l_60 > close and
        cam_M_position < -cam_M_threshold
    )
    if not revised:
        return base_trigger

    revised_trigger = (
        (rsi_slope_1D < 2 and levels_dist_to_next_up_pct < 0.008) or
        (rsi_slope_1D < 2 and levels_dist_to_next_up_pct > 0.008 and market_cap_cat < 4.5) or
        (rsi_slope_1D > 2 and bband_h_1D_dist_pct_atr > 2.8 and bband_l_1D_dist_pct_atr < 42)
    )
    return base_trigger and revised_trigger

@njit
def bb_rsi_trigger_bear_fast(rsi_5, rsi_15, rsi_60, rsi, bband_l_5, bband_l_15, bband_l_60, close, cam_M_position, rsi_slope_1D, levels_dist_to_next_up_pct, market_cap_cat,
                             bband_h_1D_dist_pct_atr, bband_l_1D_dist_pct_atr, rsi_threshold=75, cam_M_threshold=4, revised=False) -> bool:
    base_trigger = (
        rsi_5 > rsi_threshold and
        rsi_15 > rsi_threshold and
        rsi_60 > rsi_threshold and
        rsi > rsi_threshold and
        bband_l_5 < close and
        bband_l_15 < close and
        bband_l_60 < close and
        cam_M_position > cam_M_threshold
    )
    if not revised:
        return base_trigger

    revised_trigger = (
        (rsi_slope_1D < 2 and levels_dist_to_next_up_pct < 0.008) or
        (rsi_slope_1D < 2 and levels_dist_to_next_up_pct > 0.008 and market_cap_cat < 4.5) or
        (rsi_slope_1D > 2 and bband_h_1D_dist_pct_atr > 2.8 and bband_l_1D_dist_pct_atr < 42)
    )
    return base_trigger and revised_trigger


# class BBRSIReversalStrategyBull(BBRSIReversalStrategy):

#     def __init__(self, rsi_threshold=75, cam_M_threshold=4, revised=False):
#         super().__init__(direction='Bull', rsi_threshold=rsi_threshold, cam_M_threshold=cam_M_threshold, revised=revised)

#     def evaluate_trigger(self, row):
#         return bb_rsi_trigger_bull(row, rsi_threshold=self.rsi_threshold, cam_M_threshold=self.cam_M_threshold, revised=self.revised)

#     def apply(self, df):
#         if df.empty: return df, []
#         # Check df timeframe
#         if helpers.get_df_timeframe(df) not in ['1 min']:
#             print("Timeframe must be 1 min")
#             return df, []

#         df[self.trigger_columns[0]] = df.apply(self.evaluate_trigger, axis=1)

#         return df.copy(), self.trigger_columns


# class BBRSIReversalStrategyBear(BBRSIReversalStrategy):

#     def __init__(self, rsi_threshold=75, cam_M_threshold=4, revised=False):

#         super().__init__(direction='Bear', rsi_threshold=rsi_threshold, cam_M_threshold=cam_M_threshold, revised=revised)

#     def apply(self, df):
#         if df.empty: return df, []
#         # Check df timeframe
#         if helpers.get_df_timeframe(df) not in ['1 min']:
#             print("Timeframe must be 1 min")
#             return df, []

#         else:
#             df.rename(columns={'cam_M_position': 'cam_M'}, inplace=True)
#             df['rsi_trigger_bear'] = (df['rsi_5'] > self.rsi_threshold) & (df['rsi_15'] > self.rsi_threshold) & (df['rsi_60'] > self.rsi_threshold) & (df['rsi'] > self.rsi_threshold) & (df['cam_M'] > self.cam_M_threshold)
#             df['bb_trigger_bear'] = (df['close'] > df['bband_h_5']) & (df['close'] > df['bband_h_15']) & (df['close'] > df['bband_h_60'])

#             if self.revised:
#                 df['revised_trigger_bear'] = ((df['rsi_1D'] < 71.5) & (df['hammer_up_since_last'] > 33.5)) | \
#                     ((df['rsi_1D'] > 71.5) & (df['sr_1h_dist_to_next_down'] < 0.0065)) | \
#                     ((df['rsi_1D'] > 71.5) & (df['sr_1h_dist_to_next_down'] > 0.0065) & (df['avg_volume'] < 243))
#             else:
#                 df['revised_trigger_bear'] = True

#             df[self.trigger_columns[0]] = df['rsi_trigger_bear'] & df['bb_trigger_bear'] & df['revised_trigger_bear']
#             df.drop(columns=['rsi_trigger_bear', 'bb_trigger_bear', 'revised_trigger_bear'], inplace=True)

#             return df.copy(), self.trigger_columns





# class BBRSIReversalStrategyBull(BBRSIReversalStrategy):

#     def __init__(self, rsi_threshold=75, cam_M_threshold=4, revised=False):

#         super().__init__(side='Bull', rsi_threshold=rsi_threshold, cam_M_threshold=cam_M_threshold, revised=revised)

#     def apply(self, df):
#         if df.empty: return df, []
#         # Check df timeframe
#         if helpers.get_df_timeframe(df) not in ['1 min']:
#             print("Timeframe must be 1 min")
#             return df, []

#         else:
#             # df_tf = df.resample(tf, on='date').agg({'open':'first', 'close':'last', 'high':'max', 'low':'min', 'volume':'sum'}).dropna(subset=['close']).reset_index(
#             df.rename(columns={'cam_M_position': 'cam_M'}, inplace=True)
#             # df['rsi_trigger_down'] = (df['rsi_5'] > self.rsi_threshold) & (df['rsi_15'] > self.rsi_threshold) & (df['rsi_60'] > self.rsi_threshold) & (df['rsi'] > self.rsi_threshold) & (df['cam_M'] > self.cam_M_threshold)
#             df['rsi_trigger_bull'] = (df['rsi_5'] < (100 - self.rsi_threshold)) & (df['rsi_15'] < (100 - self.rsi_threshold)) & (df['rsi_60'] < (100 - self.rsi_threshold)) & (df['rsi'] < (100 - self.rsi_threshold)) & (df['cam_M'] < -self.cam_M_threshold)
#             # df['bb_trigger_down'] = (df['close'] > df['bband_h_5']) & (df['close'] > df['bband_h_15']) & (df['close'] > df['bband_h_60'])# & (df['close'] > df['bband_h'])
#             df['bb_trigger_bull'] = (df['close'] < df['bband_l_5']) & (df['close'] < df['bband_l_15']) & (df['close'] < df['bband_l_60'])# & (df['close'] > df['bband_l'])
#             # df[self.trigger_columns[0]] = df['rsi_trigger_bull'] & df['bb_trigger_bull']# & (df['date'].dt.time >= pd.to_datetime("10:00:00").time()) & (df['date'].dt.time < pd.to_datetime("16:00:00").time())
#             # df[self.trigger_columns[1]] = df['rsi_trigger_down'] & df['bb_trigger_down']# & (df['date'].dt.time >= pd.to_datetime("10:00:00").time()) & (df['date'].dt.time < pd.to_datetime("16:00:00").time())

#             if self.revised:
#                 # df['revised_trigger_bull'] = ((df['rsi_1D'] < 33.5) & (df['levels_M_dist_to_next_up'] > 6) & (df['change'] < 49)) | \
#                 #     ((df['rsi_1D'] > 33.5) & (df['breakout_up_since_last'] < 6100))
#                 # df['revised_trigger_bull'] = ((df['rsi_1D'] < 30) & (df['avg_volume_log'] > 9) & (df['gap'] > -2.5) & (df['pivots_M_dist_to_next_up_pct'] < 0.1))
#                 df['revised_trigger_bull'] = ((df['rsi_1D'] < 28) & (df['pivots_M_dist_to_next_up'] > 5) & (df['pivots_M_dist_to_next_up'] < 7.5)) | \
#                     ((df['rsi_1D'] > 28) & (df['pivots_M_dist_to_next_up'] < 2.5) & (df['atr_D'] < 4)) | \
#                     ((df['rsi_1D'] > 28) & (df['pivots_M_dist_to_next_up'] > 2.5) & (df['avg_volume'] < 240000))
#             else:
#                 df['revised_trigger_bull'] = True

#             df[self.trigger_columns[0]] = df['rsi_trigger_bull'] & df['bb_trigger_bull'] & df['revised_trigger_bull']
#             df.drop(columns=['rsi_trigger_bull', 'bb_trigger_bull', 'revised_trigger_bull'], inplace=True)

#             return df.copy(), self.trigger_columns