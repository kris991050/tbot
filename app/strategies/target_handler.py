import sys, os, pandas as pd, numpy as np, pytz
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Optional, Union

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.constants import CONSTANTS
from utils.timeframe import Timeframe
from features import indicators



def resolve_max_time(trigger_time:pd.Timestamp, max_time:timedelta, timezone=CONSTANTS.TZ_WORK) -> Optional[pd.Timestamp]:
    """
    Resolves a max_time constraint relative to trigger_time.

    max_time can be:
        - None → no limit
        - pd.Timedelta → adds to trigger_time
        - 'eod' or 'eod_rth' → session-based end
    """
    if max_time is None:
        return None
    elif isinstance(max_time, timedelta):
        return trigger_time + max_time
    elif isinstance(max_time, str):
        lower = max_time.lower()
        if lower == 'eod':
            return pd.Timestamp.combine(trigger_time.date(), CONSTANTS.TH_TIMES['end_of_day']).tz_localize(timezone)
        elif lower == 'eod_rth':
            return pd.Timestamp.combine(trigger_time.date(), CONSTANTS.TH_TIMES['post-market']).tz_localize(timezone)
    raise ValueError(f"Invalid max_time: {max_time}")


def resolve_level_types(timeframe:Timeframe):
    sr_tfs = indicators.IndicatorsUtils.resolve_sr_timeframes(timeframe)
    level_types = [f'sr_{sr_tf}' for sr_tf in sr_tfs]
    if timeframe.to_seconds < Timeframe('1M').to_seconds:
        level_types.extend(['levels_M', 'pivots_M'])
    if timeframe.to_seconds < Timeframe('1D').to_seconds:
        level_types.extend(['pivots', 'pivots_D', 'levels'])
    return level_types


class TargetHandler(ABC):
    """
    Abstract base class for handling different types of post-trigger targets (time-based, event-based, etc.).
    """
    def __init__(self, max_time:timedelta=None, timezone=CONSTANTS.TZ_WORK):
        self.target_str = None
        self.required_columns = []
        self.entry_time = None
        self.timezone = timezone
        self.max_time = max_time
    
    def set_entry_time(self, time:pd.Timestamp):
        self.entry_time = time

    @abstractmethod
    def get_target_time(self, df:pd.DataFrame, trigger_time:pd.Timestamp) -> pd.Timestamp:
        pass

    @abstractmethod
    def get_target_event(self, prev_row:pd.Series, curr_row:pd.Series, entry_time:pd.Timestamp=None) -> Optional[str]:
        """
        Returns the exit reason if exit condition is met, else None.
        """
        pass

    # @staticmethod
    # def from_target(target:Union[str, pd.Timedelta, 'TargetHandler'], timezone:pytz=CONSTANTS.TZ_WORK) -> 'TargetHandler':
    #     """
    #     Factory method to return appropriate TargetHandler based on `target` input.
    #     """
    #     if isinstance(target, TargetHandler):
    #         return target
    #     elif isinstance(target, pd.Timedelta):
    #         return TimeDeltaTargetHandler(target, timezone=timezone)
    #     elif isinstance(target, str):
    #         lower = target.lower()
    #         if lower in ['eod', 'eod_rth']:
    #             return EODTargetHandler(rth_only=(lower == 'eod_rth'), timezone=timezone)
    #         elif 'vwap' in lower:
    #             timeframe = helpers.extract_timeframe_from_df_column(lower)
    #             return VWAPCrossTargetHandler(timeframe=timeframe, max_time='eod_rth', timezone=timezone)
    #         else:
    #             # Try parsing string like "5 min", "1h", etc.
    #             try:
    #                 parsed_delta = pd.to_timedelta(target)
    #                 return TimeDeltaTargetHandler(parsed_delta)
    #             except ValueError:
    #                 raise ValueError(f"Unsupported target type: {target}")
    #     else:
    #         raise ValueError(f"Unsupported target type: {target}")


class TimeDeltaTargetHandler(TargetHandler):
    def __init__(self, delta:pd.Timedelta, timezone=CONSTANTS.TZ_WORK):
        super().__init__(max_time=delta, timezone=timezone)
        self.target_str = str(self.max_time.total_seconds() / 60) + 'min'

    def get_target_time(self, df:pd.DataFrame, trigger_time:pd.Timestamp) -> pd.Timestamp:
        return resolve_max_time(trigger_time, self.max_time, self.timezone)
    
    def get_target_event(self, prev_row:pd.Series, curr_row:pd.Series, entry_time:pd.Timestamp=None) -> Optional[str]:
        entry_time = entry_time or self.entry_time
        if entry_time is None:
            return None

        end_time = resolve_max_time(entry_time, self.max_time, timezone=self.timezone)
        if curr_row['date'] >= end_time:
            return 'timedelta_exit'
        return None


# class EODTargetHandler(TargetHandler):
#     def __init__(self, rth_only: bool = False, timezone=CONSTANTS.TZ_WORK):
#         self.end_time = CONSTANTS.TH_TIMES['post-market'] if rth_only else CONSTANTS.TH_TIMES['end_of_day']
#         self.timezone = timezone

#     def get_target_time(self, df:pd.DataFrame, trigger_time: pd.Timestamp) -> pd.Timestamp:
#         date = trigger_time.date()
#         return pd.Timestamp.combine(date, self.end_time).tz_localize(self.timezone)

class EODTargetHandler(TargetHandler):
    def __init__(self, rth_only:bool=False, timezone=CONSTANTS.TZ_WORK):
        super().__init__(timezone=timezone)
        self.max_time = 'eod_rth' if rth_only else 'eod'
        self.target_str = 'eodt'

    def get_target_time(self, df:pd.DataFrame, trigger_time:pd.Timestamp) -> pd.Timestamp:
        return resolve_max_time(trigger_time, self.max_time, self.timezone)

    def get_target_event(self, prev_row:pd.Series, curr_row:pd.Series, entry_time:pd.Timestamp=None) -> Optional[str]:
        entry_time = entry_time or self.entry_time
        if entry_time is None:
            return None

        eod_time = resolve_max_time(entry_time, self.max_time, timezone=self.timezone)
        if curr_row['date'] >= eod_time:
            return 'eod_exit'
        return None


class FixedTargetHandler(TargetHandler):
    def __init__(self, direction:str, max_time:timedelta=None, timezone=CONSTANTS.TZ_WORK):
        super().__init__(max_time=max_time, timezone=timezone)
        self.direction = direction
        self.dir = {'bull': 'up', 'bear': 'down'}[self.direction]
        self.dir_int = {'bull': 1, 'bear': -1}[self.direction]
        self.target_price = None
        self.target_str = 'fixed'
    
    def set_target_price(self, row:pd.Series):
        self.target_price = row['close'] # Placeholder for now
    
    def get_target_time(self,  df:pd.DataFrame, trigger_time:pd.Timestamp) -> pd.Timestamp:
        # Ensure trigger_time exists in the dataframe index
        if trigger_time not in df.index:
            raise ValueError(f"trigger_time {trigger_time} not found in dataframe index")

        # Resolve the max target time based on the given max_time and timezone
        max_target_time = resolve_max_time(trigger_time, self.max_time, timezone=self.timezone)

        # Slice the dataframe from the trigger time onward
        post_df = df.loc[trigger_time:]
        if max_target_time:
            post_df = post_df[post_df.index <= max_target_time]

        if post_df.empty:
            return trigger_time  # Return trigger_time if no data after it

        # if not self.target_price:
        self.set_target_price(df.loc[trigger_time])

        # trig_close = df.at[trigger_time, 'close']
        # direction = np.sign(trig_close - self.target_price)
        direction = -self.dir_int
        
        sign_diff = (post_df['close'] - self.target_price).apply(np.sign).diff()
        cross_rows = post_df[sign_diff == -2 * direction]

        if not cross_rows.empty:
            return cross_rows.index[0]
        else:
            return post_df.index[-1]
    
    def get_target_event(self, prev_row:pd.Series, curr_row:pd.Series, entry_time:pd.Timestamp=None) -> Optional[str]:
        entry_time = entry_time or self.entry_time
        if prev_row is None or not self.target_price or entry_time is None:
            return None

        max_target_time = resolve_max_time(entry_time, self.max_time, timezone=self.timezone)

        if not self.target_price:
            self.set_target_price(curr_row)

        prev_diff = prev_row['close'] - self.target_price
        curr_diff = curr_row['close'] - self.target_price

        if curr_row['date'] > max_target_time:
            return 'max_time_overshoot'
        if np.sign(prev_diff) != np.sign(curr_diff):
            return f'{self.target_str}_cross_exit'
        return None

class PercGainTargetHandler(FixedTargetHandler):
    def __init__(self, perc_gain:int, direction:str, max_time:timedelta=None, timezone=CONSTANTS.TZ_WORK):
        super().__init__(direction=direction, max_time=max_time, timezone=timezone)
        self.perc_gain = perc_gain
        self.target_price = None
        self.target_str = f'perc_gain_{perc_gain}%'

    def set_target_price(self, row:pd.Series):
        self.target_price = row['close'] * (1 + (self.perc_gain / 100))


class NextLevelTargetHandler(FixedTargetHandler):
    def __init__(self, level_types:str, timeframe:Timeframe, direction:str, max_time:timedelta=None, timezone=CONSTANTS.TZ_WORK):
        super().__init__(direction=direction, max_time=max_time, timezone=timezone)
        self.level_types = level_types or resolve_level_types(timeframe)
        self.target_str = f'next_level_{self.dir}'
        self.required_columns = self._generate_required_columns()
        self.next_level = None
    
    def _generate_required_columns(self):
        cols = []
        for level in self.level_types:
            cols.append(f"{level}_dist_to_next_{self.dir}")
            # cols.append(f"{level}_dist_to_next_{self.dir}_pct") # Include both raw and pct versions for flexibility
        return cols

    @staticmethod
    def _get_next_round_price_level(current_price:float) -> float:
        """
        Calculate the next psychological level above the current price.
        Automatically adjusts the rounding increment based on the current price range.
        """
        if current_price < 10: increment = 1  # For prices below 10, we round to the nearest 1.
        elif current_price < 100: increment = 5  # For prices between 10 and 100, we round to the nearest 5.
        elif current_price < 1000: increment = 10  # For prices between 100 and 1000, we round to the nearest 10.
        elif current_price < 10000: increment = 50  # For prices between 1000 and 10000, we round to the nearest 50.
        else: increment = 100  # For prices above 10000, we round to the nearest 100 (or use larger increments).

        # Round up to the next psychological level using the determined increment
        psychological_level = (current_price // increment + 1) * increment
        return psychological_level

    def set_target_price(self, row:pd.Series):
        dist_next_levels = [{'value': row[f"{level}_dist_to_next_{self.dir}"], 'level': level} for level in self.level_types]
        dist_next_level = min(dist_next_levels, key=lambda x: x['value'])

        if pd.isna(dist_next_level['value']):
            # If all distance values are NaN, calculate the next psychological level
            self.target_price = self._get_next_round_price_level(row['close'])
        else:
            self.target_price = row['close'] + self.dir_int * dist_next_level['value']
            self.next_level = dist_next_level['level']
    
    # def get_target_time(self, df:pd.DataFrame, trigger_time:pd.Timestamp) -> pd.Timestamp:
    #     # Ensure all required columns are present in the dataframe
    #     if not set(self.required_columns).issubset(df.columns):
    #         raise ValueError(f"Required columns {self.required_columns} not found in dataframe")

    #     # Ensure trigger_time exists in the dataframe index
    #     if trigger_time not in df.index:
    #         raise ValueError(f"trigger_time {trigger_time} not found in dataframe index")

    #     # Resolve the max target time based on the given max_time and timezone
    #     max_target_time = resolve_max_time(trigger_time, self.max_time, timezone=self.timezone)

    #     # Slice the dataframe from the trigger time onward
    #     post_df = df.loc[trigger_time:]
    #     if max_target_time:
    #         post_df = post_df[post_df.index <= max_target_time]

    #     if post_df.empty:
    #         return trigger_time  # Return trigger_time if no data after it

    #     if not self.target_price:
    #         self.set_target_price(df.loc[trigger_time])

    #     # trig_close = df.at[trigger_time, 'close']
    #     # direction = np.sign(trig_close - self.target_price)
    #     direction = -self.dir_int
        
    #     sign_diff = (post_df['close'] - self.target_price).apply(np.sign).diff()
    #     cross_rows = post_df[sign_diff == -2 * direction]

    #     if not cross_rows.empty:
    #         return cross_rows.index[0]
    #     else:
    #         return post_df.index[-1]
        
    # def get_target_event(self, prev_row:pd.Series, curr_row:pd.Series, entry_time:pd.Timestamp=None) -> Optional[str]:
    #     entry_time = entry_time or self.entry_time
    #     if prev_row is None or entry_time is None:
    #         return None

    #     max_target_time = resolve_max_time(entry_time, self.max_time, timezone=self.timezone)

    #     if not self.target_price:
    #         self.set_target_price(curr_row)
        
    #     prev_diff = prev_row['close'] - self.target_price
    #     curr_diff = curr_row['close'] - self.target_price

    #     if max_target_time and curr_row['date'] > max_target_time:
    #         return 'max_time_overshoot'
    #     if np.sign(prev_diff) != np.sign(curr_diff) and curr_diff * self.dir_int > 0:
    #         return f'{self.target_str}_cross_exit'
    #     return None
    

class VWAPCrossTargetHandler(TargetHandler):
    def __init__(self, timeframe:Timeframe, direction:str, max_time:timedelta=None, timezone=CONSTANTS.TZ_WORK):
        super().__init__(max_time=max_time, timezone=timezone)
        self.timeframe = timeframe or Timeframe()
        self.target_str = f'vwap_{self.timeframe}'
        self.required_columns = [self.target_str]
        self.dir_int = {'bull': 1, 'bear': -1}[direction]
        
    def get_target_time(self, df:pd.DataFrame, trigger_time:pd.Timestamp) -> pd.Timestamp:
        # Ensure all required columns are present in the dataframe
        if not set(self.required_columns).issubset(df.columns):
            raise ValueError(f"Required columns {self.required_columns} not found in dataframe")

        if trigger_time not in df.index:
            raise ValueError(f"trigger_time {trigger_time} not found in dataframe index")

        max_target_time = resolve_max_time(trigger_time, self.max_time, timezone=self.timezone)

        post_df = df.loc[trigger_time:]
        if max_target_time:
            post_df = post_df[post_df.index <= max_target_time]

        if post_df.empty:
            return trigger_time

        trig_close = df.at[trigger_time, 'close']
        trig_vwap = df.at[trigger_time, self.target_str]
        direction = np.sign(trig_close - trig_vwap)

        vwap_diff = post_df['close'] - post_df[self.target_str]
        sign_diff = vwap_diff.apply(np.sign).diff()

        cross_rows = post_df[sign_diff == -2 * direction]

        if not cross_rows.empty:
            return cross_rows.index[0]
        else:
            return post_df.index[-1]

    def get_target_event(self, prev_row:pd.Series, curr_row:pd.Series, entry_time:pd.Timestamp=None) -> Optional[str]:
        entry_time = entry_time or self.entry_time
        if self.target_str not in curr_row or prev_row is None or self.target_str not in prev_row or entry_time is None:
            return None

        max_target_time = resolve_max_time(entry_time, self.max_time, timezone=self.timezone)

        prev_diff = prev_row['close'] - prev_row[self.target_str]
        curr_diff = curr_row['close'] - curr_row[self.target_str]

        if curr_row['date'] > max_target_time:
            return 'max_time_overshoot'
        if np.sign(prev_diff) != np.sign(curr_diff) and curr_diff * self.dir_int < 0:
            return f'{self.target_str}_cross_exit'
        return None


class StopLossHandler(ABC):
    """
    Abstract base class for handling different types of stop-losses.
    """
    def __init__(self):
        self.required_columns = []
        self.stop_type = ''

    def check_stop_loss(self, curr_row:pd.Series, stop_price:pd.Series, direction:int) -> Optional[str]:
        close = curr_row['close']
        if (direction == 1 and close <= stop_price) or (direction == -1 and close >= stop_price):
            return f"{self.stop_type.capitalize()}"
        return None


class FixedStopLossHandler(StopLossHandler):
    def __init__(self, sl_pct:float):
        self.sl_pct = sl_pct
        self.required_columns = []
        self.stop_type = 'fixed'

    # def check_stop_loss(self, curr_row, entry_price, direction) -> Optional[str]:
    #     stop_loss = self.resolve_stop_price(entry_price, direction)
    #     if (curr_row['close'] <= stop_loss and direction == 1) or (curr_row['close'] >= stop_loss and direction == -1):
    #         return f'Fixed stop_loss reached {stop_loss}'
    #     return None
    
    def resolve_stop_price(self, entry_row:float, direction:int):
        stop_price = entry_row['close'] * (1 - direction * self.sl_pct)
        return stop_price, self.stop_type

    # def check_stop_loss(self, curr_row, stop_price, direction) -> Optional[str]:
    #     if (curr_row['close'] <= stop_price and direction == 1) or (curr_row['close'] >= stop_price and direction == -1):
    #         return f'Fixed stop_loss reached at {stop_price}'
    #     return None


class NextLevelStopLossHandler(StopLossHandler):
    # def __init__(self, level_types:list[str]=None, offset:float=0.05):
    def __init__(self, timeframe:Timeframe, offset:float=0.05):
        """
        :param level_types: List of level types to use (e.g., 'sr_1D', 'pivots', etc.)
        :param offset: Percentage offset beyond the level for the stop-loss (e.g., 0.05 = 5%)
        """
        self.level_types = resolve_level_types(timeframe)#level_types or ['sr_1h', 'sr_1D', 'sr_1W', 'pivots', 'pivots_D', 'pivots_M', 'levels', 'levels_M']
        self.offset = offset
        self.required_columns = self._generate_required_columns()
        self.stop_type = 'level'

    def _generate_required_columns(self):
        cols = []
        for level in self.level_types:
            # Include both raw and pct versions for flexibility
            cols.append(f"{level}_dist_to_next_up")
            cols.append(f"{level}_dist_to_next_down")
            # cols.append(f"{level}_dist_to_next_up_pct")
            # cols.append(f"{level}_dist_to_next_down_pct")
        return cols

    def resolve_stop_price(self, entry_row:pd.Series, direction:int, nth_closest:int=1) -> Optional[tuple[float, str]]:
        """
        Returns the stop loss price and level type used, based on the N-th closest level to current price.
        
        :param entry_row: Row of data at trade entry
        :param direction: 1 for long, -1 for short
        :param nth_closest: Which closest level to use (1 = closest, 2 = second closest, etc.)
        :return: (stop_price, level_type), or (None, None)
        """
        close = entry_row['close']
        valid_levels = []

        for level_type in self.level_types:
            # Determine which column to use based on direction and availability
            suffix_raw = "down" if direction == 1 else "up"
            col_raw = f"{level_type}_dist_to_next_{suffix_raw}"
            col_pct = f"{level_type}_dist_to_next_{suffix_raw}_pct"

            value = None
            is_pct = False

            # Priority: use raw if available, else pct
            if pd.notna(entry_row.get(col_raw)):
                value = entry_row[col_raw]
                is_pct = False
            elif pd.notna(entry_row.get(col_pct)):
                value = entry_row[col_pct]
                is_pct = True

            # Skip if no usable value
            if value is None or value <= 0:
                continue

            # Calculate actual level
            level = close - direction * (value if not is_pct else value * close)
            # Apply stop offset in direction
            stop_price = level * (1 - direction * self.offset)
            distance_to_price = abs(level - close)

            valid_levels.append((distance_to_price, stop_price, level_type))

        # Sort by closest level
        valid_levels.sort(key=lambda x: x[0])

        if len(valid_levels) >= nth_closest:
            _, stop_price, level_type = valid_levels[nth_closest - 1]
            return stop_price, level_type

        return None, None
    
    # def check_stop_loss(self, curr_row, stop_price, direction) -> Optional[str]:
    #     """
    #     Compares current price to stop loss threshold.
    #     :param curr_row: Current candle data
    #     :param stop_price: Calculated stop level
    #     :param direction: 1 for long, -1 for short
    #     """
    #     close = curr_row['close']
    #     if (direction == 1 and close <= stop_price) or (direction == -1 and close >= stop_price):
    #         return f'Level stop_loss reached at {stop_price}'
    #     return None
    

class HighOfDayStopLossHandler(StopLossHandler):
    def __init__(self, direction:str='bull', offset:float=0.05):
        self.offset = offset
        self.direction = direction
        self.stop_type = 'low_of_day' if self.direction  == 'bull' else 'high_of_day' if self.direction  == 'bear' else None
        self.required_columns = [self.stop_type]

    def resolve_stop_price(self, entry_row, direction:int) -> Optional[tuple[float, str]]:
        stop_price = entry_row[self.stop_type] * (1 - direction * self.offset)
        return stop_price, self.stop_type
        
    # def check_stop_loss(self, curr_row, stop_price, direction) -> Optional[str]:
    #     """
    #     Compares current price to stop loss threshold.
    #     :param curr_row: Current candle data
    #     :param stop_price: Calculated stop level
    #     :param direction: 1 for long, -1 for short
    #     """
    #     close = curr_row['close']
    #     if (direction == 1 and close <= stop_price) or (direction == -1 and close >= stop_price):
    #         return 'stop_loss'
    #     return None


class PredictedDrawdownStopLossHandler(StopLossHandler):
    def __init__(self, dd_col:str='predicted_drawdown'):
        self.dd_col = dd_col
        self.required_columns = [self.dd_col]
        self.stop_type = 'predicted_drawdown'

    def resolve_stop_price(self, entry_row, direction) -> tuple[Optional[float], Optional[str]]:
        """
        Compute dynamic stop-loss price based on predicted drawdown.
        """
        predicted_dd = entry_row.get(self.dd_col)

        if predicted_dd is None or np.isnan(predicted_dd):
            return None, None

        entry_price = entry_row['close']
        # predicted_dd = predicted_dd * entry_price
        stop_price = entry_price - direction * abs(predicted_dd)

        return stop_price, self.stop_type

    # def check_stop_loss(self, curr_row, stop_price, direction) -> Optional[str]:
    #     """
    #     Check if the stop-loss level has been hit based on the current bar.
    #     """
    #     if (direction == 1 and curr_row['low'] <= stop_price) or (direction == -1 and curr_row['high'] >= stop_price):
    #         return f"{self.stop_type.capitalize()} stop loss reached at {stop_price:.2f}"
    #     return None


# class NextLevelStopLossHandler(StopLossHandler):
#     def __init__(self, level_types: list[str], offset: float = 0.05):
#         self.level_types = level_types or ['sr_1h', 'sr_1D', 'sr_1W', 'pivots', 'pivots_D', 'pivots_M', 'levels', 'levels_M']
#         self.offset = offset
#         self.required_columns = self.level_types

#     def resolve_stop_price(self, entry_row, direction) -> Optional[tuple[float, str]]:
#         """
#         Determine the fixed stop-loss level based on entry_row and direction.

#         Returns:
#             Tuple of (stop_price, level_type_used), or (None, None) if none found.
#         """

#         row_df = pd.DataFrame([entry_row])
#         for level_type in self.level_types:
#             level_data = entry_row.get(level_type)
#             if isinstance(level_data, (list, np.ndarray)) and len(level_data) > 0:
#                 enriched_df = indicators.calculate_next_levels(row_df.copy(), levels_col=level_type, show_next_level=True)
#                 enriched_row = enriched_df.iloc[0]

#                 if direction == 1 and pd.notna(enriched_row.get(f"next_{level_type}_down")):
#                     level = enriched_row[f"next_{level_type}_down"]
#                     stop_price = level * (1 - self.offset)
#                     return stop_price, level_type

#                 elif direction == -1 and pd.notna(enriched_row.get(f"next_{level_type}_up")):
#                     level = enriched_row[f"next_{level_type}_up"]
#                     stop_price = level * (1 + self.offset)
#                     return stop_price, level_type
#         return None, None

#     def check_stop_loss(self, curr_row, stop_price, direction) -> Optional[str]:
#         """
#         Compares current close to a fixed stop_price.
#         """
#         close = curr_row['close']
#         if direction == 1 and close <= stop_price:
#             return 'stop_loss'
#         elif direction == -1 and close >= stop_price:
#             return 'stop_loss'
#         return None


# class NextLevelStopLossHandler(StopLossHandler):
#     def __init__(self, level_types: list[str], offset: float = 0.01):
#         """
#         Parameters:
#             level_types: list[str]
#                 Priority-ordered list of level columns (e.g., ['sr_1h', 'cam']).
#             offset: float
#                 Buffer percentage added/subtracted from level to trigger the stop-loss.
#         """
#         self.level_types = level_types
#         self.offset = offset

#     def check_stop_loss(self, curr_row, entry_price, direction) -> Optional[str]:
#         """
#         Dynamically checks stop-loss condition using next-level resistance/support logic.

#         Parameters:
#             curr_row (pd.Series): The current row with price + level data.
#             entry_price (float): Entry price of the trade.
#             direction (int): 1 for long, -1 for short.

#         Returns:
#             Optional[str]: A string like 'stop_loss_sr_1h' if stop triggered, else None.
#         """

#         # Convert the single row to a DataFrame for processing
#         row_df = pd.DataFrame([curr_row])
#         stop_price = None

#         for level_type in self.level_types:
#             level_data = curr_row.get(level_type)

#             if isinstance(level_data, (list, np.ndarray)) and len(level_data) > 0:
#                 # try:
#                 # Compute next level distances for this level_type
#                 enriched_df = indicators.calculate_next_levels(row_df.copy(), levels_col=level_type, show_next_level=True)
#                 enriched_row = enriched_df.iloc[0]

#                 next_up = enriched_row.get(f"next_{level_type}_up")
#                 next_down = enriched_row.get(f"next_{level_type}_down")
#                 close = curr_row['close']

#                 if direction == 1 and pd.notna(next_down):
#                     stop_price = next_down * (1 - self.offset)
#                     if close <= stop_price:
#                         return f"stop_loss_{level_type}", stop_price

#                 elif direction == -1 and pd.notna(next_up):
#                     stop_price = next_up * (1 + self.offset)
#                     if close >= stop_price:
#                         return f"stop_loss_{level_type}", stop_price
#                 # except Exception as e:
#                 #     print(f"[Warning] Error processing levels for {level_type}: {e}")
#                 #     continue  # If any level type fails, try the next one

#         return None, stop_price  # No stop triggered
