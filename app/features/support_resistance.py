import os, sys, datetime, pandas as pd, numpy as np, scipy, matplotlib.pyplot as plt, sklearn, plotly
from ib_insync import *

current_folder = os.path.dirname(os.path.realpath(__file__))
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(current_folder)
sys.path.append(parent_folder)
from utils import helpers
from utils.constants import CONSTANTS, FORMATS, PATHS
from utils.timeframe import Timeframe
from features import indicators
from data import hist_market_data_handler


def get_SR_levels(df:pd.DataFrame, levels_timeframe:Timeframe, use_rth:bool=False, level_granularity:int=5, count_threshold:int=2, 
                  look_forward_reaction:int=10, display:bool=False):

    df = helpers.format_df_date(df) # Make sure date column is in datetime format

    if levels_timeframe.to_timedelta == datetime.timedelta(days=1) and not use_rth:
        df_tf = helpers.get_daily_df(df, rename_volume=False)
    elif levels_timeframe.to_timedelta == datetime.timedelta(days=1) and use_rth:
        df_tf = helpers.get_daily_df(df, th='rth', rename_volume=False)
    else:
        df_tf = df.resample(levels_timeframe.pandas, on='date').agg({'open':'first', 'close':'last', 'high':'max', 'low':'min', 'volume':'sum'}).dropna(subset=['close']).reset_index()

    # Detect local peaks and troughs (resistance and support)
    peaks, _ = scipy.signal.find_peaks(df_tf['high'], distance=10, prominence=1)
    troughs, _ = scipy.signal.find_peaks(-df_tf['low'], distance=10, prominence=1)

    df_tf['is_peak'] = False
    # df_tf['is_peak'].iloc[peaks] = True
    df_tf.loc[peaks, 'is_peak'] = True

    df_tf['is_trough'] = False
    # df_tf['is_trough'].iloc[troughs] = True
    df_tf.loc[troughs, 'is_trough'] = True

    df_tf['swing_price'] = np.nan
    df_tf.loc[df_tf['is_peak'] == True, 'swing_price'] = df_tf['high']
    df_tf.loc[df_tf['is_trough'] == True, 'swing_price'] = df_tf['low']

    # df_tf['rounded_level'] = df_tf['swing_price'].round(-int(np.log10(level_granularity)))
    df_tf['rounded_level'] = np.round(df_tf['swing_price'] / level_granularity) * level_granularity


    # Only use peaks and troughs
    df_pivots = df_tf[df_tf['is_peak'] | df_tf['is_trough']]
    level_counts = df_pivots['rounded_level'].value_counts().sort_index()

    # Basic scoring based on reaction count
    significant_levels = level_counts[level_counts > count_threshold * level_counts.std()] # Using std
    # significant_levels = level_counts[level_counts >= count_threshold]

    # Only use valid swing points
    swing_points = df_pivots.copy()
    swing_idx = swing_points.index.to_numpy()
    num_points = len(swing_idx)

    # Build future windows of close prices using as_strided
    close_array = df_tf['close'].values
    volume_array = df_tf['volume'].values
    window_size = look_forward_reaction + 1

    # Ensure bounds
    max_index = len(close_array) - window_size
    valid_idx = swing_idx[swing_idx <= max_index]

    # Initialize arrays
    pct_changes = []
    cum_volumes = []
    dates = []
    level_vals = []

    for idx in valid_idx:
        future_prices = close_array[idx:idx + window_size]
        future_volumes = volume_array[idx:idx + window_size]
        base_price = close_array[idx]

        if df_tf['is_trough'].iloc[idx]:
            target_price = np.max(future_prices)
        elif df_tf['is_peak'].iloc[idx]:
            target_price = np.min(future_prices)
        else:
            continue  # skip non-pivots (shouldn’t occur)

        pct_change = 100 * (target_price - base_price) / base_price
        cum_volume = np.sum(future_volumes)

        pct_changes.append(pct_change)
        cum_volumes.append(cum_volume)
        dates.append(df_tf['date'].iloc[idx])#.date())
        level_vals.append(df_tf['rounded_level'].iloc[idx])

    df_tf_reaction = pd.DataFrame({'date': dates, 'level': level_vals, 'cum_volume': cum_volumes, 'pct_move': pct_changes})

    # Aggregate strength
    # df_tf_reaction = pd.DataFrame(reaction_strengths, columns=['date', 'level', 'cum_volume', 'pct_move']).sort_values(by='date', ascending=True)
    df_tf_ranking = df_tf_reaction.groupby('level').agg(level_count=('level', 'count'), last_date=('date', 'last'), mean_cum_volume=('cum_volume', 'mean'), mean_pct_move=('pct_move', 'mean')).sort_values(by='mean_pct_move', key=abs, ascending=False).reset_index()
    df_tf_ranking['date_score'] = (pd.to_datetime(df_tf_ranking['last_date']) - pd.to_datetime(df_tf_reaction['date'].max())).dt.days
    df_tf_ranking['mean_pct_move_abs'] = df_tf_ranking['mean_pct_move'].abs()

    # Normalize
    ranking_columns = ['level_count', 'date_score', 'mean_cum_volume', 'mean_pct_move_abs']

    if not df_tf_ranking.empty:
        normalized = sklearn.preprocessing.MinMaxScaler().fit_transform(df_tf_ranking[ranking_columns])
        df_normalized = pd.DataFrame(normalized, columns=[col + '_norm' for col in ranking_columns])
        df_tf_ranking = pd.concat([df_tf_ranking, df_normalized], axis=1)

        # df_tf_ranking['score'] = df_normalized.mean(axis=1) # Case equal weights
        weights = {'level_count': 0.25, 'date_score': 0.25, 'mean_cum_volume': 0.25, 'mean_pct_move_abs': 0.25}
        df_tf_ranking['score'] = sum(df_normalized[col+'_norm'] * weight for col, weight in weights.items())

        df_tf_ranking = df_tf_ranking.sort_values(by='score', ascending=False)#.reset_index(drop=True)

        # Display levels
        if display:
            # print(df_tf_reaction)
            print("\n", helpers.df_to_table(df_tf_ranking[['level', 'level_count', 'last_date', 'mean_cum_volume', 'mean_pct_move', 'score']].round(2)), "\n")

            plt.figure(figsize=(14,6))
            plt.plot(df_tf['close'], label='Price')
            for level in df_tf_ranking['level'].head(10):
                plt.axhline(y=level, color='orange', linestyle='--', alpha=0.5)
            plt.legend()
            plt.title("Strong Support & Resistance Levels")
            plt.show()

    # return df_tf_ranking['level'].values.tolist()
    return df_tf_ranking['level'].astype('float32').values.tolist()



class RecursiveSRBuilder:

    def __init__(self, df:pd.DataFrame, ib:IB, symbol:str=None, rounding_precision:int=None, 
                 file_format:str=None, hist_folder:str=None, sr_types:list=['all'], drop_levels:bool=True, save_levels:bool=False):#, use_atr=False):
        """
        Parameters:
        - df: DataFrame with OHLC data (index will be forced to datetime)
        - timeframes: dict like {'weekly': 'W', 'daily': 'D', '4h': '4H'}
        - proximity_thresholds: fixed % thresholds per timeframe
        - use_atr: use ATR-based proximity if True
        - atr_period: period for ATR calculation
        """

        self.ib = ib
        self.symbol = symbol or (df.attrs['symbol'] if 'symbol' in df.attrs else None)
        self.df = helpers.format_df_date(df, set_index=True)
        self.sr_types = self._filter_sr_types_by_df_resolution(self.df, sr_types)
        self.file_format = file_format or FORMATS.DEFAULT_FILE_FORMAT
        self.hist_folder = hist_folder if hist_folder else PATHS.folders_path['hist_market_data']
        self.rounding_precision = rounding_precision or CONSTANTS.LEVELS_ROUNDING_PRECISION
        self.drop_levels = drop_levels
        self.save_levels = save_levels

        # self.proximity_thresholds = proximity_thresholds or {'weekly': 0.01, 'daily': 0.005, '1h': 0.001}

    # @staticmethod
    # def _format_df(df):
    #     # Make sure date column is in datetime format and is set as index
    #     df = helpers.format_df_date(df)
    #     df.index = df['date']
    #     return df

    def _filter_sr_types_by_df_resolution(self, df, sr_types):
        """Filter out timeframes that are of finer resolution (lower duration in seconds) than the DataFrame itself."""
        timeframe_df_seconds = helpers.get_df_timeframe(df).to_seconds
        if 'all' in sr_types: sr_types = [sr_type['timeframe'] for sr_type in CONSTANTS.SR_SETTINGS]

        filtered = []
        for tf in sr_types:
            try:
                # timeframe_levels_seconds = Timeframe(tf['timeframe']).to_seconds
                timeframe_levels_seconds = Timeframe(tf).to_seconds
            except Exception as e:
                # print(f"Skipping timeframe '{tf['timeframe']}' due to parsing error: {e}")
                print(f"Skipping timeframe '{tf}' due to parsing error: {e}")
                continue

            if timeframe_levels_seconds > timeframe_df_seconds:
                filtered.append(tf)

        return filtered

    # def _get_df_levels(self, df, df_levels, tf):
    #     """
    #     Returns a DataFrame to be used for SR level calculation for the given timeframe.
    #     Will use df_levels if it's sufficiently long; otherwise will fetch extra data.
    #     """

    #     lookback_td = helpers.parse_timedelta(tf['lookback'])
    #     lookback_start = df.index.min() - lookback_td

    #     if not df_levels.empty:
    #         if df_levels.index.min() <= lookback_start:
    #             return df_levels
    #         else:
    #             print(f"Warning: Provided df_levels does not cover required lookback ({tf['timeframe']}). Falling back to fetch.")

    #     # Fetch new data
    #     to_time = df.index.max()
    #     from_time = lookback_start

    #     print(f"Fetching historical data for timeframe {tf['timeframe']} from {to_time} to {from_time}...")

    #     fetcher = hist_market_data_handler.HistMarketDataFetcher(self.ib, helpers.pandas_to_ibkr_timeframe(tf['timeframe']), save_to_file=True, delete_existing=True)

    #     params = {'symbol': self.symbol, 'to_time': to_time, 'from_time': from_time, 'step_duration': helpers.pandas_duration_to_ibkr_duration_str(tf['lookback'])}
    #     df_fetched = helpers.load_df_from_file(fetcher.run(params))

    #     if df_fetched is not None and not df_fetched.empty:
    #         df_fetched = self._format_df(df_fetched)
    #         # df_levels_combined = pd.concat([df_fetched, df])
    #         # df_levels_combined = df_levels_combined[~df_levels_combined.index.duplicated(keep='last')].sort_index()
    #         return df_fetched

    #     print(f"Warning: Could not fetch data for timeframe {tf['timeframe']}. Using df only.")
    #     return df

    def _calculate_recursive_levels(self, df):

        for sr_type in self.sr_types:
            
            tf = next((item for item in CONSTANTS.SR_SETTINGS if item['timeframe'] == sr_type), None)
            timeframe = Timeframe(tf['timeframe'])
            print("Calculating recursive levels ", timeframe, "...")
            df_tf = indicators.IndicatorsUtils.get_level_compatible_df(self.ib, self.symbol, df, timeframe, tf['lookback'], self.file_format, self.hist_folder)
            # df_res = self._get_df_levels(df, df_levels, tf)
            # timeframe_df_levels_seconds = helpers.timeframe_to_seconds(helpers.get_df_timeframe(df_levels)) if not df_levels.empty else None
            # df_res = df_levels if (not df_levels.empty and timeframe_df_levels_seconds <= helpers.timeframe_to_seconds(tf['timeframe'])) else df

            # df_tf = df_res.resample(tf['timeframe']).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            refresh_points = df_tf.resample(tf['refresh_rate']).first().index if not df_tf.empty else []

            levels_dict = {}
            # due_flags_dict = {}
            for idx in refresh_points:
                df_lookback = df_tf.loc[(df_tf.index <= idx) & (df_tf.index >= idx - helpers.parse_timedelta(tf['lookback']))]
                use_rth = 'rth' in tf['timeframe']
                levels = get_SR_levels(df_lookback, timeframe, use_rth, tf['granularity'], tf['count_threshold']) if not df_lookback.empty else []
                # levels = get_SR_levels(df_lookback, timeframe, level_granularity, count_threshold) if not df_lookback.empty else []

                levels_dict[idx] = levels

                # Add due flag logic: Flag will be True for the first row of each new S/R level period. Mark as due at the first occurrence
                # if levels:
                #     due_flags_dict[idx] = True
                # else:
                #     due_flags_dict[idx] = False

            # Create Serie from dict, reindex it to match df's index and forward-fill the values. Also format to float32 for memory efficiency
            # levels_series = pd.Series(levels_dict, name=f'sr_{timeframe}_list').reindex(df.index, method='ffill').apply(lambda l: [] if l is np.nan else l) if levels_dict else pd.Series()
            levels_series = pd.Series(levels_dict, name=f'sr_{timeframe}_list').reindex(df.index, method='ffill')\
                .apply(lambda l: [] if l is np.nan else [np.float32(round(x, self.rounding_precision)) for x in l]) if levels_dict else pd.Series()
            
            df[f'sr_{timeframe}_list'] = levels_series

            # Create due flags
            df[f'~sr_{sr_type}_due'] = False
            for refresh_point in refresh_points:
                if refresh_point >= df['date'].min() and refresh_point <= df['date'].max():
                    # For each refresh point, find the closest row in df
                    time_differences = (df['date'] - refresh_point).abs()
                    closest_index = time_differences.idxmin()
                    df.loc[closest_index, f'~sr_{sr_type}_due'] = True

            # df[f'~sr_{sr_type}_due'] = False
            # for refresh_point in refresh_points:
            #     # Find the first row in df where the date matches the refresh_point
            #     first_matching_index = df[df['date'].dt.date == refresh_point.date()].index.min()
            #     if pd.notna(first_matching_index):  # Ensure there's at least one match
            #         df.loc[first_matching_index, f'~sr_{sr_type}_due'] = True
            # due_flags_series = pd.Series(due_flags_dict, name=f'~sr_{timeframe}_due').reindex(df.index).fillna(False)  # Fill missing due flags with False (no new level yet)
            # df[f'~sr_{timeframe}_due'] = due_flags_series

        return df

    def apply_sr(self):
        df = self._calculate_recursive_levels(self.df)

        for sr_type in self.sr_types:
            col = f"sr_{sr_type}_list"
            df = indicators.IndicatorsUtils.calculate_next_levels(df, col)

            # Save levels list in separate Dataframe
            if self.save_levels: df = indicators.IndicatorsUtils.save_levels(df=df, timeframe=Timeframe(sr_type), symbol=self.symbol, col=col, 
                                                                             hist_folder=self.hist_folder, data_type=f'sr_{sr_type}', file_format='csv')
            
            # Drop SR list column
            if self.drop_levels: df.drop(columns=col, inplace=True)

        # for tf in self.timeframes:
        #     col = f"sr_{tf['timeframe']}_list"
        #     df = indicators.IndicatorsUtils.calculate_next_levels(df, col)
        


        # df = self._calculate_next_sr_levels(df)
        # df = self._check_proximity(df)

        return df

    def _plot_levels(self, df, price_col='close', levels_from='weekly', start=None, end=None, show=True):
        """
        Plot price data with horizontal lines for closest levels.
        """
        plot_df = df.copy()
        if start or end:
            plot_df = plot_df.loc[start:end]

        fig = plotly.graph_objects.Figure()

        # Plot price
        fig.add_trace(plotly.graph_objects.Scatter(
            x=plot_df.index, y=plot_df[price_col],
            mode='lines', name='Price', line=dict(color='blue')
        ))

        # Plot levels (horizontal lines)
        last_levels = None
        for i, row in plot_df.iterrows():
            levels = row.get(f'{levels_from}_levels', [])
            if levels != last_levels and levels:
                for level in levels:
                    fig.add_hline(y=level, line=dict(dash='dot', color='gray', width=1), opacity=0.3)
                last_levels = levels

        fig.update_layout(
            title=f"{levels_from.title()} Support/Resistance Levels",
            xaxis_title='Time',
            yaxis_title='Price',
            height=600
        )

        if show: fig.show()
        else: return fig


if __name__ == "__main__":

    args = sys.argv

    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    symbol = 'TSLA'
    to_time = pd.to_datetime('2025-05-06T19:59:59').tz_localize(CONSTANTS.TZ_WORK)
    from_time = pd.to_datetime('2023-01-01T04:00:00').tz_localize(CONSTANTS.TZ_WORK)
    paperTrading = True
    display_res = False
    strategy = ''
    if len(args) > 1:
        if 'live' in args: paperTrading = False
        if 'display' in args: display_res = True
        for arg in args:
            if 'sym' in arg: symbol = arg[3:]

    # TWS Connection
    ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)
    if not ibConnection:
        paperTrading = not paperTrading
        ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)


    timeframe_1m = Timeframe('1min')
    df_1m, _ = hist_market_data_handler.HistMarketDataLoader(ib, symbol, timeframe_1m, to_time=to_time, from_time=from_time).load_and_trim()

    levels = get_SR_levels(df_1m, Timeframe('1D'), display=True)

    # sr = RecursiveSRBuilder(df_1m)
    # df = sr.get_SR()
    # # sr._plot_levels(df_1m, price_col='close', levels_from='weekly', start=None, end=None, show=True)


    input("\nEnter anything to exit")


    # def _map_levels(self, df, levels_dict, timeframe):
    #     level_series = []
    #     for date in df.index:
    #         keys = [k for k in levels_dict.keys() if k <= date]
    #         key = max(keys) if keys else None
    #         level_series.append(levels_dict.get(key, []))
    #     df[f'{timeframe}_levels'] = level_series
    #     return df


    # def _check_proximity(self, close, levels, threshold, atr=None):
    #     for level in levels:
    #         if self.use_atr and atr is not None and not np.isnan(atr):
    #             dynamic_threshold = atr / close
    #         else:
    #             dynamic_threshold = threshold
    #         diff = (close - level) / level
    #         if abs(diff) <= dynamic_threshold:
    #             return {
    #                 'near_level': level,
    #                 'direction': 'above' if close >= level else 'below',
    #                 'distance': diff,
    #                 'threshold_used': dynamic_threshold
    #             }
    #     return None


    # def annotate(self):
    #     df = self.df.copy()

    #     for timeframe, freq in self.timeframes.items():
    #         print("Calculating...")
    #         levels = self._calculate_recursive_levels(df, timeframe, freq)
    #         print("Mapping...")
    #         df = self._map_levels(df, levels, timeframe)
    #         print("Checking Proximity...")
    #         prox_col = f'{timeframe}_proximity'
    #         df[prox_col] = df.apply(
    #             lambda row: self._check_proximity(
    #                 row['close'],
    #                 row.get(f'{timeframe}_levels', []),
    #                 self.proximity_thresholds.get(timeframe, 0.01),
    #                 row['atr'] if self.use_atr else None
    #             ),
    #             axis=1
    #         )

    #     return df









# def _get_near_col_names(self, tf):
    #     return ["near_sr_" + tf['timeframe'] + '_up', "near_sr_" + tf['timeframe'] + '_down']

    # def _check_proximity(self, df):#, timeframe, threshold, col_up, col_down):

    #     for tf in self.timeframes:

    #         print("Calculating proximity to levels ", tf['timeframe'], "...")

    #         level_col = f'sr_{tf['timeframe']}'
    #         flat_levels = np.concatenate(df[level_col].values)
    #         row_indices = np.concatenate([[i]*len(lvls) for i, lvls in enumerate(df[level_col])]).astype(int)
    #         flat_close = df['close'].values[row_indices.astype(int)]

    #         is_near_up = (np.abs(flat_levels - flat_close) < tf['proximity_threshold']) & (flat_levels < flat_close)
    #         is_near_down = (np.abs(flat_levels - flat_close) < tf['proximity_threshold']) & (flat_levels > flat_close)
    #         result_up = pd.Series(is_near_up).groupby(row_indices).any()
    #         result_down = pd.Series(is_near_down).groupby(row_indices).any()

    #         [col_near_up, col_near_down] = self._get_near_col_names(tf)
    #         df[col_near_up] = False
    #         mask_up = np.zeros(len(df), dtype=bool) # Avoid using the expression "df[col_near_up].iloc[result_up.index] = result_up.values", which works but throws warning
    #         mask_up[result_up.index.astype(int)] = result_up.values
    #         df[col_near_up] = mask_up

    #         df[col_near_down] = False
    #         mask_down = np.zeros(len(df), dtype=bool) # Avoid using the expression "df[col_near_down].iloc[result_down.index] = result_down.values", which works but throws warning
    #         mask_down[result_down.index.astype(int)] = result_down.values
    #         df[col_near_down] = mask_down

    #     return df






# class SupportResistance:


#     def __init__(self, df, timeframes=None, proximity_thresholds=None, use_atr=False):
#         """
#         Parameters:
#         - df: DataFrame with OHLC data (index will be forced to datetime)
#         - timeframes: dict like {'weekly': 'W', 'daily': 'D', '4h': '4H'}
#         - proximity_thresholds: fixed % thresholds per timeframe
#         - use_atr: use ATR-based proximity if True
#         - atr_period: period for ATR calculation
#         """
#         self.df = df.copy()

#         # Make sure date column is in datetime format and is set as index
#         self.df['date'] = pd.to_datetime(df['date'], utc=True)
#         self.df['date'] = df['date'].dt.tz_convert(CONSTANTS.TZ_WORK)
#         # self.df.set_index('date', inplace=False)
#         self.df.index = self.df['date']

#         self.timeframes = timeframes or {'weekly': 'W', 'daily': 'D', '4h': '240min'}
#         self.proximity_thresholds = proximity_thresholds or {'weekly': 0.01, 'daily': 0.005, '4h': 0.003}

#         self.use_atr = use_atr
#         self.atr_period = 14
#         self.cache = collections.defaultdict(dict)

#         if self.use_atr:
#             self.df['atr'] = self._calculate_atr(self.df, period=self.atr_period)


#     def _calculate_atr(self, df, period=14):
#         high_low = df['high'] - df['low']
#         high_close = np.abs(df['high'] - df['close'].shift(1))
#         low_close = np.abs(df['low'] - df['close'].shift(1))
#         tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
#         atr = tr.rolling(window=period).mean()
#         return atr


#     def _calculate_recursive_levels(self, df, timeframe, freq):
#         df_tf = df.resample(freq).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})

#         # print(df_tf)
#         # input()

#         levels_dict = {}
#         for idx in df_tf.index:
#             # print("idx = ", idx)
#             if idx in self.cache[timeframe]:
#                 levels = self.cache[timeframe][idx]
#             else:
#                 freq_str = '1'+freq
#                 # print("timeframe = ", timeframe)
#                 # print("freq = ", freq)
#                 # print("pd.Timedelta(freq) = ", pd.Timedelta(freq_str))
#                 # input()
#                 period_df = df.loc[(df.index <= idx) & (df.index >= idx - pd.Timedelta(freq_str))]
#                 # print(period_df)
#                 if not period_df.empty:
#                     levels = get_SR_levels(period_df, freq)
#                     self.cache[timeframe][idx] = levels
#                 else:
#                     levels = []
#             levels_dict[idx] = levels
#         return levels_dict


#     def _map_levels(self, df, levels_dict, timeframe):
#         level_series = []
#         for date in df.index:
#             keys = [k for k in levels_dict.keys() if k <= date]
#             key = max(keys) if keys else None
#             level_series.append(levels_dict.get(key, []))
#         df[f'{timeframe}_levels'] = level_series
#         return df


#     def _check_proximity(self, close, levels, threshold, atr=None):
#         for level in levels:
#             if self.use_atr and atr is not None and not np.isnan(atr):
#                 dynamic_threshold = atr / close
#             else:
#                 dynamic_threshold = threshold
#             diff = (close - level) / level
#             if abs(diff) <= dynamic_threshold:
#                 return {
#                     'near_level': level,
#                     'direction': 'above' if close > level else 'below',
#                     'distance': diff,
#                     'threshold_used': dynamic_threshold
#                 }
#         return None

    # def _check_proximity2(self, row):#, timeframe, threshold, col_up, col_down):

    #     # print(row['date'])

    #     col_names = []
    #     near_list = []
    #     for tf in self.timeframes:
    #         timeframe = tf['timeframe']
    #         threshold = tf['proximity_threshold']

    #         levels = np.array(row[f'sr_{timeframe}'])
    #         close = row['close']
    #         diff = np.abs(close - levels)

    #         [col_near_up, col_near_down] = self._get_near_col_names(tf)
    #         col_names.append(col_near_up)
    #         col_names.append(col_near_down)

    #         near_up = np.any((diff < threshold) & (close > levels))
    #         near_down = np.any((diff < threshold) & (close < levels))
    #         near_list.append(near_up)
    #         near_list.append(near_down)

    #     return pd.Series(near_list, index=col_names)


    # def get_SR(self):
    #     df = self.df.copy()

    #     # for tf in self.timeframes:
    #     print("Calculating...")
    #     df = self._calculate_recursive_levels(df)#, tf['timeframe'], tf['lookback'], tf['refresh_rate'], tf['granularity'], tf['count_threshold'])

    #     # print("Mapping...")
    #     # df = self._map_levels(df, levels, timeframe)
    #     print("Checking Proximity...")
    #     # col_names = []
    #     # near_list = []
    #     # for tf in self.timeframes:
    #     #     col_near_up, col_near_down = self._get_near_col_names(tf)
    #     #     col_names.append(col_near_up)
    #     #     col_names.append(col_near_down)
        # col_names = [col for tf in self.timeframes for col in self._get_near_col_names(tf)]


    #     time_now_start = datetime.datetime.now()

    #     col_names_2 = []
    #     for c in col_names: col_names_2.append(c + '_2')
    #     df[col_names_2] = df.apply(lambda row: self._check_proximity2(row), axis=1)
    #     # # df[col_names] = list(map(self._check_proximity, row), axis=1)

    #     time_now_end = datetime.datetime.now()
    #     print("Time elapsed: ", time_now_end - time_now_start)

    #     # input()

    #     time_now_start = datetime.datetime.now()

    #     df = self._check_proximity(df)

    #     time_now_end = datetime.datetime.now()
    #     print("Time elapsed: ", time_now_end - time_now_start)

    #     df_1 = df[col_names]
    #     df_2 = df[col_names_2].copy()
    #     df_2.columns = col_names  # rename to match for comparison

    #     comparison = df_1 == df_2
    #     print("comparison = ", comparison)
    #     all_match = comparison.all().all()
    #     print("all macth = ", all_match)

    #     # input()

    #     for tf in self.timeframes:
    #         print()
    #         print(tf['timeframe'], '  ', tf['lookback'], '   ', tf['refresh_rate'], '   ', tf['granularity'])
    #         # input()

    #     # df[prox_col] = df.apply(
    #     #     lambda row: self._check_proximity(
    #     #         row['close'],
    #     #         row.get(f'{timeframe}_levels', []),
    #     #         self.proximity_thresholds.get(timeframe, 0.01),
    #     #         row['atr'] if self.use_atr else None
    #     #     ),
    #     #     axis=1
    #     # )

    #     return df


#     def annotate(self):
#         df = self.df.copy()

#         for timeframe, freq in self.timeframes.items():
#             print("Calculating...")
#             levels = self._calculate_recursive_levels(df, timeframe, freq)
#             print("Mapping...")
#             df = self._map_levels(df, levels, timeframe)
#             print("Checking Proximity...")
#             prox_col = f'{timeframe}_proximity'
#             df[prox_col] = df.apply(
#                 lambda row: self._check_proximity(
#                     row['close'],
#                     row.get(f'{timeframe}_levels', []),
#                     self.proximity_thresholds.get(timeframe, 0.01),
#                     row['atr'] if self.use_atr else None
#                 ),
#                 axis=1
#             )

#         return df


#     def _plot_levels(self, df, price_col='close', levels_from='weekly', start=None, end=None, show=True):
#         """
#         Plot price data with horizontal lines for closest levels.
#         """
#         plot_df = df.copy()
#         if start or end:
#             plot_df = plot_df.loc[start:end]

#         fig = plotly.graph_objects.Figure()

#         # Plot price
#         fig.add_trace(plotly.graph_objects.Scatter(
#             x=plot_df.index, y=plot_df[price_col],
#             mode='lines', name='Price', line=dict(color='blue')
#         ))

#         # Plot levels (horizontal lines)
#         last_levels = None
#         for i, row in plot_df.iterrows():
#             levels = row.get(f'{levels_from}_levels', [])
#             if levels != last_levels and levels:
#                 for level in levels:
#                     fig.add_hline(y=level, line=dict(dash='dot', color='gray', width=1), opacity=0.3)
#                 last_levels = levels

#         fig.update_layout(
#             title=f"{levels_from.title()} Support/Resistance Levels",
#             xaxis_title='Time',
#             yaxis_title='Price',
#             height=600
#         )

#         if show: fig.show()
#         else: return fig












# def get_SR_levels2(df, levels_timeframe, level_granularity=5, count_threshold=2, look_forward_reaction=10, display=False):

#     # Make sure date column is in datetime format
#     df['date'] = pd.to_datetime(df['date'], utc=True)
#     df['date'] = df['date'].dt.tz_convert(CONSTANTS.TZ_WORK)

#     if levels_timeframe == 'D':
#         df_tf = helpers.get_daily_df(df, rename_volume=False)
#     elif levels_timeframe == 'D_rth':
#         df_tf = helpers.get_daily_df(df, th='rth', rename_volume=False)
#     else:
#         df_tf = df.resample(levels_timeframe, on='date').agg({'open':'first', 'close':'last', 'high':'max', 'low':'min', 'volume':'sum'}).dropna(subset=['close']).reset_index()

#     # Detect local peaks and troughs (resistance and support)
#     peaks, _ = scipy.signal.find_peaks(df_tf['high'], distance=10, prominence=1)
#     troughs, _ = scipy.signal.find_peaks(-df_tf['low'], distance=10, prominence=1)

#     df_tf['is_peak'] = False
#     # df_tf['is_peak'].iloc[peaks] = True
#     df_tf.loc[peaks, 'is_peak'] = True

#     df_tf['is_trough'] = False
#     # df_tf['is_trough'].iloc[troughs] = True
#     df_tf.loc[troughs, 'is_trough'] = True

#     df_tf['swing_price'] = np.nan
#     df_tf.loc[df_tf['is_peak'] == True, 'swing_price'] = df_tf['high']
#     df_tf.loc[df_tf['is_trough'] == True, 'swing_price'] = df_tf['low']

#     # df_tf['rounded_level'] = df_tf['swing_price'].round(-int(np.log10(level_granularity)))
#     df_tf['rounded_level'] = np.round(df_tf['swing_price'] / level_granularity) * level_granularity


#     # Only use peaks and troughs
#     df_pivots = df_tf[df_tf['is_peak'] | df_tf['is_trough']]
#     level_counts = df_pivots['rounded_level'].value_counts().sort_index()

#     # Basic scoring based on reaction count
#     significant_levels = level_counts[level_counts > count_threshold * level_counts.std()] # Using std
#     # significant_levels = level_counts[level_counts >= count_threshold]

#     # Calculate reversal strength
#     reaction_strengths = []
#     for level in significant_levels.index:
#         swing_points = df_pivots[df_pivots['rounded_level'] == level]
#         for idx in swing_points.index:
#             # look forward or backward N bars and measure max % move
#             future_df_subset = df_tf.iloc[idx:idx+look_forward_reaction+1]
#             future_price_max = future_df_subset['close'].max() if future_df_subset['is_trough'].iloc[0] else future_df_subset['close'].min() if future_df_subset['is_peak'].iloc[0] else None
#             future_price_max_index = future_df_subset['close'].idxmax()
#             future_volume_cum = df_tf['volume'].iloc[idx:future_price_max_index+1].sum()
#             current_price = df_tf['close'].iloc[idx]
#             pct_change = 100 * (future_price_max - current_price) / current_price
#             date_change = df_tf['date'].iloc[idx].date()
#             reaction_strengths.append((date_change, level, future_volume_cum, pct_change))


#     # Aggregate strength
#     df_tf_reaction = pd.DataFrame(reaction_strengths, columns=['date', 'level', 'cum_volume', 'pct_move']).sort_values(by='date', ascending=True)
#     df_tf_ranking = df_tf_reaction.groupby('level').agg(level_count=('level', 'count'), last_date=('date', 'last'), mean_cum_volume=('cum_volume', 'mean'), mean_pct_move=('pct_move', 'mean')).sort_values(by='mean_pct_move', key=abs, ascending=False).reset_index()
#     df_tf_ranking['date_score'] = (pd.to_datetime(df_tf_ranking['last_date']) - pd.to_datetime(df_tf_reaction['date'].max())).dt.days
#     df_tf_ranking['mean_pct_move_abs'] = df_tf_ranking['mean_pct_move'].abs()

#     # Normalize
#     ranking_columns = ['level_count', 'date_score', 'mean_cum_volume', 'mean_pct_move_abs']

#     if not df_tf_ranking.empty:
#         normalized = sklearn.preprocessing.MinMaxScaler().fit_transform(df_tf_ranking[ranking_columns])
#         df_normalized = pd.DataFrame(normalized, columns=[col + '_norm' for col in ranking_columns])
#         df_tf_ranking = pd.concat([df_tf_ranking, df_normalized], axis=1)

#         # df_tf_ranking['score'] = df_normalized.mean(axis=1) # Case equal weights
#         weights = {'level_count': 0.25, 'date_score': 0.25, 'mean_cum_volume': 0.25, 'mean_pct_move_abs': 0.25}
#         df_tf_ranking['score'] = sum(df_normalized[col+'_norm'] * weight for col, weight in weights.items())

#         df_tf_ranking = df_tf_ranking.sort_values(by='score', ascending=False)#.reset_index(drop=True)

#         # Display levels
#         if display:
#             # print(df_tf_reaction)
#             print("\n", helpers.df_to_table(df_tf_ranking[['level', 'level_count', 'last_date', 'mean_cum_volume', 'mean_pct_move', 'score']].round(2)), "\n")

#             plt.figure(figsize=(14,6))
#             plt.plot(df_tf['close'], label='Price')
#             for level in df_tf_ranking['level'].head(10):
#                 plt.axhline(y=level, color='orange', linestyle='--', alpha=0.5)
#             plt.legend()
#             plt.title("Strong Support & Resistance Levels")
#             plt.show()

#     return df_tf_ranking['level'].values.tolist()



# def add_support_resistance(df):

#     # Make sure date column is in datetime format and is set as index (while keeping the date column, just in case)
#     df['date'] = pd.to_datetime(df['date'], utc=True)
#     df['date'] = df['date'].dt.tz_convert(CONSTANTS.TZ_WORK)
#     df.index = df['date']


#     for timeframe, freq in self.timeframes.items():
#         print("Calculating...")
#         levels = self._calculate_recursive_levels(df, timeframe, freq)
#         print("Mapping...")
#         df = self._map_levels(df, levels, timeframe)
#         print("Checking Proximity...")
#         prox_col = f'{timeframe}_proximity'
#         df[prox_col] = df.apply(
#             lambda row: self._check_proximity(
#                 row['close'],
#                 row.get(f'{timeframe}_levels', []),
#                 self.proximity_thresholds.get(timeframe, 0.01),
#                 row['atr'] if self.use_atr else None
#             ),
#             axis=1
#         )
