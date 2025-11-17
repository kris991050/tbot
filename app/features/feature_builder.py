import sys, os, pandas as pd, numpy as np, gc
from datetime import datetime, timedelta
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.constants import CONSTANTS, PATHS, FORMATS
from utils.timeframe import Timeframe
from features import indicators, patterns, support_resistance, levels_pivots
from data import hist_market_data_handler


class FeatureBuilder:
    def __init__(self, df=pd.DataFrame(), ib=None, symbol=None, timeframe:Timeframe=None, feature_types:list=['all'], mtf:list=None, fill_method:str='ffill',
                 display_time:bool=True, file_format:str=None, hist_folder:str=None, save_to_file:bool=True, rounding_precision:int=None,
                 drop_levels:bool=False, save_levels:bool=False):
        self.df = helpers.format_df_date(df)
        self.date_from = self.df['date'].iloc[0]
        self.date_to = self.df['date'].iloc[-1]
        self.ib = ib
        self.symbol = symbol or (df.attrs['symbol'] if 'symbol' in df.attrs else None)
        self.contract, self.symbol = helpers.get_symbol_contract(self.ib, self.symbol)
        self.feature_types = self._build_feature_types(feature_types)
        self.timeframe = timeframe or helpers.get_df_timeframe(self.df)
        self.mtf = self._resolve_mtf(mtf)
        self.fill_method = fill_method
        self.display_time = display_time
        self.file_format = file_format or FORMATS.DEFAULT_FILE_FORMAT
        self.save_to_file = save_to_file
        self.hist_folder = hist_folder if hist_folder else PATHS.folders_path['hist_market_data']
        self.attrs = df.attrs if hasattr(df, 'attrs') else {}
        self.wmap = CONSTANTS.WARMUP_MAP
        # self.dfs_list = self._resolve_dfs_list()
        self.rounding_precision = rounding_precision or CONSTANTS.LEVELS_ROUNDING_PRECISION
        self.drop_levels = drop_levels
        self.save_levels = save_levels

    def _build_feature_types(self, feature_types):
        if feature_types == ['all']:
            return {
                'indicator_types': ['all'], 
                'pattern_types': ['all'], 
                'candle_pattern_list': ['all'], 
                'level_types': ['all'], 
                'sr_types': ['all']
            }
        return feature_types

    def _resolve_mtf(self, mtf):
        mtf = mtf if mtf else CONSTANTS.TIMEFRAMES_STD
        mtf = sorted(mtf, key=lambda tf: Timeframe(tf).to_seconds)
        mtf.insert(0, self.timeframe.pandas)
        mtf = list(dict.fromkeys(mtf))  # Remove duplicates, while keeping elements order

        tf_to_remove = []
        for tf in mtf:
            timeframe_mtf = Timeframe(tf)
            if timeframe_mtf.to_seconds < self.timeframe.to_seconds:
                print(f"Skipping timeframe {timeframe_mtf}, as < current timeframe {self.timeframe}")
                tf_to_remove.append(tf)
                continue
        mtf = [tf for tf in mtf.copy() if tf not in tf_to_remove]
        return mtf
    
    def _get_warmup_df(self, df:pd.DataFrame, timeframe:Timeframe) -> pd.DataFrame:
        print(f"⛏️ Fetching warmup data for {self.symbol}")
        wfetcher = hist_market_data_handler.HistMarketDataFetcher(ib=self.ib, timeframe=timeframe, file_format=self.file_format,
                                         validate=True, delete_existing=False, save_to_file=False)
        wparams = {'timeframe': timeframe,
                   'symbol': self.symbol, 'step_duration': 'auto',
                   'from_time': Timeframe(self.wmap[timeframe.pandas]).subtract_from_date(df['date'].min()),
                   'to_time': df['date'].min()}
        results = wfetcher.run(wparams)
        return results[self.symbol]['df']

    def _add_indicators(self, df):
        df = indicators.Indicators(df, ib=self.ib, symbol=self.symbol, types=self.feature_types['indicator_types']).apply_indicators()
        return df

    def _add_patterns(self, df):
        df = patterns.PatternBuilder(df, self.ib, self.symbol, candle_pattern_list=self.feature_types['candle_pattern_list'], 
                                     pattern_types=self.feature_types['pattern_types']).apply_patterns()
        return df

    def _add_levels(self, df):
        df = levels_pivots.LevelsAndPivotsCalculator(df, self.ib, self.symbol, rounding_precision=self.rounding_precision,
                                                     file_format=self.file_format, hist_folder=self.hist_folder, drop_levels=self.drop_levels,
                                                     save_levels=self.save_levels, level_types=self.feature_types['level_types']).apply_levels()
        return df

    def _add_support_resistance(self, df):
        df = support_resistance.RecursiveSRBuilder(df, self.ib, self.symbol, rounding_precision=self.rounding_precision,
                                                   file_format=self.file_format, hist_folder=self.hist_folder, drop_levels=self.drop_levels,
                                                   save_levels=self.save_levels, sr_types=self.feature_types['sr_types']).apply_sr()
        return df

    def add_features(self, add_indicators=True, add_levels=True, add_patterns=True, add_sr=True):
        print(f"🔧 Starting full feature pipeline... Current timeframe: {self.timeframe}, mutli timeframes: {self.mtf}")

        df_list = []
        for tf in self.mtf:
            times = {}
            timeframe_mtf = Timeframe(tf)
            df_tf = self.df if timeframe_mtf.to_seconds == self.timeframe.to_seconds else self.df.resample(timeframe_mtf.pandas, on='date').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                                                        'close': 'last', 'volume':'sum'}).dropna().reset_index()
            df_tf_warmup = self._get_warmup_df(df_tf, timeframe_mtf)
            df_tf = pd.concat([df_tf, df_tf_warmup]).drop_duplicates(subset=['date']).sort_values('date', inplace=False)

            # Indicators
            if add_indicators:
                print(f"🔧 Calculating indicators... Current timeframe {self.timeframe} | Current timeframe {timeframe_mtf}")
                t0 = datetime.now()
                df_tf = self._add_indicators(df_tf)
                t1 = datetime.now()
                times['indicators'] = t1 - t0
                print(f"⏱️ Time for indicators calculation: {times['indicators']}")

            # Levels
            if add_levels and self.timeframe.to_seconds == timeframe_mtf.to_seconds:
                print(f"🔧 Calculating levels and pivots... Current timeframe {self.timeframe}")
                t2 = datetime.now()
                df_tf = self._add_levels(df_tf)
                t3 = datetime.now()
                times['levels'] = t3 - t2
                print(f"⏱️ Time for levels and pivots calculation: {times['levels']}")

            # Patterns
            if add_patterns:
                print(f"🔧 Calculating patterns... Current timeframe {self.timeframe} | Current timeframe {timeframe_mtf}")
                t4 = datetime.now()
                df_tf = self._add_patterns(df_tf)
                t5 = datetime.now()
                times['patterns'] = t5 - t4
                print(f"⏱️ Time for patterns calculation: {times['patterns']}")

            # Support / Resistance
            if add_sr and self.timeframe.to_seconds == timeframe_mtf.to_seconds:
                print(f"🔧 Calculating support/resistance... Current timeframe {self.timeframe}")
                t6 = datetime.now()
                df_tf = self._add_support_resistance(df_tf)
                t7 = datetime.now()
                times['support_resistance'] = t7 - t6
                print(f"⏱️ Time for support/resistance calculation: {times['support_resistance']}")

            print(f"Trimming and downsizing df for {self.symbol} {tf}")
            keep_last = timeframe_mtf.to_seconds != self.timeframe.to_seconds
            df_tf = helpers.trim_df(df_tf, self.date_from, self.date_to, keep_last=keep_last)
            df_tf = self._set_df_float_type(df_tf) # Downsizing data type for disk space saving

            # Remove temporary columns
            columns_with_tilde = [col for col in df_tf.columns if col.startswith('~')]
            df_tf.drop(columns=columns_with_tilde, axis=1, inplace=True)
            gc.collect()

            if self.display_time and times:
                print(f"\n⏱️  Time summary for {tf}:")
                for step, duration in times.items():
                    print(f"  • {step.capitalize()}: {duration}")
                print(f"⏱️ Total time for enrichment: {sum([duration for step, duration in times.items()], timedelta(0))}")
                print()

            df_list.append({'df': df_tf, 'length': len(df_tf), 'width': len(df_tf.columns), 'timeframe': timeframe_mtf, 
                            'memory': helpers.get_df_memory_usage(df_tf, verbose=False)[0]})

        self.df = self._merge_mtf(df_list)
        self.df = self.ffill_df(self.df)
        self.df = self._set_df_float_type(self.df) # Downsizing data type for disk space saving

        self._restore_attrs()
        if self.save_to_file: self.save()

        return self.df

    def _restore_attrs(self):
        self.df.attrs = self.attrs
        return self.df

    def _set_df_float_type(self, df:pd.DataFrame):
        # Downsizing data type for disk space saving
        for col in df.select_dtypes(include='float'):
            df[col] = df[col].astype('float32').round(self.rounding_precision)
        return df

    def save(self, df=pd.DataFrame()):
        df = df if not df.empty else self.df
        attrs = df.attrs or self.attrs
        df_memory_usage_MB, _ = helpers.get_df_memory_usage(df, verbose=False)
        # if df.empty or not attrs:
        #     print("Dataframe empty or no attributes, data not saved")
        # else:
        print(f"💾 Saving enrichment results... Dataframe size: {df_memory_usage_MB:.2f} MB")
        symbol = self.symbol#attrs['symbol']
        timeframe = helpers.get_df_timeframe(df)#attrs['timeframe']
        hist_folder_symbol = os.path.join(self.hist_folder, symbol)

        data_path = helpers.construct_data_path(hist_folder_symbol, symbol, timeframe, to_time=df['date'].iloc[-1],
                                                        from_time=df['date'].iloc[0], file_format=self.file_format, data_type='enriched_data')

        helpers.save_df_to_file(df, data_path, file_format=self.file_format)

    def _merge_mtf(self, df_list:list):
        if len(df_list) == 0:
            print("⚠️ No dataframe found for merging.")
            return pd.DataFrame()
        elif len(df_list) == 1:
            return df_list[0]['df']
        
        df_list = sorted(df_list, key=lambda l: l['timeframe'].to_seconds) # make sure df_list is sorted based on timeframe length
        # print(f"🔗 Merging multitimeframe dfs for {self.symbol}:")
        df_list_df = pd.DataFrame([{key: value for key, value in item.items() if key not in 'df'} for item in df_list]) # Transform into dataframe without the dfs
        df_list_df['memory'] = df_list_df['memory'].round(self.rounding_precision)
        df_list_df.rename(columns = {'timeframe': 'Timeframe', 'length': 'Length', 'width': 'Width', 'memory': 'Memory (MB)'}, inplace=True)
        # df_list_df.drop(columns=['timeframe', 'width', 'length', 'memory'])
        print(helpers.df_to_table(df_list_df[['Timeframe', 'Length', 'Width', 'Memory (MB)']], title=f"🔗 Merging multitimeframe dfs for {self.symbol}"))
        print()
        # for entry in df_list:
        #     print(f"{entry['timeframe']}: size {entry['length']} x {entry['width']}, memory usage {entry['memory']:.2f} MB")

        # Merge step-by-step from larger timeframes to smaller ones
        df_merged = df_list[-1]['df']
        tf_merged = df_list[-1]['timeframe']
        for entry in reversed(df_list[:-1]): # Skip the last (smallest timeframe) DataFrame
            df = entry['df']
            tf = entry['timeframe']
            print(f"🔗 Merging to df {entry['timeframe']}...")
            df_merged = self._merge_dfs(df, df_merged, tf, tf_merged, entry['memory'])
            tf_merged = tf

        return df_merged

    def _merge_dfs(self, df1:pd.DataFrame, df2:pd.DataFrame, tf1:Timeframe, tf2:Timeframe, memory_MB:float):
        # Merge the resampled data back with the main dataframe (drop common columns from df_tf)
        common_cols = [col for col in df1.columns if col in df2.columns and col not in ['date']]
        df2.drop(columns=common_cols, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        df1.reset_index(drop=True, inplace=True)
        on = 'date'

        if tf2.to_timedelta >= timedelta(days=1):
            # Create the 'date_only' column for the first instance of each date in df1 (similar to date_D but only on every first instance of date, so that after merging with a 1D df_tf, we can interpolate the missing values)
            df2['date_only'] = df2['date'].dt.date
            df2.drop(columns='date', inplace=True)
            df1['date_only'] = df1['date'].dt.date
            df1['date_only'] = df1['date_only'].where(~df1['date_only'].duplicated(), np.nan)
            on = 'date_only'

        # Edge case where higher tf df2 in only 1 row and which date does not match any of df1 dates, as timeframes are differents 
        elif len(df2) == 1 and not df1['date'].isin(df2['date']).any() and tf2.to_seconds > tf1.to_seconds and df1['date'].iloc[0] > df2['date'].iloc[-1] and df1['date'].iloc[-1] < df2['date'].iloc[-1] + tf2.to_timedelta:
            df2['date'].iloc[-1] = df1['date'].iloc[0]

        if memory_MB > CONSTANTS.MEMORY_THRESHOLD_MB:
            df = helpers.merge_mtf_df_in_chunks(df1, df2, on=on)
        else:
            df = pd.merge(df1, df2, on=on, how='left')

        if on == 'date_only':
            df = self.ffill_df(df) # ffill early for df > 1D to avoid incorrect data ffill
            df.drop(columns=['date_only'], inplace=True)
        return df

    def ffill_df(self, df):
        print(f"⏭️ Forward fill df with method {self.fill_method}...")
        # Fill missing values in relevant columns after the merge
        # fill_columns = [col for col in df.columns if col.endswith(f"_{timeframe_mtf.pandas}")]# and not col.startswith("sr"))]
        fill_columns = [col for col in df.columns if any(col.endswith(f"_{Timeframe(tf).pandas}") for tf in self.mtf)]


        # Record boolean column (type is object and not boolean as contain (True, False, NaN))
        bool_cols, non_bool_cols = [], []
        for col in fill_columns:
            if df[col].dtype == object and df[col].isin([True, False, np.nan]).all(): # If dtype is object (could include NaN)
                bool_cols.append(col)
            else:
                non_bool_cols.append(col)

        # Fill boolean columns with forward fill (ffill)
        if bool_cols:
            df[bool_cols] = df[bool_cols].ffill().astype('bool') # Note: ffill before changing type otherwise NaN are set to True


        # For non-boolean columns, apply the specified fill method
        if non_bool_cols:
            if self.fill_method == "ffill":
                df[non_bool_cols] = df[non_bool_cols].ffill() # Forward fill
            elif self.fill_method == "interpolate":
                df[non_bool_cols] = df[non_bool_cols].interpolate(method='linear', limit_direction='both') # Interpolate
            else:
                print(f"Fill method {self.fill_method} not recognized for non-boolean columns.")

        return df





    # @staticmethod
    # def _merge_mtf_to_df(df, df_tf, timeframe_mtf, fill_method):

    #     print(f"⚙️ Merging multitimeframe df {timeframe_mtf} to base df...")
    #     # Merge the resampled data back with the main dataframe (drop common columns from df_tf)
    #     common_cols = [col for col in df.columns if col in df_tf.columns and col not in ['date']]
    #     df_tf.drop(columns=common_cols, inplace=True)
    #     df_tf.reset_index(drop=True, inplace=True)

    #     if timeframe_mtf.to_timedelta >= timedelta(days=1):
    #         # Create the 'date_only' column for the first instance of each date in self.df (similar to date_D but only on every first instance of date, so that after merging with a 1D df_tf, we can interpolate thhe missing values)
    #         df_tf['date_only'] = df_tf['date'].dt.date
    #         df_tf.drop(columns='date', inplace=True)
    #         df['date_only'] = df['date'].dt.date
    #         df['date_only'] = df['date_only'].where(~df['date_only'].duplicated(), np.nan)
    #         # df['date_only'] = df.groupby('date_only')['date_only'].transform('first')

    #         # df = pd.merge(df, df_tf, on='date_only', how='left')#, suffixes=('', f'_{tf}'))
    #         df = helpers.merge_mtf_df_in_chunks(df, df_tf, on='date_only')#, suffixes=('', f'_{tf}'))
    #         df.drop(columns=['date_only'], inplace=True)
    #     else:
    #         # # df2 = pd.merge(df.copy(), df_tf, on='date', how='left')
    #         df = helpers.merge_mtf_df_in_chunks(df, df_tf, on='date')

    #     # Fill missing values in relevant columns after the merge
    #     fill_columns = [col for col in df.columns if col.endswith(f"_{timeframe_mtf.pandas}")]# and not col.startswith("sr"))]

    #     # Convert columns to boolean if they contain only boolean values (True, False, NaN)
    #     for col in fill_columns:
    #         if df[col].dtype == object:  # If dtype is object (could include NaN)
    #             # Check if the column contains only booleans and NaNs
    #             # if df[col].apply(lambda x: x in [True, False, np.nan]).all():
    #             if df[col].isin([True, False, np.nan]).all():
    #                 df[col] = df[col].astype('bool')  # Convert to nullable boolean dtype

    #     # Separate boolean and non-boolean columns
    #     bool_cols = [col for col in fill_columns if df[col].dtype == bool]
    #     non_bool_cols = [col for col in fill_columns if df[col].dtype != bool]

    #     # Fill boolean columns with forward fill (ffill)
    #     if bool_cols:
    #         df[bool_cols] = df[bool_cols].ffill()

    #     # For non-boolean columns, apply the specified fill method
    #     if non_bool_cols:
    #         if fill_method == "ffill":
    #             df[non_bool_cols] = df[non_bool_cols].ffill()  # Forward fill
    #         elif fill_method == "interpolate":
    #             df[non_bool_cols] = df[non_bool_cols].interpolate(method='linear', limit_direction='both')  # Interpolate
    #         else:
    #             print(f"Fill method {fill_method} not recognized for non-boolean columns.")

    #     return df



    # def _resolve_dfs_list(self):
    #     ''' Get a dictionary containing dataframes and warmup dataframes for each timeframe in mtf '''
    #     dfs_list = {f'{self.timeframe}': {'df': self.df, 'df_warmup': self._get_warmup_df(self.df, self.timeframe)}}

    #     tf_to_remove = []
    #     for tf in self.mtf:
    #         timeframe_mtf = Timeframe(tf)
    #         if Timeframe(tf).to_seconds <= self.timeframe.to_seconds:
    #             if Timeframe(tf).to_seconds < self.timeframe.to_seconds:
    #                 print(f"Skipping timeframe {timeframe_mtf}, as < base timeframe {self.timeframe}")
    #             tf_to_remove.append(tf)
    #             continue
    #         df_tf = self.df.resample(timeframe_mtf.pandas, on='date').agg({'open': 'first', 'high': 'max', 'low': 'min',
    #                                                                     'close': 'last', 'volume':'sum'}).dropna().reset_index()
    #         df_tf_warmup = self._get_warmup_df(df_tf, timeframe_mtf)
    #         dfs_list[f'{timeframe_mtf}'] = {'df': df_tf, 'df_warmup': df_tf_warmup}
    #     self.mtf = [tf for tf in self.mtf.copy() if tf not in tf_to_remove]
    #     return dfs_list


    # def _apply_func_with_warmup(self, timeframe, func, class_type, *args, **kwargs):
    #     # df_warmup = self._get_warmup_df(df, timeframe)
    #     df = self.dfs_list[timeframe.pandas]['df']
    #     df_warmup = self.dfs_list[timeframe.pandas]['df_warmup']
    #     df_full = pd.concat([df, df_warmup]).drop_duplicates(subset=['date']).sort_values('date', inplace=False)

    #     class_instance = class_type(df_full, *args, **kwargs)
    #     df_full = getattr(class_instance, func)()  # Call the method dynamically

    #     df = helpers.trim_df(df_full, df['date'].iloc[0], df['date'].iloc[-1])
    #     # df = df_full[~df_full['date'].isin(df_warmup['date'])] # Remove warmup rows
    #     return df

    # def _apply_func_mtf(self, mtf, func, class_type, *args, **kwargs):
    #     """
    #     Apply a method (func) of a class (class_type) to the dataframe for multiple timeframes.

    #     Parameters:
    #     - func: The method to apply (e.g., apply_indicators or run).
    #     - class_type: The class type (e.g., Indicators or PatternBuilder).
    #     - df: The dataframe to which the method should be applied.
    #     - *args: Additional positional arguments to be passed to the class constructor.
    #     - **kwargs: Keyword arguments to be passed to the class constructor.
    #     """

    #     # Apply for the base timeframe
    #     # df_warmup = self._get_warmup(self.df, self.timeframe)
    #     # df_full = pd.concat([self.df, df_warmup]).drop_duplicates(subset=['date']).sort_values('date', inplace=False)
    #     # class_instance = class_type(df_full, *args, **kwargs)
    #     # df_full = getattr(class_instance, func)()  # Call the method dynamically
    #     # self.df = df_full[~df_full['date'].isin(df_warmup['date'])] # Remove warmup rows
    #     self.df = self._apply_func_with_warmup(self.timeframe, func, class_type, *args, **kwargs)

    #     # Apply for additional timeframes
    #     for tf in mtf:
    #         timeframe_mtf = Timeframe(tf)


    #         # Resample and aggregate data
    #         # df_tf = self.df.resample(timeframe_mtf.pandas, on='date').agg({'open': 'first', 'high': 'max', 'low': 'min',
    #         #                                                           'close': 'last', 'volume':'sum'}).dropna().reset_index()

    #         # Apply the method to the resampled data
    #         # df_warmup_mtf = self._get_warmup(df_tf, timeframe_mtf)
    #         # df_tf_full = pd.concat([df_tf, df_warmup_mtf]).drop_duplicates(subset=['date']).sort_values('date', inplace=False)
    #         # class_instance_tf = class_type(df_tf_full, *args, **kwargs)
    #         # df_tf_full = getattr(class_instance_tf, func)() # Call the method dynamically
    #         # df_tf_full.reset_index()
    #         # df_tf = df_tf_full[~df_tf_full['date'].isin(df_warmup_mtf['date'])] # Remove warmup rows
    #         df_tf = self._apply_func_with_warmup(timeframe_mtf, func, class_type, *args, **kwargs)

    #         # Merge the resampled data back with the main dataframe (drop common columns from df_tf)
    #         common_cols = [col for col in self.df.columns if col in df_tf.columns and col not in ['date']]
    #         df_tf.drop(columns=common_cols, inplace=True)

    #         if timeframe_mtf.to_timedelta >= timedelta(days=1):
    #             # Create the 'date_only' column for the first instance of each date in self.df (similar to date_D but only on every first instance of date, so that after merging with a 1D df_tf, we can interpolate thhe missing values)
    #             df_tf['date_only'] = df_tf['date'].dt.date
    #             df_tf.drop(columns='date', inplace=True)
    #             self.df['date_only'] = self.df['date'].dt.date
    #             self.df['date_only'] = self.df['date_only'].where(~self.df['date_only'].duplicated(), np.nan)
    #             # self.df['date_only'] = self.df.groupby('date_only')['date_only'].transform('first')

    #             self.df = pd.merge(self.df, df_tf, on='date_only', how='left')#, suffixes=('', f'_{tf}'))
    #             self.df.drop(columns=['date_only'], inplace=True)
    #         else:
    #             self.df = pd.merge(self.df, df_tf, on='date', how='left')

    #         # Fill missing values in relevant columns after the merge
    #         fill_columns = [col for col in self.df.columns if col.endswith(f"{tf}")]

    #         if self.fill_method == "ffill":
    #             self.df[fill_columns] = self.df[fill_columns].ffill()  # Forward fill
    #         elif self.fill_method == "interpolate":
    #             self.df[fill_columns] = self.df[fill_columns].interpolate(method='linear', limit_direction='both')  # Interpolate
    #             # self.df[fill_columns].interpolate(method='linear', limit_direction='both')  # Interpolate

    #     return self.df

    # def _apply_mtf(self, func)
    #     for tf in self.mtf:
    #         timeframe_mtf = Timeframe(tf)
    #         if timeframe_mtf.to_seconds <= self.timeframe.to_seconds():
    #             print(f"Skipping timeframe {tf}, < base timeframe {self.timeframe}")
    #             continue
    #         df_tf = df.resample(timeframe_mtf.pandas, on='date').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()
    #         df_tf = indicators.Indicators(df_tf, ib=self.ib, symbol=self.symbol, ind_types=self.ind_types).apply_indicators()

    #     df = pd.merge(df, df_tf, on='date', direction='backward')


    #     df = helpers.format_df_date(df)
    #     attrs = df.attrs
    #     tf_seconds = self.timeframe.to_seconds

    #     # List of indicator columns that could conflict
    #     indicator_cols = ['rsi', 'bband_h', 'bband_l', 'bband_mavg']
    #     temp_renamed = {}

    #     # Temporarily rename existing columns that would be overwritten
    #     for col in indicator_cols:
    #         if col in df.columns:
    #             temp_name = f"__tmp_{col}"
    #             df.rename(columns={col: temp_name}, inplace=True)
    #             temp_renamed[temp_name] = col

    #     for tf in self.mtf:
    #         timeframe = Timeframe(tf)
    #         if timeframe.to_seconds <= tf_seconds:
    #             print(f"Skipping timeframe {tf}, < base timeframe {self.timeframe}")
    #             continue

    #         df_tf = df.resample(timeframe.pandas, on='date').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()

    #         # df_tf = add_indicator(df_tf, ib, contract, types=['rsi', 'bollinger_bands'], bb_window=bb_window, rsi_window=rsi_window)
    #         df_tf = Indicators(df_tf, ib, contract, types=['rsi', 'bollinger_bands']).apply_indicators()

    #         rename_map = get_bb_rsi_tf_columns(tf)
    #         df_tf.rename(columns=rename_map, inplace=True)
    #         df = pd.merge_asof(df, df_tf[['date'] + list(rename_map.values())], on='date', direction='backward')

            # tf_str = tf if 'min' not in tf else tf[:-3]

            # df_tf.rename(columns={
            #     'rsi': f'rsi_{tf_str}',
            #     'rsi_slope': f'rsi_slope_{tf_str}',
            #     'bband_h': f'bband_h_{tf_str}',
            #     'bband_l': f'bband_l_{tf_str}',
            #     'bband_mavg': f'bband_mavg_{tf_str}',
            #     'bband_z_score': f'bband_z_score_{tf_str}',
            #     'bband_width_ratio': f'bband_width_ratio_{tf_str}'
            # }, inplace=True)

            # df = pd.merge_asof(df,
            #     df_tf[['date', f'rsi_{tf_str}', f'rsi_slope_{tf_str}', f'bband_h_{tf_str}', f'bband_l_{tf_str}',
            #            f'bband_z_score_{tf_str}', f'bband_width_ratio_{tf_str}']],
            #            on='date', direction='backward')

        # Restore any temporarily renamed original columns
        # if temp_renamed:
        #     df.rename(columns={tmp: orig for tmp, orig in temp_renamed.items()}, inplace=True)

        # df.attrs = attrs

        # return df.copy()