import os, sys, pandas as pd, numpy as np
from datetime import timedelta
from ib_insync import *

current_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_folder)
from utils import helpers
from utils.constants import CONSTANTS, FORMATS, PATHS
from utils.timeframe import Timeframe
import indicators

os.environ["MODIN_CPUS"] = '4'


class LevelsAndPivotsCalculator:
    def __init__(self, df:pd.DataFrame, ib:IB, symbol:str=None, lookback_factor:int=None, rounding_precision:int=None, file_format:str=None,
                 hist_folder:str=None, level_types:list=['all'], drop_levels:bool=True, save_levels:bool=False, next_levels:bool=True):
        self.ib = ib
        self.df = helpers.format_df_date(df, set_index=True)
        self.attrs = df.attrs
        self.symbol = symbol or (df.attrs['symbol'] if 'symbol' in df.attrs else None)
        self.lookback_factor = int(lookback_factor) if lookback_factor else 2
        self.monthly_lookback = f"{self.lookback_factor}M"
        self.daily_lookback = f"{2 * self.lookback_factor}D"
        self.rounding_precision = rounding_precision or CONSTANTS.LEVELS_ROUNDING_PRECISION
        self.file_format = file_format or FORMATS.DEFAULT_FILE_FORMAT
        self.hist_folder = hist_folder if hist_folder else PATHS.folders_path['hist_market_data']
        self.piv_addon = []
        self.level_types = level_types
        self.drop_levels = drop_levels
        self.save_levels = save_levels
        self.next_levels = next_levels

    @staticmethod
    def _get_levels_by_resampling(df, levels, timeframe_str:str, df2=pd.DataFrame()):

        timeframe = Timeframe(timeframe_str)
        if df2.empty: df2 = df
        timeframe_str_col = timeframe_str.lstrip('0123456789') if len(timeframe_str) > 1 else timeframe_str[1:]
        date_tf_col = f'date_{timeframe_str_col}'

        # Resample data by timeframe
        levels_aggregate = {}
        for l in levels: levels_aggregate[l['col']] = l['funct']

        df2_resampled = df2.resample(timeframe.pandas, on='date').agg(levels_aggregate).dropna().reset_index()
        # df2_resampled = df2.resample('ME' if timeframe == 'M' else timeframe, on='date').agg(levels_aggregate).dropna().reset_index()
        # if timeframe == '1D': df2_resampled[date_tf_col] = df2_resampled['date'].dt.date
        # elif timeframe == 'M': df2_resampled[date_tf_col] = df2_resampled['date'].dt.strftime('%Y-%m')
        if timeframe.to_timedelta == timedelta(days=1): df2_resampled[date_tf_col] = df2_resampled['date'].dt.date
        elif timeframe.to_timedelta == timedelta(days=30): df2_resampled[date_tf_col] = df2_resampled['date'].dt.strftime('%Y-%m')

        # Shift to get previous row's data
        for l in levels: df2_resampled[l['name']] = df2_resampled[l['col']].shift(l['shift'])

        # Map the previous row's data to the original DataFrame
        levels_merge = [date_tf_col]
        for l in levels: levels_merge += [l['name']]

        df = pd.merge(df, df2_resampled[levels_merge], on=date_tf_col, how='left')
        df = helpers.format_df_date(df)#, set_index=True)

        return df

    @staticmethod
    def _add_pm_levels(df:pd.DataFrame, df_extended:pd.DataFrame=pd.DataFrame()):
        if df_extended.empty: df_extended = df
        df_extended.reset_index(drop=True, inplace=True)

        # Extract premarket high/low per day
        pm_levels = (df_extended[df_extended['session'] == 'pre-market'].groupby('date_D').agg(pmh=('high', 'max'), pml=('low', 'min')))

        # Shift to align with *next day’s* premarket
        pm_levels_shifted = pm_levels.shift(1)

        # Map the PMH/PML to all rows based on session
        df_extended = df_extended.merge(pm_levels, how='left', left_on='date_D', right_index=True)
        df_extended = df_extended.merge(pm_levels_shifted, how='left', left_on='date_D', right_index=True, suffixes=('', '_prev'))


        # conditions = [df_extended['session'].isin(['rth', 'post-market']), df_extended['session'] == 'pre-market']

        # # Define corresponding values (use pm_levels for today and pm_levels_shifted for the previous day)
        # pmh_values = [df_extended['date_D'].map(pm_levels['pmh']), df_extended['date_D'].map(pm_levels_shifted['pmh'])]
        # pml_values = [df_extended['date_D'].map(pm_levels['pml']), df_extended['date_D'].map(pm_levels_shifted['pml'])]

        # # Use np.select to assign the values based on conditions
        # df_extended['pmh_final'] = np.select(conditions, pmh_values, default=np.nan)
        # df_extended['pml_final'] = np.select(conditions, pml_values, default=np.nan)


        # Initialize pmh/pml columns with NaN
        df_extended['~pmh'] = np.nan
        df_extended['~pml'] = np.nan
        # df_extended['~pmh_due'] = False  # Initialize the flag as False
        # df_extended['~pml_due'] = False

        # Assign based on session type
        df_extended.loc[df_extended['session'].isin(['rth', 'post-market']), '~pmh'] = df_extended['pmh']
        df_extended.loc[df_extended['session'].isin(['rth', 'post-market']), '~pml'] = df_extended['pml']
        df_extended.loc[df_extended['session'] == 'pre-market', '~pmh'] = df_extended['pmh_prev']
        df_extended.loc[df_extended['session'] == 'pre-market', '~pml'] = df_extended['pml_prev']

        # Set flag to True for the end of pre-market session (when levels are updated)
        # df_extended.loc[df_extended['session'] == 'pre-market', '~pmh_due'] = True
        # df_extended.loc[df_extended['session'] == 'pre-market', '~pml_due'] = True
        # df_extended['~pmh_due'] = df_extended.groupby('date_D')['session'].apply(lambda x: x.eq('pre-market').cumsum().eq(1))
        # df_extended['~pml_due'] = df_extended.groupby('date_D')['session'].apply(lambda x: x.eq('pre-market').cumsum().eq(1))
        # df_extended['~pmh_due'] = (df_extended.groupby('date_D')['session'].transform(lambda x: (x == 'rth').cumsum() == 1))
        # df_extended['~pml_due'] = df_extended['~pmh_due']

        # Drop helper columns and merge back to original df
        df_extended = df_extended.drop(columns=['pmh', 'pml', 'pmh_prev', 'pml_prev'])
        df = df.merge(df_extended[['date', 'session', '~pmh', '~pml']], how='left', on=['date', 'session'])

        return df

    def _add_daily_levels(self, df):

        timeframe_df = helpers.get_df_timeframe(df)
        if not(timeframe_df.to_timedelta < timedelta(days=1)):
            print(f"Daily levels not added, timeframe must be < 1 day. Current timeframe: {timeframe_df}")
            return df

        df = indicators.IndicatorsUtils.add_market_sessions(df)

        # Prepare df for resampling
        if 'date_D' not in df.columns: df['date_D'] = df['date'].dt.date

        self.piv_addon.append('')
        self.piv_addon.append('_D')

        # ✳️ Ensure sufficient data
        df_extended = indicators.IndicatorsUtils.get_level_compatible_df(self.ib, self.symbol, df, timeframe_df, lookback=self.daily_lookback,
                                                                         file_format=self.file_format, hist_folder=self.hist_folder)
        if df_extended.equals(df):
            print(f"⚠️ Could not fetch additional data for level calculation. Calculating based on original df...")
        df_extended = indicators.IndicatorsUtils.add_market_sessions(df_extended)
        df_extended['date_D'] = df_extended['date'].dt.date

        # Calculate pdh and pdl
        levels_list = [{'name':'~pdh', 'col':'high', 'funct':'max', 'shift':1},
                    {'name':'~pdl', 'col':'low', 'funct':'min', 'shift':1}]
        df = self._get_levels_by_resampling(df, levels_list, '1D', df2=df_extended)

        # Calculate pmh and pml
        # df_extended_pre_market = df_extended[df_extended['session'] == 'pre-market']
        # levels_list = [{'name':'~pmh', 'col':'high', 'funct':'max', 'shift':0},
        #             {'name':'~pml', 'col':'low', 'funct':'min', 'shift':0}]
        # df = self._get_levels_by_resampling(df, levels_list, '1D', df2=df_extended_pre_market)

        # # Calculate shift to apply to pmh and pml
        # th_times = CONSTANTS.TH_TIMES
        # # time_interval = df['date'].iloc[1] - df['date'].iloc[0]
        # # time_interval2 = datetime(1, 1, 1, 9, 30) - datetime(1, 1, 1, 4, 0)
        # time_interval = df['date'].iloc[1] - df['date'].iloc[0]
        # time_interval2 = datetime(1, 1, 1, th_times['rth'].hour, th_times['rth'].minute) - datetime(1, 1, 1, th_times['pre-market'].hour, th_times['pre-market'].minute)
        # shift_pm = int(round(time_interval2 / time_interval, 0))
        # df_shift = df[df["date"].between(pd.to_datetime("2024-12-30 04:00:00-05:00"), pd.to_datetime("2024-12-30 09:30:00-05:00"))]

        # # df[['~pmh', '~pml']] = df[['~pmh', '~pml']].shift(shift_pm)

        df = self._add_pm_levels(df, df_extended)

        # Calculate pdc, do, pdh_D, pdl_D
        df_extended_rth = df_extended[df_extended['session'] == 'rth']
        levels_list = [{'name':'~pdc', 'col':'close', 'funct':'last', 'shift':1},
                    {'name':'~do', 'col':'open', 'funct':'first', 'shift':0},
                    {'name':'~pdh_D', 'col':'high', 'funct':'max', 'shift':1},
                    {'name':'~pdl_D', 'col':'low', 'funct':'min', 'shift':1}]
        df = self._get_levels_by_resampling(df, levels_list, '1D', df2=df_extended_rth)

        df['levels_list'] = [sorted(p) for p in zip(df['~pml'], df['~pmh'], df['~pdl'], df['~pdh'], df['~pdl_D'], df['~pdh_D'], df['~pdc'], df['~do'])]
        df['levels_list'] = df['levels_list'].apply(lambda lst: [np.float32(round(x, self.rounding_precision)) for x in lst]) # Downsizing to float 32 to save doisk space
        if self.next_levels: df = indicators.IndicatorsUtils.calculate_next_levels(df, 'levels_list')

        # Create due flags
        df['~pmh_due'] = (df.groupby('date_D')['session'].transform(lambda x: (x == 'rth').cumsum() == 1))
        df['~pdh_due'] = (df.groupby('date_D')['session'].transform(lambda x: (x == 'pre-market').cumsum() == 1))
        df['~levels_due'] = (df['~pmh_due'] | df['~pdh_due'])
        df.drop(columns=['~pmh_due', '~pdh_due'], inplace=True)

        # Save levels list in separate Dataframe
        col = 'levels_list'
        tf = '1D'
        if self.save_levels: df = indicators.IndicatorsUtils.save_levels(df=df, timeframe=Timeframe(tf), symbol=self.symbol, col=col,
                                                                         hist_folder=self.hist_folder, data_type='levels', file_format='csv')
        if self.drop_levels: df.drop(columns=col, inplace=True)
        return df

    def _add_monthly_levels(self, df):

        timeframe_df = helpers.get_df_timeframe(df)
        if not(timeframe_df.to_timedelta < timedelta(days=30)):
            print(f"Monthly levels not added, timeframe must be < 1 month. Current timeframe: {timeframe_df}")
            return df

        # Prepare df for resampling
        if 'date_M' not in df.columns: df['date_M'] = df['date'].dt.strftime('%Y-%m')

        self.piv_addon.append('_M')

        # ✳️ Ensure sufficient data
        df_extended = indicators.IndicatorsUtils.get_level_compatible_df(self.ib, self.symbol, df, Timeframe('1M'), lookback=self.monthly_lookback,
                                                                         file_format=self.file_format, hist_folder=self.hist_folder)

        # Calculate pMc, pMh, pMl
        levels_list = [{'name':'~pMh', 'col':'high', 'funct':'max', 'shift':1},
                    {'name':'~pMl', 'col':'low', 'funct':'min', 'shift':1},
                    {'name':'~pMc', 'col':'close', 'funct':'last', 'shift':1},
                    #    {'name':'~Mo', 'col':'open', 'funct':'first', 'shift':0},
                    {'name':'~Mo', 'col':'open', 'funct':'first', 'shift':0}]
        df = self._get_levels_by_resampling(df, levels_list, '1M', df2=df_extended)

        df['levels_M_list'] = [sorted(p) for p in zip(df['~pMl'], df['~pMh'], df['~pMc'], df['~Mo'])]
        df['levels_M_list'] = df['levels_M_list'].apply(lambda lst: [np.float32(round(x, self.rounding_precision)) for x in lst]) # Downsizing to float 32 to save doisk space
        if self.next_levels: df = indicators.IndicatorsUtils.calculate_next_levels(df, 'levels_M_list')

        # Create due flags
        # df['levels_M_due'] = df.groupby('date_M')['session'].transform(lambda x: x.idxmin() == x.index[0])
        df['~levels_M_due'] = df.groupby('date_M').cumcount() == 0


        # Save levels list in separate Dataframe
        col = 'levels_M_list'
        tf = '1M'
        if self.save_levels: df = indicators.IndicatorsUtils.save_levels(df=df, timeframe=Timeframe(tf), symbol=self.symbol, col=col,
                                                                         hist_folder=self.hist_folder, data_type='levels', file_format='csv')
        if self.drop_levels: df.drop(columns=col, inplace=True)
        return df

    def _add_camarilla_pivots(self, df):

        # Calculate Camarilla Pivots
        for p in self.piv_addon:
            pc = '~pMc' if p == '_M' else '~pdc'
            ph = '~pMh' if p == '_M' else '~pdh_D' if p == '_D' else '~pdh'
            pl = '~pMl' if p == '_M' else '~pdl_D' if p == '_D' else '~pdl'
            pp = (df[ph] + df[pl] + df[pc]) / 3
            r1 = df[pc] + 1.1 * (df[ph] - df[pl]) / 12
            r2 = df[pc] + 1.1 * (df[ph] - df[pl]) / 6
            r3 = df[pc] + 1.1 * (df[ph] - df[pl]) / 4
            r4 = df[pc] + 1.1 * (df[ph] - df[pl]) / 2
            r5 = r4 + 1.168 * (r4 - r3)
            r6 = df[pc] * df[ph] / df[pl]
            s1 = df[pc] - 1.1 * (df[ph] - df[pl]) / 12
            s2 = df[pc] - 1.1 * (df[ph] - df[pl]) / 6
            s3 = df[pc] - 1.1 * (df[ph] - df[pl]) / 4
            s4 = df[pc] - 1.1 * (df[ph] - df[pl]) / 2
            s5 = s4 - 1.168 * (s3 - s4)
            s6 = df[pc] - (r6 - df[pc])

            df[f'pivots{p}_list'] = [sorted(p) for p in zip(s6, s5, s4, s3, s2, s1, r1, r2, r3, r4, r5, r6)]
            df[f'pivots{p}_list'] = df[f'pivots{p}_list'].apply(lambda lst: [np.float32(round(x, self.rounding_precision)) for x in lst]) # Downsizing to float 32 to save doisk space

            # df['pp'+p], df['r1'+p], df['r2'+p], df['r3'+p], df['r4'+p], df['r5'+p], df['r6'+p] = pp, r1, r2, r3, r4, r5, r6
            # df['s1'+p], df['s2'+p], df['s3'+p], df['s4'+p], df['s5'+p], df['s6'+p] = s1, s2, s3, s4, s5, s6

            # Calculate CPR
            df['cpr_size_to_yst_%'] = (((r2 - s2) * 100 / (r2.shift(1) - s2.shift(1))) - 100).round(2)
            df['cpr_midpoint_to_yst_%'] = ((0.5 * (s2 + r2) * 100 / (0.5 * (s2.shift(1) + r2.shift(1)))) - 100).round(2)

            # Create Camarilla position column
            conditions_cam_positions = [
                (df['close'] < s6), (r6 < df['close']), (s6 < df['close']) & (df['close'] < s5), (s5 < df['close']) & (df['close'] < s4),
                (s4 < df['close']) & (df['close'] < s3), (s3 < df['close']) & (df['close'] < s2), (s2 < df['close']) & (df['close'] < s1),
                (s1 < df['close']) & (df['close'] < r1), (r1 < df['close']) & (df['close'] < r2), (r2 < df['close']) & (df['close'] < r3),
                (r3 < df['close']) & (df['close'] < r4), (r4 < df['close']) & (df['close'] < r5), (r5 < df['close']) & (df['close'] < r6)]

            choices_cam_positions = [-6, 6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
            df['cam'+p+'_position'] = np.select(conditions_cam_positions, choices_cam_positions, default=np.nan)
            if self.next_levels: df = indicators.IndicatorsUtils.calculate_next_levels(df, f'pivots{p}_list')

            # Create due flags
            pp = '_M' if p == '_M' else '_D'
            # df[f'pivots{p}_due'] = df.groupby(f'date{p}')['session'].transform(lambda x: x.idxmin() == x.index[0])
            df[f'~pivots{pp}_due'] = df.groupby(f'date{pp}').cumcount() == 0

            # Save levels list in separate Dataframe
            col = f'pivots{p}_list'
            tf = '1M' if p == '_M' else '1D'
            if self.save_levels: df = indicators.IndicatorsUtils.save_levels(df=df, timeframe=Timeframe(tf), symbol=self.symbol, col=col,
                                                                         hist_folder=self.hist_folder, data_type=f'pivots{p}', file_format='csv')
            if self.drop_levels: df.drop(columns=col, inplace=True)

        return df

    def apply_levels(self):

        # df = self.df.copy()
        if not self.df.empty:
            if 'daily' in self.level_types or 'all' in self.level_types:
                print(f"Calculating daily levels...")
                self.df = self._add_daily_levels(self.df)
            if 'monthly' in self.level_types or 'all' in self.level_types:
                print(f"Calculating monthly levels...")
                self.df = self._add_monthly_levels(self.df)
            if 'camarilla' in self.level_types or 'all' in self.level_types:
                print(f"Calculating camarilla pivots...")
                self.df = self._add_camarilla_pivots(self.df)

        if self.df.attrs != self.attrs: self.df.attrs = self.attrs
        # self.df = df
        return self.df


if __name__ == "__main__":

    # current_path = os.path.realpath(__file__)
    # path = path_current_setup(current_path)

    input("\nEnter anything to exit")



# def _add_levels(self):

#     df = self.df.copy()
#     timeframe_df = helpers.get_df_timeframe(df)

#     if not df.empty:
#         df = add_market_sessions(df)

#         # Prepare df for resampling
#         if 'date_D' not in df.columns: df['date_D'] = df['date'].dt.date
#         if 'date_M' not in df.columns: df['date_M'] = df['date'].dt.strftime('%Y-%m')

#         piv_addon = []

#         if helpers.timeframe_to_seconds(timeframe_df) < helpers.timeframe_to_seconds('1 day'):

#             piv_addon.append('')
#             piv_addon.append('_D')

#             # Calculate pdc, pdh and pdl
#             levels_list = [{'name':'pdh', 'col':'high', 'funct':'max', 'shift':1},
#                         {'name':'pdl', 'col':'low', 'funct':'min', 'shift':1}]
#             df = self._get_levels_by_resampling(df, levels_list, '1D')

#             # Calculate pmh and pml
#             df_pre_market = df[df['session'] == 'pre-market']
#             levels_list = [{'name':'pmh', 'col':'high', 'funct':'max', 'shift':0},
#                         {'name':'pml', 'col':'low', 'funct':'min', 'shift':0}]
#             df = self._get_levels_by_resampling(df, levels_list, '1D', df2=df_pre_market)

#             # Calculate shift to apply to pmh and pml
#             time_interval = df['date'].iloc[1] - df['date'].iloc[0]
#             time_interval2 = datetime(1, 1, 1, 9, 30) - datetime(1, 1, 1, 4, 0)
#             shift_pm = int(round(time_interval2 / time_interval, 0))
#             df_shift = df[df["date"].between(pd.to_datetime("2024-12-30 04:00:00-05:00"), pd.to_datetime("2024-12-30 09:30:00-05:00"))]

#             df[['pmh', 'pml']] = df[['pmh', 'pml']].shift(shift_pm)

#             # Calculate pdc, do, pdh_D, pdl_D
#             df_rth = df[df['session'] == 'rth']
#             levels_list = [{'name':'pdc', 'col':'close', 'funct':'last', 'shift':1},
#                         {'name':'do', 'col':'open', 'funct':'first', 'shift':0},
#                         {'name':'pdh_D', 'col':'high', 'funct':'max', 'shift':1},
#                         {'name':'pdl_D', 'col':'low', 'funct':'min', 'shift':1}]
#             df = self._get_levels_by_resampling(df, levels_list, '1D', df2=df_rth)

#             df['levels'] = [sorted(p) for p in zip(df['pml'], df['pmh'], df['pdl'], df['pdh'], df['pdl_D'], df['pdh_D'], df['pdc'], df['do'])]
#             df['levels'] = df['levels'].apply(lambda lst: [np.float32(round(x, self.rounding_precision)) for x in lst]) # Downsizing to float 32 to save doisk space
#             df = calculate_next_levels(df, 'levels')

#         # if timeframe_df not in ['1M']:
#         # if timeframe_df < timedelta(days=30):
#         if helpers.timeframe_to_seconds(timeframe_df) < helpers.timeframe_to_seconds('1 month'):

#             # ✳️ Ensure sufficient data
#             df_extended = get_level_compatible_df(self.ib, self.symbol, df, self.df_levels, 'M', lookback='2M')

#             piv_addon.append('_M')

#             # Calculate pMc, pMh, pMl
#             levels_list = [{'name':'pMh', 'col':'high', 'funct':'max', 'shift':1},
#                         {'name':'pMl', 'col':'low', 'funct':'min', 'shift':1},
#                         {'name':'pMc', 'col':'close', 'funct':'last', 'shift':1},
#                         #    {'name':'Mo', 'col':'open', 'funct':'first', 'shift':0},
#                         {'name':'Mo', 'col':'open', 'funct':'first', 'shift':0}]
#             df = self._get_levels_by_resampling(df, levels_list, 'M', df2=df_extended)

#             df['levels_M'] = [sorted(p) for p in zip(df['pMl'], df['pMh'], df['pMc'], df['Mo'])]
#             df['levels_M'] = df['levels_M'].apply(lambda lst: [np.float32(round(x, self.rounding_precision)) for x in lst]) # Downsizing to float 32 to save doisk space
#             df = calculate_next_levels(df, 'levels_M')

#         # Calculate Camarilla Pivots
#         for p in piv_addon:
#             pc = 'pMc' if p == '_M' else 'pdc'
#             ph = 'pMh' if p == '_M' else 'pdh_D' if p == '_D' else 'pdh'
#             pl = 'pMl' if p == '_M' else 'pdl_D' if p == '_D' else 'pdl'
#             pp = (df[ph] + df[pl] + df[pc]) / 3
#             r1 = df[pc] + 1.1 * (df[ph] - df[pl]) / 12
#             r2 = df[pc] + 1.1 * (df[ph] - df[pl]) / 6
#             r3 = df[pc] + 1.1 * (df[ph] - df[pl]) / 4
#             r4 = df[pc] + 1.1 * (df[ph] - df[pl]) / 2
#             r5 = r4 + 1.168 * (r4 - r3)
#             r6 = df[pc] * df[ph] / df[pl]
#             s1 = df[pc] - 1.1 * (df[ph] - df[pl]) / 12
#             s2 = df[pc] - 1.1 * (df[ph] - df[pl]) / 6
#             s3 = df[pc] - 1.1 * (df[ph] - df[pl]) / 4
#             s4 = df[pc] - 1.1 * (df[ph] - df[pl]) / 2
#             s5 = s4 - 1.168 * (s3 - s4)
#             s6 = df[pc] - (r6 - df[pc])

#             df['pivots'+p] = [sorted(p) for p in zip(s6, s5, s4, s3, s2, s1, r1, r2, r3, r4, r5, r6)]
#             df['pivots'+p] = df['pivots'+p].apply(lambda lst: [np.float32(round(x, self.rounding_precision)) for x in lst]) # Downsizing to float 32 to save doisk space

#             # df['pp'+p], df['r1'+p], df['r2'+p], df['r3'+p], df['r4'+p], df['r5'+p], df['r6'+p] = pp, r1, r2, r3, r4, r5, r6
#             # df['s1'+p], df['s2'+p], df['s3'+p], df['s4'+p], df['s5'+p], df['s6'+p] = s1, s2, s3, s4, s5, s6

#             # Calculate CPR
#             df['cpr_size_to_yst_%'] = (((r2 - s2) * 100 / (r2.shift(1) - s2.shift(1))) - 100).round(2)
#             df['cpr_midpoint_to_yst_%'] = ((0.5 * (s2 + r2) * 100 / (0.5 * (s2.shift(1) + r2.shift(1)))) - 100).round(2)

#             # Create Camarilla position column
#             conditions_cam_positions = [
#                 (df['close'] < s6), (r6 < df['close']), (s6 < df['close']) & (df['close'] < s5), (s5 < df['close']) & (df['close'] < s4),
#                 (s4 < df['close']) & (df['close'] < s3), (s3 < df['close']) & (df['close'] < s2), (s2 < df['close']) & (df['close'] < s1),
#                 (s1 < df['close']) & (df['close'] < r1), (r1 < df['close']) & (df['close'] < r2), (r2 < df['close']) & (df['close'] < r3),
#                 (r3 < df['close']) & (df['close'] < r4), (r4 < df['close']) & (df['close'] < r5), (r5 < df['close']) & (df['close'] < r6)]

#             choices_cam_positions = [-6, 6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

#             df['cam'+p+'_position'] = np.select(conditions_cam_positions, choices_cam_positions, default=np.nan)

#             df = calculate_next_levels(df, 'pivots'+p)

#         # print(df[['date', 'date_D', 'date_M', 'pdc', 'pdh', 'pdl', 'pmh', 'pml', 'pdh_D', 'pdl_D', 'pMc', 'pMh', 'pMl']].to_string())
#         # print(df[['date', 'date_D', 'date_M', 'pMc', 'pMh', 'pMl', 'cam_position', 'cam_position_D', 'cam_position_M']].to_string())

#     if df.attrs != self.attrs: df.attrs = self.attrs

#     return df.copy()



        # if "levels_old" in indicators_list:
        #     # iterate through df using and create list with previous days values
        #     pdc_list, pdh_list, pdl_list, pmh_list, pml_list, pdp_list, pdp_list_D, pdp_list_M = [], [], [], [], [], [], [], []
        #     date_prev = pd.to_datetime(df['date'].iloc[0]).strftime('%Y-%m-%d')
        #     month_prev = pd.to_datetime(df['date'].iloc[0]).month
        #     index_new_day, index_end_premarket, index_start_postmarket = 0, 0, 0
        #     pdc, pdh, pdl, pmh, pml, pdh_D, pdl_D, pMc, pMh, pMl = None, None, None, None, None, None, None, None, None, None
        #     pp, s1, s2, s3, s4, s5, s6, r1, r2, r3, r4, r5, r6 = None, None, None, None, None, None, None, None, None, None, None, None, None
        #     pp_D, s1_D, s2_D, s3_D, s4_D, s5_D, s6_D, r1_D, r2_D, r3_D, r4_D, r5_D, r6_D = None, None, None, None, None, None, None, None, None, None, None, None, None
        #     pp_M, s1_M, s2_M, s3_M, s4_M, s5_M, s6_M, r1_M, r2_M, r3_M, r4_M, r5_M, r6_M = None, None, None, None, None, None, None, None, None, None, None, None, None

        #     # Get monthly data for monthly pivots calculation

        #     date_init = pd.to_datetime(df['date'].iloc[0])#.strftime('%Y-%m-%d'))
        #     query_time = pd.to_datetime(df["date"].iloc[len(df["date"])-1])#.strftime('%Y-%m-%d'))
        #     num_months = (query_time.year - date_init.year) * 12 + (query_time.month - date_init.month) + 2
        #     whatToShow = "TRADES" if contract.symbol not in get_forex_symbols_list() else "MIDPOINT"
        #     if num_months >= 12: duration = str(np.floor(num_months / 12)) + ' Y'
        #     else: duration = str(num_months) + ' M'
        #     bars_M = ib.reqHistoricalData(contract, endDateTime=query_time, durationStr=duration, barSizeSetting='1 month', whatToShow=whatToShow, useRTH=False)
        #     ib.sleep(CONSTANTS.PROCESS_TIME['long'])
        #     df_M = util.df(bars_M)

        #     # pdc_triggered = Trues
        #     for i, d in enumerate(df["date"]):
        #         date = pd.to_datetime(df['date'].iloc[i]).strftime('%Y-%m-%d')
        #         time = pd.to_datetime(df['date'].iloc[i]).strftime('%H-%M-%S')
        #         month = pd.to_datetime(date).month
        #         if i > 0: time_prev = pd.to_datetime(df['date'].iloc[i-1]).strftime('%H-%M-%S')
        #         else: time_prev = time

        #         # Previous day high and low calculation
        #         if date != date_prev:
        #             # print(date)
        #             date_prev = date
        #             # pdh = max([df.loc[j, "high"] for j in range(index_end_premarket, index_start_postmarket)])
        #             # pdl = min([df.loc[j, "low"] for j in range(index_end_premarket, index_start_postmarket)])
        #             pdh = max([df['high'].iloc[j] for j in range(index_new_day, i)])
        #             pdl = min([df['low'].iloc[j] for j in range(index_new_day, i)])
        #             index_new_day = i

        #             # Camarilla Pivots (Non-Daily Based Values)
        #             if pdh and pdl and pdc:
        #                 base_pivots = (pdh - pdl) * 1.1
        #                 pp = (pdh + pdl + pdc) / 3
        #                 r1, r2, r3, r4 = pdc + base_pivots / 12, pdc + base_pivots / 6, pdc + base_pivots / 4, pdc + base_pivots / 2
        #                 r5, r6 = r4 + 1.168 * (r4 - r3), (pdh / pdl) * pdc
        #                 s1, s2, s3, s4 = pdc - base_pivots / 12, pdc - base_pivots / 6, pdc - base_pivots / 4, pdc - base_pivots / 2
        #                 s5, s6 = s4 - 1.168 * (s3 - s4), pdc - (r6 - pdc)

        #         # Previous day close calculation
        #         if time >= "16-00-00" and time_prev < "16-00-00":
        #             pdc = df['close'].iloc[i-1]
        #             index_start_postmarket = i

        #         # Premarket high and low calculation
        #         if time >= "09-30-00" and time_prev < "09-30-00":
        #             pmh = max([df['high'].iloc[j] for j in range(index_new_day, i)])
        #             pml = min([df['low'].iloc[j] for j in range(index_new_day, i)])
        #             # pdh_D = max([df.loc[j, "high"] for j in range(index_end_premarket, i)])
        #             # pdl_D = min([df.loc[j, "low"] for j in range(index_end_premarket, i)])
        #             pdh_D = max([df['high'].iloc[j] for j in range(index_end_premarket, index_start_postmarket)], default=None)
        #             pdl_D = min([df['low'].iloc[j] for j in range(index_end_premarket, index_start_postmarket)], default=None)
        #             index_end_premarket = i

        #             # Camarilla Pivots (Daily Based Values)
        #             if pdh_D and pdl_D and pdc:
        #                 base_pivots_D = (pdh_D - pdl_D) * 1.1
        #                 pp_D = (pdh_D + pdl_D + pdc) / 3
        #                 r1_D, r2_D, r3_D, r4_D = pdc + base_pivots_D / 12, pdc + base_pivots_D / 6, pdc + base_pivots_D / 4, pdc + base_pivots_D / 2
        #                 r5_D, r6_D = r4_D + 1.168 * (r4_D - r3_D), (pdh_D / pdl_D) * pdc
        #                 s1_D, s2_D, s3_D, s4_D = pdc - base_pivots_D / 12, pdc - base_pivots_D / 6, pdc - base_pivots_D / 4, pdc - base_pivots_D / 2
        #                 s5_D, s6_D = s4_D - 1.168 * (s3_D - s4_D), pdc - (r6_D - pdc)

        #         # Previous month high and low calculation
        #         prev_month = lambda d: 12 if pd.to_datetime(d).month == 1 else pd.to_datetime(d).month - 1
        #         pMc = df_M.loc[pd.to_datetime(df_M['date']).dt.month == prev_month(date), 'close'].iloc[0]
        #         pMh = df_M.loc[pd.to_datetime(df_M['date']).dt.month == prev_month(date), 'high'].iloc[0]
        #         pMl = df_M.loc[pd.to_datetime(df_M['date']).dt.month == prev_month(date), 'low'].iloc[0]

        #         if pMh and pMl and pMc:
        #             base_pivots_M = (pMh - pMl) * 1.1
        #             pp_M = (pMh + pMl + pMc) / 3
        #             r1_M, r2_M, r3_M, r4_M = pMc + base_pivots_M / 12, pMc + base_pivots_M / 6, pMc + base_pivots_M / 4, pMc + base_pivots_M / 2
        #             r5_M, r6_M = r4_M + 1.168 * (r4_M - r3_M), (pMh / pMl) * pMc
        #             s1_M, s2_M, s3_M, s4_M = pMc - base_pivots_M / 12, pMc - base_pivots_M / 6, pMc - base_pivots_M / 4, pMc - base_pivots_M / 2
        #             s5_M, s6_M = s4_M - 1.168 * (s3_M - s4_M), pMc - (r6_M - pMc)

        #             pdp_list_M.append([pp_M, s1_M, s2_M, s3_M, s4_M, s5_M, s6_M, r1_M, r2_M, r3_M, r4_M, r5_M, r6_M])
        #         else:
        #             pdp_list_M.append([None, None, None, None, None, None, None, None, None, None, None, None, None])


        #         if index_start_postmarket == i:
        #             pdc_list.append(None)
        #         else:
        #             pdc_list.append(pdc)

        #         if index_new_day == i:
        #             pdh_list.append(None)
        #             pdl_list.append(None)
        #             pdp_list.append([None, None, None, None, None, None, None, None, None, None, None, None, None])
        #         else:
        #             pdh_list.append(pdh)
        #             pdl_list.append(pdl)
        #             pdp_list.append([pp, s1, s2, s3, s4, s5, s6, r1, r2, r3, r4, r5, r6])

        #         if index_end_premarket == i:
        #             pmh_list.append(None)
        #             pml_list.append(None)
        #             pdp_list_D.append([None, None, None, None, None, None, None, None, None, None, None, None, None])
        #         else:
        #             pmh_list.append(pmh)
        #             pml_list.append(pml)
        #             pdp_list_D.append([pp_D, s1_D, s2_D, s3_D, s4_D, s5_D, s6_D, r1_D, r2_D, r3_D, r4_D, r5_D, r6_D])

        #     levels_list = [pdc_list, pdh_list, pdl_list, pmh_list, pml_list] + [list(x) for x in zip(*pdp_list)] + [list(x) for x in zip(*pdp_list_D)] + [list(x) for x in zip(*pdp_list_M)]
        #     levels_titles = ['pdc', 'pdh', 'pdl', 'pmh', 'pml', 'pp', 's1', 's2', 's3', 's4', 's5', 's6', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'pp_D', 's1_D', 's2_D', 's3_D', 's4_D', 's5_D', 's6_D', 'r1_D', 'r2_D', 'r3_D', 'r4_D', 'r5_D', 'r6_D', 'pp_M', 's1_M', 's2_M', 's3_M', 's4_M', 's5_M', 's6_M', 'r1_M', 'r2_M', 'r3_M', 'r4_M', 'r5_M', 'r6_M']

        #     for i, level in enumerate(levels_list):
        #         df_level = pd.DataFrame(level, columns=[levels_titles[i]])
        #         df[levels_titles[i]] = df_level[levels_titles[i]]


            # print(df.to_string())
            # print(df[["date", "close", "pdc", "pdh", "pdl", "pmh", "pml"]].to_string())
            # input()


        # df["diff"] = bars_df.close.diff()
        # df["delta"] = bars_df.close - close_previous_day

    # else: print("\nCould not add indicators, DataFrame is empty.\n")

    # return df
