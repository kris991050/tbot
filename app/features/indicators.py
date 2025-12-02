import os, sys, pandas as pd, ta, numpy as np, traceback, tqdm, arch
from datetime import timedelta
from ib_insync import *

current_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_folder)
from utils import helpers
from utils.constants import CONSTANTS
from utils.timeframe import Timeframe
from data import hist_market_data_handler
from features import levels_pivots

# Doc ib_insync: https://ib-insync.readthedocs.io/api.html
# Doc IBKR API: https://interactivebrokers.github.io/tws-api/introduction.html
# util.startLoop()  # uncomment this line when in a notebook
# Doc TA library: https://technical-analysis-library-in-python.readthedocs.io/en/latest/
# Available timframes and durations for historical data: https://interactivebrokers.github.io/tws-api/historical_bars.html


# ======================
# Indicators
# ======================

class Indicators:
    def __init__(self, df, ib, symbol, types:list=['all'], rsi_window:int=14, adx_window:int=14, atr_window:int=14, vol_window:int=20, 
                 vpa_window:int=20, bb_window:int=20, bb_dev:int=2, pred_vlty_type:str='ewma', 
                 macd_windows:dict={'slow':26, 'fast':12, 'sign':9, 'roll':20}, awesome_windows:dict={'slow':34, 'fast':5}):
        self.df = df
        self.ib = ib
        self.symbol = symbol or (self.df.attrs['symbol'] if 'symbol' in self.df.attrs else None)
        self.contract, self.symbol = helpers.get_symbol_contract(self.ib, self.symbol)
        self.types = types
        self.rsi_window = rsi_window
        self.adx_window = adx_window
        self.atr_window = atr_window
        self.vol_window = vol_window
        self.vpa_window = vpa_window
        self.bb_window = bb_window
        self.bb_dev = bb_dev
        self.macd_windows = macd_windows
        self.awesome_windows = awesome_windows
        self.pred_vlty_type = pred_vlty_type

        # Store original attributes to preserve them in final df
        self.attrs = df.attrs
        self.df = helpers.format_df_date(self.df)
        self.tf = helpers.get_df_timeframe(self.df)
        self._add_time_related_columns()
        self.df_day = self._calculate_daily_df()

    def _add_time_related_columns(self):
        # Add day and month info columns
        if 'date_D' not in self.df.columns:
            self.df['date_D'] = self.df['date'].dt.date

        if self.tf.to_timedelta < timedelta(days=1):
            self.df = IndicatorsUtils.add_market_sessions(self.df)
            self.df['hour_of_day'] = self.df['date'].dt.hour
            self.df['day_of_week'] = self.df['date'].dt.dayofweek  # (0 = Monday, 6 = Sunday)
            self.df['low_of_day'] = self.df.groupby('date_D')['low'].cummin()
            self.df['high_of_day'] = self.df.groupby('date_D')['high'].cummax()

        if self.tf.to_timedelta < timedelta(days=30):
            self.df['week#'] = self.df['date'].dt.isocalendar().week

    def _calculate_daily_df(self):
        df_day = pd.DataFrame()
        if self.tf.to_timedelta < timedelta(days=1):
            print(f"Fetching additional data for daily based indicators...")
            # Create daily df
            df_day = helpers.get_daily_df(self.df)
            query_time = df_day['date'].iloc[0]

            timeframe_1D = Timeframe('1D')
            from_time_1D = Timeframe('1M').subtract_from_date(query_time)
            # lookback_start = query_time - lookback_td
            fetcher = hist_market_data_handler.HistMarketDataFetcher(ib=self.ib, ftype='auto', timeframe=timeframe_1D, save_to_file=False, delete_existing=False)
            params = {'symbol': self.contract.symbol, 'timeframe': timeframe_1D, 'to_time': query_time, 'from_time': from_time_1D, 'step_duration': 'auto'}
            df_day_additional = fetcher.run(params)[self.contract.symbol]['df']
            df_day = pd.concat([df_day, df_day_additional], ignore_index=True).drop_duplicates(subset=['date'])
            df_day.sort_values(by=['date'], ascending=True, inplace=True)
            df_day.reset_index(inplace=True, drop=True)
            df_day['date_D'] = df_day['date'].dt.date

        return df_day

    def _apply_trend_indicators(self):
        print(f"Applying trend indicators for {self.tf}...")
        self.apply_ema_indicators()
        self.apply_adx()
        self.apply_macd()
        # if any(item in self.types for item in ['all', 'trend', 'trend']):
        #     self.df = self._apply_trend(self.df)
        # if any(item in self.types for item in ['all', 'index_trend', 'trend']):
        #     self.df = self._apply_index_trend(self.df)

    def _apply_volume_indicators(self):
        print(f"Applying volume indicators for {self.tf}...")
        self.apply_vwap()
        self.apply_r_vol()
        self.apply_pm_vol()

    def _apply_volatility_indicators(self):
        print(f"Applying volatility indicators for {self.tf}...")
        self.apply_bollinger_bands()
        self.apply_atr()
        self.apply_volatility_ratio()
        # self.apply_pred_vlty()

    def _apply_momentum_indicators(self):
        print(f"Applying momentum indicators for {self.tf}...")
        self.apply_rsi()
        self.apply_awesome()

    def _apply_price_indicators(self):
        print(f"Applying price indicators for {self.tf}...")
        self.apply_gap()
        self.apply_change()

    def apply_indicators(self):
        if any(item in self.types for item in ['all', 'trend']):
            self._apply_trend_indicators()
        if any(item in self.types for item in ['all', 'volume']):
            self._apply_volume_indicators()
        if any(item in self.types for item in ['all', 'volatility']):
            self._apply_volatility_indicators()
        if any(item in self.types for item in ['all', 'momentum']):
            self._apply_momentum_indicators()
        if any(item in self.types for item in ['all', 'price']):
            self._apply_price_indicators()
        if any(item in self.types for item in ['all', 'vpa']):
            self.apply_vpa()

        # Reapply attributes to preserve original df attributes
        if self.df.attrs != self.attrs:
            self.df.attrs = self.attrs
        return self.df

    def apply_ema_indicators(self):
        if self.df.empty: return self.df
        ema_params = [(9, 'ema9'), (20, 'ema20'), (50, 'sma50'), (200, 'sma200')]
        for ma_window, prefix in ema_params:
            ema_indicator = ta.trend.EMAIndicator(close=self.df['close'], window=ma_window, fillna=False)
            self.df[f"{prefix}_{self.tf}"] = ema_indicator.ema_indicator()
            self.df[f"{prefix}_slope_{self.tf}"] = self.df[f"{prefix}_{self.tf}"].diff()
        return self.df

    def apply_vwap(self):
        if self.df.empty or self.tf.to_timedelta >= timedelta(days=1): return self.df
        # Calculate window parameter as the average candle per day in the whole df
        vwap_window = int(np.floor(np.average([len(group) for date, group in self.df.groupby(self.df['date'].dt.date)])))
        vwap_indicator = ta.volume.VolumeWeightedAveragePrice(high=self.df["high"], low=self.df["low"], close=self.df["close"],
                                                              volume=self.df["volume"], window=vwap_window, fillna=False)
        self.df[f"vwap_{self.tf}"] = vwap_indicator.volume_weighted_average_price()
        return self.df

    def apply_macd(self, macd_windows:dict=None):
        if self.df.empty: return self.df
        macd_windows = macd_windows if macd_windows else self.macd_windows
        macd_indicator = ta.trend.MACD(close=self.df["close"], window_slow=self.macd_windows['slow'], window_fast=self.macd_windows['fast'],
                                       window_sign=self.macd_windows['sign'], fillna=False)
        self.df[f"macd_{self.tf}"] = macd_indicator.macd()
        self.df[f'macd_signal_{self.tf}'] = macd_indicator.macd_signal()
        self.df[f'macd_diff_{self.tf}'] = macd_indicator.macd_diff()
        self.df[f'macd_z_score_{self.tf}'] = (self.df[f'macd_{self.tf}'] - self.df[f'macd_{self.tf}'].rolling(window=self.macd_windows['roll']).mean()) / self.df[f'macd_{self.tf}'].rolling(window=self.macd_windows['roll']).std()
        return self.df

    def apply_rsi(self):
        if self.df.empty: return self.df
        rsi_indicator = ta.momentum.RSIIndicator(close=self.df['close'], window=self.rsi_window, fillna=False)
        self.df[f'rsi_{self.tf}'] = rsi_indicator.rsi()
        self.df[f'rsi_slope_{self.tf}'] = self.df[f'rsi_{self.tf}'].diff()
        return self.df

    def apply_adx(self):
        if self.df.empty: return self.df
        adx_indicator = ta.trend.ADXIndicator(high=self.df['high'], low=self.df['low'], close=self.df['close'], window=self.adx_window)
        self.df[f'adx_{self.tf}'] = adx_indicator.adx()
        self.df[f'adx_pos_{self.tf}'] = adx_indicator.adx_pos()
        self.df[f'adx_neg_{self.tf}'] = adx_indicator.adx_neg()
        return self.df

    def apply_awesome(self):
        if self.df.empty: return self.df
        awesome_indicator = ta.momentum.AwesomeOscillatorIndicator(high=self.df['high'], low=self.df['low'], window1=self.awesome_windows['fast'],
                                                                   window2=self.awesome_windows['slow'])
        self.df[f'awesome_{self.tf}'] = awesome_indicator.awesome_oscillator()
        return self.df

    def apply_bollinger_bands(self):
        if self.df.empty: return self.df
        bb_indicator = ta.volatility.BollingerBands(close=self.df["close"], window=self.bb_window, window_dev=self.bb_dev, fillna=False)
        self.df[f'bband_h_{self.tf}'] = bb_indicator.bollinger_hband()
        self.df[f'bband_l_{self.tf}'] = bb_indicator.bollinger_lband()
        self.df[f'bband_mavg_{self.tf}'] = bb_indicator.bollinger_mavg()
        self.df[f'bband_z_score_{self.tf}'] = (self.df['close'] - self.df[f'bband_mavg_{self.tf}']) / (self.df[f'bband_h_{self.tf}'] - self.df[f'bband_mavg_{self.tf}'])
        self.df[f'bband_width_ratio_{self.tf}'] = (self.df[f'bband_h_{self.tf}'] - self.df[f'bband_l_{self.tf}']) / self.df['close']
        # self.df["bband_h_ind"] = indicator_bbands.bollinger_hband_indicator()
        # self.df["bband_l_ind"] = indicator_bbands.bollinger_lband_indicator()
        return self.df

    def apply_atr(self):
        if self.df.empty: return self.df
        atr_indicator = ta.volatility.AverageTrueRange(self.df['high'], self.df['low'], self.df['close'], window=self.atr_window, fillna=False)
        self.df[f'atr_{self.tf}'] = atr_indicator.average_true_range()

        if self.tf.to_timedelta < timedelta(days=1):
            # Getting ATR values from daily candlesticks
            atr_window = min(self.atr_window, len(self.df_day))
            indicator_ATR_D = ta.volatility.AverageTrueRange(high=self.df_day['high'], low=self.df_day['low'], close=self.df_day['close'], window=atr_window, fillna=False)
            self.df_day[f'atr_D_{self.tf}'] = indicator_ATR_D.average_true_range().shift(1) # shit ensure there is no info leak from today's data

            # Adding ATR values to main df dataframe
            # df_day.rename(columns={'date': 'date_D'}, inplace=True)
            self.df = pd.merge(self.df, self.df_day[['date_D', f'atr_D_{self.tf}']], on='date_D', how='left')

            self.df[f'~atr_D_band_high_{self.tf}'] = self.df['low_of_day'] + self.df[f'atr_D_{self.tf}']
            self.df[f'~atr_D_band_low_{self.tf}'] = self.df['high_of_day'] - self.df[f'atr_D_{self.tf}']

            # Get pre-market open (day open at 04:00)
            day_open = self.df[self.df['session'] == 'pre-market'].groupby('date_D')['close'].first().rename('day_open')
            self.df = self.df.merge(day_open, on='date_D', how='left')
            self.df[f'atr_D_from_open_scaled_{self.tf}'] = (self.df['close'] - self.df['day_open']) / self.df[f'atr_D_{self.tf}']
            self.df[f'atr_D_from_mid_scaled_{self.tf}'] = (self.df['close'] - (self.df['high_of_day'] + self.df['low_of_day']) / 2) / self.df[f'atr_D_{self.tf}']
            self.df[f'atr_D_from_low_scaled_{self.tf}'] = (self.df['close'] - self.df['low_of_day']) / self.df[f'atr_D_{self.tf}']
            self.df[f'atr_D_from_high_scaled_{self.tf}'] = (self.df['close'] - self.df['high_of_day']) / self.df[f'atr_D_{self.tf}']
            self.df[f'day_range_{self.tf}'] = (self.df['high_of_day'] - self.df['low_of_day']) / self.df[f'atr_D_{self.tf}']

            self.df_day[f'atr_D_change_{self.tf}'] = self.df_day[f'atr_D_{self.tf}'].pct_change()
            self.df = self.df.merge(self.df_day[['date_D', f'atr_D_change_{self.tf}']], on='date_D', how='left')

            self.df = self.df.drop('day_open', axis=1)
        return self.df

    def apply_volatility_ratio(self):
        if self.df.empty: return self.df
        self.df[f'volatility_ratio_{self.tf}'] = self.df.volume / (self.df.volume - self.df.volume.diff())
        self.df[f'volatility_change_{self.tf}'] = self.df['volume'].pct_change()
        return self.df
    
    def apply_pred_vlty(self):
        if self.df.empty: return self.df
        lookback_period = Timeframe(CONSTANTS.WARMUP_MAP.get(self.tf.pandas, None)).to_timedelta
        close_series = self.df[['date', 'close']].set_index('date', inplace=False)
        self.df[f'pred_vlty_{self.tf}'] = IndicatorsUtils.calculate_pred_vlty_recursive(close_series=close_series, window=lookback_period, 
                                                                                                              type=self.pred_vlty_type)
        return self.df

    def apply_vpa(self):
        if self.df.empty: return self.df
        self.df[f'volatility_{self.tf}'] = (self.df['high'] - self.df['low']) / self.df['close']
        self.df[f'body_{self.tf}'] = abs(self.df['close'] - self.df['open'])
        self.df[f'vpa_ratio_h_{self.tf}'] = self.df['volume'] / self.df[f'body_{self.tf}']
        self.df[f'vpa_ratio_l_{self.tf}'] = self.df[f'body_{self.tf}'] / self.df['volume']
        self.df[f'vpa_z_score_h_{self.tf}'] = (self.df[f'vpa_ratio_h_{self.tf}'] - self.df[f'vpa_ratio_h_{self.tf}'].rolling(window=self.vpa_window, min_periods=1).mean()) / self.df[f'vpa_ratio_h_{self.tf}'].rolling(window=self.vpa_window, min_periods=1).std()
        self.df[f'vpa_z_score_l_{self.tf}'] = (self.df[f'vpa_ratio_l_{self.tf}'] - self.df[f'vpa_ratio_l_{self.tf}'].rolling(window=self.vpa_window, min_periods=1).mean()) / self.df[f'vpa_ratio_l_{self.tf}'].rolling(window=self.vpa_window, min_periods=1).std()
        return self.df

    def apply_r_vol(self):
        if self.df.empty: return self.df
        self.df[f'avg_vol_{self.tf}'] = self.df['volume'].rolling(window=self.vol_window).mean()
        self.df[f'r_vol_{self.tf}'] = self.df['volume'] / self.df[f'avg_vol_{self.tf}']
        self.df.set_index('date', inplace=True)

        if self.tf.to_timedelta < timedelta(days=1):
            self.df[f'avg_vol_at_time_{self.tf}'] = self.df.groupby(self.df.index.time).apply(lambda d: d['volume'].rolling(self.vol_window, min_periods=1).mean()).reset_index(level=0, drop=True).sort_index()
            self.df[f'r_vol_at_time_{self.tf}'] = self.df['volume'] / self.df[f'avg_vol_at_time_{self.tf}']

        self.df = self.df.reset_index()
        return self.df

    def apply_pm_vol(self):
        if self.df.empty: return self.df
        if self.tf.to_timedelta < timedelta(days=1):
            # Total and relative pre-market volume per day
            pm_vol = self.df[self.df['session'] == 'pre-market'].groupby('date_D')['volume'].sum().rename(f'pm_vol_{self.tf}')
            pm_vol_df = pm_vol.to_frame()
            pm_vol_df[f'pm_r_vol_{self.tf}'] = (pm_vol_df[f'pm_vol_{self.tf}'] / pm_vol_df[f'pm_vol_{self.tf}'].rolling(self.vol_window).mean())
            self.df = self.df.merge(pm_vol_df, on='date_D', how='left')

            # Mask pre-market rows with NaN
            self.df.loc[self.df['session'] == 'pre-market', [f'pm_vol_{self.tf}', f'pm_r_vol_{self.tf}']] = np.nan
        return self.df

    def apply_gap(self):
        if self.df.empty or self.tf.to_timedelta > timedelta(days=1): return self.df

        if self.tf.to_timedelta == timedelta(days=1):
            # Calculate the gap as the percentage difference between today's open and the previous day's close
            gap = (100 * (self.df['open'] - self.df['close'].shift(1)) / self.df['close'].shift(1)).rename(f'gap_{self.tf}')
            self.df = self.df.assign(**{f'gap_{self.tf}': gap})  # Add the gap column directly to df
            if 'rth' in self.tf.pandas:
                self.df.rename(columns={f'gap_{self.tf}': f'gap_rth_{self.tf}'}, inplace=True)

        elif self.tf.to_timedelta < timedelta(days=1):
            sessions = self.df['session'].unique()
            session_open = self.df.groupby(['date_D', 'session'])['close'].first().unstack()
            session_close = self.df.groupby(['date_D', 'session'])['close'].last().unstack()

            # Calculate 'gap' if both 'pre-market' and 'post-market' exist
            if {'pre-market', 'post-market'}.issubset(sessions):
                gap = (100 * (session_open['pre-market'] - session_close['post-market'].shift(1)) / session_close['post-market'].shift(1)).rename(f'gap_{self.tf}')
                self.df = self.df.merge(gap, on='date_D', how='left')
            else:
                self.df[f'gap_{self.tf}'] = np.nan

            # Calculate 'gap_rth' if 'rth' exists
            if 'rth' in sessions:
                gap_rth = (100 * (session_open['rth'] - session_close['rth'].shift(1)) / session_close['rth'].shift(1)).rename(f'gap_rth_{self.tf}')
                self.df = self.df.merge(gap_rth, on='date_D', how='left')
            else:
                self.df[f'gap_rth_{self.tf}'] = np.nan

            # Mask gap_rth during pre-market
            self.df.loc[self.df['session'] == 'pre-market', f'gap_rth_{self.tf}'] = np.nan

        return self.df

    def apply_change(self):
        if self.df.empty or self.tf.to_timedelta >= timedelta(days=1): return self.df
        if not '~pdc' in self.df.columns:
            print("Daily levels needed for apply_change feature. Acquiring daily levels...")
            df_temp = levels_pivots.LevelsAndPivotsCalculator(self.df.copy(), self.ib, self.symbol, level_types=['daily'], 
                                                              drop_levels=False, save_levels=False, next_levels=False).apply_levels()
            # levels_list = [{'name':'~pdc', 'col':'close', 'funct':'last', 'shift':1}]
            # df_rth = self.df[self.df['session'] == 'rth']
            # self.df = levels_pivots.LevelsAndPivotsCalculator._get_levels_by_resampling(self.df, levels_list, '1D', df2=df_rth)
        self.df[f"change_{self.tf}"] = (100 * self.df.close / df_temp['~pdc']) - 100
        self.df[f"change_diff_{self.tf}"] = self.df[f"change_{self.tf}"].diff()
        # self.df = self.df.drop(['~pmh', '~pml', '~pdh', '~pdl', '~pdc', '~pdh_D', '~pdl_D', 'do'], axis=1)
        return self.df

    # def _apply_trend(self, df):
    #     # Assess trend
    #     df_temp = df.copy()
    #     df_temp = self._apply_macd(df_temp, macd_windows={'slow':50, 'fast':20, 'sign':self.macd_windows['sign'], 'roll':self.macd_windows['roll']}).rename(columns={'macd_diff': 'macd_diff_fast'}, inplace=False)
    #     df_temp = self._apply_macd(df_temp, macd_windows={'slow':200, 'fast':50, 'sign':50, 'roll':self.macd_windows['roll']}).rename(columns={'macd_diff': 'macd_diff_slow'}, inplace=False)
    #     if 'rsi' not in df_temp.columns: df_temp = self._apply_rsi(df_temp, rsi_window=self.rsi_window)
    #     if 'adx' not in df_temp.columns: df_temp = self._apply_adx(df_temp, adx_window=self.adx_window)

    #     # Trend analysis based on RSI and ADX
    #     df_temp['trend'] = np.where((df_temp['trend_short'] == 1) & (df_temp['trend_long'] == 1) & (df_temp['adx'] > 25) & (df_temp['rsi'] < 70), 1,
    #                                       np.where((df_temp['trend_short'] == -1) & (df_temp['trend_long'] == -1) & (df_temp['adx'] > 25) & (df_temp['rsi'] > 30), -1, 0))

    #     # Short-Term Trend Signal based on MACD
    #     condition_short_term_up = (df_temp['macd_diff_fast'] > df_temp['macd_diff_fast'].shift()) & (df_temp['macd_diff_fast'].shift() > df_temp['macd_diff_fast'].shift(2))
    #     condition_short_term_down = (df_temp['macd_diff_fast'] < df_temp['macd_diff_fast'].shift()) & (df_temp['macd_diff_fast'].shift() < df_temp['macd_diff_fast'].shift(2))
    #     df_temp['index_trend_fast'] = np.where(condition_short_term_up, 1, np.where(condition_short_term_down, -1, 0))

    #     # Long-Term Trend Signal based on MACD
    #     condition_long_term_up = (df_temp['macd_diff_slow'] > df_temp['macd_diff_slow'].shift()) & (df_temp['macd_diff_slow'].shift() > df_temp['macd_diff_slow'].shift(2))
    #     condition_long_term_down = (df_temp['macd_diff_slow'] < df_temp['macd_diff_slow'].shift()) & (df_temp['macd_diff_slow'].shift() < df_temp['macd_diff_slow'].shift(2))
    #     df_temp['index_trend_slow'] = np.where(condition_long_term_up, 1, np.where(condition_long_term_down, -1, 0))

    #     # Adding trend strength (for short-term)
    #     df_temp['index_trend_strength_fast'] = df_temp['macd_diff_fast'] - df_temp['macd_diff_fast'].shift(1)
    #     df_temp['index_trend_strength_slow'] = df_temp['macd_diff_slow'] - df_temp['macd_diff_slow'].shift(1)

    #     # condition_index_trend_up = (df_index['macd_diff'] > df_index['macd_diff'].shift()) & (df_index['macd_diff'].shift() > df_index['macd_diff'].shift(2))
    #     # condition_index_trend_down = (df_index['macd_diff'] < df_index['macd_diff'].shift()) & (df_index['macd_diff'].shift() < df_index['macd_diff'].shift(2))
    #     # df_index['index_trend'] = np.where(condition_index_trend_up, 1, np.where(condition_index_trend_down, -1, 0))

    #     # df_index['index_trend'] = np.where((df_index['macd_diff'] > 0) & (df_index['rsi'] > 50) & (df_index['adx'] > 25), 1,
    #     #                                    np.where((df_index['macd_diff'] < 0) & (df_index['rsi'] < 50) & (df_index['adx'] > 25), -1, 0))
    #     # df_index['index_trend_strength'] = df_index['macd_diff'] - df_index['macd_diff'].shift(1)

    #     # # Combining Short-Term and Long-Term Trends
    #     # df_index['index_trend'] = np.where((df_index['trend_fast'] == 1) & (df_index['trend_slow'] == 1), 1,
    #     #                                    np.where((df_index['trend_fast'] == -1) & (df_index['trend_slow'] == -1), -1, 0))

    #     df = pd.merge(df, df_temp[['date', 'index_trend', 'index_trend_fast', 'index_trend_slow', 'index_trend_strength_fast', 'index_trend_strength_slow']], on='date', how='left')
    #     return df

    # def _apply_index_trend(self, df):
    #     # Get symbol related Index and ETF
    #     symbol_index = helpers.get_index_from_symbol(self.ib, self.contract.symbol)
    #     if len(symbol_index) > 0: symbol_index = symbol_index[0]
    #     symbol_index_ETF = helpers.get_index_etf(symbol_index)

    #     if symbol_index_ETF:
    #         # Gathering Index historical data
    #         query_time = self.timeframe_df.add_to_date(df['date'].iloc[-1])
    #         from_time = df['date'].iloc[0]
    #         index_fetcher = hist_market_data_handler.HistMarketDataFetcher(ib=self.ib, ftype='auto', save_to_file=True, delete_existing=True)
    #         index_params = {'symbol': symbol_index_ETF, 'timeframe': self.timeframe_df, 'from_time': from_time, 'to_time': query_time, 'step_duration': 'auto'}
    #         df_index = index_fetcher.run(params=index_params)[symbol_index_ETF]['df']

    #         df_index = self._apply_trend(df_index)

    #         # Adding the trends to main df dataframe
    #         df = pd.merge(df, df_index[['date', 'index_trend_fast', 'index_trend_slow', 'index_trend_strength_fast', 'index_trend_strength_slow']], on='date', how='left')
    #         return df


def get_bb_rsi_tf_columns(tf: str):
    tf_str = tf if 'min' not in tf else tf[:-3]
    return {
        'rsi': f'rsi_{tf_str}',
        'rsi_slope': f'rsi_slope_{tf_str}',
        'bband_h': f'bband_h_{tf_str}',
        'bband_l': f'bband_l_{tf_str}',
        'bband_mavg': f'bband_mavg_{tf_str}',
        'bband_z_score': f'bband_z_score_{tf_str}',
        'bband_width_ratio': f'bband_width_ratio_{tf_str}'
    }


def add_bb_rsi_mtf(df, ib, contract, timeframes=[], bb_window=20, rsi_window=14):

    df = helpers.format_df_date(df)
    attrs = df.attrs
    tf_seconds = helpers.get_df_timeframe(df).to_seconds

    # List of indicator columns that could conflict
    indicator_cols = ['rsi', 'bband_h', 'bband_l', 'bband_mavg']
    temp_renamed = {}

    # Temporarily rename existing columns that would be overwritten
    for col in indicator_cols:
        if col in df.columns:
            temp_name = f"__tmp_{col}"
            df.rename(columns={col: temp_name}, inplace=True)
            temp_renamed[temp_name] = col

    for tf in timeframes:
        timeframe = Timeframe(tf)
        if timeframe.to_seconds <= tf_seconds:
            print(f"Skipping {tf}, not higher than base timeframe")
            continue

        df_tf = df.resample(timeframe.pandas, on='date').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()

        # df_tf = add_indicator(df_tf, ib, contract, types=['rsi', 'bollinger_bands'], bb_window=bb_window, rsi_window=rsi_window)
        df_tf = Indicators(df_tf, ib, contract, types=['rsi', 'bollinger_bands']).apply_indicators()

        rename_map = get_bb_rsi_tf_columns(tf)
        df_tf.rename(columns=rename_map, inplace=True)
        df = pd.merge_asof(df, df_tf[['date'] + list(rename_map.values())], on='date', direction='backward')

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
    if temp_renamed:
        df.rename(columns={tmp: orig for tmp, orig in temp_renamed.items()}, inplace=True)

    df.attrs = attrs

    return df#.copy()


class IndicatorsUtils:

    @staticmethod
    def get_indicator(df, indicator, query_time):

        query_time_df = helpers.adjust_time_to_df(df, query_time)
        ind = ''

        try:
            if indicator == 'pivots' or indicator == 'pivots_D' or indicator == 'pivots_M':
                ind = {}
                if indicator == 'pivots': pivots_list = ['s1', 's2', 's3', 's4', 's5', 's6', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6']
                elif indicator == 'pivots_D': pivots_list = ['s1_D', 's2_D', 's3_D', 's4_D', 's5_D', 's6_D', 'r1_D', 'r2_D', 'r3_D', 'r4_D', 'r5_D', 'r6_D']
                elif indicator == 'pivots_M': pivots_list = ['s1_M', 's2_M', 's3_M', 's4_M', 's5_M', 's6_M', 'r1_M', 'r2_M', 'r3_M', 'r4_M', 'r5_M', 'r6_M']
                for p in pivots_list:
                    ind[p] = df.loc[df['date'] == query_time_df, p].iloc[0]
            else:
                ind = df.loc[df['date'] == query_time_df, indicator].iloc[0]

        except Exception as e: print("Could not fetch ", indicator, ". Error: ", e, "  |  Full error: ", traceback.format_exc())

        return ind

    @staticmethod
    def get_CPR(df, query_time, pivots):

        char = '_D' if '_D' in list(pivots.keys())[0] else ''

        # Get CPR size and level comparisons
        cpr_size_to_yst_perc, cpr_midpoint_to_yst_perc = '', ''
        try:
            query_time_yst = helpers.get_previous_date_from_df(df, query_time, day_offset=1)
            r2_yst, s2_yst = float(IndicatorsUtils.get_indicator(df, 'r2' + char, query_time_yst)), float(IndicatorsUtils.get_indicator(df, 's2' + char, query_time_yst))
            cpr_size_to_yst_perc = round(((pivots['r2' + char] - pivots['s2' + char]) * 100 / (r2_yst - s2_yst)) - 100, 1)
            cpr_midpoint_to_yst_perc = round((0.5 * (pivots['s2' + char] + pivots['r2' + char]) * 100 / (0.5 * (s2_yst + r2_yst))) - 100, 1)
        except Exception as e: print("Could not calculate CPR. Error: ", e, "  |  Full error: ", traceback.format_exc())

        return cpr_size_to_yst_perc, cpr_midpoint_to_yst_perc

    @staticmethod
    def add_market_sessions(df, th_times=None):
        if df.empty:
            return df
        th_times = th_times or CONSTANTS.TH_TIMES
        timeframe_df = helpers.get_df_timeframe(df)
        if timeframe_df.to_timedelta < timedelta(days=1) and all(col not in df.columns for col in ['pre-market', 'post-market', 'rth']):
            # Define market sessions
            session_conditions = [(df['date'].dt.time >= th_times['pre-market']) & (df['date'].dt.time < th_times['rth']), (df['date'].dt.time >= th_times['post-market']) & (df['date'].dt.time <= th_times['end_of_day'])]
            session_choices = ['pre-market', 'post-market']
            df['session'] = np.select(session_conditions, session_choices, default='rth')
            # df['session'] = ['pre-market' if (t >= th_times['pre-market'] and t < th_times['rth']) else 'post-market' if (t >= th_times['post-market'] and t <= th_times['end_of_day']) else 'rth' for t in df['date'].dt.time]
            # df['session2'] = pd.cut(df.date.dt.hour + df.date.dt.minute / 60.0, bins=[-float('inf'), 4, 9.5, 16, 20, float('inf')], labels=['Pre-market', 'RTH', 'Post-market', 'RTH', 'Post-market'], right=False)
        else:
            print(f"Market sessions not added to df. Timeframe needs to be < 1 day and sessions not already present. Current timeframe: {timeframe_df}.")

        return df

    @staticmethod
    def calculate_next_levels(df, levels_col, show_next_level=False):

        print(f"⏳ Calculating levels derivatives columns for {levels_col}")
        df = df.copy()
        close_prices = df['close'].values
        levels_series = df[levels_col]

        # Preallocate output arrays
        next_up = np.full(len(df), np.nan)
        next_down = np.full(len(df), np.nan)
        dist_up = np.full(len(df), np.nan)
        dist_down = np.full(len(df), np.nan)
        pct_dist_up = np.full(len(df), np.nan)
        pct_dist_down = np.full(len(df), np.nan)

        for i in range(len(df)):
            levels = levels_series.iat[i]
            if levels is None or (isinstance(levels, (list, np.ndarray)) and len(levels) == 0):
                continue

            levels_array = np.array(levels)
            close = close_prices[i]

            ups = levels_array[levels_array > close]
            downs = levels_array[levels_array < close]

            if ups.size > 0:
                next_up[i] = ups.min()
                dist_up[i] = next_up[i] - close
                pct_dist_up[i] = dist_up[i] / close

            if downs.size > 0:
                next_down[i] = downs.max()
                dist_down[i] = close - next_down[i]
                pct_dist_down[i] = dist_down[i] / close

        # Normalized position in SR range: -1 (support) to +1 (resistance)
        position_in_range = np.full(len(df), np.nan)
        valid_mask = (~np.isnan(next_up)) & (~np.isnan(next_down)) & ((next_up - next_down) != 0)
        position_in_range[valid_mask] = 2 * ((close_prices[valid_mask] - next_down[valid_mask]) /
                                            (next_up[valid_mask] - next_down[valid_mask])) - 1

        # Assign back to DataFrame
        # string_to_remove_end = '_list'
        # levels_col_modif = levels_col[:-len(string_to_remove_end)] if levels_col.endswith(string_to_remove_end) else levels_col
        # string_to_remove_start = '~'
        # levels_col_modif = levels_col_modif[len(string_to_remove_start):] if levels_col.startswith(string_to_remove_start) else levels_col_modif
        levels_col_modif = levels_col.lstrip('~').removesuffix('_list')
        df[f'{levels_col_modif}_pos_in_range'] = position_in_range
        if show_next_level:
            df[f"next_{levels_col_modif}_up"] = next_up
            df[f"next_{levels_col_modif}_down"] = next_down
        df[f"{levels_col_modif}_dist_to_next_up"] = dist_up
        df[f"{levels_col_modif}_dist_to_next_down"] = dist_down
        # df[f"{levels_col}_pct_dist_to_next_up"] = pct_dist_up
        # df[f"{levels_col}_pct_dist_to_next_down"] = pct_dist_down

        return df

    @staticmethod
    def get_level_compatible_df(ib:IB, symbol:str, df:pd.DataFrame, timeframe:Timeframe, lookback:str, file_format:str, hist_folder:str) -> pd.DataFrame:
        """
        Ensure df has enough history for level calculation (e.g., 1 day or 1 month back).
        If df_levels is missing or not long enough, fetch additional data.
        """

        # lookback_td = helpers.parse_timedelta(lookback)
        # lookback_start = df['date'].min() - lookback_td
        lookback_start = Timeframe(lookback).subtract_from_date(df['date'].min())

        # Find exisintg hist data for level calculation
        for tf_i in CONSTANTS.TIMEFRAMES_STD:
            tf = Timeframe(tf_i)
            if tf.to_seconds <= timeframe.to_seconds:
                df_existing_hist, existing_hist_to, existing_hist_from, _ = \
                    helpers.check_existing_data_file(symbol, tf, hist_folder, data_type='hist_data', delete_file=False, file_format=file_format)
                if existing_hist_from and existing_hist_to and existing_hist_from <= lookback_start and existing_hist_to >= df['date'].max():
                    return helpers.format_df_date(df_existing_hist, col='date', set_index=True)
        
        # Fetch new data if no sufficient existing data
        print(f"Fetching additional data for level calculation. Timeframe: {timeframe}...")
        save_levels_to_file = helpers.get_df_timeframe(df).to_seconds != timeframe.to_seconds
        fetcher = hist_market_data_handler.HistMarketDataFetcher(ib=ib, ftype='auto', timeframe=timeframe, file_format=file_format,
                                                                 save_to_file=save_levels_to_file, delete_existing=save_levels_to_file)

        params = {'symbol': symbol, 'timeframe': timeframe, 'to_time': df['date'].min(), 'from_time': lookback_start, 'step_duration': Timeframe(lookback).pandas}

        df_fetched = fetcher.run(params)[symbol]['df']

        if df_fetched is not None and not df_fetched.empty:
            df_fetched = helpers.format_df_date(df_fetched, set_index=False)
            df_resampled = df.resample(timeframe.pandas, on='date').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna().reset_index()
            df_combined = pd.concat([df_fetched, df_resampled], axis=0).drop_duplicates(subset=['date']).sort_values(by=['date'], ascending=True)

            return helpers.format_df_date(df_combined, col='date', set_index=True)

        print(f"Warning: Could not fetch data for {timeframe}. Returning original dataframe")#empty dataframe")
        return df#pd.DataFrame()

    @staticmethod
    def detect_last_sr_change(df: pd.DataFrame, offset=None) -> dict:
        """
        Detects if SR values have not changed in the window of (refresh_rate - offset) for each timeframe.
        If unchanged, returns True for that timeframe, meaning SR should be recalculated.
        """
        add_sr = {}

        if df.empty or 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise ValueError("DataFrame must have a non-empty datetime 'date' column.")

        # if df.empty or not isinstance(df['date'], datetime.datetime):
        #     raise ValueError("DataFrame must have a DatetimeIndex and not be empty.")

        now = df['date'].iloc[-1]

        for setting in CONSTANTS.SR_SETTINGS:
            timeframe = Timeframe(setting['timeframe'])
            refresh_rate_str = setting['refresh_rate']
            sr_column = f"sr_{timeframe}_list"

            if sr_column not in df.columns:
                add_sr[timeframe] = True
                continue

            # Convert refresh_rate string to Timedelta
            try:
                # refresh_delta = pd.to_timedelta(pd.tseries.frequencies.to_offset(refresh_rate_str))
                refresh_offset = pd.tseries.frequencies.to_offset(refresh_rate_str)
            except Exception as e:
                raise ValueError(f"Invalid refresh_rate: {refresh_rate_str}") from e

            # Define time window: last `refresh_rate` worth of data
            window_start = now - refresh_offset + offset
            # window_start = now - (refresh_delta - offset)
            # recent_df = df.loc[window_start:now]
            recent_df = df[(df['date'] >= window_start) & (df['date'] <= now)]

            if recent_df.empty:
                add_sr[timeframe] = True
                continue

            sr_values = recent_df[sr_column]

            def safe_to_tuple(x):
                if isinstance(x, np.ndarray): return tuple(x.tolist())
                elif isinstance(x, list): return tuple(x)
                else: return x  # NaN, None, float, etc.

            sr_values_as_tuples = sr_values.apply(safe_to_tuple)

            # Add S/R only if all values not equals
            add_sr[timeframe] = True if sr_values_as_tuples.nunique(dropna=False) <= 1 else False

        return add_sr
    
    @staticmethod
    def save_levels(df:pd.DataFrame, timeframe:Timeframe, symbol:str, col:str, hist_folder:str, data_type:str, file_format:str='csv'):
        print(f"💾 Saving {data_type} levels to separate Dataframe...")
        hist_folder_symbol = os.path.join(hist_folder, symbol)
        data_path = helpers.construct_data_path(hist_folder_symbol, symbol, timeframe, to_time=df['date'].iloc[-1], from_time=df['date'].iloc[0], 
                                                file_format=file_format, data_type=data_type)
        helpers.save_df_to_file(df[['date', col]], data_path, file_format=file_format)
        return df
    
    @staticmethod
    def resolve_sr_timeframes(timeframe:Timeframe, sr_tfs:list=None):
        sr_tfs = sr_tfs if sr_tfs else [sr_setting['timeframe'] for sr_setting in CONSTANTS.SR_SETTINGS]
        sr_tfs = sorted(sr_tfs, key=lambda tf: Timeframe(tf).to_seconds)
        sr_tfs = list(dict.fromkeys(sr_tfs))  # Remove duplicates, while keeping elements order

        tf_to_remove = []
        for tf in sr_tfs:
            timeframe_sr = Timeframe(tf)
            if timeframe_sr.to_seconds <= timeframe.to_seconds:
                print(f"Skipping timeframe {timeframe_sr}, as < current timeframe {timeframe}")
                tf_to_remove.append(tf)
                continue
        sr_tfs = [tf for tf in sr_tfs.copy() if tf not in tf_to_remove]
        return sr_tfs
    
    @staticmethod
    def calculate_pred_vlty_recursive(close_series:pd.Series, window:timedelta, p:int=1, q:int=1, lambda_:float=0.94, type:str='ewma'):
        """
        Calculate prediucted volatility recursively for each row using a rolling window defined by timedelta.
        
        :param close_series: The pandas Series of historical close prices with a datetime index.
        :param window: A timedelta object representing the rolling window size.
        :param p: The order of the GARCH model (GARCH(p, q)).
        :param q: The order of the GARCH model (GARCH(p, q)).
        :param lambda_: Decay factor for the EWMA model.
        :return: A pandas Series with the GARCH volatility for each row.
        """
        print(f"🎢 Calculating predicted volatility recursively using {type.upper()} method with a time window of {window}...")
        
        # List to store the volatilities for each row
        vol_list = [np.nan] * len(close_series)
        prev_volatility = 0.0 # For calculation with EWMA method

        # Iterate through each row, starting from the point where the window is fully populated
        # for i in range(len(close_series)):
        for i in tqdm.tqdm(range(len(close_series)), desc="Calculating Predicted Volatility", ncols=100, ascii=True):
            window_start_time = close_series.index[i] - window

            # Select data in the window: filter out rows where timestamp is older than window_start_time
            window_prices = close_series[close_series.index >= window_start_time].iloc[:i+1]
            
            # If there is enough data in the window to fit the GARCH model, calculate volatility
            if len(window_prices) > 1:  # Ensure there's more than 1 data point to calculate returns
                if type == 'garch':
                    prev_volatility = IndicatorsUtils.calculate_garch_volatility(window_prices, p=p, q=q)
                elif type == 'ewma':
                    prev_volatility = IndicatorsUtils.calculate_ewma_volatility(window_prices, lambda_=lambda_, prev_volatility=prev_volatility)
                vol_list[i] = prev_volatility
        
        # Return the volatility as a pandas Series
        return pd.Series(vol_list)#, index=close_series.index)
    
    @staticmethod
    def calculate_garch_volatility(price_data:pd.Series, p:int=1, q:int=1, forecast_horizon:int=1) -> float:
        """
        Calculate the volatility using a GARCH model.

        :param price_data: A pandas Series of historical price data.
        :param p: The order of the GARCH model (GARCH(p, q)).
        :param q: The order of the GARCH model (GARCH(p, q)).
        :param forecast_horizon: The number of steps ahead to forecast volatility.
        :return: Forecasted volatility.
        """
        # Calculate returns
        returns = price_data.pct_change().dropna() * 100  # Percentage returns
        returns_scaled = returns * 10  # Scaling factor recommended to avoid model convergence issues

        # Fit GARCH model
        model = arch.arch_model(returns_scaled, vol='Garch', p=p, q=q)
        model_fit = model.fit(disp="off")

        # Forecast volatility for the next forecast_horizon periods
        forecast = model_fit.forecast(horizon=forecast_horizon)
        pred_vlty = forecast.variance.values[-1, 0]  # Forecasted variance for the last period
        pred_vlty = np.sqrt(pred_vlty)  / 10  # Rescale the volatility  (Volatility is the square root of variance)

        return pred_vlty

    @staticmethod
    def calculate_ewma_volatility(price_data:pd.Series, lambda_:float=0.94, prev_volatility:float=0.0) -> float:
        """
        Calculate the volatility using the EWMA method.

        :param price_data: A pandas Series of historical price data.
        :param lambda_: The smoothing factor (default is 0.94).
        :param prev_volatility: The previous volatility value (default is 0.0 for the first period).
        :return: Forecasted volatility.
        """
        # Calculate returns
        returns = price_data.pct_change().dropna() * 100  # Percentage returns
        squared_returns = returns ** 2

        # Calculate the EWMA volatility
        if prev_volatility == 0.0:  # If it's the first calculation, initialize volatility
            volatility = np.sqrt(squared_returns.mean()).values[0]  # Use the mean of squared returns as initial volatility
        else:
            volatility = np.sqrt(lambda_ * squared_returns.iloc[-1] + (1 - lambda_) * prev_volatility ** 2).values[0]

        return volatility

    @staticmethod
    def calculate_weighted_atr(atr_series:pd.Series, lambda_:float=0.94):
        """
        Calculate a weighted ATR using an exponential moving average (EMA).
        
        :param atr_series: The ATR series.
        :param lambda_: The smoothing factor (default 0.94).
        :return: The weighted ATR series.
        """
        # Apply the EMA to the ATR series
        return atr_series.ewm(span=(2 / (1 - lambda_)) - 1, adjust=False).mean()


if __name__ == "__main__":

    # current_path = os.path.realpath(__file__)
    # path = path_current_setup(current_path)

    input("\nEnter anything to exit")







# @staticmethod
#     def get_level_compatible_df(ib, symbol, df, df_levels, timeframe:Timeframe, lookback, file_format):
#         """
#         Ensure df has enough history for level calculation (e.g., 1 day or 1 month back).
#         If df_levels is missing or not long enough, fetch additional data.
#         """

#         lookback_td = helpers.parse_timedelta(lookback)
#         # lookback_start = df.index.min() - lookback_td
#         lookback_start = df['date'].min() - lookback_td

#         timeframe_df = helpers.get_df_timeframe(df)
#         timeframe_df_levels = helpers.get_df_timeframe(df_levels)
#         timeframe_sec = timeframe.to_seconds
#         timeframe_df_sec = timeframe_df.to_seconds
#         timeframe_df_levels_sec = timeframe_df_levels.to_seconds if timeframe_df_levels else None

#         # Check if df_levels (historical) is sufficient
#         if not df_levels.empty and timeframe_df_levels_sec <= timeframe_sec and timeframe_df_sec <= timeframe_df_levels_sec:
#             if df_levels.index.min() <= lookback_start and df_levels.index.max() >= df.index.max():
#                 return df_levels
#             else:
#                 print(f"Warning: df_levels does not cover required lookback for {timeframe}. Fetching additional data...")

#         # # Find exisintg hist data for level calculation
#         # for tf_i in CONSTANTS.TIMEFRAMES_STD:
#         #     tf = Timeframe(tf_i)
#         #     if tf.to_seconds <= timeframe.to_seconds:
#         #         df_existing_hist, existing_hist_to, existing_hist_from, _ = \
#         #             helpers.check_existing_data_file(symbol, tf, symbol, delete_file=False, file_format=file_format, data_type='hist')
#         #         if existing_hist_from <= lookback_start and existing_hist_to >= df['date'].max():
#         #             return df_existing_hist
        
#         # Fetch new data
#         print(f"Fetching additional data for {timeframe} timeframe...")
#         fetcher = hist_market_data_handler.HistMarketDataFetcher(ib=ib, ftype='auto', timeframe=timeframe, file_format=file_format,
#                                                                  save_to_file=True, delete_existing=True)
#         # fetcher = hist_market_data_handler.HistMarketDataFetcher(ib=ib, ftype='auto', timeframe=timeframe, file_format=file_format,
#         #                                                          save_to_file=True, delete_existing=True)

#         # params = {'symbol': symbol, 'timeframe': timeframe, 'to_time': df.index.min(), 'from_time': lookback_start, 'step_duration': Timeframe(lookback).pandas}
#         params = {'symbol': symbol, 'timeframe': timeframe, 'to_time': df['date'].min(), 'from_time': lookback_start, 'step_duration': Timeframe(lookback).pandas}

#         # #####################################################################
#         # if params['from_time'] == pd.Timestamp('2020-09-21 00:00:00-0400', tz='US/Eastern') or params['to_time'] == pd.Timestamp('2020-09-21 04:00:00-0400', tz='US/Eastern'):
#         #     print("from_time = ", params['from_time'])
#         #     print("to_time = ", params['to_time'])
#         #     print()
#         # #####################################################################


#         df_fetched = fetcher.run(params)[symbol]['df']

#         if df_fetched is not None and not df_fetched.empty:
#             df_fetched = helpers.format_df_date(df_fetched, set_index=False)
#             df_resampled = df.resample(timeframe.pandas, on='date').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna().reset_index()
#             df_combined = pd.concat([df_fetched, df_resampled], axis=0).drop_duplicates(subset=['date']).sort_values(by=['date'], ascending=True)
#             # df_combined = df_combined[~df_combined.index.duplicated(keep='last')]

#             # df_day = pd.concat([df_day, df_day_additional], ignore_index=True).drop_duplicates(subset=['date'])
#             # df_day.sort_values(by=['date'], ascending=True, inplace=True)
#             # df_day.reset_index(inplace=True, drop=True)

#             return helpers.format_df_date(df_combined, col='date', set_index=True)

#         print(f"Warning: Could not fetch data for {timeframe}. Using current df_levels only.")
#         return df_levels


        # if "atr_old" in types:
            # atr_list = [df_day.loc[df_day['date'] == datetime.datetime.strptime(pd.to_datetime(row['date']).strftime('%Y-%m-%d'), '%Y-%m-%d').date(), 'atr_D'].to_frame().iloc[-1]['atr_D'] for index, row in df.iterrows()]
            # df_atr = pd.DataFrame(atr_list, columns=["atr"])
            # df["atr"] = df_atr["atr"]

            # print(df[["date", "close", "atr"]].to_string())
            # print(df_day[["date", "close", "atr"]].to_string())

            # ATR bands
            # # Method 1
            # atr_band_high_list, atr_band_low_list = [], []
            # # date_prev = pd.to_datetime(df.loc[0, "date"]).strftime('%Y-%m-%d')
            # # day_low, day_high = df.loc[0, "low"], df.loc[0, "high"]
            # date_prev = pd.to_datetime(df['date'].iloc[0]).strftime('%Y-%m-%d')
            # day_low, day_high = df['low'].iloc[0], df['high'].iloc[0]
            # for i, d in enumerate(df["date"]):
            #     date = pd.to_datetime(df['date'].iloc[i]).strftime('%Y-%m-%d')

            #     if date != date_prev:
            #         date_prev = date
            #         # day_low = df.loc[i, "low"]
            #         # day_high = df.loc[i, "high"]
            #         day_low = df['low'].iloc[i]
            #         day_high = df['high'].iloc[i]
            #     else:
            #         day_low = min(df['low'].iloc[i], day_low)
            #         day_high = max(df['high'].iloc[i], day_high)

            #     atr_band_high_list.append(day_low + df['atr_D'].iloc[i])
            #     atr_band_low_list.append(day_high - df['atr_D'].iloc[i])

            # df_atr_band_high = pd.DataFrame(atr_band_high_list, columns=["atr_band_high"])
            # df_atr_band_low = pd.DataFrame(atr_band_low_list, columns=["atr_band_low"])
            # df["atr_band_high"] = df_atr_band_high["atr_band_high"]
            # df["atr_band_low"] = df_atr_band_low["atr_band_low"]
