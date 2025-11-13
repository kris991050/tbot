import os, sys, datetime, pandas as pd, numpy as np, scipy, hdbscan
from ib_insync import *
# from patternpy.tradingpatterns import head_and_shoulders

current_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(current_folder)

from utils import helpers, constants
from features import indicators
from data import hist_market_data_handler

# Doc ib_insync: https://ib-insync.readthedocs.io/api.html
# Doc IBKR API: https://interactivebrokers.github.io/tws-api/introduction.html
# util.startLoop()  # uncomment this line when in a notebook
# Doc TA library: https://technical-analysis-library-in-python.readthedocs.io/en/latest/
# Available timframes and duurations for historical data: https://interactivebrokers.github.io/tws-api/historical_bars.html


class PatternBuilder():
    def __init__(self, df:pd.DataFrame, ib:IB, symbol:str, pattern_window:int=10, pattern_decay_span:int=10, atr_period:int=14, candle_atr_multiplier:float=0.7, 
                 breakouts_atr_multiplier:int=1, volume_factor:float=1.5, range_window:int=20, min_cluster_size:int=50, confirmation_window:int=3, 
                 volume_weight:float=1.0, body_weight:float=1.0, price_weight:float=1.0, rsi_weight:float=1.0, breakout_lookback:int=5, extrema_order:int=5, 
                 candle_pattern_list:list=['all'], pattern_types:list=['all']):
        self.df = df.reset_index(drop=True)
        self.timeframe = helpers.get_df_timeframe(self.df)
        self.ib = ib
        self.contract, self.symbol = helpers.get_symbol_contract(self.ib, symbol)
        self.pattern_window = pattern_window
        self.decay_span = pattern_decay_span
        self.atr_period = atr_period
        self.candle_atr_multiplier = candle_atr_multiplier
        self.breakouts_atr_multiplier = breakouts_atr_multiplier
        self.volume_factor = volume_factor
        self.range_window = range_window
        self.min_cluster_size = min_cluster_size
        self.confirmation_window = confirmation_window
        self.volume_weight = volume_weight
        self.body_weight = body_weight
        self.price_weight = price_weight
        self.rsi_weight = rsi_weight
        self.breakout_lookback = breakout_lookback
        self.extrema_order = extrema_order
        self.candle_pattern_list = candle_pattern_list
        self.pattern_types = pattern_types

        # if 'avg_vol' not in self.df.columns:
        #     self.df['avg_vol'] = self.df['volume'].rolling(window=20).mean().fillna(method='bfill')

    # === Utility methods ===

    def _compute_pattern_features(self, df, pattern_col):
        tf = self.timeframe
        # Rolling count of occurrences
        df[f'{pattern_col}_roll_count_{tf}'] = df[f'{pattern_col}_{tf}'].rolling(window=self.pattern_window).sum()

        # Exponentially weighted pattern presence (decaying memory of recent patterns)
        df[f'{pattern_col}_ewm_{tf}'] = df[f'{pattern_col}_{tf}'].ewm(span=self.decay_span, adjust=False).mean()

        # Time since last occurrence
        df[f'{pattern_col}_since_last_{tf}'] = (~df[f'{pattern_col}_{tf}']).cumsum() - (~df[f'{pattern_col}_{tf}']).cumsum().where(df[f'{pattern_col}_{tf}']).ffill().fillna(0).astype(int)

        return df

    def _score_pattern(self, cond, body_size, volume, atr, volume_avg):
        """Returns a score for the pattern based on body size and volume, normalized by ATR and avg volume."""
        score = (self.body_weight * (body_size / atr)) + (self.volume_weight * (volume / volume_avg))
        return np.where(cond, score, 0)

    def _score_breakout(self, breakout_cond, price_dist, volume, avg_volume, atr):
        """Compute breakout score based on price extension and volume."""
        price_score = self.price_weight * (price_dist / atr)
        volume_score = self.volume_weight * (volume / avg_volume)

        return np.where(breakout_cond, price_score + volume_score, 0)

    def _score_divergence(self, divergence_cond, price_diff, rsi_diff, atr):
        """Compute divergence score based on normalized price movement and RSI shift."""
        price_score = self.price_weight * (price_diff / atr)
        rsi_score = self.rsi_weight * (rsi_diff / 100)  # normalize RSI difference to 0–1 range
        return np.where(divergence_cond, price_score + rsi_score, 0)


    # === Main Feature Extractors ===

    def _add_candle_patterns(self, df):
        tf = self.timeframe

        if f'atr_{tf}' not in df.columns:
            print(f"ATR column not found for timeframe {tf}.")
            return df

        if f'avg_vol_{tf}' not in df.columns:
            df[f'avg_vol_{tf}'] = df['volume'].rolling(window=20).mean().fillna(method='bfill')

        atr_filter = (df['high'] - df['low']) >= df[f'atr_{tf}'] * self.candle_atr_multiplier
        upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
        lower_wick = df[['close', 'open']].min(axis=1) - df['low']
        body_size = (df['close'] - df['open']).abs()

        # Hammer
        if 'hammer' in self.candle_pattern_list or 'all' in self.candle_pattern_list:
            hammer_body_ratio = 3
            hammer_wick_ratio = 0.6

            hammer_up_cond = ((df['high'] - df['low'] > hammer_body_ratio * body_size) &
                ((df['close'] - df['low']) / (df['high'] - df['low']) > hammer_wick_ratio) &
                atr_filter)
            hammer_down_cond = (
                (df['high'] - df['low'] > hammer_body_ratio * body_size) &
                ((df['high'] - df['close']) / (df['high'] - df['low']) > hammer_wick_ratio) &
                atr_filter)

            df[f'hammer_up_{tf}'] = np.where(hammer_up_cond, True, False)
            df[f'hammer_down_{tf}'] = np.where(hammer_down_cond, True, False)
            df[f'hammer_up_score_{tf}'] = self._score_pattern(hammer_up_cond, body_size, df['volume'], df[f'atr_{tf}'], df[f'avg_vol_{tf}'])
            df[f'hammer_down_score_{tf}'] = self._score_pattern(hammer_down_cond, body_size, df['volume'], df[f'atr_{tf}'], df[f'avg_vol_{tf}'])
            df = self._compute_pattern_features(df, 'hammer_up')
            df = self._compute_pattern_features(df, 'hammer_down')

        # Engulfing
        if 'engulfing' in self.candle_pattern_list or 'all' in self.candle_pattern_list:
            engulfing_up_cond = (
                (df['close'].shift(1) < df['open'].shift(1)) &
                (df['close'] > df['open']) &
                (df['open'] < df['close'].shift(1)) &
                (df['close'] > df['open'].shift(1)) &
                atr_filter)
            engulfing_down_cond = (
                (df['close'].shift(1) > df['open'].shift(1)) &
                (df['close'] < df['open']) &
                (df['open'] > df['close'].shift(1)) &
                (df['close'] < df['open'].shift(1)) &
                atr_filter)

            df[f'engulfing_up_{tf}'] = np.where(engulfing_up_cond, True, False)
            df[f'engulfing_down_{tf}'] = np.where(engulfing_down_cond, True, False)
            df[f'engulfing_up_score_{tf}'] = self._score_pattern(engulfing_up_cond, body_size, df['volume'], df[f'atr_{tf}'], df[f'avg_vol_{tf}'])
            df[f'engulfing_down_score_{tf}'] = self._score_pattern(engulfing_down_cond, body_size, df['volume'], df[f'atr_{tf}'], df[f'avg_vol_{tf}'])
            df = self._compute_pattern_features(df, 'engulfing_up')
            df = self._compute_pattern_features(df, 'engulfing_down')

        # Marubozu
        if 'marubozu' in self.candle_pattern_list or 'all' in self.candle_pattern_list:
            marubozu_wick_thresh = 0.05
            range_ = df['high'] - df['low']
            marubozu_up_cond = (
                (df['close'] > df['open']) &
                (upper_wick <= (marubozu_wick_thresh * range_)) &
                (lower_wick <= (marubozu_wick_thresh * range_)) &
                atr_filter)
            marubozu_down_cond = (
                (df['close'] < df['open']) &
                (upper_wick <= (marubozu_wick_thresh * range_)) &
                (lower_wick <= (marubozu_wick_thresh * range_)) &
                atr_filter)

            df[f'marubozu_up_{tf}'] = np.where(marubozu_up_cond, True, False)
            df[f'marubozu_down_{tf}'] = np.where(marubozu_down_cond, True, False)
            df[f'marubozu_up_score_{tf}'] = self._score_pattern(marubozu_up_cond, body_size, df['volume'], df[f'atr_{tf}'], df[f'avg_vol_{tf}'])
            df[f'marubozu_down_score_{tf}'] = self._score_pattern(marubozu_down_cond, body_size, df['volume'], df[f'atr_{tf}'], df[f'avg_vol_{tf}'])
            df = self._compute_pattern_features(df, 'marubozu_up')
            df = self._compute_pattern_features(df, 'marubozu_down')

        # Doji
        if 'doji' in self.candle_pattern_list or 'all' in self.candle_pattern_list:
            range_ = df['high'] - df['low']
            doji_cond = (
                (range_ > 0) &
                (body_size <= range_ * 0.1) &
                (upper_wick >= range_ * 0.1) &
                (lower_wick >= range_ * 0.1) &
                ((upper_wick - lower_wick).abs() / range_ <= 0.2) &
                atr_filter)

            df[f'doji_{tf}'] = np.where(doji_cond, True, False)
            df[f'doji_score_{tf}'] = self._score_pattern(doji_cond, body_size, df['volume'], df[f'atr_{tf}'], df[f'avg_vol_{tf}'])
            df = self._compute_pattern_features(df, 'doji')

        # Volume spike (treated separately)
        if 'volume_spike' in self.candle_pattern_list or 'all' in self.candle_pattern_list:
            df[f'volume_spike_{tf}'] = df['volume'] > df[f'avg_vol_{tf}'] * self.volume_factor
            df[f'volume_spike_score_{tf}'] = np.where(df['volume'] > df[f'avg_vol_{tf}'] * self.volume_factor,
                                                (df['volume'] / df[f'avg_vol_{tf}']), 0)
            df = self._compute_pattern_features(df, 'volume_spike')

        # Total Bullish/Bearish Scores
        df[f'bullish_score_{tf}'] = df.get(f'engulfing_up_score_{tf}', 0) + df.get(f'hammer_up_score_{tf}', 0) + \
                            df.get(f'marubozu_up_score_{tf}', 0)
        df[f'bearish_score_{tf}'] = df.get(f'engulfing_down_score_{tf}', 0) + df.get(f'hammer_down_score_{tf}', 0) + \
                            df.get(f'marubozu_down_score_{tf}', 0)

        # Normalized score ratio
        total_score = df[f'bullish_score_{tf}'] + df[f'bearish_score_{tf}']
        df[f'score_bias_{tf}'] = np.where(total_score > 0, (df[f'bullish_score_{tf}'] - df[f'bearish_score_{tf}']) / total_score, 0)

        # Simple directional bias using smoothed returns
        df[f'return_{tf}'] = df['close'].pct_change()
        df[f'directional_bias_{tf}'] = df[f'return_{tf}'].ewm(span=10, adjust=False).mean()
        df[f'bias_trend_{tf}'] = np.where(df[f'directional_bias_{tf}'] > 0, 1, -1)

        # Hybrid bias diraction
        df[f'hybrid_bias_{tf}'] = df[f'score_bias_{tf}'] + df[f'directional_bias_{tf}']
        df[f'hybrid_direction_{tf}'] = np.where(df[f'hybrid_bias_{tf}'] > 0, 1, np.where(df[f'hybrid_bias_{tf}'] < 0, -1, 0))

        return df

    def _add_ranges(self, df):
        tf = self.timeframe
        if f'bband_h_{tf}' not in df.columns or f'bband_l_{tf}' not in df.columns: print("\nBollinger Bands columns not found.\n")
        else:
            df[f'~bband_width_{tf}'] = df[f'bband_h_{tf}'] - df[f'bband_l_{tf}']
            df[f'bband_width_pct_{tf}'] = df[f'~bband_width_{tf}'].rolling(100).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min()), raw=True)
            # df['volume_mean'] = df['volume'].rolling(window=20).mean()
            df[f'low_volume_{tf}'] = df['volume'] < df['volume'].rolling(window=20).mean()

            # === atr-Based Range Detection ===
            df[f'~rolling_high_{tf}'] = df['high'].rolling(window=self.range_window).max()
            df[f'~rolling_low_{tf}'] = df['low'].rolling(window=self.range_window).min()
            # df['range_width'] = df['rolling_high'] - df['rolling_low']
            # df['atr_range'] = 4 * df['atr']
            # df['atr_in_range'] = df['range_width'] <= df['atr_range']
            df[f'atr_in_range_{tf}'] = df[f'atr_{tf}'] < df[f'atr_{tf}'].rolling(window=15).mean()

            # === Price Action Filter: Inside Bars ===
            df[f'inside_bar_{tf}'] = (df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))

            # === DBSCAN Clustering on Price + Time ===
            price_time_data = df[['close']].copy()
            price_time_data['time'] = np.arange(len(df))
            price_time_data_scaled = (price_time_data - price_time_data.mean()) / price_time_data.std()

            # db = sklearn.cluster.DBSCAN(eps=0.3, min_samples=5)
            # df['dbscan_cluster'] = db.fit_predict(price_time_data_scaled)

            db = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)#, core_dist_n_jobs=-1)
            df[f'dbscan_cluster_{tf}'] = db.fit_predict(price_time_data_scaled)

            # === Hybrid Consolidation Flag ===
            df[f'consolidation_{tf}'] = (
                (df[f'atr_in_range_{tf}']) &
                (df[f'bband_width_pct_{tf}'] < 0.2) &
                df[f'rsi_{tf}'].between(45, 55) &
                df[f'low_volume_{tf}'] &
                (df[f'dbscan_cluster_{tf}'] != -1)
            )

            # Optional: Add stronger signal when Inside Bar is also present
            df[f'strong_consolidation_{tf}'] = df[f'consolidation_{tf}'] & df[f'inside_bar_{tf}']

        return df

    def _add_breakouts(self, df):

        # df = df_original.copy()
        tf = self.timeframe
        # df.attrs = df_original.attrs

        if f'consolidation_{tf}' not in df.columns:
            print(f"Missing 'consolidation_{tf}' column. Run add_ranges() first.")
            return df

        # Calculate ATR-based dynamic price buffer
        df['tr'] = np.maximum(df['high'] - df['low'],
                            np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
        if f'atr_{tf}' not in df.columns: df[f'atr_{tf}'] = df['tr'].rolling(window=self.atr_period).mean()
        df[f'price_buffer_pct_{tf}'] = (df[f'atr_{tf}'] / df['close']) * self.breakouts_atr_multiplier

        if f'avg_vol_{tf}' not in df.columns:
            df[f'avg_vol_{tf}'] = df['volume'].rolling(window=20).mean().fillna(method='bfill')

        breakout_up = np.zeros(len(df), dtype=bool)
        breakout_down = np.zeros(len(df), dtype=bool)
        breakout_level_up = np.full(len(df), np.nan)
        breakout_level_down = np.full(len(df), np.nan)

        for i in range(1, self.breakout_lookback + 1):
            price_buffer = df[f'price_buffer_pct_{tf}']
            cond_up = (df[f'consolidation_{tf}'].shift(i) & (df['close'] > df[f'~rolling_high_{tf}'].shift(i) * (1 + price_buffer)))
            cond_down = (df[f'consolidation_{tf}'].shift(i) & (df['close'] < df[f'~rolling_low_{tf}'].shift(i) * (1 - price_buffer)))

            breakout_up |= cond_up
            breakout_down |= cond_down

            breakout_level_up = np.where(cond_up, df[f'~rolling_high_{tf}'].shift(i) * (1 + price_buffer), breakout_level_up)
            breakout_level_down = np.where(cond_down, df[f'~rolling_low_{tf}'].shift(i) * (1 - price_buffer), breakout_level_down)

        volume_condition = df['volume'] > df[f'avg_vol_{tf}'] * self.volume_factor
        df[f'breakout_up_{tf}'] = breakout_up & volume_condition
        df[f'breakout_down_{tf}'] = breakout_down & volume_condition

        breakout_up &= volume_condition
        breakout_down &= volume_condition

        df[f'breakout_level_up_{tf}'] = breakout_level_up
        df[f'breakout_level_down_{tf}'] = breakout_level_down

        # df['breakout_strength_up'] = np.where(df['breakout_up'],
        #                                       (df['close'] - df['breakout_level_up']) / df['breakout_level_up'], np.nan)
        # df['breakout_strength_down'] = np.where(df['breakout_down'],
        #                                         (df['close'] - df['breakout_level_down']) / df['breakout_level_down'], np.nan)

        # Breakout strength
        breakout_dist_up = (df['close'] - df[f'breakout_level_up_{tf}']).clip(lower=0)
        breakout_dist_down = (df[f'breakout_level_down_{tf}'] - df['close']).clip(lower=0)

        df[f'breakout_up_score_{tf}'] = self._score_breakout(breakout_up, breakout_dist_up, df['volume'], df[f'avg_vol_{tf}'], df[f'atr_{tf}'])
        df[f'breakout_down_score_{tf}'] = self._score_breakout(breakout_down, breakout_dist_down, df['volume'], df[f'avg_vol_{tf}'], df[f'atr_{tf}'])

        # Confirmed breakout logic (no future lookahead)
        confirmed_up = np.zeros(len(df), dtype=bool)
        confirmed_down = np.zeros(len(df), dtype=bool)
        confirmed_up_score = np.full(len(df), np.nan)
        confirmed_down_score = np.full(len(df), np.nan)

        if self.confirmation_window > 0:
            for i in range(self.confirmation_window, len(df)):
                past_window = df['close'].iloc[i - self.confirmation_window + 1:i + 1]
                past_level_up = df[f'breakout_level_up_{tf}'].iloc[i - self.confirmation_window]
                past_level_down = df[f'breakout_level_down_{tf}'].iloc[i - self.confirmation_window]

                if df[f'breakout_up_{tf}'].iloc[i - self.confirmation_window] and not np.isnan(past_level_up):
                    if (past_window > past_level_up).all():
                        confirmed_up[i] = True

                if df[f'breakout_down_{tf}'].iloc[i - self.confirmation_window] and not np.isnan(past_level_down):
                    if (past_window < past_level_down).all():
                        confirmed_down[i] = True

                if not np.isnan(past_level_up) and (past_window > past_level_up).all():
                    confirmed_up_score[i] = df[f'breakout_up_score_{tf}'].iloc[i - self.confirmation_window]

                if not np.isnan(past_level_down) and (past_window < past_level_down).all():
                    confirmed_down_score[i] = df[f'breakout_down_score_{tf}'].iloc[i - self.confirmation_window]

            df[f'breakout_up_{tf}'] = confirmed_up
            df[f'breakout_down_{tf}'] = confirmed_down

            df[f'breakout_up_score_{tf}'] = confirmed_up_score
            df[f'breakout_down_score_{tf}'] = confirmed_down_score

        df = self._compute_pattern_features(df, 'breakout_up')
        df = self._compute_pattern_features(df, 'breakout_down')

        # Clean up temporary columns
        df.drop(['tr'], axis=1, inplace=True)

        return df

    def _add_rsi_divergence(self, df):
        # df = df_original.copy()
        tf = self.timeframe

        # === Calculate RSI and ATR if missing ===
        if f'rsi_{tf}' not in df.columns or f'atr_{tf}' not in df.columns:
            print(f"RSI and ATR columns not found for timeframe {tf}.")
            return df

        # === Detect extrema ===
        # peaks_idx = scipy.signal.argrelextrema(df['close'].values, np.greater_equal, order=self.extrema_order)[0]
        peaks_idx = scipy.signal.argrelextrema(df['high'].values, np.greater_equal, order=self.extrema_order)[0]
        # troughs_idx = scipy.signal.argrelextrema(df['close'].values, np.less_equal, order=self.extrema_order)[0]
        troughs_idx = scipy.signal.argrelextrema(df['low'].values, np.less_equal, order=self.extrema_order)[0]

        # === Prepare pairwise extrema ===
        bullish_idx = []
        # price_prev_list = []
        bullish_hid_idx = []
        bearish_idx = []
        bearish_hid_idx = []
        divergence_up_scores = np.zeros(len(df))
        divergence_hid_up_scores = np.zeros(len(df))
        divergence_down_scores = np.zeros(len(df))
        divergence_hid_down_scores = np.zeros(len(df))

        # -- Bullish Divergences
        for i in range(1, len(troughs_idx)):
            prev_i, curr_i = troughs_idx[i - 1], troughs_idx[i]
            if np.isnan(df[f'rsi_{tf}'].iloc[prev_i]) or np.isnan(df[f'rsi_{tf}'].iloc[curr_i]):
                continue
            price_prev, price_curr = df['low'].iloc[prev_i], df['low'].iloc[curr_i]
            rsi_prev, rsi_curr = df[f'rsi_{tf}'].iloc[prev_i], df[f'rsi_{tf}'].iloc[curr_i]
            atr_val = df[f'atr_{tf}'].iloc[curr_i]

            # -- Bullish Divergence: price lower low, RSI higher low
            if price_curr < price_prev and rsi_curr > rsi_prev and not np.isnan(atr_val):
                price_diff = abs(price_prev - price_curr)
                rsi_diff = rsi_curr - rsi_prev
                score = self.price_weight * (price_diff / atr_val) + self.rsi_weight * (rsi_diff / 100)
                bullish_idx.append(curr_i)
                # price_prev_list.append(price_prev)
                divergence_up_scores[curr_i] = score

            # -- Bullish Hidden Divergence: price higher low, RSI lower low
            elif price_curr > price_prev and rsi_curr < rsi_prev and not np.isnan(atr_val):
                price_diff = abs(price_prev - price_curr)
                rsi_diff = rsi_curr - rsi_prev
                score = self.price_weight * (price_diff / atr_val) + self.rsi_weight * (rsi_diff / 100)
                bullish_hid_idx.append(curr_i)
                divergence_hid_up_scores[curr_i] = score

        # -- Bearish Divergences
        for i in range(1, len(peaks_idx)):
            prev_i, curr_i = peaks_idx[i - 1], peaks_idx[i]
            if np.isnan(df[f'rsi_{tf}'].iloc[prev_i]) or np.isnan(df[f'rsi_{tf}'].iloc[curr_i]):
                continue
            price_prev, price_curr = df['high'].iloc[prev_i], df['high'].iloc[curr_i]
            rsi_prev, rsi_curr = df[f'rsi_{tf}'].iloc[prev_i], df[f'rsi_{tf}'].iloc[curr_i]
            atr_val = df[f'atr_{tf}'].iloc[curr_i]

            # -- Bearish Divergence: price higher high, RSI lower high
            if price_curr > price_prev and rsi_curr < rsi_prev and not np.isnan(atr_val):
                price_diff = abs(price_curr - price_prev)
                rsi_diff = rsi_prev - rsi_curr
                score = self.price_weight * (price_diff / atr_val) + self.rsi_weight * (rsi_diff / 100)
                bearish_idx.append(curr_i)
                divergence_down_scores[curr_i] = score

            # -- Bearish Hidden Divergence: price lower high, RSI higher high
            elif price_curr < price_prev and rsi_curr > rsi_prev and not np.isnan(atr_val):
                price_diff = abs(price_curr - price_prev)
                rsi_diff = rsi_prev - rsi_curr
                score = self.price_weight * (price_diff / atr_val) + self.rsi_weight * (rsi_diff / 100)
                bearish_hid_idx.append(curr_i)
                divergence_hid_down_scores[curr_i] = score

        # === Final output ===
        df[f'divergence_up_{tf}'] = False
        df[f'divergence_hid_up_{tf}'] = False
        df[f'divergence_down_{tf}'] = False
        df[f'divergence_hid_down_{tf}'] = False
        df.loc[bullish_idx, f'divergence_up_{tf}'] = True
        # df['divergence_up_prev'] = np.nan  # Initialize column
        # for idx, price_prev in zip(bullish_idx, price_prev_list):
        #     df.loc[idx, 'divergence_up_prev'] = price_prev
        df.loc[bullish_hid_idx, f'divergence_hid_up_{tf}'] = True
        df.loc[bearish_idx, f'divergence_down_{tf}'] = True
        df.loc[bearish_hid_idx, f'divergence_hid_down_{tf}'] = True
        df[f'divergence_up_score_{tf}'] = divergence_up_scores
        df[f'divergence_hid_up_score_{tf}'] = divergence_hid_up_scores
        df[f'divergence_down_score_{tf}'] = divergence_down_scores
        df[f'divergence_hid_down_score_{tf}'] = divergence_hid_down_scores

        return df

    def _add_trend(self, df, symbol=None):
        # Assess trend
        # df = df_original.copy()
        tf = self.timeframe
        symbol = symbol or self.symbol
        indicators_handler = indicators.Indicators(df=df.copy(), ib=self.ib, symbol=symbol)
        df_macd_fast = indicators_handler.apply_macd(macd_windows={'slow':50, 'fast':20, 'sign':9, 'roll':20})
        df_macd_slow = indicators_handler.apply_macd(macd_windows = {'slow':200, 'fast':50, 'sign':9, 'roll':20})

        # Short-Term Trend Signal based on MACD
        condition_short_term_up = (df_macd_fast[f'macd_diff_{tf}'] > df_macd_fast[f'macd_diff_{tf}'].shift()) & (df_macd_fast[f'macd_diff_{tf}'].shift() > df_macd_fast[f'macd_diff_{tf}'].shift(2))
        condition_short_term_down = (df_macd_fast[f'macd_diff_{tf}'] < df_macd_fast[f'macd_diff_{tf}'].shift()) & (df_macd_fast[f'macd_diff_{tf}'].shift() < df_macd_fast[f'macd_diff_{tf}'].shift(2))
        df_macd_fast[f'trend_fast_{tf}'] = np.where(condition_short_term_up, 1, np.where(condition_short_term_down, -1, 0))
        df_macd_fast[f'trend_strength_fast_{tf}'] = df_macd_fast[f'macd_diff_{tf}'] - df_macd_fast[f'macd_diff_{tf}'].shift(1)
        df = pd.merge(df, df_macd_fast[['date', f'trend_fast_{tf}', f'trend_strength_fast_{tf}']], on='date', how='left')

        # Long-Term Trend Signal based on MACD
        condition_short_term_up = (df_macd_slow[f'macd_diff_{tf}'] > df_macd_slow[f'macd_diff_{tf}'].shift()) & (df_macd_slow[f'macd_diff_{tf}'].shift() > df_macd_slow[f'macd_diff_{tf}'].shift(2))
        condition_short_term_down = (df_macd_slow[f'macd_diff_{tf}'] < df_macd_slow[f'macd_diff_{tf}'].shift()) & (df_macd_slow[f'macd_diff_{tf}'].shift() < df_macd_slow[f'macd_diff_{tf}'].shift(2))
        df_macd_slow[f'trend_slow_{tf}'] = np.where(condition_short_term_up, 1, np.where(condition_short_term_down, -1, 0))
        df_macd_slow[f'trend_strength_slow_{tf}'] = df_macd_slow[f'macd_diff_{tf}'] - df_macd_slow[f'macd_diff_{tf}'].shift(1)
        df = pd.merge(df, df_macd_slow[['date', f'trend_slow_{tf}', f'trend_strength_slow_{tf}']], on='date', how='left')

        # Trend analysis based on RSI and ADX
        if f'rsi_{tf}' not in df.columns: df = indicators_handler.apply_rsi()
        if f'adx_{tf}' not in df.columns: df = indicators_handler.apply_adx()
        df[f'trend_{tf}'] = np.where((df[f'trend_fast_{tf}'] == 1) & (df[f'trend_slow_{tf}'] == 1) & (df[f'adx_{tf}'] > 25) & (df[f'rsi_{tf}'] < 70), 1,
                                          np.where((df[f'trend_fast_{tf}'] == -1) & (df[f'trend_slow_{tf}'] == -1) & (df[f'adx_{tf}'] > 25) & (df[f'rsi_{tf}'] > 30), -1, 0))


        # condition_index_trend_up = (df_index['macd_diff'] > df_index['macd_diff'].shift()) & (df_index['macd_diff'].shift() > df_index['macd_diff'].shift(2))
        # condition_index_trend_down = (df_index['macd_diff'] < df_index['macd_diff'].shift()) & (df_index['macd_diff'].shift() < df_index['macd_diff'].shift(2))
        # df_index['index_trend'] = np.where(condition_index_trend_up, 1, np.where(condition_index_trend_down, -1, 0))

        # df_index['index_trend'] = np.where((df_index['macd_diff'] > 0) & (df_index['rsi'] > 50) & (df_index['adx'] > 25), 1,
        #                                    np.where((df_index['macd_diff'] < 0) & (df_index['rsi'] < 50) & (df_index['adx'] > 25), -1, 0))
        # df_index['index_trend_strength'] = df_index['macd_diff'] - df_index['macd_diff'].shift(1)

        # # Combining Short-Term and Long-Term Trends
        # df_index['index_trend'] = np.where((df_index['trend_fast'] == 1) & (df_index['trend_slow'] == 1), 1,
        #                                    np.where((df_index['trend_fast'] == -1) & (df_index['trend_slow'] == -1), -1, 0))

        # df = pd.merge(df, df_temp[['date', 'index_trend', 'index_trend_fast', 'index_trend_slow', 'index_trend_strength_fast', 'index_trend_strength_slow']], on='date', how='left')
        return df

    def _add_index_trend(self, df):
        # Get symbol related Index and ETF
        symbol_index = helpers.get_index_from_symbol(self.ib, self.contract.symbol)
        if len(symbol_index) > 0: symbol_index = symbol_index[0]
        symbol_index_ETF = helpers.get_index_etf(symbol_index)

        if symbol_index_ETF:
            # Gathering Index historical data
            tf = self.timeframe
            query_time = tf.add_to_date(df['date'].iloc[-1])
            from_time = df['date'].iloc[0]
            index_fetcher = hist_market_data_handler.HistMarketDataFetcher(ib=self.ib, ftype='auto', save_to_file=True, delete_existing=True)
            index_params = {'symbol': symbol_index_ETF, 'timeframe': tf, 'from_time': from_time, 'to_time': query_time, 'step_duration': 'auto'}
            df_index = index_fetcher.run(params=index_params)[symbol_index_ETF]['df']

            df_index = self._add_trend(df_index, symbol=symbol_index_ETF)
            df_index.rename(columns={
                f'trend_{tf}': f'index_trend_{tf}',
                f'trend_fast_{tf}': f'index_trend_fast_{tf}',
                f'trend_slow_{tf}': f'index_trend_slow_{tf}',
                f'trend_strength_fast_{tf}': f'index_trend_strength_fast_{tf}',
                f'trend_strength_slow_{tf}': f'index_trend_strength_slow_{tf}',
                }, inplace=True)

            # Adding the trends to main df dataframe
            df = pd.merge(df, df_index[['date', f'index_trend_{tf}', f'index_trend_fast_{tf}', f'index_trend_slow_{tf}', f'index_trend_strength_fast_{tf}', f'index_trend_strength_slow_{tf}']], on='date', how='left')
        else:
            print(f"⚠️ No Index found for {self.symbol}.")
        return df

    def apply_patterns(self):

        # df = self.df.copy()
        attrs = self.df.attrs
        if 'candle' in self.pattern_types or 'all' in self.pattern_types:
            print(f"Adding candle patterns for {self.timeframe}...")
            self.df = self._add_candle_patterns(self.df)
        if 'divergence' in self.pattern_types or 'all' in self.pattern_types:
            print(f"Adding rsi divergences for {self.timeframe}...")
            self.df = self._add_rsi_divergence(self.df)
        if 'range' in self.pattern_types or 'all' in self.pattern_types:
            print(f"Adding ranges for {self.timeframe}...")
            self.df = self._add_ranges(self.df)
        if 'breakout' in self.pattern_types or 'all' in self.pattern_types:
            print(f"Adding breakouts for {self.timeframe}...")
            self.df = self._add_breakouts(self.df)
        if 'trend' in self.pattern_types or 'all' in self.pattern_types:
            print(f"Adding trend for {self.timeframe}...")
            self.df = self._add_trend(self.df)
        if 'index_trend' in self.pattern_types or 'all' in self.pattern_types:
            print(f"Adding index trend for {self.timeframe}...")
            self.df = self._add_index_trend(self.df)

        if self.df.attrs != attrs: self.df.attrs = attrs
        return self.df


if __name__ == "__main__":

    args = sys.argv

    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    symbol = 'TSLA'
    to_time = pd.to_datetime('2025-05-06T19:59:59').tz_localize(constants.CONSTANTS.TZ_WORK)
    from_time = pd.to_datetime('2023-01-01T04:00:00').tz_localize(constants.CONSTANTS.TZ_WORK)
    timeframe_1D = '1day'
    timeframe_1m = '1min'
    timeframe_5m = '5mins'
    timeframe_15m = '15mins'
    timeframe_1h = '1hour'
    perc_limit = 3
    paperTrading = True
    strategy = ''
    if len(args) > 1:
        if 'live' in args: paperTrading = False
        for arg in args:
            if 'sym' in arg: symbol = arg[3:]

    # TWS Connection
    ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)
    if not ibConnection:
        paperTrading = not paperTrading
        ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)


    # Setup
    hist_folder = constants.PATHS.folders_path['hist_market_data']
    market_data_folder = constants.PATHS.folders_path['market_data']

    symbol_hist_folder = os.path.join(hist_folder, symbol)
    print("\nSymbol Hist Data folder: ", symbol_hist_folder, "\n")

    from stats import stats
    hist_csv_file_1m = stats.get_symbol_hist_file(symbol, timeframe_1m, symbol_hist_folder)
    df_1m = pd.read_csv(hist_csv_file_1m)
    df_1m.attrs['symbol'] = symbol

    # Make sure date column is in datetime format
    df_1m['date'] = pd.to_datetime(df_1m['date'], utc=True)
    df_1m['date'] = df_1m['date'].dt.tz_convert(constants.CONSTANTS.TZ_WORK)


    # Trimming df to desired time interval
    df_1m = df_1m[pd.to_datetime(df_1m["date"]).between(max(from_time, pd.to_datetime(df_1m['date'].iloc[0])), min(to_time, pd.to_datetime(df_1m['date'].iloc[-1])))]

    df_5m = df_1m.resample('5min', on='date').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna().reset_index()
    df_15m = df_1m.resample('15min', on='date').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna().reset_index()
    df_1D = df_1m.resample('1D', on='date').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna().reset_index()
    contract, mktData = helpers.get_symbol_mkt_data(ib, symbol, currency='USD')
    df_5m_ind = indicators.Indicators(df_5m, ib, contract, types=['volatility', 'momentum'], bb_window=10).apply_indicators()
    df_15m_ind = indicators.Indicators(df_15m, ib, contract, types=['volatility', 'momentum']).apply_indicators()
    df_1m_ind = indicators.Indicators(df_1m, ib, contract, types=['volatility', 'momentum']).apply_indicators()
    df_1D_ind = indicators.Indicators(df_1D, ib, contract, types=['volatility', 'momentum'], bb_window=5).apply_indicators()

    pb = PatternBuilder(df_1m_ind)
    time_now_start = datetime.datetime.now()
    # sr_levels = indicators.get_SR_levels(df_1m, levels_timeframe='5min', count_threshold=2, look_forward_reaction=50, display=False)
    df = pb._add_ranges(df_1m_ind)
    # df = add_ranges(df_5m_ind)
    # df = add_ranges(df_1D_ind)
    time_now_end = datetime.datetime.now()
    print("Time elapsed: ", time_now_end - time_now_start)

    time_now_start = datetime.datetime.now()
    df = pb._add_candle_patterns(df)
    time_now_end = datetime.datetime.now()
    print("Time elapsed: ", time_now_end - time_now_start)

    time_now_start = datetime.datetime.now()
    df = pb._add_breakouts(df)
    time_now_end = datetime.datetime.now()
    print("Time elapsed: ", time_now_end - time_now_start)

    # print(sr_levels)
    # print(df[['date', 'open', 'close', 'low', 'high', 'atr_in_range', 'bband_width_pct', 'rsi', 'low_volume', 'inside_bar', 'dbscan_cluster', 'consolidation', 'strong_consolidation']][df['strong_consolidation'] == True])#.tail(50))
    # print(df[['date', 'open', 'close', 'hammer_up', 'hammer_down']][(df['hammer_up'] == True) | (df['hammer_down'] == True)])#.tail(50))
    # print(df[['date', 'open', 'close', 'engulfing_up', 'engulfing_down']][(df['engulfing_up'] == True) | (df['engulfing_down'] == True)])#.tail(50))
    # print(df[['date', 'open', 'close', 'doji']][(df['doji'] == True)])#.tail(50))

    date_ranges = [['2025-04-01T00:00:00', '2025-05-06T20:59:59'], ['2025-03-05T00:00:00', '2025-03-09T20:59:59']]
    date_ranges = [['2023-01-01T00:00:00', '2025-05-06T20:59:59']]
    helpers.display_df(df, ['date', 'open', 'close', 'doji', 'doji_score'], date_ranges, ['doji'])
    input()
    helpers.display_df(df, ['date', 'open', 'close', 'engulfing_up', 'engulfing_down', 'engulfing_up_score', 'engulfing_down_score'], date_ranges, ['engulfing_up', 'engulfing_down'])
    input()
    helpers.display_df(df, ['date', 'open', 'close', 'marubozu_up', 'marubozu_down', 'marubozu_up_score', 'marubozu_down_score'], date_ranges, ['marubozu_up', 'marubozu_down'])
    input()
    helpers.display_df(df, ['date', 'open', 'close', 'breakout_level_up', 'breakout_level_down', 'breakout_up', 'breakout_down', 'breakout_up_score', 'breakout_down_score'], date_ranges, ['breakout_up', 'breakout_down'])

    # input()
    # print(df[['date', 'open', 'close', 'volume', 'volume_spike']][(df['volume_spike'] == True) | (df['volume_spike'].shift(-1) == True)].tail(20))
    # print("Time elapsed: ", time_now_end - time_now_start)
    input()

    input("\nEnter anything to exit")




# def add_rsi_divergence_old(df, lookback=30, rsi_weight=1.0, price_weight=1.0, extrema_order=3):
#     df = df.copy()

#     # Calculate RSI
#     if 'rsi' not in df.columns:
#         print("\nBollinger Bands columns not found.\n")
#         return df

#     # Local extrema
#     df['price_peak'] = df['close'].iloc[scipy.signal.argrelextrema(df['close'].values, np.greater_equal, order=extrema_order)[0]]
#     df['price_trough'] = df['close'].iloc[scipy.signal.argrelextrema(df['close'].values, np.less_equal, order=extrema_order)[0]]
#     df['rsi_peak'] = df['rsi'].iloc[scipy.signal.argrelextrema(df['rsi'].values, np.greater_equal, order=extrema_order)[0]]
#     df['rsi_trough'] = df['rsi'].iloc[scipy.signal.argrelextrema(df['rsi'].values, np.less_equal, order=extrema_order)[0]]

#     # Scoring placeholders
#     divergence_up = [False] * len(df)
#     divergence_down = [False] * len(df)
#     divergence_up_score = [0.0] * len(df)
#     divergence_down_score = [0.0] * len(df)

#     for i in range(lookback, len(df)):
#         # Extract local lows for bullish divergence
#         price_lows = df['price_trough'].iloc[i - lookback:i].dropna()
#         rsi_lows = df['rsi_trough'].iloc[i - lookback:i].dropna()

#         if len(price_lows) >= 2 and len(rsi_lows) >= 2:
#             p1, p2 = price_lows.iloc[-2], price_lows.iloc[-1]
#             r1, r2 = rsi_lows.iloc[-2], rsi_lows.iloc[-1]
#             if p2 < p1 and r2 > r1:
#                 price_diff_pct = abs((p1 - p2) / p1)
#                 rsi_diff = r2 - r1
#                 score = price_weight * price_diff_pct + rsi_weight * rsi_diff / 100
#                 divergence_up[i] = True
#                 divergence_up_score[i] = score

#         # Extract local highs for bearish divergence
#         price_highs = df['price_peak'].iloc[i - lookback:i].dropna()
#         rsi_highs = df['rsi_peak'].iloc[i - lookback:i].dropna()

#         if len(price_highs) >= 2 and len(rsi_highs) >= 2:
#             p1, p2 = price_highs.iloc[-2], price_highs.iloc[-1]
#             r1, r2 = rsi_highs.iloc[-2], rsi_highs.iloc[-1]
#             if p2 > p1 and r2 < r1:
#                 price_diff_pct = abs((p2 - p1) / p1)
#                 rsi_diff = r1 - r2
#                 score = score_divergence(True, price_diff_pct * df['close'].iloc[i], rsi_diff, df['atr'].iloc[i],
#                          price_weight=price_weight, rsi_weight=rsi_weight)

#                 divergence_down[i] = True
#                 divergence_down_score[i] = score

#     df['divergence_up'] = divergence_up
#     df['divergence_down'] = divergence_down
#     df['divergence_up_score'] = divergence_up_score
#     df['divergence_down_score'] = divergence_down_score

#     return df


# def add_candle_patterns(df, atr_multiplier=0.7, volume_factor=1.5, candle_pattern_list=['doji', 'hammer', 'engulfing', 'marubozu', 'volume_spike']):

#     if 'atr' not in df.columns: print("\nATR column not found.\n")
#     else:

#         """Detects basic candle patterns and filters noise using ATR."""
#         # Filter: Only consider bars that are at least X% of ATR
#         atr_filter = (df['high'] - df['low']) >= df['atr'] * atr_multiplier

#         upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
#         lower_wick = df[['close', 'open']].min(axis=1) - df['low']

#         # Hammer and Shooting Star Patterns (filtered)
#         if 'hammer' in candle_pattern_list:
#             hammer_body_ratio = 3
#             hammer_wick_ratio = 0.6
#             hammer_up_cond = ((df['high'] - df['low'] > hammer_body_ratio * (df['open'] - df['close']).abs()) & ((df['close'] - df['low']) / (df['high'] - df['low']) > hammer_wick_ratio) & atr_filter)
#             df['hammer_up'] = np.where(hammer_up_cond, True, False)
#             hammer_down_cond = ((df['low'] - df['high'] > hammer_body_ratio * (df['open'] - df['close']).abs()) & ((df['high'] - df['close']) / (df['high'] - df['low']) > hammer_wick_ratio) & atr_filter)
#             df['hammer_down'] = np.where(hammer_down_cond, True, False)

#         # Engulfing Patterns (filtered)
#         if 'engulfing' in candle_pattern_list:
#             engulfing_up_cond = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) & atr_filter
#             df['engulfing_up'] = np.where(engulfing_up_cond, True, False)
#             engulfing_down_cond = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1)) & atr_filter
#             df['engulfing_down'] = np.where(engulfing_down_cond, True, False)

#         # Marubozu Patterns (filtered)
#         if 'marubozu' in candle_pattern_list:
#             marubozu_wick_thresh = 0.05
#             marubozu_up_cond = (df['close'] > df['open']) & (upper_wick <= (marubozu_wick_thresh * (df['high'] - df['low']))) & (lower_wick <= (marubozu_wick_thresh * (df['high'] - df['low']))) & atr_filter
#             df['marubozu_up'] = np.where(marubozu_up_cond, True, False)
#             marubozu_down_cond = (df['close'] < df['open']) & (upper_wick <= (marubozu_wick_thresh * (df['high'] - df['low']))) & (lower_wick <= (marubozu_wick_thresh * (df['high'] - df['low']))) & atr_filter
#             df['marubozu_down'] = np.where(marubozu_down_cond, True, False)

#         # Doji (filtered)
#         if 'doji' in candle_pattern_list:
#             # doji_cond = (((df['high'] - df['low']) > 0) & ((df['close'] - df['open']).abs() <= (df['high'] - df['low']) * 0.1) & atr_filter)
#             doji_cond = (
#                 ((df['high'] - df['low']) > 0) &
#                 (((df['close'] - df['open']).abs()) <= (df['high'] - df['low']) * 0.1) &  # Real body <= 10% of range
#                 (upper_wick >= (df['high'] - df['low']) * 0.1) &  # Some shadow on both ends
#                 (lower_wick >= (df['high'] - df['low']) * 0.1) &
#                 (((upper_wick - lower_wick).abs() / (df['high'] - df['low'])) <= 0.2) &
#                 atr_filter
#             )
#             df['doji'] = np.where(doji_cond, True, False)

#         # Volume Spike
#         if 'volume_spike' in candle_pattern_list:
#             volume_spike_window = 20
#             if 'avg_vol' not in df.columns: df['avg_vol'] = df['volume'].rolling(window=volume_spike_window).mean()
#             df['volume_spike'] = df['volume'] > df['avg_vol'] * volume_factor

#         return df





        # breakout_up = np.zeros(len(df), dtype=bool)

        # for i in range(1, lookback + 1):
        #     breakout_up_cond |= (
        #         df['consolidation'].shift(i) &
        #         (df['close'] > df['rolling_high'].shift(i) * (1 + price_buffer_pct))
        #     )

        # volume_cond = df['volume'] > df['volume'].rolling(window=20).mean() * volume_multiplier
        # df['breakout_up'] = breakout_up & volume_cond

        #--------------------------------

        # breakout_up_cond = (((df['consolidation'].shift(1) & df['close'] > df['rolling_high'].shift(1) * (1 + price_buffer_pct)) |
        #                     (df['consolidation'].shift(2) & df['close'] > df['rolling_high'].shift(2) * (1 + price_buffer_pct)) |
        #                     (df['consolidation'].shift(3) & df['close'] > df['rolling_high'].shift(3) * (1 + price_buffer_pct))) &
        #                     df['volume'] > df['volume'].rolling(window=20).mean())
        # df['breakout_up'] = np.where(breakout_up_cond, True, False)

        #--------------------------------

        # # Initialize breakout columns
        # df['breakout_up'] = False
        # df['breakout_down'] = False

        # # Convert to NumPy for speed
        # close = df['close'].values
        # volume = df['volume'].values
        # volume_mean = df['volume'].rolling(window=20).mean().values
        # rolling_high = df['rolling_high'].values
        # rolling_low = df['rolling_low'].values
        # consolidation_flags = df['consolidation'].values

        # # Get only indices where consolidation occurred
        # consolidation_indices = np.where(consolidation_flags)[0]

        # for i in consolidation_indices:
        #     if i + lookahead >= len(df):
        #         continue

        #     high_thresh = rolling_high[i] * (1 + price_buffer_pct)
        #     low_thresh = rolling_low[i] * (1 - price_buffer_pct)
        #     vol_thresh = volume_mean[i] * volume_multiplier

        #     future_close = close[i + 1:i + 1 + lookahead]
        #     future_volume = volume[i + 1:i + 1 + lookahead]

        #     up_breakout = np.where((future_close > high_thresh) & (future_volume > vol_thresh))[0]
        #     down_breakout = np.where((future_close < low_thresh) & (future_volume > vol_thresh))[0]

        #     if len(up_breakout) > 0:
        #         df.at[i + 1 + up_breakout[0], 'breakout_up'] = True
        #     elif len(down_breakout) > 0:
        #         df.at[i + 1 + down_breakout[0], 'breakout_down'] = True


        # for i in range(len(df) - lookahead):
        #     if df.at[i, 'consolidation']:
        #         # Define breakout thresholds
        #         consolidation_high = df.loc[i, 'rolling_high']
        #         consolidation_low = df.loc[i, 'rolling_low']
        #         volume_avg = df.loc[i, 'volume_mean']

        #         # Check future bars for breakout
        #         for j in range(1, lookahead + 1):
        #             breakout_idx = i + j
        #             if breakout_idx >= len(df):
        #                 break

        #             close_price = df.at[breakout_idx, 'close']
        #             current_volume = df.at[breakout_idx, 'volume']

        #             if close_price > consolidation_high * (1 + price_buffer_pct) and current_volume > volume_avg * volume_multiplier:
        #                 df.at[breakout_idx, 'breakout_up'] = True
        #                 break
        #             elif close_price < consolidation_low * (1 - price_buffer_pct) and current_volume > volume_avg * volume_multiplier:
        #                 df.at[breakout_idx, 'breakout_down'] = True
        #                 break


# def add_breakouts(df, lookback=5, price_buffer_pct=0.01, volume_multiplier=1.5, confirmation_window=3):

#     if 'consolidation' not in df.columns:
#         # raise ValueError("Missing 'consolidation' column. Run add_ranges() first.")
#         print("Missing 'consolidation' column. Run add_ranges() first.")

#     else:

#         breakout_up = np.zeros(len(df), dtype=bool)
#         breakout_down = np.zeros(len(df), dtype=bool)
#         breakout_levels = np.full(len(df), np.nan)

#         for i in range(1, lookback + 1):
#             cond_up = (df['consolidation'].shift(i) & (df['close'] > df['rolling_high'].shift(i) * (1 + price_buffer_pct)))
#             breakout_up |= cond_up
#             breakout_levels_up = np.where(cond_up, df['rolling_high'].shift(i) * (1 + price_buffer_pct), breakout_levels)

#             cond_down = (df['consolidation'].shift(i) & (df['close'] < df['rolling_low'].shift(i) * (1 - price_buffer_pct)))
#             breakout_down |= cond_down
#             breakout_levels_down = np.where(cond_down, df['rolling_low'].shift(i) * (1 - price_buffer_pct), breakout_levels)

#         volume_condition = df['volume'] > df['volume'].rolling(window=20).mean() * volume_multiplier
#         df['breakout_up'] = breakout_up & volume_condition
#         df['breakout_down'] = breakout_down & volume_condition

#         # Save breakout level for calculating strength and confirmation
#         df['breakout_level_up'] = breakout_levels_up
#         df['breakout_level_down'] = breakout_levels_down

#         # Breakout strength: how far above the breakout level we closed, in %
#         df['breakout_strength_up'] = np.where(df['breakout_up'], (df['close'] - df['breakout_level_up']) / df['breakout_level_up'], np.nan)
#         df['breakout_strength_down'] = np.where(df['breakout_down'], (df['close'] - df['breakout_level_down']) / df['breakout_level_down'], np.nan)

#         # Confirmed breakout: price remains above breakout level for N candles
#         confirmed_up = np.zeros(len(df), dtype=bool)
#         confirmed_down = np.zeros(len(df), dtype=bool)
#         for i in range(len(df) - confirmation_window):
#             if df['breakout_up'].iloc[i] and not np.isnan(df['breakout_level_up'].iloc[i]):
#                 level = df['breakout_level_up'].iloc[i]
#                 window_prices = df['close'].iloc[i+1:i+1+confirmation_window]
#                 if (window_prices > level).all():
#                     confirmed_up[i] = True

#             if df['breakout_down'].iloc[i] and not np.isnan(df['breakout_level_down'].iloc[i]):
#                 level = df['breakout_level_down'].iloc[i]
#                 window_prices = df['close'].iloc[i+1:i+1+confirmation_window]
#                 if (window_prices < level).all():
#                     confirmed_down[i] = True

#         df['confirmed_breakout_up'] = confirmed_up
#         df['confirmed_breakout_down'] = confirmed_down

#     return df
