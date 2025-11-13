import sys, os, datetime, pandas as pd, numpy as np, traceback
from tqdm import tqdm
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.constants import PATHS
from utils.timeframe import Timeframe
from data import hist_market_data_handler
from strategies.target_handler import EODTargetHandler
from strategies.base_strategy import BaseStrategy
from ml import features_processor


class StrategyAnalyzer:

    def __init__(self, ib:IB, symbols:list, strategy:BaseStrategy, config:list, mtf:list, from_time:pd.Timestamp=None, to_time:pd.Timestamp=None,
                 entry_delay:int=1, base_timeframe:Timeframe=None, hist_folder:str=None, file_format='parquet', **kwargs):
        self.ib = ib
        self.symbols = symbols
        self.strategy_instance = strategy
        self.kwargs = kwargs
        self.config = config
        self.mtf = mtf
        self.to_time = to_time
        self.from_time = from_time
        self.base_timeframe = base_timeframe or Timeframe()
        self.hist_folder = hist_folder or PATHS.folders_path['hist_market_data']
        self.entry_delay = entry_delay
        self.file_format = file_format
        self.drop_levels = True
        self.df_results_list = pd.DataFrame()
        self.trigger_columns = []

    def _add_all_features_to_df(self, df, df_results, feature_cols=None):
        """
        df: the original feature-rich dataframe (with 'date' as index or column)
        df_results: output from analyze_post_trigger
        feature_cols: optional list of features to use (defaults to all except targets and labels)
        """

        if df_results.empty:
            return pd.DataFrame()

        # Ensure 'trig_time' is datetime (should retain timezone info if present)
        trigger_times = pd.to_datetime(df_results['trig_time'], errors='coerce')

        # Ensure df has a datetime column to match on
        if df.index.name == 'date' or df.index.name == pd.Index.name or 'date' not in df.columns: # 'date' is in the index
            df = df.reset_index()

        # Make sure the 'date' column is datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Filter the rows where 'date' is in trigger_times
        feature_df = df[df['date'].isin(trigger_times)].copy()
        feature_df = feature_df.rename(columns={'date': 'trig_time'})  # for joining later
        feature_df.index.name = None

        # Merge on 'trig_time'
        merged_df = pd.merge(df_results, feature_df, on='trig_time', suffixes=('', '_feat'))

        # If specific features are provided, filter to those
        if feature_cols:
            feature_cols = ['trig_time'] + feature_cols
            return merged_df[feature_cols + [col for col in df_results.columns if col != 'trig_time']]

        return merged_df

    @staticmethod
    def label(end_profit, max_profit, max_drawdown, mode):
        if mode == 'regression':
            return end_profit
        elif mode == 'binary':
            return 1 if end_profit > 0.005 else 0
        elif mode == 'multi':
            return 2 if end_profit > 0.005 else 0 if end_profit < -0.0025 else 1
        elif mode == 'reward_to_risk':
            return np.nan if max_drawdown == 0 else max_profit / abs(max_drawdown)

    def compute_stats(self, df, trigger_time, target_time, trigger_column, target_handler, side):

        TRADING_MINUTES_PER_YEAR = 252 * 390
        epsilon = 1e-6  # To prevent division by zero

        # try:
        trigger_close = df.at[trigger_time, 'close']
        target_df = df.loc[trigger_time:target_time]

        if not target_df.empty and target_df.index[0] == trigger_time:
            target_df = target_df.iloc[1:]

        if target_df.empty:
            return None

        highs = target_df['high']
        lows = target_df['low']
        closes = target_df['close']

        target_close = closes.values[-1]
        high = highs.max()
        low = lows.min()

        returns = closes.pct_change().dropna()
        log_returns = np.log(closes / closes.shift(1)).dropna()

        max_return = (high - trigger_close) / trigger_close
        min_return = (low - trigger_close) / trigger_close
        close_return = (target_close - trigger_close) / trigger_close
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio_bar = mean_return / std_return if std_return != 0 else np.nan

        target_index = target_df.index
        first_high_time = highs.idxmax()
        first_low_time = lows.idxmin()
        first_event = 'high' if first_high_time < first_low_time else 'low'
        under_water = closes < trigger_close
        drawdown_duration = under_water.sum() * Timeframe(df.attrs['timeframe']).to_seconds / 60 # duration of post_period in minutes
        time_to_max = (first_high_time - trigger_time).total_seconds() / 60  # in minutes
        time_to_min = (first_low_time - trigger_time).total_seconds() / 60

        # Time to recovery (price moves back above trigger close)
        recovery_times = target_df[closes >= trigger_close].index if side == 1 else target_df[closes <= trigger_close].index if side == -1 else None
        time_to_recovery = (recovery_times[0] - trigger_time).total_seconds() / 60 if not recovery_times.empty else np.nan

        target_volatility = (high - low) / trigger_close
        timeframe_min = Timeframe(df.attrs['timeframe']).to_seconds / 60
        event_duration = len(target_df) * timeframe_min # duration of post_period in minutes

        # Compute core values
        end_profit = side * (target_close - trigger_close)
        max_profit = side * (high - trigger_close)
        max_drawdown = -side * (trigger_close - low)
        max_drawdown_pct = max_drawdown / trigger_close

        # Compute normalized values
        end_profit_per_min = end_profit / event_duration + epsilon
        max_drawdown_per_min = max_drawdown / (event_duration + epsilon)
        end_reward_to_risk_ratio = end_profit / (abs(max_drawdown) + epsilon)
        sharpe_ratio_yearly = sharpe_ratio_bar * np.sqrt(TRADING_MINUTES_PER_YEAR / (event_duration + epsilon))

        # Downside deviation for Sortino
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio_bar = mean_return / (downside_std + epsilon)
        sortino_ratio_yearly = sortino_ratio_bar * np.sqrt(TRADING_MINUTES_PER_YEAR / (event_duration + epsilon))


        time_to_max_ratio = time_to_max / (event_duration + epsilon)
        time_to_min_ratio = time_to_min / (event_duration + epsilon)
        drawdown_duration_ratio = drawdown_duration / (event_duration + epsilon)
        time_to_recovery_ratio = time_to_recovery / (event_duration + epsilon)

        market_cap = df.attrs['market_cap']
        market_cap_cat = helpers.categorize_market_cap(market_cap)

        return {
            'symbol': df.attrs['symbol'],
            # 'market_cap': market_cap,
            'market_cap_cat': market_cap_cat,
            'timeframe': df.attrs['timeframe'],
            'timeframe_min': timeframe_min,
            'strategy': trigger_column,
            'target': target_handler.target_str,
            'data_to_time': df.attrs['data_to_time'],
            'data_from_time': df.attrs['data_from_time'],
            'trig_time': trigger_time,
            'trig_close': trigger_close,
            'target_close': target_close,
            'post_trig_high': high,
            'post_trig_low': low,
            'first_event': first_event,
            'time_to_max': time_to_max,
            'time_to_max_ratio': time_to_max_ratio,
            'time_to_min': time_to_min,
            'time_to_min_ratio': time_to_min_ratio,
            'time_to_recovery': time_to_recovery,
            'time_to_recovery_ratio': time_to_recovery_ratio,
            'target_volatility': target_volatility,
            'event_duration': event_duration,
            'max_profit': max_profit,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_per_min': max_drawdown_per_min,
            'end_profit': end_profit,
            'end_profit_per_min': end_profit_per_min,
            'is_profit': end_profit > 0,
            'end_R-R_ratio': end_reward_to_risk_ratio,
            'max_rtn_pct': side * max_return,
            'min_rtn_pct': side * min_return,
            'close_rtn_pct': side * close_return,
            'mean_rtn_pct': mean_return,
            'std_rtn_pct': std_return,
            'sharpe_ratio_bar': sharpe_ratio_bar,
            'sharpe_ratio_yearly': sharpe_ratio_yearly,
            'sortino_ratio_bar': sortino_ratio_bar,
            'sortino_ratio_yearly': sortino_ratio_yearly,
            'log_rtn_volty': log_returns.std(),
            'drawdown_duration': drawdown_duration,
            'drawdown_duration_ratio': drawdown_duration_ratio,
            'label_regression': self.label(end_profit, max_profit, max_drawdown, mode='regression'),
            'label_binary': self.label(end_profit, max_profit, max_drawdown, mode='binary'),
            'label_multi': self.label(end_profit, max_profit, max_drawdown, mode='multi'),
            'label_R-R': self.label(end_profit, max_profit, max_drawdown, mode='reward_to_risk')
        }

        # except Exception as e:
        #     print(f"Error in computing post-trigger stats: {e}.\nFull error: {traceback.format_exc()}")
        #     return None


    def analyze_post_trigger(self, df, trigger_column, target_handler, side):
        # target can now be a string (e.g. 'eod', 'vwap_cross') or timedelta, or a TargetHandler instance
        # target_handler = TargetHandler.from_target(target)

        results = []
        triggered_times = df.index[df[trigger_column]].tolist()

        # pbar = tqdm(total=triggered_times)
        i = 0
        while i < len(triggered_times):

            if i % max(1, len(triggered_times) // 20) == 0:  # every 5%
                print(f"Progress: {int((i / len(triggered_times)) * 100)}%", end="\r")

            trigger_time = triggered_times[i]
            # if trigger_time == pd.Timestamp("2024-08-07 19:59:00", tz="US/Eastern"):
            # if trigger_time == pd.Timestamp("2024-11-19 15:32:00-05:00", tz="US/Eastern"):
            #     print(trigger_time)

            try:
                trigger_time_idx = df.index.get_loc(trigger_time)

                if not trigger_time_idx + self.entry_delay >= len(df):
                    # Delay entry by one bar
                    delayed_trigger_time = df.index[trigger_time_idx + self.entry_delay]
                    # print(f"Trigger at {trigger_time} → Entry at {delayed_trigger_time}")

                    target_time = target_handler.get_target_time(df, delayed_trigger_time)
                    result = self.compute_stats(df, trigger_time, target_time, trigger_column, target_handler, side)
                    if result:
                        results.append(result)

            except Exception as e:
                print(f"Error handling trigger at {trigger_time}: {e}.\nFull error: {traceback.format_exc()}")
                continue

            # Binary search to find next trigger after target_time
            # i = bisect.bisect_left(triggered_times, target_time, lo=i+1)
            # pbar.update(1)
            i = i+1

        # pbar.close()

        return pd.DataFrame(results)#.dropna()


    def pre_analyze(self, df, targets):
        # targets is either timedelta ('5 min') either 'eod_rth', either 'eod'

        timeframe = helpers.get_df_timeframe(df)

        side_map = lambda col: 1 if '_bull' in col else -1 if '_bear' in col else None

        # df = df_original.copy()
        df.attrs = df.attrs
        df.set_index('date', inplace=True)

        # Analyze each trigger
        df_results_list = []
        for col in self.trigger_columns:
            side = side_map(col)
            for target_handler in targets:

                if timeframe.pandas in ['1D', '1W', '1M'] and isinstance(target_handler, EODTargetHandler):
                    print("Cannot use 'eod' or 'eod_rth' with daily or higher timeframes.")
                    continue

                df_results = self.analyze_post_trigger(df, trigger_column=col, target_handler=target_handler, side=side)
                df_results_list.append(df_results)

        return df_results_list

    # def _get_strategy_name(self):
    #     col = self.trigger_columns[0]
    #     return col[:-3] if '_up' in col else col[:-5] if '_down' in col else col

    def assess(self):
        df_results_list = []
        print(f"🔬 Analyzing {len(self.symbols)} symbols:\n{self.symbols}")
        for symbol in self.symbols:
            contract = helpers.get_symbol_contract(self.ib, symbol)[0]
            df, _ = hist_market_data_handler.HistMarketDataLoader(self.ib, contract.symbol, self.base_timeframe, file_format=self.file_format,
                                                               drop_levels=self.drop_levels).load_and_trim(self.from_time, self.to_time)
            for tf in self.config:
                timeframe = tf['timeframe']
                df_tf = df.resample(timeframe.pandas, on='date').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna().reset_index()

                df_tf.attrs = df.attrs
                df_tf.attrs.update({
                    'timeframe': timeframe.pandas,
                    'data_to_time': df_tf['date'].iloc[0] if not df_tf.empty else None,
                    'data_from_time': df_tf['date'].iloc[-1] if not df_tf.empty else None,
                })
                attrs_tf = df_tf.attrs

                # Add indicators
                # timeframe_ibkr = helpers.pandas_to_ibkr_timeframe(tf['timeframe'])
                enricher = hist_market_data_handler.HistMarketDataEnricher(ib=self.ib, timeframe=tf['timeframe'], file_format=self.file_format, save_to_file=True)
                results = enricher.run(symbol=contract.symbol)
                valid_result = enricher.get_valid_result(results)
                if not valid_result:
                    continue
                else:
                    df_tf = valid_result['df']
                    df_tf = helpers.trim_df(df_tf, self.from_time, self.to_time)
                    df_tf.attrs = attrs_tf

                # Remove levels list columns for memory gain
                if self.drop_levels:
                    df_tf = helpers.drop_df_columns(df_tf, '_list')

                # Prepare strategy triggers
                # df_tf, self.trigger_columns = self.strategy_instance_func(df_tf, self.kwargs)
                if self.strategy_instance.revised:
                    attrs = df_tf.attrs
                    print(f"Applying transformation to DataFrame for {contract.symbol} {tf['timeframe']}")
                    df_tf_tr, _ = features_processor.apply_feature_transformations(df_tf)
                    new_cols = [col for col in df_tf_tr.columns if col not in df_tf.columns]
                    df_tf = pd.concat([df_tf, df_tf_tr[new_cols]], axis=1) # Concatenate only those new columns
                    df_tf.attrs = attrs

                df_tf = self.strategy_instance.apply(df_tf)
                self.trigger_columns = self.strategy_instance.trigger_columns
                strategy = self.trigger_columns[0]
                # strategy = self._get_strategy_name()

                # Analyze strategy
                print("Analyzing strategy ", strategy, " for symbol ", contract.symbol, " and timeframe ", tf['timeframe'], "...")
                time_now_start = datetime.datetime.now()
                df_results_tf_list = self.pre_analyze(df_tf, targets=tf['targets'])
                print("\nTime elapsed for analyzing ", strategy, " on ", symbol, ": ", datetime.datetime.now() - time_now_start, "\n")

                # Remove transformed features from df_tf before saving
                if self.strategy_instance.revised:
                    df_tf.drop(columns=new_cols, axis=1, inplace=True)

                # Add remaining features to df_results
                print("Add remaining features to df_results for symbol ", contract.symbol, " and timeframe ", tf['timeframe'], "...")
                for df_results_tf in df_results_tf_list:
                    df_results_tf_enriched = self._add_all_features_to_df(df_tf, df_results_tf)
                    df_results_list.append(df_results_tf_enriched)

        # Concatenate results across symbols by strategy, timeframe and target period
        df_results_combined = pd.concat(df_results_list, ignore_index=True) if df_results_list else pd.DataFrame
        self.df_results_list = [group for _, group in df_results_combined.groupby(['strategy', 'target', 'timeframe'])] if not df_results_combined.empty else []

        # return df_summary, df_results_combined, df_list
        return self.df_results_list







    # def analyze_post_trigger(self, df, trigger_column, target, side):
    #     if not (helpers.is_valid_timedelta(target) or target.lower() in ['eod', 'eod_rth']):
    #         raise ValueError("target must be timedelta or 'eod' or 'eod_rth'")

    #     results = []
    #     triggered_times = df.index[df[trigger_column]].tolist()

    #     if target.lower() == 'eod_rth':
    #         triggered_times = [t for t in triggered_times if t.time() <= constants.CONSTANTS.TH_TIMES['post-market']]

    #     i = 0
    #     while i < len(triggered_times):

    #         trigger_time = triggered_times[i]

    #         if target.lower() == 'eod':
    #             target_time = pd.Timestamp.combine(trigger_time.date(),
    #                                                constants.CONSTANTS.TH_TIMES['end_of_day']).tz_localize(constants.CONSTANTS.TZ_WORK)
    #         elif target.lower() == 'eod_rth':
    #             target_time = pd.Timestamp.combine(trigger_time.date(),
    #                                                constants.CONSTANTS.TH_TIMES['post-market']).tz_localize(constants.CONSTANTS.TZ_WORK)
    #         else:
    #             target_time = trigger_time + pd.to_timedelta(target)

    #         result = self.compute_stats(df, trigger_time, target_time, trigger_column, target, side)
    #         if result:
    #             results.append(result)

    #         # Binary search to find next trigger after target_time
    #         # i = bisect.bisect_left(triggered_times, target_time, lo=i+1)
    #         i = i+1

    #     return pd.DataFrame(results).dropna()
