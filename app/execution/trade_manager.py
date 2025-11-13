import sys, os, pandas as pd, backtrader as bt, pytz, re
from datetime import datetime

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.constants import FORMATS, PATHS, CONSTANTS
from utils.timeframe import Timeframe
from strategies import target_handler
from data import hist_market_data_handler
from strategies import bb_rsi_reversal_strategy, sr_bounce_strategy
from live import trading_config
from ml import ml_trainer


def get_strategy_instance(strategy_name:str, revised:bool=False, rsi_threshold:int=75, cam_M_threshold:int=4, time_target_factor:int=10):
    
    for direction in ['bull', 'bear']:
        for tf in ['1min', '2min', '5min', '15min', '1h']:
            if f'bb_rsi_reversal_{tf}_{direction}' in strategy_name:
                return bb_rsi_reversal_strategy.BBRSIReversalStrategy(direction=direction, timeframe=Timeframe(tf), rsi_threshold=rsi_threshold, 
                                                                      cam_M_threshold=cam_M_threshold, revised=revised)
        
        for tf in ['15min', '1h']:
            if f'sr_bounce_{tf}_{direction}' in strategy_name:
                return sr_bounce_strategy.SRBounceStrategy(direction=direction, timeframe=Timeframe(tf), cam_M_threshold=cam_M_threshold, 
                                                           time_target_factor=time_target_factor, revised=revised)
            
    raise ValueError(f"Strategy '{strategy_name}' not recognized")


class TradeManager:
    def __init__(self, ib, config=None, strategy_name:str=None, stop=None, revised:bool=False, look_backward:str=None, 
                 step_duration:str=None, base_folder:str=None, selector_type:str=None, timezone:pytz=None):
        self.ib = ib
        self.config = config or trading_config.TradingConfig().set_config(locals())
        # self.revised = revised
        # self.strategy_name = strategy_name
        self.strategy_name = self.config.strategy_name + ('_R' if self.config.revised else '')
        # self.look_backward = look_backward
        # self.step_duration = step_duration
        # self.config = config or trading_config.TradingConfig
        self.base_folder = base_folder
        self.tz = self.config.timezone# or CONSTANTS.TZ_WORK
        self.strategy_instance = self._get_strategy_instance()
        # self.timeframe = self.strategy_instance.timeframe
        # self.target = self.strategy_instance.params['target']
        # self.target_handler = self.strategy_instance.target_handler
        # self.stop = stop
        # self.target_handler = self._resolve_target_handler()
        self.stop_handler = self._resolve_stop_handler()
        self.trainer, self.trainer_dd = self._load_models()
        self.direction = self._resolve_direction()
        self.all_trades = []

    def _get_strategy_instance(self):
        return get_strategy_instance(self.strategy_name, self.config.revised)
        # if 'bb_rsi_reversal_bull' in self.strategy:
        #     return bb_rsi_reversal_strategy.BBRSIReversalStrategy(direction='bull', rsi_threshold=75, cam_M_threshold=4, revised=self.config.revised)
        # elif 'bb_rsi_reversal_bear' in self.strategy:
        #     return bb_rsi_reversal_strategy.BBRSIReversalStrategy(direction='bear', rsi_threshold=75, cam_M_threshold=4, revised=self.config.revised)
        # else:
        #     raise ValueError(f"Strategy '{self.strategy}' not recognized")
        
    # def _set_targets(self, target, stop):
        
    #     if stop == 'levels':
    #         self.stop = stop
    #         self.stop_level_types = None#['sr_1h', 'sr_1D', 'sr_1W', 'pivots', 'pivots_D', 'pivots_M', 'levels', 'levels_M']
    #     else:
    #         self.stop = stop or 'predicted_drawdown'
    
    # def _resolve_target_handler(self):
    #     # Target handler
    #     if 'vwap' in self.target:
    #         target_hdlr = target_handler.VWAPCrossTargetHandler(self.timeframe, max_time='eod')
    #     else:
    #         target_hdlr = None
            
    #     return target_hdlr

    def _resolve_stop_handler(self):
        # Stop-loss handler
        if isinstance(self.config.stop, float) and 0 < self.config.stop < 1:
            stop_hdlr = target_handler.FixedStopLossHandler(self.config.stop)

        # elif isinstance(self.stop, list) and all(isinstance(s, str) for s in self.stop):
        elif self.config.stop == 'levels':
            stop_hdlr = target_handler.NextLevelStopLossHandler()
        elif self.config.stop == 'predicted_drawdown':
            stop_hdlr = target_handler.PredictedDrawdownStopLossHandler(dd_col='predicted_drawdown')
        elif self.config.stop == 'hod':
            stop_hdlr = target_handler.HighOfDayStopLossHandler(direction=self.strategy_instance.direction)
        else:
            stop_hdlr = None # Optional fallback

        return stop_hdlr

    def _load_models(self):
        target = self.strategy_instance.target_handler.target_str
        trainer = ml_trainer.ModelTrainer(model_type=self.config.model_type, selector_type=self.config.selector_type,
                                               strategy_name=self.strategy_name, target=target)
        trainer.load()
        if self.config.stop == 'predicted_drawdown':
            trainer_dd = ml_trainer.ModelTrainer(model_type='xgboost', strategy_name=self.strategy_name, target=target, drawdown=True)
            trainer_dd.load()
        else:
            trainer_dd = None

        return trainer, trainer_dd
    
    def get_strategy_required_columns(self):
        return list({
                # *getattr(self.strategy_instance, 'trigger_columns', []), 
                *getattr(self.strategy_instance, 'required_columns', []), 
                *getattr(self.strategy_instance.target_handler, 'required_columns', []), 
                *getattr(self.stop_handler, 'required_columns', []), 
                'date'
            })

    def get_required_columns(self):
        strategy_required_columns = self.get_strategy_required_columns()
        model_required_columns = self.trainer.feature_names
        model_dd_required_columns = self.trainer_dd.feature_names if self.trainer_dd is not None else []

        required_columns = list(set(strategy_required_columns + model_required_columns + model_dd_required_columns + ['date']))

        if 'market_cap_cat' in required_columns:
            required_columns.remove('market_cap_cat')
        return required_columns
    
    # def _get_timeframes_for_feature(self, feature_name, required_columns):
    #     """
    #     Helper function to extract the timeframes for the given feature from the required columns.
    #     """
    #     mtf_timeframes = []
    #     for feature in CONSTANTS.MTF_SETTINGS:
    #         if feature_name == feature['name']:
    #             # Find timeframes corresponding to this feature
    #             for timeframe in feature['timeframes']:
    #                 if any(f"{feature_name}_{timeframe}" in col for col in required_columns):
    #                     mtf_timeframes.append(timeframe)
    #     return mtf_timeframes

    def _resolve_features(self):
        # required_columns = self.get_required_columns()
        # feature_types = {'indicator_types': [], 'pattern_types': [], 'candle_pattern_list': [], 'level_types': [], 'sr_types': []}

        # if any(ind in col for ind in ['ema', 'sma', 'adx', 'macd'] for col in required_columns):
        #     feature_types['indicator_types'].append('trend')
        # if any(ind in col for ind in ['vwap', 'r_vol', 'avg_vol', 'pm_vol'] for col in required_columns):
        #     feature_types['indicator_types'].append('volume')
        # if any(ind in col for ind in ['bband', 'atr', 'day_range', 'volatility_ratio', 'volatility_change'] for col in required_columns):
        #     feature_types['indicator_types'].append('volatility')
        # if any(ind in col for ind in ['rsi', 'awesome'] for col in required_columns):
        #     feature_types['indicator_types'].append('momentum')
        # if any(ind in col for ind in ['gap', 'change'] for col in required_columns):
        #     feature_types['indicator_types'].append('price')

        # if any(ind in col for ind in ['levels'] for col in required_columns):
        #     feature_types['level_types'].append('daily')
        # if any(ind in col for ind in ['levels_M'] for col in required_columns):
        #     feature_types['level_types'].append('monthly')
        # if any(ind in col for ind in ['pivots'] for col in required_columns):
        #     feature_types['level_types'].append('camarilla')

        # if any(ind in col for ind in ['hammer', 'engulfing', 'marubozu', 'doji', 'volume_spike', 'bullish_score', 'bearish_score', 'score_bias', 'return', 'directional_bias', 'bias_trend', 'hybrid_bias', 'hybrid_direction'] for col in required_columns):
        #     feature_types['pattern_types'].append('candle')
        # if any(ind in col for ind in ['divergence'] for col in required_columns):
        #     feature_types['pattern_types'].append('divergence')
        # if any(ind in col for ind in ['low_volume', 'bband_width_pct', 'atr_in_range', 'inside_bar', 'dbscan_cluster', 'consolidation'] for col in required_columns):
        #     feature_types['pattern_types'].append('range')
        # if any(ind in col for ind in ['breakout', 'price_buffer_pct'] for col in required_columns):
        #     feature_types['pattern_types'].append('breakout', 'range')
        # if any(ind in col for ind in ['trend'] for col in required_columns):
        #     feature_types['pattern_types'].append('trend')
        # if any(ind in col for ind in ['index_trend'] for col in required_columns):
        #     feature_types['pattern_types'].append('index_trend')
        
        # feature_types['candle_pattern_list'] = ['all']

        # # sr_cols = any(ind in col for ind in ['sr_'] for col in required_columns)
        # sr_cols = [col for col in required_columns if 'sr_' in col]
        # for sr_col in sr_cols:
        #     # Extract the timeframe (1min, 1h, 1D, etc.) from the column name (e.g., sr_1min_dist_to_next)
        #     match = re.search(r'sr_(\d+(?:min|h|D|W))', sr_col)
        #     if match:
        #         timeframe = match.group(1)
        #         # Find the corresponding SR setting
        #         sr_setting = next((sr for sr in CONSTANTS.SR_SETTINGS if sr['timeframe'] == timeframe), None)
        #         if sr_setting:
        #             feature_types['sr_types'].append(sr_setting)

        required_columns = self.get_required_columns()
        feature_types = {'indicator_types': [], 'pattern_types': [], 'candle_pattern_list': ['all'], 'level_types': [], 'sr_types': []}
        mtf = []

        # Loop through indicator types and classify them
        # for indicator in CONSTANTS.INDICATOR_TYPES:
        #     if any(indicator['name'] in col for col in required_columns):
        #         feature_types['indicator_types'].append(indicator['category'])
        for indicator_category in CONSTANTS.INDICATOR_TYPES:
            # Check if any of the indicator names in this category are in required_columns
            if any(indicator in col for col in required_columns for indicator in indicator_category['names']):
                feature_types['indicator_types'].append(indicator_category['category'])

        # Loop through pattern types and classify them
        # for pattern in CONSTANTS.PATTERN_TYPES:
        #     if any(pattern['name'] in col for col in required_columns):
        #         feature_types['pattern_types'].append(pattern['category'])
        for pattern_category in CONSTANTS.PATTERN_TYPES:
            if any(pattern in col for col in required_columns for pattern in pattern_category['names']):
                feature_types['pattern_types'].append(pattern_category['category'])

        # Loop through level types and classify them
        # for level in CONSTANTS.LEVEL_TYPES:
        #     if any(level['name'] in col for col in required_columns):
        #         feature_types['level_types'].append(level['category'])
        for level_category in CONSTANTS.LEVEL_TYPES:
            if any(level in col for col in required_columns for level in level_category['names']):
                feature_types['level_types'].append(level_category['category'])

        # Handle SR Columns and Match with SR Settings
        sr_cols = [col for col in required_columns if 'sr_' in col]
        for sr_col in sr_cols:
            match = re.search(r'sr_(\d+(?:min|h|D|W))', sr_col)
            if match:
                timeframe = match.group(1)
                feature_types['sr_types'].append(timeframe)
                # sr_setting = next((sr for sr in CONSTANTS.SR_SETTINGS if sr['timeframe'] == timeframe), None)
                # if sr_setting:
                #     feature_types['sr_types'].append(sr_setting)

        # Check for corresponding timeframes for multi-timeframe
        non_sr_cols = [col for col in required_columns if col not in sr_cols]
        # for col in non_sr_cols:
        #     for tf in CONSTANTS.TIMEFRAMES_STD:
        #         if f'_{tf}' in col:
        #             mtf.append(tf)
        for col in non_sr_cols:
            mtf.extend(tf for tf in set(CONSTANTS.TIMEFRAMES_STD) if f'_{tf}' in col)

        # Remove duplicates
        mtf = list(set(mtf))
        for key in ['indicator_types', 'level_types', 'pattern_types', 'sr_types']:
            feature_types[key] = list(set(feature_types[key]))

        # feature_types = ['all']
        # mtf = None

        return feature_types, mtf    

    def _resolve_direction(self):
        if hasattr(self.strategy_instance, 'direction'):
            if self.strategy_instance.direction == 'bull':
                return 1
            elif self.strategy_instance.direction == 'bear':
                return -1
        return None

    def resolve_stop_price(self, curr_row, active_stop_price=None):
        """
        Resolve stop-loss price at trade entry and store it.
        """
        if active_stop_price is None and self.stop_handler is not None:
            if hasattr(self.stop_handler, 'resolve_stop_price'):
                active_stop_price, _ = self.stop_handler.resolve_stop_price(curr_row, self.direction)

        return active_stop_price

    def assess_reason2close(self, curr_row, prev_row, active_stop_price):
        """
        Determines if the current trade should be closed due to:
        - Cached stop-loss value
        - Target exit condition
        """
        stop_reason = None

        if active_stop_price is not None and self.stop_handler is not None:
            stop_reason = self.stop_handler.check_stop_loss(curr_row, active_stop_price, self.direction)

        target_reason = self.strategy_instance.target_handler.get_target_event(prev_row, curr_row)

        return stop_reason or target_reason

    def _evaluate_prediction(self, prediction, threshold=None):
        if not prediction:
            return None
        threshold = threshold or self.config.pred_th
        return prediction >= threshold if threshold else True

    def _evaluate_RRR(self, curr_row, stop_price):
        # Assess Risk to Reward Ratio
        target = curr_row[self.strategy_instance.target_handler.target_str]
        expected_reward = abs(target - curr_row['close']) if target and target > 0 else None
        risk = abs(curr_row['close'] - stop_price) if stop_price and stop_price > 0 else None
        rrr = expected_reward / risk if risk and expected_reward else float('inf')
        
        if self.config.rrr_threshold and rrr < self.config.rrr_threshold:
            return False, rrr  # Skip trade
        return True, rrr
        
        
        # predicted_dd = curr_row.get('predicted_drawdown')

        # # # condition_vwap = (self.target == 'vwap_cross' and 'vwap' in curr_row)
        # # condition_vwap = isinstance(self.target_handler, target_handler.VWAPCrossTargetHandler)
        # # expected_reward = abs(curr_row[f'vwap_{self.timeframe}'] - curr_row['close']) if condition_vwap else None
        # # predicted_dd = curr_row.get('predicted_drawdown') if condition_vwap else None
        # # # expected_reward = self.estimate_expected_reward(curr_row)

        # if self.config.rrr_threshold and predicted_dd and expected_reward:
        #     rrr = expected_reward / predicted_dd if predicted_dd > 0 else float('inf')
        #     if rrr < self.config.rrr_threshold:
        #         return False  # Skip trade

    def evaluate_quantity(self, prediction):
        if isinstance(self.config.size, int):
            return self.config.size

        elif self.config.size == 'auto':
            if self.config.pred_th and prediction >= self.config.pred_th:
                tier_range = (1.0 - self.config.pred_th) / self.config.tier_max
                quantity = int((prediction - self.config.pred_th) / tier_range) + 1

                return min(quantity, self.config.tier_max)
            else:
                return 1

    def apply_model_predictions(self, df, symbol, return_proba=True, shift_features=False):
        if 'market_cap_cat' not in df.columns and 'market_cap' in df.attrs:
            df['market_cap_cat'] = helpers.categorize_market_cap(df.attrs['market_cap'])
        else:
            df['market_cap_cat'] = None
        
        # Predict using model and add predictions to dataframe
        print(f"🤖 Generating trade entry predictions for {symbol}...")
        predictions = self.trainer.predict(df.copy(), return_proba=return_proba, shift_features=shift_features)
        df['model_prediction'] = predictions

        if self.trainer_dd is not None:
            print("🤖 Generating drawdown predictions...")
            predictions_dd_pct = self.trainer_dd.predict(df.copy(), return_proba=False, shift_features=shift_features)
            df['predicted_drawdown'] = predictions_dd_pct * df['close']

        return df

    def evaluate_entry_conditions(self, row, stop_price, expand=False, display_triggers=False):
        stop_price = self.resolve_stop_price(row) if not stop_price else None

        is_triggered = self.strategy_instance.evaluate_trigger(row)
        is_predicted = self._evaluate_prediction(row.get('model_prediction'), self.config.pred_th)
        is_RRR, rrr = self._evaluate_RRR(row, stop_price)
        if display_triggers and is_triggered:
            trigger_cols = [col for col in self.strategy_instance.required_columns if not any(keyword in col for keyword in ['open', 'high', 'low', 'volume'])]
            print(f"\nEntry conditions triggered at {row['date']}")#.\nTriggers: {row[trigger_cols]}")
            print(f"Prediction: {row.get('model_prediction'):.2f} (threshold: {self.config.pred_th})")
            print(f"RRR: {rrr} (close: {row['close']:.2f} | stop price: {stop_price}) | target price: {row[self.strategy_instance.target_handler.target_str]:.2f} | threshold: {self.config.rrr_threshold}")
        if not expand:
            return is_triggered and is_predicted and is_RRR
        else:
            return is_triggered, is_predicted, is_RRR
    
    def evaluate_discard_conditions(self, row):
        return self.strategy_instance.evaluate_discard(row)
    
    def get_symbols_from_daily_data_folder(self, date=datetime.now()):
        daily_data_folder = helpers.get_path_daily_data_folder(date, create_if_none=False)
        if 'bb_rsi_reversal' in self.strategy_instance.name:
            file_path = os.path.join(daily_data_folder, PATHS.daily_csv_files['bb_rsi_reversal'])
        else:
            file_path = None
        if not os.path.exists(file_path):
            print(f"❌ Path does not exist: {file_path}")
            return pd.DataFrame()
        
        df_csv_file = helpers.load_df_from_file(file_path)

        return df_csv_file
        # return [symbol for symbol in df_csv_file['Symbol']]
    
    def get_scanner_data(self, now, last_scanner_check=None):
        df_csv_file = self.get_symbols_from_daily_data_folder(date=now)

        symbols_scanner = []
        if not df_csv_file.empty:
            df_csv_file['Time'] = pd.to_datetime(df_csv_file['Time']).dt.tz_localize(self.tz)
            if last_scanner_check:
                symbols_scanner = df_csv_file.loc[(df_csv_file['Time'] >= last_scanner_check) & (df_csv_file['Time'] <= now), 'Symbol'].tolist()
            else:
                symbols_scanner = df_csv_file.loc[df_csv_file['Time'] <= now,'Symbol'].tolist()
        
        # symbols_scanner = symbols_scanner[0:4]

        return symbols_scanner

    def fetch_df(self, symbol, from_time, to_time, file_format=FORMATS.DEFAULT_FILE_FORMAT):
        fetcher = hist_market_data_handler.HistMarketDataFetcher(ib=self.ib, ftype='auto', timeframe=self.strategy_instance.timeframe, file_format=file_format, 
                                                                 delete_existing=True, base_folder=self.base_folder, timezone=self.tz)
        params = {'symbol': symbol, 'timeframe': self.strategy_instance.timeframe, 'to_time': to_time, 'from_time': from_time, 'step_duration': self.config.step_duration}
        df = fetcher.run(params)[symbol]['df']
        
        return df

    def complete_df(self, symbol, file_format='parquet'):
        completer = hist_market_data_handler.HistMarketDataCompleter(self.ib, self.strategy_instance.timeframe, file_format, [symbol], 
                                                                     self.config.step_duration, base_folder=self.base_folder, timezone=self.tz)
        results = completer.run()
        df = results[symbol]['df']
        
        return df

    def enrich_df(self, symbol, file_format:str='parquet', block_add_sr:bool=False, base_timeframe:Timeframe=None, from_time:datetime=None, to_time:datetime=None):
        # bb_rsi_tf_list = bb_rsi_tf_list or ['5min', '15min', '60min', '1D'] if 'bb_rsi_reversal' in self.strategy_instance.name else []
        feature_types, mtf = self._resolve_features()
        enricher = hist_market_data_handler.HistMarketDataEnricher(self.ib, timeframe=self.strategy_instance.timeframe, base_timeframe=base_timeframe, 
                                                                   feature_types=feature_types, mtf=mtf, file_format=file_format, base_folder=self.base_folder, 
                                                                   validate=False, block_add_sr=block_add_sr, timezone=self.tz)
        results = enricher.run(symbol=symbol, from_time=from_time, to_time=to_time)
        # if results:
        #     valid_df = next((r['df'] for r in results if r.get('df') is not None and not r['df'].empty), pd.DataFrame())
        #     df = valid_df
        #     df = df[pd.to_datetime(df["date"]).between(from_time, to_time)] if not df.empty else pd.DataFrame()
        # else:
        #     df = pd.DataFrame()
        valid_results = enricher.get_valid_result(results)

        return valid_results['df']
    
    def load_data_live(self, symbol, trig_time, to_time=None, file_format='parquet', block_add_sr=False):

        trig_time = trig_time.replace(second=0, microsecond=0) # Floor to minute
        to_time = to_time or trig_time
        from_time = helpers.substract_duration_from_time(trig_time, self.config.look_backward)
        
        # Fetch symbol
        # df = self.fetch_df(symbol, to_time, from_time, file_format=file_format)
        df = self.fetch_df(symbol, from_time, to_time, file_format=file_format)

        # Enrich data
        # df = self.enrich_df(symbol, to_time, from_time, file_format=file_format, block_add_sr=block_add_sr)
        df = self.enrich_df(symbol, from_time, to_time, file_format=file_format, block_add_sr=block_add_sr)

        if not df.empty:
            duration = f"{helpers.timeframe_to_seconds(helpers.get_df_timeframe(df))}S"
            adjusted_trig_time = helpers.substract_duration_from_time(trig_time, duration)
            df = df[(df['date'] >= adjusted_trig_time) & (df['date'] <= to_time)] if not df.empty else pd.DataFrame()

        return df
    

class IBKRCanadaCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.008),  # $0.008/share
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),  # per unit
    )

    def _getcommission(self, size, price, pseudoexec):
        trade_value = abs(size) * price
        commission = abs(size) * self.p.commission
        commission = max(commission, 1.00)         # Min CAD $1.00
        commission = min(commission, trade_value * 0.005)  # Max 0.5%
        return commission



# class TradingManager:

#     @staticmethod
#     def get_required_columns(strategy, target_handler, stop_handler):
#         return list({
#                 *getattr(strategy, 'trigger_columns', []),
#                 *getattr(strategy, 'required_columns', []),
#                 *getattr(target_handler, 'required_columns', []),
#                 *getattr(stop_handler, 'required_columns', [])
#             })

#     @staticmethod
#     def resolve_direction(strategy):
#         if hasattr(strategy, 'direction'):
#             if strategy.direction == 'bull':
#                 return 1
#             elif strategy.direction == 'bear':
#                 return -1
#         return None

#     @staticmethod
#     def resolve_stop_price(curr_row, active_stop_price, stop_handler, direction):
#         """
#         Resolve stop-loss price at trade entry and store it.
#         """
#         if active_stop_price is None and stop_handler is not None:
#             if hasattr(stop_handler, 'resolve_stop_price'):
#                 active_stop_price, le = stop_handler.resolve_stop_price(curr_row, direction)

#         return active_stop_price

#     @staticmethod
#     def assess_reason2close(curr_row, prev_row, target_handler, stop_handler, active_stop_price, direction):
#         """
#         Determines if the current trade should be closed due to:
#         - Cached stop-loss value
#         - Target exit condition
#         """
#         stop_reason = None

#         if active_stop_price is not None and stop_handler is not None:
#             stop_reason = stop_handler.check_stop_loss(curr_row, active_stop_price, direction)

#         target_reason = target_handler.get_target_event(prev_row, curr_row)

#         return stop_reason or target_reason

#     @staticmethod
#     def evaluate_prediction(prediction, pred_th, threshold=None):
#         if not prediction:
#             return None
#         threshold = threshold or pred_th
#         return prediction >= threshold if threshold else True

#     @staticmethod
#     def evaluate_RRR(curr_row, target_hdlr, rrr_threshold):
#         # Assess Risk to Reward Ratio
#         # condition_vwap = (self.target == 'vwap_cross' and 'vwap' in curr_row)
#         condition_vwap = isinstance(target_hdlr, target_handler.VWAPCrossTargetHandler)
#         expected_reward = abs(curr_row['vwap'] - curr_row['close']) if condition_vwap else None
#         predicted_dd = curr_row.get('predicted_drawdown') if condition_vwap else None
#         # expected_reward = self.estimate_expected_reward(curr_row)

#         if rrr_threshold and predicted_dd and expected_reward:
#             rrr = expected_reward / predicted_dd if predicted_dd > 0 else float('inf')
#             if rrr < rrr_threshold:
#                 return False  # Skip trade
#         return True

#     @staticmethod
#     def evaluate_quantity(prediction, size, pred_th, tier_max):
#         if isinstance(size, int):
#             return size

#         elif size == 'auto':
#             if pred_th and prediction >= pred_th:
#                 tier_range = (1.0 - pred_th) / tier_max
#                 quantity = int((prediction - pred_th) / tier_range) + 1

#                 return min(quantity, tier_max)
#             else:
#                 return 1

#     # @staticmethod
#     # def evaluate_quantity(prediction, size, pred_th, tier_max):
#     #     if isinstance(size, int):
#     #         return size

#     #     elif size == 'auto':
#     #         if pred_th and prediction <= pred_th:
#     #             tier_range = int((1.0 - pred_th) / tier_max)
#     #             quantity = int((prediction - pred_th) / tier_range) + 1

#     #             # Cap quantity to tier_max
#     #             return min(quantity, tier_max)
#     #         else: return 1

#     @staticmethod
#     def apply_model_predictions(df, trainer, trainer_dd=None, return_proba=True, shift_features=False):
#         if 'market_cap_cat' not in df.columns:
#             df['market_cap_cat'] = helpers.categorize_market_cap(df.attrs['market_cap'])

#         # Predict using model and add predictions to dataframe
#         print("🤖 Generating trade entry predictions...")
#         predictions = trainer.predict(df.copy(), return_proba=return_proba, shift_features=shift_features)
#         df['model_prediction'] = predictions

#         if trainer_dd is not None:
#             print("🤖 Generating drawdown predictions...")
#             predictions_dd_pct = trainer_dd.predict(df.copy(), return_proba=False, shift_features=shift_features)
#             df['predicted_drawdown'] = predictions_dd_pct * df['close']

#         return df

#     @staticmethod
#     def evaluate_entry_conditions(row, strategy, target_handler, pred_th, rrr_threshold):
#         is_triggered = strategy.evaluate_trigger(row)
#         is_predicted = TradingManager.evaluate_prediction(row.get('model_prediction'), pred_th)
#         is_RRR = TradingManager.evaluate_RRR(row, target_handler, rrr_threshold)
#         return is_triggered and is_predicted and is_RRR
    
#     @staticmethod
#     def get_symbols_from_daily_data_folder(strategy_name, date=datetime.datetime.now()):
#         daily_data_folder = helpers.get_path_daily_data_folder(date, create_if_none=False)
#         if 'bb_rsi_reversal' in strategy_name:
#             file_path = os.path.join(daily_data_folder, PATHS.daily_csv_files['bb_rsi_reversal'])
#         else:
#             file_path = None
#         if not os.path.exists(file_path):
#             print(f"❌ Path does not exist: {file_path}")
#             return pd.DataFrame()
        
#         df_csv_file = helpers.load_df_from_file(file_path)

#         return df_csv_file
#         # return [symbol for symbol in df_csv_file['Symbol']]

#     @staticmethod
#     def get_strategy_instance(strategy_name: str, revised: bool):
#         if 'bb_rsi_reversal_bull' in strategy_name:
#             return bb_rsi_reversal_strategy.BBRSIReversalStrategy(direction='bull', rsi_threshold=75, cam_M_threshold=4, revised=revised)
#         elif 'bb_rsi_reversal_bear' in strategy_name:
#             return bb_rsi_reversal_strategy.BBRSIReversalStrategy(direction='bear', rsi_threshold=75, cam_M_threshold=4, revised=revised)
#         else:
#             raise ValueError(f"Strategy '{strategy_name}' not recognized")

#     @staticmethod
#     def load_data_forward(ib, symbol, timeframe, trig_time, to_time, strategy_name, look_backward='2 M', file_format='parquet'):

#         # Fetch symbol
#         fetcher = hist_market_data_handler.HistMarketDataFetcher(ib, timeframe, file_format, delete_existing=True)
#         from_time = helpers.substract_duration_from_time(trig_time, look_backward)
#         params = {'symbol': symbol, 'to_time': to_time, 'from_time': from_time, 'step_duration': '1 W'}
#         df = helpers.load_df_from_file(fetcher.run(params))

#         # Enrich data
#         df = TradingManager.enrich_df(ib, symbol, timeframe, to_time, from_time, strategy_name, file_format=file_format)
#         df = df[(df['date'] >= trig_time) & (df['date'] <= to_time)] if not df.empty else pd.DataFrame()

#         return df
    
#     @staticmethod
#     def enrich_df(ib, symbol, timeframe, to_time, from_time, strategy_name, bb_rsi_tf_list=None, file_format='parquet'):
#         bb_rsi_tf_list = bb_rsi_tf_list or ['5min', '15min', '60min', '1D'] if 'bb_rsi_reversal' in strategy_name else []
#         enricher = hist_market_data_handler.HistMarketDataEnricher(ib, timeframe, file_format=file_format,
#                                                                    bb_rsi_tf_list=bb_rsi_tf_list)
#         results = enricher.run(symbol=symbol)
#         if results is not None:
#             valid_df = next((r['df'] for r in results if r.get('df') is not None and not r['df'].empty), pd.DataFrame())
#             df = valid_df
#             df = df[pd.to_datetime(df["date"]).between(from_time, to_time)]
#         else:
#             df = pd.DataFrame()

#         return df
