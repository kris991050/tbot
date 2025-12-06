import sys, os, pandas as pd, backtrader as bt, pytz, re, math
from datetime import datetime

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.constants import FORMATS, PATHS, CONSTANTS
from utils.timeframe import Timeframe
from strategies import target_handler
from data import hist_market_data_handler
from strategies import bb_rsi_reversal_strategy, sr_bounce_strategy, breakout_strategy
from live import trading_config
from ml import ml_trainer
from miscellaneous import scanner
from features import indicators


def get_strategy_instance(strategy_name:str, config:trading_config.TradingConfig=None):
    config = config or trading_config.TradingConfig()
    
    for direction in ['bull', 'bear']:
        for tf in ['1min', '2min', '5min', '15min', '1h']:
            if f'bb_rsi_reversal_{tf}_{direction}' in strategy_name:
                return bb_rsi_reversal_strategy.BBRSIReversalStrategy(direction=direction, timeframe=Timeframe(tf), rsi_threshold=config.rsi_threshold, 
                                                                      cam_M_threshold=config.cam_M_threshold, revised=config.revised)
        for tf in ['15min', '1h', '4h', '1D']:
            if f'sr_bounce_{tf}_{direction}' in strategy_name:
                return sr_bounce_strategy.SRBounceStrategy(direction=direction, timeframe=Timeframe(tf), cam_M_threshold=config.cam_M_threshold, 
                                                           revised=config.revised, target_factor=config.volatility_factor*config.profit_ratio, #target_factor=config.perc_gain, 
                                                           max_time_factor=config.max_time_factor)
        for tf in ['5min', '15min']:
            if f'breakout_{tf}_{direction}' in strategy_name:
                return breakout_strategy.BreakoutStrategy(direction=direction, timeframe=Timeframe(tf), 
                                                           revised=config.revised, target_factor=config.volatility_factor*config.profit_ratio, #target_factor=config.perc_gain, 
                                                           max_time_factor=config.max_time_factor)
            
    raise ValueError(f"Strategy '{strategy_name}' not recognized")


class TradeManager:
    def __init__(self, ib, config=None, strategy_name:str=None, stop=None, revised:bool=False, 
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
        self.entry_delay = helpers.get_entry_delay_from_timeframe(self.strategy_instance.timeframe)
        self.stop_handler = self._resolve_stop_handler()
        self.trainer, self.trainer_dd = self._load_models()
        self.direction = self._resolve_direction()
        self.all_trades = []
        self.capital = self.get_equity()

    def get_equity(self):
        if not self.config.paper_trading:
            account_values = self.ib.accountValues()
            self.ib.sleep(CONSTANTS.PROCESS_TIME['long'])
            if account_values:
                return float([accountValue.value for accountValue in self.ib.accountValues() if accountValue.tag == 'CashBalance' and accountValue.currency == self.config.currency][0])
            else:
                print(f"⚠️ Could not get account values. Returning 0.")
                return 0
        else:
            return self.config.initial_capital

    def _get_strategy_instance(self):
        return get_strategy_instance(self.strategy_name, self.config)

    def _resolve_stop_handler(self):
        # Stop-loss handler
        if isinstance(self.config.stop, float) and 0 < self.config.stop < 1:
            stop_hdlr = target_handler.FixedStopLossHandler(self.config.stop)

        # elif isinstance(self.stop, list) and all(isinstance(s, str) for s in self.stop):
        elif self.config.stop == 'levels':
            stop_hdlr = target_handler.NextLevelStopLossHandler(timeframe=self.strategy_instance.timeframe)
        elif self.config.stop == 'predicted_drawdown':
            stop_hdlr = target_handler.PredictedDrawdownStopLossHandler(dd_col='predicted_drawdown')
        elif self.config.stop == 'hod':
            stop_hdlr = target_handler.HighOfDayStopLossHandler(direction=self.strategy_instance.direction)
        elif self.config.stop == 'pred_vlty':
            stop_hdlr = target_handler.PredictedVolatilityStopLossHandler(volatility_factor=self.config.volatility_factor, timeframe=self.strategy_instance.timeframe)
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

    def _resolve_features(self, select_features:bool=True):
        if not select_features:
            return ['all'], None
        
        required_columns = self.get_required_columns()
        feature_types = {'indicator_types': [], 'pattern_types': [], 'candle_pattern_list': ['all'], 'level_types': [], 'sr_types': []}
        mtf = []

        for indicator_category in CONSTANTS.INDICATOR_TYPES:
            # Check if any of the indicator names in this category are in required_columns
            if any(indicator in col for col in required_columns for indicator in indicator_category['names']):
                feature_types['indicator_types'].append(indicator_category['category'])

        for pattern_category in CONSTANTS.PATTERN_TYPES:
            if any(pattern in col for col in required_columns for pattern in pattern_category['names']):
                feature_types['pattern_types'].append(pattern_category['category'])

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

        # Check for corresponding timeframes for multi-timeframe
        non_sr_cols = [col for col in required_columns if col not in sr_cols]
        for col in non_sr_cols:
            mtf.extend(tf for tf in set(CONSTANTS.TIMEFRAMES_STD) if f'_{tf}' in col)

        # Remove duplicates
        mtf = list(set(mtf))
        for key in ['indicator_types', 'level_types', 'pattern_types', 'sr_types']:
            feature_types[key] = list(set(feature_types[key]))

        return feature_types, mtf    

    def _resolve_direction(self):
        if hasattr(self.strategy_instance, 'direction'):
            if self.strategy_instance.direction == 'bull':
                return 1
            elif self.strategy_instance.direction == 'bear':
                return -1
        return None

    def resolve_stop_price(self, row:pd.Series, active_stop_price:float=None):
        """
        Resolve stop-loss price at trade entry and store it.
        """
        if active_stop_price is None and self.stop_handler is not None:
            if hasattr(self.stop_handler, 'resolve_stop_price'):
                active_stop_price, _ = self.stop_handler.resolve_stop_price(row, self.direction)

        return active_stop_price

    def set_target_for_entry(self, row:pd.Series, stop_price:float, symbol:str):
        if hasattr(self.strategy_instance.target_handler, 'set_entry_time'):
            self.strategy_instance.target_handler.set_entry_time(row['date'])
        if hasattr(self.strategy_instance.target_handler, 'set_target_price'):
            self.strategy_instance.target_handler.set_target_price(row=row, stop_price=stop_price, symbol=symbol)

    def assess_reason2close(self, curr_row:pd.Series, prev_row:pd.Series, active_stop_price:float):
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

    def _evaluate_prediction(self, prediction:float, threshold:float=None):
        if not prediction:
            return None
        threshold = threshold or self.config.pred_th
        return prediction >= threshold if threshold else True

    def evaluate_RRR(self, row:pd.Series, stop_price:float):
        # Get target from strategy, depending on target type
        if hasattr(self.strategy_instance.target_handler, 'set_target_price'):
            self.strategy_instance.target_handler.set_target_price(row)
        if hasattr(self.strategy_instance.target_handler, 'target_price'):
            target = self.strategy_instance.target_handler.target_price
        elif self.strategy_instance.target_handler.target_str in row:
            target = row[self.strategy_instance.target_handler.target_str]
        else:
            target = None
        
        # Assess Risk to Reward Ratio
        expected_reward = abs(target - row['close']) if target and target > 0 else None
        risk = abs(row['close'] - stop_price) if stop_price and stop_price > 0 else None
        rrr = expected_reward / risk if risk and expected_reward else float('inf')
        
        if self.config.rrr_threshold and rrr < self.config.rrr_threshold:
            return False, rrr  # Skip trade
        return True, rrr

    # def evaluate_quantity(self, prediction:float, capital:float, risk_pct:float):
    #     if isinstance(self.config.size, int):
    #         return self.config.size

    #     elif self.config.size == 'auto':
    #         if not self.config.pred_th or prediction < self.config.pred_th:
    #             return 0
            
    #         tier_range = (1.0 - self.config.pred_th) / self.config.tier_max
    #         quantity_factor = int((prediction - self.config.pred_th) / tier_range) + 1
    #         quantity_factor = min(quantity, self.config.tier_max)

    #         quantity = risk_pct * capital * quantity_factor

    #         return quantity

    def evaluate_quantity(self, prediction:float, price:float, stop_price:float, capital:float, risk_pct:float, exp_factor:float=0.5):
        '''
        **Linear Scaling**: Position increases gradually as the prediction improves above the threshold.
        **Exponential Scaling**: Small improvements in prediction lead to much larger increases in position size.
        **Logarithmic Scaling**: Position size grows at a decreasing rate, starting more aggressively, but slowing down. Offers controlled scaling with a reduced impact for very high predictions.
        **Sigmoidal Scaling**: Position size grows smoothly, with diminishing returns as the prediction improves. The position size increases quickly at first but then slows down, capping at a maximum value.
        '''
        if isinstance(self.config.size, int):
            return self.config.size

        elif isinstance(self.config.size, str):
            growth_mode = helpers.set_var_with_constraints(self.config.size, CONSTANTS.POSITION_SIZING_MODE)
            if not self.config.pred_th or prediction < self.config.pred_th:
                return 0
            
            if stop_price:
                # Calculate the tier range (distance between prediction threshold and max prediction)
                tier_range = (1.0 - self.config.pred_th) / self.config.tier_max

                linear_factor = min(int((prediction - self.config.pred_th) / tier_range) + 1, self.config.tier_max)
                exponential_factor = int(math.exp(exp_factor * (prediction - self.config.pred_th) / tier_range))
                logarithmic_factor = int(math.log((prediction - self.config.pred_th) / tier_range + 1)) + 1
                sigmoid_factor = int((1 / (1 + math.exp(-10 * (prediction - self.config.pred_th) / tier_range))) * self.config.tier_max)

                if growth_mode == 'linear':
                    quantity_factor = linear_factor    
                elif growth_mode == 'exponential':
                    quantity_factor = exponential_factor
                elif growth_mode == 'logarithmic':
                    quantity_factor = logarithmic_factor
                elif growth_mode == 'sigmoidal':
                    quantity_factor = sigmoid_factor
            else:
                quantity_factor = 1 # If no stop_price, fallback on the lowest tier

            # Calculate the quantity based on the risk percentage and capital
            risk_pct_adjusted = risk_pct * quantity_factor
            risk = price - stop_price
            quantity = int(min(math.ceil(risk_pct_adjusted * capital / risk), capital * 0.9 / price)) # 0.9 to keep a buffer and not buy more than the available capital
            return quantity


    def apply_model_predictions(self, df:pd.DataFrame, symbol:str, return_proba:bool=True, shift_features:bool=False):
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
        # stop_price = self.resolve_stop_price(row) if not stop_price else None
        stop_price = self.resolve_stop_price(row) if not stop_price else stop_price

        is_triggered = self.strategy_instance.evaluate_trigger(row)
        is_predicted = self._evaluate_prediction(row.get('model_prediction'), self.config.pred_th)
        # if is_triggered and is_predicted:
        #     print(row['date'])
        #     print()
        #     print()
        is_RRR, rrr = self.evaluate_RRR(row, stop_price)
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
        if hasattr(self.strategy_instance, 'scanner_data_file_name'):
            file_path = os.path.join(daily_data_folder, self.strategy_instance.scanner_data_file_name)
        else:
            file_path = None
        if not file_path or not os.path.exists(file_path):
            print(f"❌ Path to scanner file does not exist: {file_path}")
            return pd.DataFrame()
        
        df_csv_file = helpers.load_df_from_file(file_path)

        return df_csv_file
        # return [symbol for symbol in df_csv_file['Symbol']]
    
    # def scan_bb_rsi_reversal(self):
    #     now = helpers.calculate_now(sim_offset=self.config.sim_offset, tz=self.config.timezone)
    #     if not helpers.is_between_market_times('pre-market', 'end_of_tday', now=now, timezone=self.config.timezone):
    #         return []
    #     print('\n======== FETCHING RSI REVERSALS ========\n')
    #     symbols, _ = scanner.scannerTradingView("RSI-Reversal")#scanner.scannerFinviz("RE")
    #     return symbols

    def get_scanner_data(self, now:datetime, use_daily_data:bool=False, last_scanner_check:pd.Timestamp=None):

        symbols_scanner = []        
        if not use_daily_data:
            if hasattr(self.strategy_instance, 'scanner_func'):
                now = helpers.calculate_now(sim_offset=self.config.sim_offset, tz=self.config.timezone)
                symbols_scanner = self.strategy_instance.scanner_func(now=now, timezone=self.config.timezone)
        else:
            df_csv_file = self.get_symbols_from_daily_data_folder(date=now)

            if not df_csv_file.empty:
                df_csv_file['Time'] = pd.to_datetime(df_csv_file['Time']).dt.tz_localize(self.tz)
                if last_scanner_check:
                    symbols_scanner = df_csv_file.loc[(df_csv_file['Time'] >= last_scanner_check) & (df_csv_file['Time'] <= now), 'Symbol'].tolist()
                else:
                    symbols_scanner = df_csv_file.loc[df_csv_file['Time'] <= now,'Symbol'].tolist()

        return symbols_scanner
    
    def fetch_df(self, symbol, from_time, to_time, file_format=FORMATS.DEFAULT_FILE_FORMAT):
        fetcher = hist_market_data_handler.HistMarketDataFetcher(ib=self.ib, ftype='auto', timeframe=self.strategy_instance.timeframe, file_format=file_format, 
                                                                 delete_existing=True, base_folder=self.base_folder, timezone=self.tz)
        params = {'symbol': symbol, 'timeframe': self.strategy_instance.timeframe, 'to_time': to_time, 'from_time': from_time, 'step_duration': self.config.step_duration}
        df = fetcher.run(params)[symbol]['df']
        
        return df

    def complete_df(self, symbol, file_format=FORMATS.DEFAULT_FILE_FORMAT):
        completer = hist_market_data_handler.HistMarketDataCompleter(self.ib, self.strategy_instance.timeframe, file_format, [symbol], 
                                                                     self.config.step_duration, base_folder=self.base_folder, timezone=self.tz)
        results = completer.run()
        df = results[symbol]['df']
        
        return df

    # @staticmethod
    # def calculate_garch_volatility_recursive(self, close_series:pd.Series, window:int=250, p:int=1, q:int=1):
    #     """
    #     Calculate GARCH volatility recursively for each row. Using rolling window to limit computational cost.
    #     """
    #     print(f"🎢 Calculating GARCH volatility recursively...")
    #     vol_list = [np.nan] * len(close_series)

    #     for i in range(window, len(close_series)):
    #         window_prices = close_series.iloc[i-window:i]
    #         vol_list[i] = helpers.calculate_garch_volatility(window_prices, p=p, q=q)

    #     return pd.Series(vol_list, index=close_series.index)
    
    def add_pred_vlty(self, row:pd.Series, symbol:str):
        if not isinstance(self.strategy_instance.target_handler, target_handler.PredictedVolatilityTargetHandler) and \
            not isinstance(self.stop_handler, target_handler.PredictedVolatilityStopLossHandler):
            return row

        # Add predicted volatility
        wto_time = row['date']
        window_tf = Timeframe(CONSTANTS.WARMUP_MAP[self.strategy_instance.timeframe.pandas])
        wfrom_time = window_tf.subtract_from_date(wto_time)
        df_warmup = self.fetch_df(symbol=symbol, from_time=wfrom_time, to_time=wto_time, file_format=self.config.file_format)
        close_series = df_warmup[['date', 'close']].set_index('date', inplace=False)
        
        # combined = pd.concat([df_warmup[['date', 'close']], df[['date', 'close']]], ignore_index=True)
        # combined = combined.drop_duplicates(subset=['date']).sort_values('date')
        # combined.index = combined['date']  # use date as index

        if self.config.pred_vlty_type == 'garch':
            row['pred_vlty'] = indicators.IndicatorsUtils.calculate_garch_volatility(close_series)
        if self.config.pred_vlty_type == 'ewma':
            row['pred_vlty'] = indicators.IndicatorsUtils.calculate_pred_vlty_recursive(close_series, window=window_tf.to_timedelta, type='ewma')

        # # Merge only needed volatility column back into df
        # df = df.merge(combined[['date', 'garch_volatility']], on='date', how='left')

        return row

    def enrich_df(self, symbol, file_format:str=FORMATS.DEFAULT_FILE_FORMAT, block_add_sr:bool=False, base_timeframe:Timeframe=None, from_time:datetime=None, 
                  to_time:datetime=None, select_features:bool=True):
        feature_types, mtf = self._resolve_features(select_features)
        enricher = hist_market_data_handler.HistMarketDataEnricher(self.ib, timeframe=self.strategy_instance.timeframe, base_timeframe=base_timeframe, 
                                                                   feature_types=feature_types, mtf=mtf, file_format=file_format, base_folder=self.base_folder, 
                                                                   validate=False, block_add_sr=block_add_sr, timezone=self.tz)
        results = enricher.run(symbol=symbol, from_time=from_time, to_time=to_time)
        valid_results = enricher.get_valid_result(results)
        df = valid_results['df'] if valid_results else pd.DataFrame()

        return df 
    
    def load_data_forward(self, symbol, trig_time, to_time=None, file_format='parquet', block_add_sr=False, look_backward:str=None):

        trig_time = trig_time.replace(second=0, microsecond=0) # Floor to minute
        to_time = to_time or trig_time
        from_time = Timeframe(look_backward).subtract_from_date(trig_time)
        
        # Fetch symbol
        df = self.fetch_df(symbol, from_time, to_time, file_format=file_format)

        # Enrich data
        df = self.enrich_df(symbol, from_time, to_time, file_format=file_format, block_add_sr=block_add_sr)

        if not df.empty:
            duration = f"{Timeframe(helpers.get_df_timeframe(df)).to_seconds}S"
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