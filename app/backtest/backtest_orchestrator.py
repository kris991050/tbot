import os, sys, pandas as pd, datetime
from dataclasses import dataclass
from ib_insync import *

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

import trade_evaluator
from utils import helpers
from utils.constants import CONSTANTS, PATHS
from utils.timeframe import Timeframe
import custom_backtester, backtrader_backtester
from execution import trade_manager


class BacktestOrchestrator:
    def __init__(self, ib:IB, strategy_name:str, stop=None, revised:bool=False, symbols:list=None, seed:int=None, timeframe=Timeframe(), 
                 config=None, engine_type:str='custom', selector_type:str='rf', entry_delay:int=1, mode:str='backtest', timezone=None, 
                 look_bacward:str=None, step_duration:str=None):
        self.manager = trade_manager.TradeManager(ib, config, strategy_name, stop, revised=revised, look_backward=look_bacward, step_duration=step_duration, 
                                                  selector_type=selector_type, timezone=timezone)
        self.strategy_required_columns = self.manager.get_strategy_required_columns()
        # self.required_columns = self.manager.get_required_columns()
        self.seed = seed
        self.engine_type = engine_type
        self.entry_delay = entry_delay
        self.mode = helpers.set_var_with_constraints(mode, CONSTANTS.MODES['backtest'])
        self.setup_paths()
        self.symbols = self.get_symbols(symbols)
        self.all_trades = []

    def setup_paths(self):
        self.file_name_pattern = f"{self.engine_type}_{self.manager.strategy_name}_{self.manager.config.model_type}_{self.manager.config.selector_type}_{self.mode}_{self.manager.strategy_instance.timeframe}_seed{self.seed}_{self.manager.strategy_instance.target_handler.target_column}_{self.manager.config.stop}_pred{str(self.manager.config.pred_th)}"
        base_folder = PATHS.folders_path['strategies_data']
        self.outputs_folder = os.path.join(base_folder, self.manager.strategy_name, f'backtest_{self.file_name_pattern}')
        # self.data_folder = os.path.join(base_folder, 'backtest_data')
        # os.makedirs(self.data_folder, exist_ok=True)
        self.checkpoint_file_path = os.path.join(self.outputs_folder, 'test_checkpoint.json')
    
    # def _save_checkpoint(self, completed_symbol_dates):
    #     # Sort by the second item (the time string), parsed as datetime
    #     checkpoint_data = {
    #         'all_trades': self.all_trades,
    #         'completed_symbol_dates': sorted(list(completed_symbol_dates), key=lambda x: datetime.datetime.fromisoformat(x[1]))
    #     }

    #     helpers.save_json(checkpoint_data, self.checkpoint_file_path)
    
    def _save_checkpoint(self, completed_symbol_dates):
        def safe_parse_time(item):
            try:
                return datetime.datetime.fromisoformat(item[1]) if item[1] else datetime.datetime.max
            except Exception:
                return datetime.datetime.max  # invalid format → send to end

        checkpoint_data = {
            'all_trades': self.all_trades,
            'completed_symbol_dates': sorted(list(completed_symbol_dates), key=safe_parse_time)
        }
        helpers.save_json(checkpoint_data, self.checkpoint_file_path)
    
    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file_path):
            checkpoint_data = helpers.load_json(self.checkpoint_file_path)
            self.all_trades = checkpoint_data.get('all_trades', [])
            return set(tuple(item) for item in checkpoint_data.get('completed_symbol_dates', []))
        return set()

    def get_symbols(self, symbols):
        if symbols and isinstance(symbols, list):
            return symbols
        if self.seed:
            return helpers.get_symbol_seed_list(self.seed)
        df = pd.read_csv(self.manager.trainer.paths['data'])
        if df.empty:
            print("Results Dataframe is empty")
            return []
        else:
            return sorted(df['symbol'].unique())

    def load_data_backtest(self, symbol, data_set='test', file_format='parquet'):
        # path = os.path.join(self.data_folder, f"df_test_{symbol}_{self.manager.timeframe}.{file_format}")

        # if os.path.exists(path):
        #     return helpers.load_df_from_file(path)

        test_to = pd.to_datetime(self.manager.trainer.data_ranges[data_set][1], utc=True).tz_convert(CONSTANTS.TZ_WORK)
        test_from = pd.to_datetime(self.manager.trainer.data_ranges[data_set][0], utc=True).tz_convert(CONSTANTS.TZ_WORK)

        # test_to = pd.Timestamp('2025-10-01 20:00:00', tz=CONSTANTS.TZ_WORK)
        # test_from = pd.Timestamp('2025-08-01 04:00:00', tz=CONSTANTS.TZ_WORK)

        df = self.manager.enrich_df(symbol, file_format=file_format)
        
        # Trim again to ensure it is trimmed
        df = helpers.trim_df(df, from_time=test_from, to_time=test_to)
        # if 'bb_rsi_reversal' in self.manager.strategy:
        #     df = self.manager.enrich_df(symbol, test_start, test_end, file_format=file_format)
        # else:
        #     loader = hist_market_data_handler.HistMarketDataLoader(
        #         None, symbol, self.manager.timeframe,
        #         file_format='parquet', data_type='enriched'
        #     )
        #     df = loader.load_and_trim(test_end, test_start)

        # helpers.save_df_to_file(df, path, file_format='parquet')
        return df

    def select_engine(self, df, symbol):
        if self.engine_type == 'custom':
            return custom_backtester.CustomBacktestEngine(df, symbol, self.manager, self.entry_delay)
        elif self.engine_type == 'backtrader':
            return backtrader_backtester.BacktraderBacktestEngine(df=df, symbol=symbol, manager=self.manager, entry_delay=self.entry_delay)
        else:
            raise ValueError(f"Unknown engine type: {self.engine_type}")

    def _run_backtest_engine(self, df, symbol):
        
        # df, _ = features_processor.apply_feature_transformations(df, drop=False)
        # df = self.manager.trainer.preprocessor.prepare_features(df, self.manager.model, display_features=False, drop=False)
        df = self.manager.apply_model_predictions(df, symbol)

        # Trimming df to only required columns
        # df = df[self.required_columns]
        df = df[self.strategy_required_columns + ['model_prediction']]

        # dropped_required_columns = [col for col in required_columns if col not in df_pred.columns]
        # pred_cols = ['date', 'model_prediction', 'predicted_drawdown'] if 'predicted_drawdown' in df_pred.columns else ['date', 'model_prediction']
        # df = pd.merge(df, df_pred[pred_cols], on='date', how='left')

        # print(f"Adding dropped required columns back to main df for {symbol}:\n{dropped_required_columns}")
        # for col in dropped_required_columns:
        #     df_pred[col] = df[col]

        engine = self.select_engine(df, symbol)
        trades = engine.run()
        self.all_trades.extend(trades)
        print(f"✅ Backtest completed for {symbol}\n")
        
    def run(self):
        if self.mode == 'backtest':
            for symbol in self.symbols:
                print(f"\n▶️ Running backtest for {symbol}...")
                df = self.load_data_backtest(symbol)
                if df.empty:
                    print(f"❌ No data could be loaded for {symbol}")
                    continue
                
                self._run_backtest_engine(df, symbol)
            self.summarize_and_save(self.all_trades)

        elif self.mode == 'forward':
            completed_symbol_dates = self._load_checkpoint()
            dates_list = helpers.get_dates_list(datetime.date(2025, 9, 28), datetime.date(2025, 10, 1))#days_ago=12)
            # dates_list = helpers.get_dates_list(datetime.date(2025, 5, 29), datetime.date(2025, 6, 1))#days_ago=12)
            
            for date_assess in dates_list:
                df_csv_file = self.manager.get_symbols_from_daily_data_folder(date_assess)
                if df_csv_file.empty:
                    print(f"❌ No data found for {date_assess}")
                    continue

                for i, symbol in enumerate(df_csv_file['Symbol']):
                # for symbol in ['BPMC']:
                #     i=0
                    if not symbol:
                        continue

                    key = (symbol, str(df_csv_file["Time"].iloc[i])) # Use string date for easier serialization
                    if key in completed_symbol_dates:
                        print(f"⏭️ Skipping {symbol} at {key[1]} (already processed)")
                        continue

                    print(f"\n▶️ Running forward backtest for {symbol} on day {date_assess.date()}...")
                    trig_time = pd.to_datetime(df_csv_file["Time"].iloc[i]).tz_localize(CONSTANTS.TZ_WORK) if not pd.isna(df_csv_file["Time"].iloc[i]) else None
                    th_times = CONSTANTS.TH_TIMES['end_of_day']
                    to_time = trig_time.normalize() + pd.Timedelta(hours=th_times.hour, minutes=th_times.minute, seconds=th_times.second) if trig_time else None
                    
                    df = self.manager.load_data_live(symbol, trig_time, to_time, file_format='parquet') if (trig_time and to_time) else pd.DataFrame()
                    
                    if not df.empty:
                        self._run_backtest_engine(df, symbol)
                    else:
                        print(f"❌ No data could be loaded for {symbol}")
                    
                    completed_symbol_dates.add(key)
                    self._save_checkpoint(completed_symbol_dates)

            self.summarize_and_save(self.all_trades)

    def summarize_and_save(self, all_trades):
        df_trades = pd.DataFrame(all_trades)
        if df_trades.empty:
            print("No trades executed.")
            return

        summary = trade_evaluator.TradeEvaluator.summarize_all_trades(all_trades)
        summary_symbol = trade_evaluator.TradeEvaluator.summarize_by_group(all_trades, 'symbol')
        summary_mcap = trade_evaluator.TradeEvaluator.summarize_by_group(all_trades, 'market_cap_cat')

        paths = {
            "log": os.path.join(self.outputs_folder, f"trades_{self.file_name_pattern}.csv"),
            "summary": os.path.join(self.outputs_folder, f"summary_{self.file_name_pattern}.csv"),
            "symbol": os.path.join(self.outputs_folder, f"summary_symbol_{self.file_name_pattern}.csv"),
            "mcap": os.path.join(self.outputs_folder, f"summary_market_cap_{self.file_name_pattern}.csv")
        }
        
        os.makedirs(self.outputs_folder, exist_ok=True)
        helpers.save_df_to_file(df_trades, paths["log"], 'csv')
        helpers.save_df_to_file(summary, paths["summary"], 'csv')
        helpers.save_df_to_file(summary_symbol, paths["symbol"], 'csv')
        helpers.save_df_to_file(summary_mcap, paths["mcap"], 'csv')

        print(helpers.df_to_table(summary.round(2).astype(str)))


if __name__ == "__main__":
    args = sys.argv
    pd.options.mode.chained_assignment = None # Disable Pandas warnings


    paperTrading = 'live' not in args
    revised = 'revised' in args
    seed = next((int(arg[5:]) for arg in args if arg.startswith('seed=')), None)
    strategy_name = next((arg[9:] for arg in args if arg.startswith('strategy=')), '')
    engine_type = next((arg[7:] for arg in args if arg.startswith('engine=')), 'backtrader') # or 'custom'
    symbol = next((arg[7:] for arg in args if arg.startswith('symbol=')), [])
    # pred_th = next((float(arg[7:]) for arg in args if arg.startswith('predth=')), None)
    mode = next((arg[5:] for arg in args if arg.startswith('mode=')), 'backtest')
    entry_delay = next((arg[10:] for arg in args if arg.startswith('entry_delay=')), 1)
    stop = next((arg[5:] for arg in args if arg.startswith('stop=')), 'levels') # 'predicted_drawdown'
    selector = next((arg[9:] for arg in args if arg.startswith('selector=') and arg[9:] in ['rf', 'rfe', 'rfecv']), 'rf')


    ib, _ = helpers.IBKRConnect_any(IB(), paper=paperTrading)

    symbols = [symbol] if symbol else []

    orchestrator = BacktestOrchestrator(ib, strategy_name, revised=revised, stop=stop, symbols=symbols, seed=seed, engine_type=engine_type, 
                                        selector_type=selector, entry_delay=entry_delay, mode=mode)
    orchestrator.run()




# class BacktestOrchestrator:
#     def __init__(self, ib, strategy_name, stop, target=None, revised: bool=False, symbols=None, seed=None,
#                  timeframe=None, engine_type='custom', model_type='xgboost', selector_type='rf',
#                  model_dd_type='xgboost', size='auto', entry_delay=1, pred_th: float=0.7, rrr_threshold: float=1.5,
#                  mode='backtest'):
#         self.ib = ib
#         self.revised = revised
#         self.strategy = strategy_name + ('_R' if self.revised else '')
#         self.seed = seed
#         self.engine_type = engine_type
#         self.pred_th = pred_th
#         self.rrr_threshold = rrr_threshold
#         self.model_type = model_type
#         self.selector_type = selector_type
#         self.model_dd_type = model_dd_type
#         self.size = size
#         self.entry_delay = entry_delay
#         self.mode = mode
#         # self.strategy_instance = self.get_strategy_instance()
#         self.strategy_instance = trading_utils.TradingManager.get_strategy_instance(self.strategy, self.revised)
#         self.timeframe = timeframe or self.strategy_instance.params['timeframe']
#         self._set_targets(target, stop)
#         self.setup_paths()
#         self.load_models()
#         self.symbols = self.get_symbols(symbols)
#         self.target_handler = None
#         self.stop_handler = None
#         self._resolve_target_handlers()
#         self.all_trades = []

#     def _set_targets(self, target, stop):
#         self.target = target or self.strategy_instance.params['target']
#         if stop == 'levels':
#             self.stop = stop
#             self.stop_level_types = None#['sr_1h', 'sr_1D', 'sr_1W', 'pivots', 'pivots_D', 'pivots_M', 'levels', 'levels_M']
#         else:
#             self.stop = stop or 'predicted_drawdown'

#     def setup_paths(self):
#         self.file_name_pattern = f"{self.strategy}_{self.model_type}_{self.selector_type}_{self.mode}_{self.timeframe}_seed{self.seed}_{self.target}_{self.stop}_pred{str(self.pred_th)}"
#         base_folder = PATHS.folders_path['strategies_data']
#         self.outputs_folder = os.path.join(base_folder, self.strategy, f'backtest_{self.engine_type}_{self.file_name_pattern}')
#         self.data_folder = os.path.join(base_folder, 'backtest_data')
#         os.makedirs(self.outputs_folder, exist_ok=True)
#         os.makedirs(self.data_folder, exist_ok=True)

#     def load_models(self):
#         self.trainer = ml_trainer.ModelTrainer(model_type=self.model_type, selector_type=self.selector_type,
#                                                strategy=self.strategy, timeframe=self.timeframe, target=self.target)
#         self.trainer.load()
#         self.trainer_dd = ml_trainer.ModelTrainer(strategy=self.strategy, timeframe=self.timeframe, target=self.target,
#                                                   drawdown=True)
#         self.trainer_dd.load()

#     # def get_strategy_instance(self):
#     #     if 'bb_rsi_reversal_bull' in self.strategy:
#     #         return bb_rsi_reversal_strategy.BBRSIReversalStrategy(direction='bull', rsi_threshold=75, cam_M_threshold=4, revised=self.revised)
#     #     elif 'bb_rsi_reversal_bear' in self.strategy:
#     #         return bb_rsi_reversal_strategy.BBRSIReversalStrategy(direction='bear', rsi_threshold=75, cam_M_threshold=4, revised=self.revised)
#     #     else:
#     #         raise ValueError(f"Strategy '{self.strategy}' not recognized")

#     def get_symbols(self, symbols):
#         if symbols and isinstance(symbols, list):
#             return symbols
#         if self.seed:
#             return helpers.get_symbol_seed_list(self.seed)
#         df = pd.read_csv(self.trainer.paths['data'])
#         if df.empty:
#             print("Results Dataframe is empty")
#             return []
#         else:
#             return sorted(df['symbol'].unique())#.tolist()

#     # def enrich_df(self, symbol, to_time, from_time, bb_rsi_tf_list=None, file_format='parquet'):
#     #     bb_rsi_tf_list = bb_rsi_tf_list or ['5min', '15min', '60min', '1D'] if 'bb_rsi_reversal' in self.strategy else []
#     #     enricher = hist_market_data_handler.HistMarketDataEnricher(self.ib, self.trainer.timeframe, file_format=file_format,
#     #                                                                bb_rsi_tf_list=bb_rsi_tf_list)
#     #     results = enricher.run(symbol=symbol)
#     #     if results is not None:
#     #         valid_df = next((r['df'] for r in results if r.get('df') is not None and not r['df'].empty), pd.DataFrame())
#     #         df = valid_df
#     #         df = df[pd.to_datetime(df["date"]).between(from_time, to_time)]
#     #     else:
#     #         df = pd.DataFrame()

#     #     return df

#     def load_data_backtest(self, symbol, data_set='test', file_format='parquet'):
#         path = os.path.join(self.data_folder, f"df_test_{symbol}_{self.timeframe}.{file_format}")

#         if os.path.exists(path):
#             return helpers.load_df_from_file(path)

#         test_start = pd.to_datetime(self.trainer.data_ranges[data_set][1], utc=True).tz_convert(CONSTANTS.TZ_WORK)
#         test_end = pd.to_datetime(self.trainer.data_ranges[data_set][0], utc=True).tz_convert(CONSTANTS.TZ_WORK)

#         if 'bb_rsi_reversal' in self.strategy:
#             # df = self._enrich_df(symbol, test_start, test_end, file_format=file_format)
#             df = trading_utils.TradingManager.enrich_df(self.ib, symbol, self.timeframe, test_start, test_end, self.strategy, file_format=file_format)
#         else:
#             loader = hist_market_data_handler.HistMarketDataLoader(
#                 None, symbol, self.timeframe,
#                 file_format='parquet', data_type='enriched'
#             )
#             df = loader.load_and_trim(test_end, test_start)

#         helpers.save_df_to_file(df, path, file_format='parquet')
#         return df

#     # def load_data_forward(self, symbol, trig_time, to_time, look_backward='2 M', file_format='parquet'):

#     #     # Fetch symbol
#     #     fetcher = hist_market_data_handler.HistMarketDataFetcher(ib, self.timeframe, file_format, delete_existing=True)
#     #     from_time = helpers.substract_duration_from_time(trig_time, look_backward)
#     #     params = {'symbol': symbol, 'to_time': to_time, 'from_time': from_time, 'step_duration': '1 W'}
#     #     df = helpers.load_df_from_file(fetcher.run(params))

#     #     # Enrich data
#     #     # df = self._enrich_df(symbol, to_time, from_time, file_format=file_format)
#     #     df = trade_manager.TradeManager.enrich_df(self.ib, symbol, self.trainer.timeframe, to_time, from_time, self.strategy, file_format=file_format)
#     #     df = df[(df['date'] >= trig_time) & (df['date'] <= to_time)] if not df.empty else pd.DataFrame()

#     #     return df

#     def _resolve_target_handlers(self):
#         # Target handler
#         if self.target == 'vwap_cross':
#             self.target_handler = target_handler.VWAPCrossTargetHandler(max_time='eod')

#         # Stop-loss handler
#         if isinstance(self.stop, float) and 0 < self.stop < 1:
#             self.stop_handler = target_handler.FixedStopLossHandler(self.stop)

#         # elif isinstance(self.stop, list) and all(isinstance(s, str) for s in self.stop):
#         elif self.stop == 'levels':
#             self.stop_handler = target_handler.NextLevelStopLossHandler(level_types=self.stop_level_types)

#         elif self.stop == 'predicted_drawdown':
#             self.stop_handler = target_handler.PredictedDrawdownStopLossHandler(dd_col='predicted_drawdown')

#         else:
#             self.stop_handler = None  # Optional fallback

#     def select_engine(self, df, symbol):
#         if self.engine_type == 'custom':
#             return custom_backtester.CustomBacktestEngine(df, symbol, self.strategy_instance, self.target_handler, self.stop_handler,
#                                                           trade_manager.TradingConfig, prediction_threshold=self.pred_th,
#                                                           rrr_threshold=self.rrr_threshold, size=self.size, entry_delay=self.entry_delay)
#         elif self.engine_type == 'backtrader':
#             return backtrader_backtester.BacktraderBacktestEngine(df=df, symbol=symbol, strategy_logic=self.strategy_instance,
#                                                stop_handler=self.stop_handler, target_handler=self.target_handler,
#                                                prediction_threshold=self.pred_th, rrr_threshold=self.rrr_threshold,
#                                                size=self.size, entry_delay=self.entry_delay)
#         else:
#             raise ValueError(f"Unknown engine type: {self.engine_type}")

#     def summarize_and_save(self, all_trades):
#         df_trades = pd.DataFrame(all_trades)
#         if df_trades.empty:
#             print("No trades executed.")
#             return

#         summary = trade_evaluator.TradeEvaluator.summarize_all_trades(all_trades)
#         summary_symbol = trade_evaluator.TradeEvaluator.summarize_by_group(all_trades, 'symbol')
#         summary_mcap = trade_evaluator.TradeEvaluator.summarize_by_group(all_trades, 'market_cap_cat')

#         paths = {
#             "log": os.path.join(self.outputs_folder, f"trades_{self.file_name_pattern}.csv"),
#             "summary": os.path.join(self.outputs_folder, f"summary_{self.file_name_pattern}.csv"),
#             "symbol": os.path.join(self.outputs_folder, f"summary_symbol_{self.file_name_pattern}.csv"),
#             "mcap": os.path.join(self.outputs_folder, f"summary_market_cap_{self.file_name_pattern}.csv")
#         }

#         helpers.save_df_to_file(df_trades, paths["log"], 'csv')
#         helpers.save_df_to_file(summary, paths["summary"], 'csv')
#         helpers.save_df_to_file(summary_symbol, paths["symbol"], 'csv')
#         helpers.save_df_to_file(summary_mcap, paths["mcap"], 'csv')

#         print(helpers.df_to_table(summary.round(2).astype(str)))

#     def _run_backtest_engine(self, df, symbol):
#         df = trading_utils.TradingManager.apply_model_predictions(df, self.trainer, self.trainer_dd)
#         engine = self.select_engine(df, symbol)
#         trades = engine.run()
#         self.all_trades.extend(trades)
#         print(f"✅ Backtest completed for {symbol}\n")
        
#     def run(self):
#         if self.mode == 'backtest':
#             for symbol in self.symbols:
#                 print(f"\n▶️ Running backtest for {symbol}...")
#                 df = self.load_data_backtest(symbol)
#                 if df.empty:
#                     print(f"❌ No data could be loaded for {symbol}")
#                     continue
                
#                 self._run_backtest_engine(df, symbol)
#             self.summarize_and_save(self.all_trades)

#         elif self.mode == 'forward':
#             dates_list = helpers.get_dates_list(days_ago=12)
#             for date_assess in dates_list:
#             #     daily_data_folder = helpers.get_path_daily_data_folder(date_assess, create_if_none=False)
#             #     rsi_reversal_file_path = os.path.join(daily_data_folder, PATHS.daily_csv_files['bb_rsi_reversal'])
#             #     if os.path.exists(rsi_reversal_file_path):
#             #         df_csv_file = helpers.load_df_from_file(rsi_reversal_file_path)
#             #     else:
#             #         print(f"❌ Path does not exist: {rsi_reversal_file_path}")
#             #         continue
#             #         # return pd.DataFrame()

#             #     # i=0
#             #     for i, symbol in enumerate(df_csv_file['Symbol']):
#             #     # for symbol in ['NEE/PR']:

#                 df_csv_file = trading_utils.TradingManager.get_symbols_from_daily_data_folder(self.strategy, date_assess)
#                 if df_csv_file.empty:
#                     print(f"❌ No data found for {date_assess}")
#                     continue

#                 # i=0
#                 for i, symbol in enumerate(df_csv_file['Symbol']):
#                 # for symbol in ['OPAD']:
#                     print(f"\n▶️ Running forward backtest for {symbol} on day {date_assess.date()}...")
#                     trig_time = pd.to_datetime(df_csv_file["Time"].iloc[i]).tz_localize(CONSTANTS.TZ_WORK)
#                     th_times = CONSTANTS.TH_TIMES['end_of_day']
#                     to_time = trig_time.normalize() + pd.Timedelta(hours=th_times.hour, minutes=th_times.minute, seconds=th_times.second)
#                     # df = self.load_data_forward(symbol, trig_time, to_time)
#                     df = trading_utils.TradingManager.load_data_forward(ib, symbol, self.timeframe, trig_time, to_time, self.strategy, 
#                                                                         look_backward='2 M', file_format='parquet')
#                     if df.empty:
#                         print(f"❌ No data could be loaded for {symbol}")
#                         continue
                    
#                     self._run_backtest_engine(df, symbol)
#             self.summarize_and_save(self.all_trades)




    # symbols_seed2 = sorted(['LLY', 'INTU', 'GE', 'JPM', 'CRM', 'WMT', 'TSLA', 'BAC', 'IBM', 'MSFT', 'ABBV', 'NVDA', 'AMZN', 'V', 'PM', 'SOLV', 'CF', 'PTC', 'VTRS', 'GS', 'COP', 'CTSH', 'DVN', 'MAR', 'D', 'HLT', 'MCK', 'CMG', 'DLTR', 'COMM', 'ASPN', 'SKYE', 'RLAY', 'PLAY', 'ATRO', 'NEXT', 'ACMR', 'TMHC', 'ERAS', 'BLZE', 'ANF', 'GH', 'MSTR', 'ORGO'])
    # symbols_seed3 = sorted(['NVDA','WFC','NOW','ABT','KO','INTU','MS','PLTR','BAC','PM','CSCO','XOM','LIN','HD','JNJ','LH','HBAN','STZ','ADP','PSA','CL','AKAM','UNP','ETR','MCK','PRMB','KHC','EXPD','HCA','RCL','ARRY','RCKT','EYE''EAT','TSHA','FLNC','SEM','RAPT','PBI','DAWN','CERS','MRCY','TWI','IBRX','SKYT'])

    # def _enrich_bb_rsi_tf(self, symbol, to_time, from_time, bb_rsi_tf_list=None):
    #     bb_rsi_tf_list = bb_rsi_tf_list or ['5min', '15min', '60min', '1D']
    #     enricher = hist_market_data_handler.HistMarketDataEnricher(ib=self.ib, timeframe=self.trainer.timeframe,
    #                                                                file_format='parquet', bb_rsi_tf_list=bb_rsi_tf_list)
    #     results = enricher.run(symbol=symbol)
    #     if results is not None:
    #         valid_df = next((r['df'] for r in results if r.get('df') is not None and not r['df'].empty), pd.DataFrame())
    #         df = valid_df
    #         df = df[pd.to_datetime(df["date"]).between(from_time, to_time)]
    #     else:
    #         df = pd.DataFrame()

    #     return df

# class StrategyBacktester:
#     def __init__(self, strategy_name, model_type='xgboost', selector_type='rf',
#                  stop='levels', target='vwap_cross', timeframe='1 min',
#                  prediction_threshold=0.7, rrr_threshold=1.5, size='auto',
#                  revised=False, engine='custom', paper_trading=True, single_symbol=None):

#         self.strategy_name = strategy_name
#         self.model_type = model_type
#         self.selector_type = selector_type
#         self.stop = stop
#         self.target = target
#         self.timeframe = timeframe
#         self.pred_th = prediction_threshold
#         self.rrr_th = rrr_threshold
#         self.size = size
#         self.revised = revised
#         self.engine = engine
#         self.single_symbol = single_symbol

#         self.file_key = self._get_file_key()
#         self.outputs_folder, self.data_folder = self._setup_folders()
#         self.symbols = []

#         self.ib = helpers.IBKRConnect_any(IB(), paper=paper_trading)
#         self.trainer = ml_trainer.ModelTrainer(model_type=self.model_type, selector_type=self.selector_type,
#                                                strategy=self.strategy_name, timeframe=self.timeframe, target=self.target)
#         self.trainer.load()

#         self.trainer_dd = ml_trainer.ModelTrainer(strategy=self.strategy_name, timeframe=self.timeframe,
#                                                   target=self.target, drawdown=True)
#         self.trainer_dd.load()

#         self.strategy_class = self._load_strategy_class()

#     def _get_file_key(self):
#         suffix = '_R' if self.revised else ''
#         return f"{self.strategy_name}{suffix}_{self.model_type}_{self.selector_type}_{self.timeframe}_{self.target}_{self.stop}_pred{str(self.pred_th)}"

#     def _setup_folders(self):
#         base_path = PATHS.folders_path['strategies_data']
#         outputs = os.path.join(base_path, self.strategy_name, f'backtest_{self._get_file_key()}')
#         data = os.path.join(base_path, 'backtest_data')
#         os.makedirs(outputs, exist_ok=True)
#         os.makedirs(data, exist_ok=True)
#         return outputs, data

#     def _load_strategy_class(self):
#         if 'bb_rsi_reversal_bull' in self.strategy_name:
#             return bb_rsi_reversal_strategy.BBRSIReversalStrategy(direction='bull', revised=self.revised)
#         elif 'bb_rsi_reversal_bear' in self.strategy_name:
#             return bb_rsi_reversal_strategy.BBRSIReversalStrategy(direction='bear', revised=self.revised)
#         else:
#             raise ValueError("Unsupported strategy")

#     def run(self):
#         self.symbols = [self.single_symbol] if self.single_symbol else self._build_symbols_list()
#         all_trades = []

#         for symbol in self.symbols:
#             print(f"\n▶️ Running backtest for: {symbol}")

#             df = self._load_data(symbol)
#             df = self._add_predictions_to_df(df, return_proba=True)

#             if self.engine == 'custom':
#                 engine = custom_backtester.CustomBacktestEngine(df, symbol, self.strategy_class, self.target, self.stop,
#                                         config=custom_backtester.BacktestConfig, prediction_threshold=self.pred_th,
#                                         rrr_threshold=self.rrr_th, size=self.size)
#                 trades = engine.run()
#             elif self.engine == 'backtrader':
#                 engine = backtrader_backtester.BacktraderEngine(df, self.strategy_class, pred_col='model_prediction')
#                 trades = engine.run()
#             else:
#                 raise ValueError(f"Unknown engine type: {self.engine}")

#             all_trades.extend(trades or [])

#         self._save_results(all_trades)

#     def _load_data(self, symbol):
#         path = os.path.join(self.data_folder, f"df_test_{symbol}_{self.timeframe}.parquet")
#         if os.path.exists(path):
#             return helpers.load_df_from_file(path)

#         # Else: download
#         test_start = pd.to_datetime(self.trainer.data_ranges['test'][1], utc=True).tz_convert(CONSTANTS.TZ_WORK)
#         test_end = pd.to_datetime(self.trainer.data_ranges['test'][0], utc=True).tz_convert(CONSTANTS.TZ_WORK)

#         if 'bb_rsi_reversal' in self.strategy_name:
#             tf_list = ['5min', '15min', '60min', '1D']
#             enricher = hist_market_data_handler.HistMarketDataEnricher(self.ib, timeframe=self.timeframe, bb_rsi_tf_list=tf_list)
#             results = enricher.run(symbol=symbol)
#             df = next((r['df'] for r in results if r['df'] is not None and not r['df'].empty), pd.DataFrame())
#             df = df[pd.to_datetime(df['date']).between(test_end, test_start)]
#         else:
#             loader = hist_market_data_handler.HistMarketDataLoader(self.ib, symbol, self.timeframe)
#             df = loader.load_and_trim(test_end, test_start)

#         helpers.save_df_to_file(df, path, file_format='parquet')
#         return df

#     def _save_results(self, trades):
#         log_path = os.path.join(self.outputs_folder, f'trades_{self._get_file_key()}.csv')
#         summary_path = os.path.join(self.outputs_folder, f'summary_{self._get_file_key()}.csv')

#         df_trades = pd.DataFrame(trades)
#         if not df_trades.empty:
#             helpers.save_df_to_file(df_trades, log_path, 'csv')

#             summary = trade_evaluator.TradeEvaluator.summarize_all_trades(trades)
#             helpers.save_df_to_file(summary, summary_path, 'csv')

#             print("\n🔹 Summary:")
#             print(helpers.df_to_table(summary.round(2).astype(str)))

#     def _build_symbols_list(self):
#         df = pd.read_csv(self.trainer.paths['data'])

#         if df.empty:
#             print("Results Dataframe is empty")
#             return []
#         else:
#             return sorted(df['symbol'].unique())#.tolist()

#     def _add_predictions_to_df(self, df, return_proba=True, shift_features=False):
#         if 'market_cap_cat' not in df.columns:
#             df['market_cap_cat'] = helpers.categorize_market_cap(df.attrs['market_cap'])

#         # Predict using model and add predictions to dataframe
#         print("🤖 Generating trade entry predictions...")
#         predictions = self.trainer.predict(df.copy(), return_proba=return_proba, shift_features=shift_features)
#         df['model_prediction'] = predictions
#         print("🤖 Generating drawdown predictions...")
#         predictions_dd_pct = self.trainer_dd.predict(df.copy(), return_proba=False, shift_features=shift_features)
#         df['predicted_drawdown'] = predictions_dd_pct * df['close']

#         return df


# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser(description="Backtest ML strategy")
#     parser.add_argument('--strategy', type=str, required=True)
#     parser.add_argument('--symbol', type=str)
#     parser.add_argument('--predth', type=float, default=0.7)
#     parser.add_argument('--engine', type=str, choices=['custom', 'backtrader'], default='custom')
#     parser.add_argument('--revised', action='store_true')
#     args = parser.parse_args()

#     backtester = StrategyBacktester(strategy_name=args.strategy, single_symbol=args.symbol, prediction_threshold=args.predth,
#                                     engine=args.engine, revised=args.revised)
#     backtester.run()
