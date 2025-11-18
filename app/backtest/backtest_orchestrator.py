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
        self.file_name_pattern = f"{self.engine_type}_{self.manager.strategy_name}_{self.manager.config.model_type}_{self.manager.config.selector_type}_{self.mode}_{self.manager.strategy_instance.timeframe}_seed{self.seed}_{self.manager.strategy_instance.target_handler.target_str}_{self.manager.config.stop}_pred{str(self.manager.config.pred_th)}_delay{self.entry_delay}"
        base_folder = PATHS.folders_path['strategies_data']
        self.outputs_folder = os.path.join(base_folder, self.manager.strategy_name, f'backtest_{self.file_name_pattern}')
        # self.data_folder = os.path.join(base_folder, 'backtest_data')
        # os.makedirs(self.data_folder, exist_ok=True)
        self.checkpoint_file_path = os.path.join(self.outputs_folder, 'test_checkpoint.json')
    
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

        df = self.manager.enrich_df(symbol, file_format=file_format, select_features=False)
        
        # Trim again to ensure it is trimmed
        df = helpers.trim_df(df, from_time=test_from, to_time=test_to)
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
        failed_symbols = []
        if self.mode == 'backtest':
            for symbol in self.symbols:
                print(f"\n▶️ Running backtest for {symbol}...")
                df = self.load_data_backtest(symbol)
                if df.empty:
                    print(f"❌ No data could be loaded for {symbol}")
                    continue
                
                try:
                    self._run_backtest_engine(df, symbol)
                except Exception as e:
                    print(f"Could not run backtest for {symbol}. Error: {e}")#  |  Full error: ", traceback.format_exc())
                    failed_symbols.append(symbol)
            
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
                        try:
                            self._run_backtest_engine(df, symbol)
                        except Exception as e:
                            print(f"Could not run backtest for {symbol}. Error: {e}")#  |  Full error: ", traceback.format_exc())
                            failed_symbols.append(symbol)
                    else:
                        print(f"❌ No data could be loaded for {symbol}")
                    
                    completed_symbol_dates.add(key)
                    self._save_checkpoint(completed_symbol_dates)

            self.summarize_and_save(self.all_trades)
        
        print(f"\n🚩 Failed symbols: {failed_symbols}\n")

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


    paperTrading = not 'live' in args
    local_ib = 'local' in args
    revised = 'revised' in args
    seed = next((int(arg[5:]) for arg in args if arg.startswith('seed=')), None)
    strategy_name = next((arg[9:] for arg in args if arg.startswith('strategy=')), '')
    engine_type = next((arg[7:] for arg in args if arg.startswith('engine=')), 'backtrader') # or 'custom'
    symbol = next((arg[7:] for arg in args if arg.startswith('symbol=')), [])
    # pred_th = next((float(arg[7:]) for arg in args if arg.startswith('predth=')), None)
    mode = next((arg[5:] for arg in args if arg.startswith('mode=')), 'backtest')
    entry_delay = next((int(arg[12:]) for arg in args if arg.startswith('entry_delay=')), 1)
    stop = next((arg[5:] for arg in args if arg.startswith('stop=')), 'levels') # 'predicted_drawdown'
    selector = next((arg[9:] for arg in args if arg.startswith('selector=') and arg[9:] in ['rf', 'rfe', 'rfecv']), 'rf')


    ib, _ = helpers.IBKRConnect_any(IB(), paper=paperTrading, remote=not local_ib)

    symbols = [symbol] if symbol else []

    orchestrator = BacktestOrchestrator(ib, strategy_name, revised=revised, stop=stop, symbols=symbols, seed=seed, engine_type=engine_type, 
                                        selector_type=selector, entry_delay=entry_delay, mode=mode)
    orchestrator.run()
