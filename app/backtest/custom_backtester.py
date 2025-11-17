import os, sys, pandas as pd, tqdm, datetime, numpy as np
from numba import njit
from ib_insync import *

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

import trade_evaluator
from execution import trade_manager
from utils import constants, helpers
from ml import ml_trainer, features_processor
from data import hist_market_data_handler
from strategies import bb_rsi_reversal_strategy

# Link: https://www.youtube.com/watch?v=UNkH1TQl7qo&list=PLvzuUVysUFOs2kZB5KbsAPi0XgqemA5s9
#       https://www.youtube.com/watch?v=aTSD_SviPLY&list=PLozLnT2Cz6R_mJbfdJTvOSl-3ocafIKEZ
# backtrader doc: https://www.backtrader.com/docu/
# some yfiance link: https://algotrading101.com/learn/yfinance-guide/



# def summarize_results(df, symbol, combined=False):
#         if df.empty:
#             return {}

#         return {
#             'Symbol': 'ALL' if combined else symbol,
#             'Total PnL': df['Total PnL'].sum() if combined else df['pnl'].sum(),
#             'Win Rate': (df['Win Rate'] * df['Trades']).sum() / df['Trades'].sum() if combined else (df['pnl'] > 0).mean(),
#             'Avg Return %': df['Avg Return %'].mean() if combined else df['return_pct'].mean(),
#             'Sharpe': df['Sharpe'].mean() if combined else df['return_pct'].mean() / df['return_pct'].std() * (252**0.5) if df['return_pct'].std() else 0,
#             'Trades': df['Trades'].sum() if combined else len(df)
#         }


def build_symbols_list(trainer):
    df = pd.read_csv(trainer.paths['data'])

    if df.empty:
        print("Results Dataframe is empty")
        return []
    else:
        return sorted(df['symbol'].unique())#.tolist()


def load_test_data(ib, symbol, trainer, data_path, file_format='parquet'):

    if os.path.exists(data_path):
        df_test = helpers.load_df_from_file(data_path)
    else:
        # df_test = load_test_data_with_predictions(ib, symbol, trainer, file_format='parquet', return_proba=True, shift_features=False)
        test_to_time = pd.to_datetime(trainer.data_ranges['test'][1], utc=True).tz_convert(constants.CONSTANTS.TZ_WORK)
        test_from_time = pd.to_datetime(trainer.data_ranges['test'][0], utc=True).tz_convert(constants.CONSTANTS.TZ_WORK)

        if 'bb_rsi_reversal' in strategy:
            bb_rsi_tf_list = ['5min', '15min', '60min', '1D']
            enricher = hist_market_data_handler.HistMarketDataEnricher(ib, timeframe=trainer.timeframe, file_format='parquet',
                                                                    bb_rsi_tf_list=bb_rsi_tf_list, save_to_file=False)
            results = enricher.run(symbol=symbol)
            valid_result = enricher.get_valid_result(results)
            df = valid_result['df'] if valid_result else pd.DataFrame()
            df_to_time = pd.to_datetime(df['date'].iloc[-1]) if not df.empty else None
            df_from_time = pd.to_datetime(df['date'].iloc[0]) if not df.empty else None
            df_test = df[pd.to_datetime(df["date"]).between(max(test_from_time, df_from_time), min(test_to_time, df_to_time))]
        else:
            df_test, _ = hist_market_data_handler.HistMarketDataLoader(ib, symbol, trainer.timeframe, file_format=file_format,
                                                                    data_type='enriched').load_and_trim(test_from_time, test_to_time)

        helpers.save_df_to_file(df_test, data_path, file_format='parquet')

    return df_test


class CustomBacktestEngine:
    def __init__(self, df, symbol, manager, entry_delay=1):#, mode='live'):
        self.df = df.reset_index(drop=True)#.copy()
        self.symbol = symbol
        self.manager = manager
        self.active_stop_price = None  # stores fixed stop-loss value for current trade
        self.trades = []
        self.entry_delay = entry_delay
        self.df_tr = self._build_df_tr()
        self.entry_exec_price = None
        self.exit_exec_price = None
        self.entry_commission = 0.0
        self.exit_commission = 0.0
        self.slippage_pct = 0.001
    
    def _build_df_tr(self):
        if self.manager.strategy_instance.revised:
            df_tr, _ = features_processor.apply_feature_transformations(self.df)
            new_cols = [col for col in df_tr.columns if col not in self.df.columns]
            df_tr = pd.concat([self.df, df_tr[new_cols]], axis=1) # Concatenate only those new columns
            df_tr.attrs = self.df.attrs
        else:
            df_tr = pd.DataFrame()
        
        return df_tr

    def run(self):
        in_position = False
        entry_idx = None
        entry_price = None

        df = self.df if not self.manager.strategy_instance.revised else self.df_tr

        for i in tqdm.tqdm(range(self.entry_delay, len(df)), desc=f"Backtesting {self.symbol}"):

            decision_row = df.iloc[i - self.entry_delay]
            prev_decision_row = df.iloc[i - self.entry_delay - 1] if i - self.entry_delay - 1 >= 0 else decision_row
            # prev_row = df.iloc[i - 1]
            curr_row = df.iloc[i]

            # time_test = pd.Timestamp('2024-11-06 11:43:00-0500', tz='US/Eastern')
            # if curr_row['date'] == time_test:
            #     print()
            
            if not in_position:
                if self.manager.evaluate_entry_conditions(decision_row, self.active_stop_price, display_triggers=False):
                    entry_idx = i
                    entry_price = curr_row['close']
                    self.entry_exec_price = entry_price * (1 + self.manager.direction * self.slippage_pct)
                    decision_prediction = decision_row['model_prediction']
                    entry_prediction = curr_row['model_prediction']
                    quantity = self.manager.evaluate_quantity(entry_prediction)

                    if hasattr(self.manager.strategy_instance.target_handler, 'set_entry_time'):
                        self.manager.strategy_instance.target_handler.set_entry_time(curr_row['date'])
                    if hasattr(self.manager.strategy_instance.target_handler, 'set_target_price'):
                        self.manager.strategy_instance.target_handler.set_target_price(row=decision_row)

                    # Resolve stop once here
                    self.active_stop_price = self.manager.resolve_stop_price(curr_row, self.active_stop_price)

                    reason2close = self.manager.assess_reason2close(decision_row, prev_decision_row, self.active_stop_price)
                    if not reason2close:
                        in_position = True
                    else:
                        self.active_stop_price = None
            else:
                reason2close = self.manager.assess_reason2close(decision_row, prev_decision_row, self.active_stop_price)
                if reason2close:
                    exit_idx = i
                    exit_price = curr_row['close']
                    self.exit_exec_price = exit_price * (1 - self.manager.direction * self.slippage_pct)
                    trade_evaluator.TradeEvaluator.log_trade(self.manager.direction, self.trades, entry_idx, exit_idx, self.df, entry_price, exit_price, quantity, 
                                             decision_prediction, entry_prediction, self.active_stop_price, reason2close, self.symbol, 
                                             self.entry_exec_price, self.exit_exec_price, self.entry_commission, self.exit_commission)
                    in_position = False
                    self.active_stop_price = None  # Reset for next trade

        if self.trades:
            print(helpers.df_to_table(pd.DataFrame(self.trades).round(2).astype(str)))
        else:
            print(f"No trade for symbol {self.symbol}")
        # print(helpers.df_to_table(pd.DataFrame.from_dict([self.summarize_trades()]).round(2).astype(str)))

        return self.trades

    # def summarize_trades(self):
    #     trade_log = self.trades
    #     df = pd.DataFrame(trade_log)
    #     return summarize_results(df, self.symbol, combined=False)

    # def summarize_trades(self):
    #     df_trade_log = pd.DataFrame(self.trades)
    #     return trade_evaluator.TradeEvaluator.summarize_results(df_trade_log, self.symbol)



# class CustomBacktestEngine:
#     def __init__(self, df, symbol, strategy, target_handler, stop_handler, config, prediction_threshold: float=0.7,
#                  rrr_threshold: float=1.5, size='auto', entry_delay=1, tier_max=5):#, mode='live'):
#         self.df = df.reset_index(drop=True).copy()
#         self.df_tr, _ = features_processor.apply_feature_transformations(self.df)
#         self.config = config
#         self.symbol = symbol
#         self.strategy = strategy
#         # self.target = target
#         # if stop == 'levels':
#         #     self.stop = ['sr_1h', 'sr_1D', 'sr_1W', 'pivots', 'pivots_D', 'pivots_M', 'levels', 'levels_M']
#         # else:
#         #     self.stop = stop
#         self.active_stop_price = None  # stores fixed stop-loss value for current trade
#         self.trades = []
#         # self.target_handler = None
#         # self.stop_handler = None
#         # self._resolve_target_handlers()
#         self.target_handler = target_handler
#         self.stop_handler = stop_handler
#         self.direction = trade_manager.TradingManager.resolve_direction(self.strategy)
#         self.prediction_threshold = prediction_threshold
#         self.rrr_threshold = rrr_threshold
#         self.size = size
#         self.entry_delay = entry_delay
#         self.tier_max = tier_max
#         # self.mode = mode
#         if self.strategy.revised:
#             self.df_tr, _ = features_processor.apply_feature_transformations(self.df)
#             new_cols = [col for col in self.df_tr.columns if col not in self.df.columns]
#             self.df_tr = pd.concat([self.df, self.df_tr[new_cols]], axis=1) # Concatenate only those new columns
#             self.df_tr.attrs = self.df.attrs
#         else:
#             self.df_tr = pd.DataFrame()
#         self.entry_exec_price = None
#         self.exit_exec_price = None
#         self.entry_commission = 0.0
#         self.exit_commission = 0.0
#         self.slippage_pct = 0.001

#     def run(self):
#         in_position = False
#         entry_idx = None
#         entry_price = None

#         df = self.df if not self.strategy.revised else self.df_tr

#         for i in tqdm.tqdm(range(self.entry_delay, len(df)), desc=f"Backtesting {self.symbol}"):

#             decision_row = df.iloc[i - self.entry_delay]
#             prev_decision_row = df.iloc[i - self.entry_delay - 1] if i - self.entry_delay - 1 >= 0 else decision_row
#             # prev_row = df.iloc[i - 1]
#             curr_row = df.iloc[i]

#             # time_test = pd.Timestamp('2024-11-06 11:43:00-0500', tz='US/Eastern')
#             # if curr_row['date'] == time_test:
#             #     print()

#             if not in_position:
#                 if trade_manager.TradingManager.evaluate_entry_conditions(decision_row, self.strategy, self.target_handler,
#                                                              self.prediction_threshold, self.rrr_threshold):
#                     entry_idx = i
#                     entry_price = curr_row['close']
#                     self.entry_exec_price = entry_price * (1 + self.direction * self.slippage_pct)
#                     entry_prediction = curr_row['model_prediction']
#                     quantity = trade_manager.TradingManager.evaluate_quantity(entry_prediction, self.size, self.prediction_threshold,
#                                                                 self.tier_max)

#                     if hasattr(self.target_handler, 'set_entry_time'):
#                         self.target_handler.set_entry_time(curr_row['date'])

#                     # Resolve stop once here
#                     self.active_stop_price = trade_manager.TradingManager.resolve_stop_price(curr_row, self.active_stop_price,
#                                                                                 self.stop_handler, self.direction)

#                     reason2close = trade_manager.TradingManager.assess_reason2close(decision_row, prev_decision_row, self.target_handler,
#                                                         self.stop_handler, self.active_stop_price, self.direction)
#                     if not reason2close:
#                         in_position = True
#                     else:
#                         self.active_stop_price = None
#             else:
#                 reason2close = trade_manager.TradingManager.assess_reason2close(decision_row, prev_decision_row, self.target_handler,
#                                                         self.stop_handler, self.active_stop_price, self.direction)
#                 if reason2close:
#                     exit_idx = i
#                     exit_price = curr_row['close']
#                     self.exit_exec_price = exit_price * (1 - self.direction * self.slippage_pct)
#                     trade_evaluator.TradeEvaluator.log_trade(self.direction, self.trades, entry_idx, exit_idx, self.df, entry_price, exit_price, quantity, 
#                                              entry_prediction, self.active_stop_price, reason2close, self.symbol, self.entry_exec_price, 
#                                              self.exit_exec_price, self.entry_commission, self.exit_commission)
#                     in_position = False
#                     self.active_stop_price = None  # Reset for next trade

#         if self.trades:
#             print(helpers.df_to_table(pd.DataFrame(self.trades).round(2).astype(str)))
#         else:
#             print(f"No trade for symbol {self.symbol}")
#         # print(helpers.df_to_table(pd.DataFrame.from_dict([self.summarize_trades()]).round(2).astype(str)))

#         return self.trades

#     # def summarize_trades(self):
#     #     trade_log = self.trades
#     #     df = pd.DataFrame(trade_log)
#     #     return summarize_results(df, self.symbol, combined=False)

#     # def summarize_trades(self):
#     #     df_trade_log = pd.DataFrame(self.trades)
#     #     return trade_evaluator.TradeEvaluator.summarize_results(df_trade_log, self.symbol)



if __name__ == "__main__":

    args = sys.argv

    pd.options.mode.chained_assignment = None # Disable Pandas warnings
    # pd.set_option('future.no_silent_downcasting', True)


    # Args Setup
    args = sys.argv
    paperTrading = not 'live' in args
    revised = 'revised' in args
    strategy = next((arg[9:] for arg in args if arg.startswith('strategy=')), '')
    single_symbol = next((arg[7:] for arg in args if arg.startswith('symbol=')), None)
    pred_th = next((float(arg[7:]) for arg in args if arg.startswith('predth=')), None)
    # mode = next((arg[5:] for arg in args if arg.startswith('mode=')), 'live')

    rev = '' if not revised else '_R'
    strategy = strategy + rev

    model_type = 'xgboost'
    # model_type = 'stacking'
    model_drawdown_type = 'xgboost'
    selector_type = 'rf'
    timeframe = '1min'
    target = 'vwap_cross'
    stop = 'levels'
    # stop = 'predicted_drawdown'

    file_name_pattern = f"{strategy}_{model_type}_{selector_type}_{timeframe}_{target}_{stop}_pred{str(pred_th)}"
    startegies_folder = constants.PATHS.folders_path['strategies_data']
    outputs_folder = os.path.join(startegies_folder, strategy, f'backtest_{file_name_pattern}')
    data_folder = os.path.join(startegies_folder, 'backtest_data')
    os.makedirs(outputs_folder, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)
    all_trades_log_file_path = os.path.join(outputs_folder, f'trades_{file_name_pattern}.csv')
    trades_summary_file_path = os.path.join(outputs_folder, f'summary_{file_name_pattern}.csv')
    trades_summary_symbol_file_path = os.path.join(outputs_folder, f'summary_symbol_{file_name_pattern}.csv')
    trades_summary_market_cap_cat_file_path = os.path.join(outputs_folder, f'summary_market_cap_{file_name_pattern}.csv')

    # TWS Connection
    ib, ibConnection = helpers.IBKRConnect_any(IB(), paper=paperTrading)
    # ib = IB()

    # Load trained model
    trainer = ml_trainer.ModelTrainer(model_type=model_type, selector_type=selector_type, strategy_name=strategy, target=target)
    trainer.load()
    trainer_drawdown = ml_trainer.ModelTrainer(strategy_name=strategy, target=target, drawdown=True)
    trainer_drawdown.load()
    # trainer.feature_names.remove('market_cap_cat')

    if 'bb_rsi_reversal_bull' in strategy:
        strategy_class = bb_rsi_reversal_strategy.BBRSIReversalStrategy(direction='bull', rsi_threshold=75, cam_M_threshold=4, revised=revised)
    elif 'bb_rsi_reversal_bear' in strategy:
        strategy_class = bb_rsi_reversal_strategy.BBRSIReversalStrategy(direction='bear', rsi_threshold=75, cam_M_threshold=4, revised=revised)

    all_trades_log = []
    all_trade_summaries = []
    symbols_list = [single_symbol] if single_symbol else build_symbols_list(trainer)
    for symbol in symbols_list:
        print(f"Running backtest for {symbol}...\n")
        symbol_data_path = os.path.join(data_folder, f"df_test_{symbol}_{timeframe}.parquet")

        df_test = load_test_data(ib, symbol, trainer, symbol_data_path, file_format='parquet')
        # df_test = add_predictions_to_df(df_test, trainer, trainer_drawdown, return_proba=True, shift_features=False)
        df_test = trade_manager.TradeManager.apply_model_predictions(df_test, symbol)

        t0 = datetime.datetime.now()
        bt_engine = CustomBacktestEngine(df_test, symbol, strategy_class, target, stop, config=trade_manager.TradingConfig,
                                   prediction_threshold=pred_th, rrr_threshold=1.5, size=1)#, mode=mode)
        trade_log = bt_engine.run()
        # trades_summary = bt_engine.summarize_trades()

        if trade_log:
            all_trades_log.extend(trade_log)
        print(f"\n⏱️  Elapsed time for backtest - {symbol}: {datetime.datetime.now() - t0}\n")
        print()

    # Final Combined Summary
    # ----------------------
    df_trades_summary = trading_evaluator.TradeEvaluator.summarize_all_trades(all_trades_log)
    df_trades_summary_symbol = trading_utils.TradeEvaluator.summarize_by_group(all_trades_log, 'symbol')
    df_trades_summary_market_cap_cat = trading_utils.TradeEvaluator.summarize_by_group(all_trades_log, 'market_cap_cat')
    df_all_trades_log = pd.DataFrame(all_trades_log)

    if not df_all_trades_log.empty:
        df_all_trades_log.sort_values(by='entry_time', ascending=True)
        print(helpers.df_to_table(df_all_trades_log.round(2).astype(str)), "\n")
        helpers.save_df_to_file(df_all_trades_log, all_trades_log_file_path, file_format='csv')

    if not df_trades_summary.empty:
        print("🔹 Overall Summary:")
        print(helpers.df_to_table(df_trades_summary.round(2).astype(str)), "\n")
        helpers.save_df_to_file(df_trades_summary, trades_summary_file_path, file_format='csv')

    if not df_trades_summary_symbol.empty:
        print("🔹 Overall Summary per Symbol:")
        print(helpers.df_to_table(df_trades_summary_symbol.round(2).astype(str)), "\n")
        helpers.save_df_to_file(df_trades_summary_symbol, trades_summary_symbol_file_path, file_format='csv')

    if not df_trades_summary_market_cap_cat.empty:
        print("🔹 Overall Summary per Market Cap:")
        print(helpers.df_to_table(df_trades_summary_market_cap_cat.round(2).astype(str)), "\n")
        helpers.save_df_to_file(df_trades_summary_market_cap_cat, trades_summary_market_cap_cat_file_path, file_format='csv')




# def run(self):
#     if self.mode == 'live':
#         return self.run_live()
#     elif self.mode == 'backtest':
#         return self.run_backtest()
#     else:
#         raise ValueError(f"Mode is set as '{self.mode}', must be either 'live' or 'backtest'.")

# @njit
# def run_backtest(close, vwap, trigger_flags, sl_pct, max_bars=390):
#     trades = []
#     in_position = False
#     entry_idx = 0
#     entry_price = 0.0
#     bars_in_trade = 0

#     for i in range(1, len(close)):
#         if not in_position:
#             if trigger_flags[i]:
#                 entry_idx = i
#                 entry_price = close[i]
#                 in_position = True
#                 bars_in_trade = 0
#         else:
#             bars_in_trade += 1

#             if target_handler.check_stop_loss_fast(close[i], entry_price, sl_pct):
#                 trades.append((entry_idx, i, entry_price, close[i], 0))  # stop_loss
#                 in_position = False
#                 continue

#             if target_handler.check_vwap_cross_fast(close[i-1], vwap[i-1], close[i], vwap[i]):
#                 trades.append((entry_idx, i, entry_price, close[i], 1))  # vwap_cross
#                 in_position = False
#                 continue

#             if bars_in_trade >= max_bars:
#                 trades.append((entry_idx, i, entry_price, close[i], 2))  # time exit
#                 in_position = False

#     return trades







    # # Path Setup
    # # path = path_setup.path_current_setup(os.path.realpath(__file__))

    # # tsla1M_CSVFile = "/Volumes/untitled/Trading/Market_Data/tsla1M.csv"
    # # tsla5M_CSVFile = "/Volumes/untitled/Trading/Market_Data/tsla5M.csv"

    # tsla1m_CSVFile = os.path.join(constants.PATHS.folders_path['stats_hist'], 'hist_data_TSLA_1 min_2023-12-29-04-00-00_2025-03-07-19-59-00.csv')
    # tsla1h_CSVFile = os.path.join(constants.PATHS.folders_path['stats_hist'], 'hist_data_TSLA_1 hour_2023-12-11-04-00-00_2025-03-07-19-00-00.csv')
    # tsll1h_CSVFile = os.path.join(constants.PATHS.folders_path['stats_hist'], 'hist_data_TSLL_1 hour_2023-12-11-04-00-00_2025-03-07-19-00-00.csv')

    # csv_file = tsll1h_CSVFile
    # # getSP500Tickers()

    # if os.path.isfile(csv_file):

    #     print("\nLoading data from CSV...\n")
    #     # dateparse = lambda x: datetime.datetime.strptime(x, 'datetime64[ns, America/New_York]')
    #     # df = pd.read_csv(tsla1m_CSVFile, date_parser=dateparse)
    #     df = pd.read_csv(csv_file)
    #     df['date'] = pd.to_datetime(df['date'])
    #     df['date'] = df['date'].apply(lambda dt: dt.replace(tzinfo=None))
    #     df = df.set_index('date')
    #     df = df[['open', 'high', 'low', 'close', 'volume']]
    #     df.columns = ['open', 'high', 'low', 'close', 'volume']
    #     df.index = pd.to_datetime(df.index)
    #     tsla_df_1m_parsed = bt.feeds.PandasData(dataname=df)


    #     # tsla_df_1m_parsed = pd.read_csv(tsla1m_CSVFile, parse_dates=True, index_col="date")
    #     # tsla_df_1m_parsed = bt.feeds.GenericCSVData(dataname=tsla1m_CSVFile, datetime=0, open=1, high=2, low=3, close=5, volume=6)#, openinterest=-1, dtformat="%Y-%m-%d")
    #     # # tsla_df_1M_parsed = bt.feeds.YahooFinanceCSVData(dataname=tsla5m_CSVFile, datetime=0, open=1, high=2, low=3, close=5, volume=6)#, openinterest=-1, dtformat="%Y-%m-%d")
    #     # # tsla_df_parsed = bt.feeds.YahooFinanceCSVData(dataname=tsla1m_CSVFile)#, reverse=False)#, fromdate=datetime.datetime(2024, 1, 1))
    #     # # tsla_df_1M_parsed = bt.feeds.YahooFinanceCSVData(dataname=tsla1m_CSVFile)#, reverse=False)#, fromdate=datetime.datetime(2024, 1, 1))
    #     # # tsla_df_5M_parsed = bt.feeds.YahooFinanceCSVData(dataname=tsla5m_CSVFile)

    # else:

    #     print("\nDownloading data from Yfinance...\n")
    #     time.sleep(2)

    #     # tsla_df = yf.download('TSLA', start="2023-01-01", prepost=True, interval="60m")

    #     tsla_df_5m = yf.download('TSLA', prepost=True, interval="5m", start = "2024-06-05", end = "2024-06-12")
    #     tsla_df_1m_parsed = bt.feeds.PandasData(dataname=tsla_df_1m)
    #     tsla_df_5m_parsed = bt.feeds.PandasData(dataname=tsla_df_5M)
    #     # print(tsla_df_1M)
    #     # print(tsla_df_5M)

    #     tsla_df_1m.to_csv(tsla1m_CSVFile)#, index=True)#"Datetime")#, sep=' ', mode='a')
    #     tsla_df_5m.to_csv(tsla5m_CSVFile)

    #     tsla_csv_1m_parsed = bt.feeds.YahooFinanceCSVData(dataname=tsla1m_CSVFile)#, reverse=False)#, fromdate=datetime.datetime(2024, 1, 1))

    #     dateparse = lambda x: datetime.datetime.strptime(x, 'datetime64[ns, America/New_York]')
    #     tsla_csv_1m = pd.read_csv(tsla1m_CSVFile, parse_dates=True, date_parser=dateparse, index_col="Datetime")#, dtype='datetime64[ns, America/New_York]'
    #     print(tsla_df_1m.equals(tsla_csv_1m))
    #     print()

    #     print(tsla_df_1m)
    #     print("---------------------------------------------")
    #     print(tsla_csv_1M)
    #     print(tsla_df_1m.index)
    #     print(tsla_csv_1m.index)
    #     print("---------------------------------------------")
    #     # print(tsla_df_1M.compare(tsla_csv_1M))
    #     input()



    # # aapl_df = yfinance.download('AAPL')#, start="2020-01-01", end="2021-01-01", timezone='America/New_York', valid=True)
    # # meta_ticker = yfinance.Ticker('AMZN')
    # # print("META = ", meta_ticker)

    # # Panda dataframe
    # # tsla_df_parsed = bt.feeds.PandasData(dataname=tsla_df, datetime=None, open=0, high=1, low=2, close=4, volume=5, openinterest=-1)

    # # tsla_df_parsed_h = bt.feeds.PandasData(dataname=tsla_df, datetime=None, open=0, high=1, low=2, close=4, volume=5, openinterest=-1, timeframe=bt.TimeFrame.Minutes)





    # cerebro = bt.Cerebro()

    # # cerebro.broker.set_cash(5000)
    # start_cash_value = cerebro.broker.getvalue()

    # cerebro.adddata(tsla_df_1m_parsed)
    # # cerebro.adddata(tsla_df_5M_parsed)
    # # cerebro.addstrategy(TestStrategy)
    # cerebro.addstrategy(SmaCross)
    # # cerebro.addstrategy(BBPullbackStrategy)
    # cerebro.addsizer(bt.sizers.FixedSize, stake = 1000)
    # cerebro.run()

    # print("Starting Value: %.2f" % start_cash_value)
    # print("Final Value: %.2f" % cerebro.broker.getvalue())

    # cerebro.plot(iplot=False)






    # # Get tckers infos from S&P500
    # # tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    # # print(tickers.head())


    # # data = yf.download(tickers.Symbol.to_list(),'2024-6-1','2024-6-6', auto_adjust=True)['Close']
    # # print(data.head())











    # class TestStrategy(bt.Strategy):

#     def log(self, txt, dt=None):
#         dt = dt or self.datas[0].datetime.datetime()
#         print('%s, %s' % (dt.isoformat(), txt))

#     def __init__(self):
#         self.dataclose = self.datas[0].close
#         self.order = None

#     def notify_order(self, order):
#         if order.status in [order.Submitted, order.Accepted]:
#             return

#         if order.status in [order.Completed]:
#             if order.isbuy():
#                 self.log('BUY EXECUTED {}'.format(order.executed.price))
#             elif order.issell():
#                 self.log('SELL EXECUTED {}'.format(order.executed.price))

#             self.bar_executed = len(self)

#         self.order = None

#     def next(self):
#         self.log('Close, %.2f' % self.dataclose[0])

#         if self.order:
#             return

#         if not self.position:
#             if self.dataclose[0] < self.dataclose[-1]:
#                 if self.dataclose[-1] < self.dataclose[-2]:
#                     self.log('BUY CREATE, %.2f' % self.dataclose[1])
#                     self.order = self.buy()

#         else:
#             if len(self) >= self.bar_executed + 5:
#                 self.log('SELL CREATE {}'.format(self.dataclose[0]))
#                 self.order = self.sell()

# class SmaCross(bt.Strategy):
#     # list of parameters which are configurable for the strategy
#     params = dict(
#         pfast=10,  # period for the fast moving average
#         pslow=30   # period for the slow moving average
#     )

#     def __init__(self):
#         sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
#         sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
#         self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

#     def next(self):
#         if not self.position:  # not in the market
#             if self.crossover > 0:  # if fast crosses slow to the upside
#                 self.buy()  # enter long

#         elif self.crossover < 0:  # in the market & cross to the downside
#             self.close()  # close long position

# class BBPullbackStrategy(bt.Strategy):

#     # list of parameters which are configurable for the strategy
#     params = dict(
#         periodMA=9,  # period for the fast moving average
#         higherTFActive=True
#     )

#     def __init__(self):
#         self.MA1M = bt.ind.EMA(self.data[0], period=self.p.periodMA)
#         self.MA5M = bt.ind.EMA(self.data[1], period=self.p.periodMA)
#         self.source1M = self.datas[0].close
#         self.source5M = self.datas[1].close
#         self.crossoverMA = bt.ind.CrossOver(self.source, self.MA)  # crossover signal
#         self.order = None

#     def next(self):
#         if not self.position:  # not in the market
#             if self.crossover > 0:  # if fast crosses slow to the upside
#                 self.buy()  # enter long

#         elif self.crossover < 0:  # in the market & cross to the downside
#             self.close()  # close long position

# def getSP500Tickers():
#     # also try this: https://www.youtube.com/watch?v=CvV261GpsBg

#     # Read and print the stock tickers that make up S&P500
#     tickers = pd.read_html(
#         'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
#     print(tickers.head())

#     # Get the data for this tickers from yahoo finance
#     data = yf.download(tickers.Symbol.to_list(),'2021-1-1','2021-7-12', auto_adjust=True)['Close']
#     print(data.head())