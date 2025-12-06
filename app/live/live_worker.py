import sys, os, pandas as pd
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from live import live_loop_base
from data import hist_market_data_handler
from utils import helpers, logs
from datetime import datetime
from utils.constants import CONSTANTS, PATHS
from utils.timeframe import Timeframe
from execution import trade_executor, orders


class LiveWorker(live_loop_base.LiveLoopBase):
    def __init__(self, action:str, wait_seconds:int=None, continuous:bool=True, single_symbol=None, initialize:bool=True,
                 tickers_list:dict={}, base_folder:str=None, look_backward:str=None, config=None, strategy_name:str=None, live_mode:str=None, paper_trading:bool=None,
                 remote_ib:bool=None, timezone=None):

            super().__init__(worker_type=f"{action}er", wait_seconds=wait_seconds, continuous=continuous, single_symbol=single_symbol, ib_disconnect=False,
                             live_mode=live_mode, config=config, strategy_name=strategy_name, paper_trading=paper_trading, remote_ib=remote_ib, timezone=timezone)

            self.action = helpers.set_var_with_constraints(action, CONSTANTS.LIVE_ACTIONS)
            self.base_folder = base_folder or PATHS.folders_path['live_data']
            self.tmanager.base_folder = self.base_folder
            self.executor = trade_executor.TradeExecutor(self.ib, config=self.config)
            self.look_backward = look_backward or CONSTANTS.WARMUP_MAP[self.tmanager.strategy_instance.timeframe.pandas]
            self.initialize = initialize
            # self.queue = queue or multiprocessing.Queue()
            self.tickers_list = tickers_list or self.logger.load_tickers_list(lock=True)
            self.base_wait_seconds = self.wait_seconds

    def _fetch_symbol(self, symbol, from_time, to_time, init):
        if init:
            df = self.tmanager.fetch_df(symbol=symbol, from_time=from_time, to_time=to_time, file_format=self.config.file_format)
        else:
            df = self.tmanager.complete_df(symbol=symbol, file_format=self.config.file_format)
        return df

    def _assess_block_add_sr(self, symbol:str, trig_time:datetime) -> bool:
        # Get the min refresh_rate from settings
        # min_refresh_str = min(CONSTANTS.SR_SETTINGS, key=lambda setting: helpers.get_ibkr_duration_as_timedelta(setting['refresh_rate']))['refresh_rate']
        min_refresh_str = min(CONSTANTS.SR_SETTINGS, key=lambda setting: Timeframe(setting['refresh_rate']).to_timedelta)['refresh_rate']
        min_refresh_td = helpers.parse_timedelta(min_refresh_str)

        info = self.tickers_list[symbol]
        if not info['initialized'] or not info['last_enriched']:
            return False
        # if not last_enriched:
        #     return False # Case symbol not present yet or no last_enriched yet
        return not ((trig_time - pd.to_datetime(info['last_enriched'])) > min_refresh_td)

    def _enrich_symbol(self, symbol, from_time, to_time, init):
        block_add_sr = False if init else self._assess_block_add_sr(symbol, to_time)
        df = self.tmanager.enrich_df(symbol, self.config.file_format, block_add_sr=block_add_sr, base_timeframe=self.tmanager.strategy_instance.timeframe,
                                    from_time=from_time, to_time=to_time)
        return df

    def _load_symbol_data(self, symbol):
        # try:
        loader = hist_market_data_handler.HistMarketDataLoader(self.ib, symbol, self.tmanager.strategy_instance.timeframe, data_type='enriched_data',
                                                                   base_folder=self.base_folder)
        return loader.load_and_trim()[0]
        # except Exception as e:
        #     print(f"Could not load data for symbol {symbol}: {e}")
        #     return pd.DataFrame()

    def _evaluate_entry(self, symbol):
        df = self._load_symbol_data(symbol)
        if df.empty:
            print(f"⚠️ No data could be loaded for {symbol}")
            return False, pd.Series()
        df = self.tmanager.apply_model_predictions(df, symbol)
        last_row = df.iloc[-1]
        stop_price = self.tmanager.resolve_stop_price(last_row)
        is_triggered, is_predicted, is_RRR = self.tmanager.evaluate_entry_conditions(last_row, stop_price=stop_price, expand=True)

        if is_triggered:
            self.tickers_list = self.logger.update_ticker(symbol, 'priority', 3, lock=True, log=True)

        df_entry = pd.DataFrame({
            'Triggered': [is_triggered, ''],
            'Predicted': [is_predicted, last_row['model_prediction'].round(3)],
            'RRR': [is_RRR, self.tmanager.evaluate_RRR(row=last_row, stop_price=stop_price)[1]]
            })
        print(helpers.df_to_table(df_entry, title=f"Assessing {symbol} for entry"), "\n")

        # print(f"Assessing {symbol} for entry:")
        # print(f"is_triggered: {is_triggered}")
        # print(f"is_predicted: {is_predicted}  ({last_row['model_prediction']})")
        # print(f"is_RRR: {is_RRR}")

        # return True, last_row
        return is_triggered and is_predicted and is_RRR, last_row

    def _evaluate_discard(self, symbol, df=pd.DataFrame()):
        df = df if not df.empty else self._load_symbol_data(symbol)
        if not df.empty:
            last_row = df.iloc[-1]
            discard_condition = self.tmanager.evaluate_discard_conditions(last_row)
            # if discard_condition:
            #     print(f"Discard conditions met for {symbol}")
            #     print(f"rsi_15: {last_row['rsi_15']}\nrsi_60: {last_row['rsi_60']}")
            return discard_condition
        else: return False

    def _execute_symbol(self, symbol):
        # logging.basicConfig(filename=self.logger.trade_log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s' )
        # with logs.LogContext(self.logger.trade_log_file_path, overwrite=True):  # Logging trades specifically in separate trade log

        entry_condition, last_row = self._evaluate_entry(symbol)
        if entry_condition:
            print(f"Executing Order for {symbol}")
            stop_price = self.tmanager.resolve_stop_price(last_row)
            # target_price = last_row[self.tmanager.strategy_instance.params['target_indicator']] if self.tmanager.strategy_instance.params['target_indicator'] else ''
            # Set target entry time and price
            self.tmanager.set_target_for_entry(row=last_row, stop_price=stop_price, symbol=symbol)
            target_price = self.tmanager.strategy_instance.target_handler.target_price
            quantity = self.tmanager.evaluate_quantity(last_row['model_prediction'], last_row['close'], stop_price, self.tmanager.capital, self.config.risk_pct)
            # values = self._create_trade_params(symbol, stop_price, target_price, quantity)

            open_position = orders.get_positions_by_symbol(self.ib, symbol)

            if not open_position:
                oorder = trade_executor.OOrder(symbol=symbol, stop_loss=stop_price, take_profit=target_price, quantity=quantity, config=self.config)
                now = helpers.calculate_now(self.config.sim_offset, self.config.timezone)
                if self.config.live_mode == 'live':
                    # price_diff_threshold = abs(target_price - last_row['close']) / self.config.rrr_threshold if (target_price and self.config.rrr_threshold) \
                    #     else 0.05 * last_row['close'] # Fallback if no stop_price. In case no target_price as well, max price diff is 5% of price.
                    price_diff_threshold = abs(((target_price + self.config.rrr_threshold * stop_price) / (1 + self.config.rrr_threshold)) - last_row['close']) \
                        if (target_price and stop_price and self.config.rrr_threshold) else 0.05 * last_row['close'] # Fallback if no stop_price. In case no target_price as well, max price diff is 5% of price.
                    trade, TPSL_trades = self.executor.execute_order(self.tmanager.direction, oorder, last_row['close'], price_diff_threshold)
                    self.ib.sleep(CONSTANTS.PROCESS_TIME['long'])
                    fill_time = trade.fills[0].time if trade.fills else now
                    order_status = trade.orderStatus.status if hasattr(trade, 'orderStatus') else None
                    order_avg_fill_price = trade.orderStatus.avgFillPrice if hasattr(trade, 'orderStatus') else None
                    self.ib.sleep(CONSTANTS.PROCESS_TIME['long'])
                elif self.config.live_mode == 'sim':
                    order_status = 'Simulated'
                    order_avg_fill_price = last_row['close']
                    fill_time = now
                    trade = oorder

                self.logger.save_to_trade_log_csv(self.ib, fill_time, symbol, last_row, quantity, target_price, stop_price, order_status, order_avg_fill_price,  self.tmanager.get_required_columns())
            else:
                trade = None

            if trade is not None and order_status:
                print(f"Order placed with status {order_status}")
                # print(TPSL_order.orderStatus.status)
                if order_status in ['Filled', 'Submitted', 'Simulated']:
                    # Log order execution
                    message = f"Executed order for {symbol} with quantity: {quantity}, stop price: {stop_price}, target price: {target_price}, model prediction: {last_row['model_prediction']}."
                    # logging.info(message)
                    print(message)
                    self.tickers_list = self.logger.update_ticker(symbol, 'priority', 4, lock=True, log=True)

                open_position = orders.get_positions_by_symbol(self.ib, symbol)
                self.ib.sleep(CONSTANTS.PROCESS_TIME['long'])
                print(f"Open position: {open_position}")
                if open_position:
                    if self.config.paper_trading:
                        self.tmanager.capital -= quantity * order_avg_fill_price
                    else:
                        self.tmanager.get_equity()
            else:
                print(f"Could not execute order for {symbol} or already existing open position.")

        return last_row.to_frame().T
        # else:
        #     return pd.DataFrame()

    def _run_worker(self):
        df = pd.DataFrame()

        queue = self.logger.get_queue(self.action, lock=True)
        if not queue:
            print(f"{self.action.capitalize()} queue empty")
            self.wait_seconds = self.base_wait_seconds * 3
            return df
        else:
            print(f"Current {self.action} queue: {queue}")
            self.wait_seconds = self.base_wait_seconds

        symbol = self.logger.pop_queue(queue, self.action, lock=True)

        self.tickers_list = self.logger.load_tickers_list(lock=True)
        if symbol not in self.tickers_list:
            print(f"⚠️ Symbol {symbol} not in tickers_list")
            return df

        init = not self.tickers_list[symbol]['initialized']
        if init and not self.initialize:
            print(f"Passing {symbol} as initialize option not active")
            self.logger.put_queue(symbol, self.action, lock=True)
            return df

        trig_time = helpers.calculate_now(self.config.sim_offset, self.config.timezone)
        trig_time_min = trig_time.replace(second=0, microsecond=0) # Floor to minute
        # from_time = helpers.substract_duration_from_time(trig_time_min, self.config.look_backward)
        from_time = Timeframe(self.look_backward).subtract_from_date(trig_time_min)

        # print("tickers_list = ", self.tickers_list)
        print("init = ", init)
        print("trig_time / from_time = ", trig_time, "     ", from_time)
        if not self.tickers_list[symbol][f'{self.action}ing']:
            self.tickers_list = self.logger.update_ticker(symbol, f'{self.action}ing', True, lock=True, log=True)

            if self.action == 'fetch':
                # df = self._fetch_symbol(symbol, trig_time, from_time, init)
                df = self._fetch_symbol(symbol, from_time, trig_time, init)

            elif self.action == 'enrich':
                df = self._enrich_symbol(symbol, from_time, trig_time, init)
                if self._evaluate_discard(symbol, df):
                    self.tickers_list = self.logger.update_ticker(symbol, 'priority', 0, lock=True, log=True)

            elif self.action == 'execut':
                df = self._execute_symbol(symbol)

            else:
                 print("Handler mode must be 'fetch' or 'enrich'")

            self.tickers_list = self.logger.update_ticker(symbol, f'{self.action}ing', False, lock=True, log=True)

            # if not df.empty:
            date_action = df['date'].iloc[-1] if not df.empty else helpers.calculate_now(self.config.sim_offset, self.config.timezone)
            self.tickers_list = self.logger.update_ticker(symbol, f'last_{self.action}ed', date_action, lock=True, log=True)

        return df

    def _execute_main_task(self):
        self._run_worker()

if __name__ == "__main__":

    args = sys.argv
    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    paper_trading = not 'live' in args
    remote_ib = 'remote' in args
    # revised = 'revised' in args
    continuous = not 'snapshot' in args
    no_initialize = 'noinit' in args
    wait_seconds = next((int(float(arg[5:])) for arg in args if arg.startswith('wait=')), 2)
    strategy_name = next((arg[9:] for arg in args if arg.startswith('strategy=')), None)
    action = next((arg[7:] for arg in args if arg.startswith('action=')), '')
    mode = next((arg[5:] for arg in args if arg.startswith('mode=')), 'live')

    # args_manager = {'strategy_name': strategy_name, 'stop': stop, 'target': None, 'revised': False, 'timeframe': None, 'look_backward': '1 M',
    #                           'step_duration': '1 W', 'config': None}
    # args_logger = {'mode': 'live', 'sim_offset': datetime.timedelta(0)}

    # import json
    # args_trade_manager_json = json.dumps({'strategy_name': strategy_name, 'stop': stop, 'target': None,
    #                           'revised': revised, 'timeframe': None, 'look_backward': None,
    #                           'step_duration': None})

    # args_logger_json = json.dumps({'mode': 'live', 'sim_offset': datetime.timedelta(0).total_seconds()})

    # trade_manager = trade_manager.TradeManager(IB(), strategy_name, stop, revised=revised)

    worker = LiveWorker(action=action, live_mode=mode, initialize=not no_initialize, strategy_name=strategy_name,
                              paper_trading=paper_trading, wait_seconds=wait_seconds, continuous=continuous, remote_ib=remote_ib)
    worker.run()






# def _create_trade_params(self, symbol, stop, target, quantity):
    #     values = {
    #         '-symbol-': symbol,
    #         '-take_profit-': target,
    #         '-stop_loss-': stop,
    #         '-auto_quantity-': False,
    #         '-quantity-': quantity,
    #         '-currency-': helpers.get_stock_currency_yf(symbol) or self.config.currency,
    #         '-profit_ratio-': self.config.profit_ratio,
    #         '-offset_targets-': self.config.offset_targets,
    #         '-auto_stop_loss-': False
    #         }
    #     return values



# def _fecth_data(self):
#         df = pd.DataFrame()

#         if self.queue and not self.queue.Empty:
#             symbol = self.queue.get(timeout=2)

#             if not self.initialize:
#                 trig_time = helpers.calculate_now(self.logger.sim_offset, self.logger.mode, self.tz)
#                 trig_time_min = trig_time.replace(second=0, microsecond=0) # Floor to minute
#                 from_time = helpers.substract_duration_from_time(trig_time_min, self.look_backward)

#                 df = self.tmanager.fetch_df(symbol, trig_time, from_time, self.file_format, self.step_duration)
#             else:
#                 df = self.tmanager.complete_df(symbol, self.file_format, self.step_duration)

#             self._update_ticker(symbol, 'fetching', False)

#         return df



# def is_between_market_times(start_label, end_label):
#     now = pd.Timestamp.now(tz=CONSTANTS.TZ_WORK).time()
#     return CONSTANTS.TH_TIMES[start_label] < now < CONSTANTS.TH_TIMES[end_label]


# # === SCANNER TASKS === #

# def scan_gappers():
#     if not is_between_market_times('pre-market', 'rth'):
#         return
#     print('\n======== FETCHING GAPPERS UP ========\n')
#     symbols_up, df_up = scanner.scannerTradingView("GapperUp")
#     print("\n", helpers.df_to_table(df_up.round(2)), "\n")
#     print("\nSymbols from TradingView scanner GapperUp=\n", symbols_up)
#     helpers.save_to_daily_csv(ib, symbols_up, PATHS.daily_csv_files['gapper_up'])

#     print('\n======== FETCHING GAPPERS DOWN ========\n')
#     symbols_down, df_down = scanner.scannerTradingView("GapperDown")
#     print("\n", helpers.df_to_table(df_down.round(2)), "\n")
#     print("\nSymbols from TradingView scanner GapperDown=\n", symbols_down)
#     helpers.save_to_daily_csv(ib, symbols_down, PATHS.daily_csv_files['gapper_down'])


# def scan_earnings():
#     if not is_between_market_times('pre-market', 'rth'):
#         return
#     print('\n======== FETCHING RECENT EARNINGS ========\n')
#     symbols_earnings = scanner.scannerFinviz("RE")
#     print("\nSymbols from Finviz scanner Recent Earnings=\n", symbols_earnings)
#     helpers.save_to_daily_csv(ib, symbols_earnings, PATHS.daily_csv_files['earnings'])


# def scan_bb_rsi_reversal():
#     if not is_between_market_times('pre-market', 'post-market'):
#         return
#     print('\n======== FETCHING RSI REVERSALS ========\n')
#     symbols_rsi, df_rsi = scanner.scannerTradingView("RSI-Reversal")
#     print("\nSymbols from TradingView scanner RSI_Reversal=\n", symbols_rsi)
#     helpers.save_to_daily_csv(ib, symbols_rsi, PATHS.daily_csv_files['bb_rsi_reversal'])


# if __name__ == "__main__":

#     # # Path Setup
#     # path = helpers.path_setup.path_current_setup(parent_folder)

#     args = sys.argv
#     paperTrading = not 'live' in args
#     continuous = 'cont' in args
#     time_wait = next((int(float(arg[5:])) * 60 for arg in args if arg.startswith('wait=')), 5 * 60)


#     # TWS Connection
#     paperTrading = False if len(args) > 1 and 'live' in args else True
#     ib, ibConnection = helpers.IBKRConnect_any(IB(), paper=paperTrading)

#     # Setup
#     daily_data_folder = helpers.get_path_daily_data_folder()

#     counter = 1000
#     while counter > 0:

#         time_now = pd.Timestamp.now(tz=CONSTANTS.TZ_WORK)

#         # Scanner Gappers
#         scan_gappers()
#         scan_earnings()
#         scan_bb_rsi_reversal()

#         if continuous:

#             counter = counter - 1
#             helpers.sleep_display(time_wait, ib)

#         else: counter = 0

# print("\n\n")


# from apscheduler.schedulers.blocking import BlockingScheduler
# from apscheduler.triggers.interval import IntervalTrigger

# def display_next_runs():
#     now = datetime.now(datetime.timezone.utc)  # Use UTC for consistency with APScheduler for delta calcuation
#     print("\n📅 Scheduled Jobs:")
#     for job in scheduler.get_jobs():
#         if job.id != 'job_monitor':
#             if job.next_run_time and job.id != 'job_monitor':
#                 delta = (job.next_run_time - now).total_seconds()
#                 seconds_remaining = int(delta) if delta > 0 else 0
#                 print(f"  🔔 Job ID: {job.id:<15} | Runs in: {seconds_remaining:>4} seconds")
#             else:
#                 print(f"  ⚠️  Job ID: {job.id:<15} | No next run scheduled.")

# if __name__ == "__main__":

#     # # Path Setup
#     # path = helpers.path_setup.path_current_setup(parent_folder)

#     args = sys.argv
#     paperTrading = not 'live' in args

#     # TWS Connection
#     paperTrading = False if len(args) > 1 and 'live' in args else True
#     ib, ibConnection = helpers.IBKRConnect_any(IB(), paper=paperTrading)

#     # Setup
#     daily_data_folder = helpers.get_path_daily_data_folder() + '_2'
#     scheduler = BlockingScheduler(timezone=CONSTANTS.TZ_WORK)
#     gappers_delay = 5
#     earnings_delay = 10
#     bb_rsi_reversal_delay = 3

#     # === JOB SCHEDULES === #

#     time_now = pd.Timestamp.now(tz=CONSTANTS.TZ_WORK)

#     scheduler.add_job(display_next_runs, IntervalTrigger(seconds=10), id='job_monitor', next_run_time=time_now)
#     # scheduler.add_job(scan_earnings, IntervalTrigger(minutes=earnings_delay), id='earnings', next_run_time=time_now + datetime.timedelta(seconds=5))
#     # scheduler.add_job(scan_gappers, IntervalTrigger(minutes=gappers_delay), id='gappers', next_run_time=time_now + datetime.timedelta(seconds=35))
#     # scheduler.add_job(scan_bb_rsi_reversal, IntervalTrigger(minutes=bb_rsi_reversal_delay), id='bb_rsi_reversal', next_run_time=time_now + datetime.timedelta(seconds=65))
#     scheduler.add_job(run_all_scans, IntervalTrigger(minutes=bb_rsi_reversal_delay), id='all_scans', next_run_time=time_now + datetime.timedelta(seconds=5))

#     # === START SCHEDULER === #
#     print("Starting scheduler. Press Ctrl+C to exit.\n")
#     try:
#         scheduler.start()
#     except (KeyboardInterrupt, SystemExit):
#         print("Shutting down scheduler...")
#         ib.disconnect()

# print("\n\n")
