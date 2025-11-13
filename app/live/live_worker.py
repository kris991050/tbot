import sys, os, pandas as pd, logging
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

import live_loop_base, live_data_logger
from data import hist_market_data_handler
from utils import helpers
from datetime import datetime
from utils.constants import CONSTANTS, PATHS
from utils.timeframe import Timeframe
from execution import trade_manager, trade_executor, orders


class LiveWorker(live_loop_base.LiveLoopBase):
    def __init__(self, worker_type:str, wait_seconds:int=None, continuous:bool=True, single_symbol=None, ib_disconnect:bool=False, initialize:bool=True, 
                 tickers_list:dict={}, base_folder:str=None, config=None, strategy_name:str=None, stop:str=None, revised:bool=None, look_backward:str=None, 
                 step_duration:str=None, live_mode:str=None, file_format: str=None, paper_trading:bool=None, timezone=None):
            
            super().__init__(wait_seconds=wait_seconds, continuous=continuous, single_symbol=single_symbol, ib_disconnect=ib_disconnect, 
                             live_mode=live_mode, config=config, paper_trading=paper_trading, timezone=timezone)
            # self.manager = trade_manager.TradeManager(self.ib, strategy_name, stop, revised=revised, look_backward=look_backward, 
            #                                           step_duration=step_duration, timezone=timezone)
            # self.manager = self._build_manager(args_manager, self.ib, timezone)
            # self.logger = self._build_logger(args_logger, timezone)
            self.worker_type = helpers.set_var_with_constraints(worker_type, CONSTANTS.LIVE_WORKER_TYPES)
            self.base_folder = base_folder or PATHS.folders_path['live_data']
            self.manager = trade_manager.TradeManager(self.ib, config=self.config, base_folder=self.base_folder)
            self.logger = live_data_logger.LiveDataLogger(config=self.config)
            self.executor = trade_executor.TradeExecutor(self.ib, config=self.config)
            self.look_backward = look_backward or CONSTANTS.WARMUP_MAP[self.manager.strategy_instance.timeframe.pandas]
            # self.logger = live_data_logger.LiveDataLogger(self_live.mode, datetime.timedelta(sim_offset_seconds), timezone)
            # self.file_format = file_format
            self.initialize = initialize
            # self.queue = queue or multiprocessing.Queue()
            self.tickers_list = tickers_list or self.logger.load_tickers_list(lock=True)
            self.base_wait_seconds = self.wait_seconds

    def _fetch_symbol(self, symbol, from_time, to_time, init):
        if init:
            df = self.manager.fetch_df(symbol=symbol, from_time=from_time, to_time=to_time, file_format=self.config.file_format)
        else:
            df = self.manager.complete_df(symbol=symbol, file_format=self.config.file_format)
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
        df = self.manager.enrich_df(symbol, self.config.file_format, block_add_sr=block_add_sr, base_timeframe=self.manager.strategy_instance.timeframe, 
                                    from_time=from_time, to_time=to_time)
        return df
    
    def _load_symbol_data(self, symbol):
        # try:
        loader = hist_market_data_handler.HistMarketDataLoader(self.ib, symbol, self.manager.strategy_instance.timeframe, data_type='enriched_data', 
                                                                   base_folder=self.base_folder)
        return loader.load_and_trim()[0]
        # except Exception as e:
        #     print(f"Could not load data for symbol {symbol}: {e}")
        #     return pd.DataFrame()

    def _evaluate_entry(self, symbol):
        df = self._load_symbol_data(symbol)
        df = self.manager.apply_model_predictions(df, symbol)
        last_row = df.iloc[-1]
        stop_price = self.manager.resolve_stop_price(last_row)
        is_triggered, is_predicted, is_RRR = self.manager.evaluate_entry_conditions(last_row, stop_price=stop_price, expand=True)

        if is_triggered:
            self.tickers_list = self.logger.update_ticker(symbol, 'priority', 3, lock=True, log=True)
        # print("last_row = ", last_row)
        print(f"Assessing {symbol} for entry:")
        # print(f"Prediction: {last_row['model_prediction']}")
        print(f"is_triggered: {is_triggered}")
        print(f"is_predicted: {is_predicted}  ({last_row['model_prediction']})")
        print(f"is_RRR: {is_RRR}")
        # return True, last_row
        return is_triggered and is_predicted and is_RRR, last_row
    
    def _evaluate_discard(self, symbol, df=pd.DataFrame()):
        df = df if not df.empty else self._load_symbol_data(symbol)
        if not df.empty:
            last_row = df.iloc[-1]
            discard_condition = self.manager.evaluate_discard_conditions(last_row)
            # if discard_condition:
            #     print(f"Discard conditions met for {symbol}")
            #     print(f"rsi_15: {last_row['rsi_15']}\nrsi_60: {last_row['rsi_60']}")
            return discard_condition
        else: return False

    def _execute_symbol(self, symbol):
        logging.basicConfig(filename=self.logger.trade_log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s' )
        
        entry_condition, last_row = self._evaluate_entry(symbol)
        if entry_condition:
            print(f"Executing Order for {symbol}")
            stop_price = self.manager.resolve_stop_price(last_row)
            target_price = last_row[self.manager.strategy_instance.params['target_indicator']] if self.manager.strategy_instance.params['target_indicator'] else ''
            quantity = self.manager.evaluate_quantity(last_row['model_prediction'])
            # values = self._create_trade_params(symbol, stop_price, target_price, quantity)
            oorder = trade_executor.OOrder(symbol=symbol, stop_loss=stop_price, take_profit=target_price, quantity=quantity, config=self.config)
            
            order, TPSL_order = self.executor.execute_order(self.manager.direction, oorder)
            self.ib.sleep(CONSTANTS.PROCESS_TIME['long'])
            if order is not None and hasattr(order, 'orderStatus'):
                print(f"Order placed with status {order.orderStatus.status}")
                # print(TPSL_order.orderStatus.status)
                if order.orderStatus.status in ['Filled', 'Submitted']:
                    # Log order execution
                    message = f"Executed order for {symbol} with quantity: {quantity}, stop price: {stop_price}, target price: {target_price}, model prediction: {last_row['model_prediction']}."
                    logging.info(message)
                    print(message)
                    self.tickers_list = self.logger.update_ticker(symbol, 'priority', 4, lock=True, log=True)
                
                open_position = orders.get_positions_by_symbol(self.ib, symbol)
                print(open_position)
            else:
                print(f"Could not execute order for {symbol}")

            return last_row.to_frame().T
        else: return pd.DataFrame()

    def _run(self):
        df = pd.DataFrame()

        queue = self.logger.get_queue(self.worker_type, lock=True)
        if not queue:
            print(f"{self.worker_type.capitalize()} queue empty")
            self.wait_seconds = self.base_wait_seconds * 3
            return df
        else:
            print(f"Current {self.worker_type} queue: {queue}")
            self.wait_seconds = self.base_wait_seconds

        symbol = self.logger.pop_queue(queue, self.worker_type, lock=True)

        self.tickers_list = self.logger.load_tickers_list(lock=True)
        if symbol not in self.tickers_list:
            print(f"⚠️ Symbol {symbol} not in tickers_list")
            return df

        init = not self.tickers_list[symbol]['initialized']
        if init and not self.initialize:
            print(f"Passing {symbol} as initialize option not active")
            self.logger.put_queue(symbol, self.worker_type, lock=True)
            return df
        
        trig_time = helpers.calculate_now(self.config.sim_offset, self.config.timezone)
        trig_time_min = trig_time.replace(second=0, microsecond=0) # Floor to minute
        # from_time = helpers.substract_duration_from_time(trig_time_min, self.config.look_backward)
        from_time = Timeframe(self.look_backward).subtract_from_date(trig_time_min)

        # print("tickers_list = ", self.tickers_list)
        print("init = ", init)
        print("trig_time / from_time = ", trig_time, "     ", from_time)
        if not self.tickers_list[symbol][f'{self.worker_type}ing']:
            self.tickers_list = self.logger.update_ticker(symbol, f'{self.worker_type}ing', True, lock=True, log=True)

            if self.worker_type == 'fetch':
                # df = self._fetch_symbol(symbol, trig_time, from_time, init)
                df = self._fetch_symbol(symbol, from_time, trig_time, init)

            elif self.worker_type == 'enrich':
                df = self._enrich_symbol(symbol, from_time, trig_time, init)
                if self._evaluate_discard(symbol, df):
                    self.tickers_list = self.logger.update_ticker(symbol, 'priority', 0, lock=True, log=True)
            
            elif self.worker_type == 'execut':
                df = self._execute_symbol(symbol)

            else:
                 print("Handler mode must be 'fetch' or 'enrich'")
            
            self.tickers_list = self.logger.update_ticker(symbol, f'{self.worker_type}ing', False, lock=True, log=True)

            if not df.empty:
                self.tickers_list = self.logger.update_ticker(symbol, f'last_{self.worker_type}ed', df['date'].iloc[-1], lock=True, log=True)

        return df
    
    def _execute_main_task(self):
        self._run()

if __name__ == "__main__":

    args = sys.argv
    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    paper_trading = not 'live' in args
    revised = 'revised' in args
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

    worker = LiveWorker(worker_type=action, strategy_name=strategy_name, revised=revised, live_mode=mode, initialize=not no_initialize, 
                              paper_trading=paper_trading, wait_seconds=wait_seconds, continuous=continuous)
    worker.run()






# def _create_trade_params(self, symbol, stop, target, quantity):
    #     values = {
    #         '-symbol-': symbol, 
    #         '-take_profit-': target,  
    #         '-stop_loss-': stop, 
    #         '-auto_quantity-': False, 
    #         '-quantity-': quantity, 
    #         '-currency-': helpers.get_stock_currency_yf(symbol) or self.config.default_currency,
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

#                 df = self.manager.fetch_df(symbol, trig_time, from_time, self.file_format, self.step_duration)
#             else:
#                 df = self.manager.complete_df(symbol, self.file_format, self.step_duration)
            
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

