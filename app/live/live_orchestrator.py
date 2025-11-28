import os, sys, pandas as pd, subprocess, multiprocessing, psutil#, tqdm, asyncio, concurrent.futures
from datetime import datetime
from ib_insync import *

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.constants import CONSTANTS
from execution import trade_manager
import live_data_logger, process_launcher, trading_config


# class Simulator:
#     def __init__(self, timezone:pytz, mode:str='live'):
#         self.tz = timezone or CONSTANTS.TZ_WORK
#         self.mode = helpers.set_var_with_constraints(mode, CONSTANTS.MODES['live'])
#         self.sim_start = self.tz.localize(datetime(2025, 9, 25, 9, 14, 0))
#         self.sim_max_time = timedelta(hours=4, minutes=0)
#         self.sim_offset = datetime.now(self.tz) - self.sim_start if self.mode == 'sim' else timedelta(0) if self.mode == 'live' else None

class LiveOrchestrator:
    def __init__(self, ib, max_fetchers:int=2, max_enrichers:int=2, new_terminal:bool=True, tail_log:bool=True, seed:int=None, strategy_name:str=None, stop=None, 
                 step_duration:str=None, revised:bool=None, live_mode:str='live', paper_trading:bool=None, file_format:str=None, remote_ib:bool=None, timezone=None):
    # def __init__(self, ib, strategy_name:str, stop, target:str=None, timeframe:str=None, look_backward:str='1M', step_duration:str='1W', revised:bool=False, 
    #              max_fetchers:int=2, max_enrichers:int=2, mode:str='live', paper_trading:bool=True, timezone=None):
        self.ib = ib
        self.live_mode = helpers.set_var_with_constraints(live_mode, CONSTANTS.MODES['live'])
        self.config = trading_config.TradingConfig(live_mode=self.live_mode).set_config(locals())

        # self.live_mode = helpers.set_var_with_constraints(self.config.live_mode, CONSTANTS.MODES['live'])
        self.tmanager = trade_manager.TradeManager(ib, config=self.config)
        self.logger = live_data_logger.LiveDataLogger(config=self.config)
        self.config.save_config(self.logger.config_file_path)
        
        # self.tmanager = trade_manager.TradeManager(ib, strategy_name, stop, revised=revised, look_backward=self.config.look_backward, 
        #                                           step_duration=self.config.step_duration, config=None, timezone=self.config.timezone)
        # self.simulator = Simulator(self.tmanager.tz, self.mode)
        # self.logger = live_data_logger.LiveDataLogger(mode=self.mode, sim_offset=self.config.sim_offset, timezone=self.config.timezone)

        # self.scan_rate = timeframe_handler.TimeframeHandler.timeframe_to_seconds(self.tmanager.timeframe)
        self.scan_rate = self.tmanager.strategy_instance.timeframe.to_seconds
        self.max_fetchers = max_fetchers
        self.max_enrichers = max_enrichers
        self.new_terminal = new_terminal
        self.tail_log = tail_log
        # List to store subprocess.Popen objects (or their PIDs)
        self.processes = []
        self.proc_params = self._set_processes_params()
        # self.mtp_manager = multiprocessing.Manager()
        # self.tickers_list = self.mtp_manager.dict()
        # self.fetch_queue = self.mtp_manager.Queue()
        # self.enrich_queue = self.mtp_manager.Queue()
        # self.fetch_queue = multiprocessing.Queue()
        # self.enrich_queue = multiprocessing.Queue()
        # self.tickers_list = self.logger.load_tickers_list(lock=True)
        # self.fetch_queue = self.logger.get_queue('fetch', lock=True)
        # self.enrich_queue = self.logger.get_queue('enrich', lock=True)
        # self.execite_queue = self.logger.get_queue('execut', lock=True)

    def _set_processes_params(self):
        log_folder = helpers.get_path_daily_logs_folder()
        scan_rate_min = int(self.scan_rate / 60)
        return {
            'scans': {'wait_seconds': 3 * 60, 'log_path': os.path.join(log_folder, 'process_scans_fetcher.log'), 'ib_client_id': 9}, 
            'L2': {'wait_seconds': 5 * 60, 'log_path': os.path.join(log_folder, 'process_L2_fetcher.log'), 'ib_client_id': 10}, 
            'fetch': {'wait_seconds': 1 * scan_rate_min, 'log_path': os.path.join(log_folder, 'process_data_fetcher.log'), 'ib_client_id': None}, 
            'enrich': {'wait_seconds': 1 * scan_rate_min, 'log_path': os.path.join(log_folder, 'process_data_enricher.log'), 'ib_client_id': None},
            'queue': {'wait_seconds': 20, 'log_path': os.path.join(log_folder, 'process_queue_manager.log'), 'ib_client_id': 8},
            'orchestrator': {'wait_seconds': 20}#5 * scan_rate_min}
        }

    # def _organize_tickers_list(self, symbols_scanner:list):
    #     # Add new tickers
    #     for symbol in self.symbols_seed + symbols_scanner:
    #         self.tickers_list = self.logger.load_tickers_list(lock=True)
    #         if symbol not in self.tickers_list:
    #             self.tickers_list[symbol] = self.logger.initialize_ticker()
    #             self.logger.save_tickers_list(self.tickers_list, lock=True)
    #         else:
    #             if not self.tickers_list[symbol]['active']:
    #                 self.tickers_list = self.logger.update_ticker(symbol, 'active', True, lock=True, log=True)
        
    #     if self.live_mode:
    #         # Deactivate tickers not in scanner results anymore
    #         self.tickers_list = self.logger.load_tickers_list(lock=True)
    #         active_symbols = [symbol for symbol in self.tickers_list if self.tickers_list[symbol]['active']]
    #         for symbol in active_symbols:
    #             if symbol not in symbols_scanner and symbol not in self.symbols_seed:
    #                 self.tickers_list = self.logger.update_ticker(symbol, 'active', False, lock=True, log=True)

    # def orchestrate_queues(self):
    #     print("🧠 Orchestration dispatcher started.")

    #     while True:
    #         now = helpers.calculate_now(self.config.sim_offset, self.tmanager.tz)
    #         print(f"\n⏱️ Current Time: {now}")
    #         symbols_scanner = self.tmanager.get_scanner_data(now, use_daily_data=self.live_mode=='sim')
    #         print(f"📡 Fetched symbols from scanner:\n{symbols_scanner}")

    #         # symbols_scanner = ['CPB']#, 'SHFS', 'UUUU']

    #         self._organize_tickers_list(symbols_scanner)

    #         self.tickers_list = self.logger.load_tickers_list(lock=True)
    #         for symbol, info in list(self.tickers_list.items()):
    #             if not info['active']:
    #                 continue
                
    #             condition_handling = not (self.tickers_list[symbol]['fetching'] or self.tickers_list[symbol]['enriching'])
    #             # --- Initialization Phase ---
    #             if not self.tickers_list[symbol]['initialized']:
    #                 if condition_handling and not self.tickers_list[symbol]['last_fetched']:
    #                     self.logger.put_queue(symbol, 'fetch', lock=True)
    #                     continue
    #                 if self.tickers_list[symbol]['last_fetched'] and condition_handling and not self.tickers_list[symbol]['last_enriched']:
    #                     self.logger.put_queue(symbol, 'enrich', lock=True)
    #                     continue

    #                 if self.tickers_list[symbol]['last_fetched'] and self.tickers_list[symbol]['last_enriched']:
    #                     self.logger.update_ticker(symbol, 'initialized', True)
    #                     self.tickers_list = self.logger.update_ticker(symbol, 'priority', 1, lock=True, log=True)
    #                     # self._update_ticker(symbol, 'status', 'ready')
    #                     # self._update_ticker(symbol, 'last_updated', now)
    #                     continue

    #             # --- Recurrent Phase ---
    #             if self.tickers_list[symbol]['initialized']:
    #                 # time_since_update = (now - self.tickers_list[symbol]['last_updated']).total_seconds() if self.tickers_list[symbol]['last_updated'] else float('inf')
    #                 time_since_fetched = (now - pd.to_datetime(self.tickers_list[symbol]['last_fetched'])).total_seconds() if self.tickers_list[symbol]['last_fetched'] else float('inf')
    #                 time_since_enriched = (now - pd.to_datetime(self.tickers_list[symbol]['last_enriched'])).total_seconds() if self.tickers_list[symbol]['last_enriched'] else float('inf')
    #                 time_since_executed = (now - pd.to_datetime(self.tickers_list[symbol]['last_executed'])).total_seconds() if self.tickers_list[symbol]['last_executed'] else float('inf')

    #                 # p = self.tickers_list[symbol].get("priority", 2) if symbol in self.tickers_list else 2
    #                 # if p > 0:
    #                 #     print()

    #                 if condition_handling and time_since_fetched > self.scan_rate:
    #                     self.logger.put_queue(symbol, 'fetch', lock=True)
    #                     continue

    #                 if condition_handling and time_since_enriched > self.scan_rate and time_since_fetched < self.scan_rate:
    #                     self.logger.put_queue(symbol, 'enrich', lock=True)
    #                     continue

    #                 if condition_handling and time_since_executed > self.scan_rate and time_since_enriched < self.scan_rate and time_since_fetched < self.scan_rate:
    #                     self.logger.put_queue(symbol, 'execut', lock=True)
    #                     continue

    #         helpers.sleep_display(self.proc_params['orchestrator']['wait_seconds'], self.ib)

    def _launch_process(self, target, args=(), log_path:str=None, pname:str=None):
        processes, pids = [], []
        if self.new_terminal:
            command = process_launcher.get_terminal_command(target, args)
            if command:
                print(f"▶️ Starting process {pname}...")
                process = subprocess.Popen(command, shell=True)
                ib.sleep(5 * CONSTANTS.PROCESS_TIME['long'])
                processes = helpers.get_process_by('cmdline', pname)
                pids = [proc['pid'] for proc in processes if isinstance(proc['cmdline'], list)]
            else:
                print("❌ Could not determine terminal command for your OS.")
        else:
            # Add timestamp to log path if provided
            if log_path:
                ts = datetime.now(self.tmanager.tz).strftime("%Y%m%d_%H%M%S_%z")
                base, ext = os.path.splitext(log_path)
                log_path = f"{base}_{ts}{ext}"
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
            else:
                log_path = os.devnull  # silence output if no log specified

            # Start a wrapper process that sets up logging
            p = multiprocessing.Process(target=process_launcher.run_with_logging, args=(target, args, log_path))
            p.start()
            processes.append(p)
            pids.append(p.pid)

            if self.tail_log and log_path != os.devnull:
                process_launcher.tail_log_in_new_terminal(log_path)
                ib.sleep(5 * CONSTANTS.PROCESS_TIME['long'])
                processes_log = helpers.get_process_by('cmdline', pname)
                pids.extend([proc['pid'] for proc in processes_log if isinstance(proc['cmdline'], list)])
                processes.extend(processes_log)
            
        self.processes.append({'name': pname, 'pids': pids, 'processes': processes})
    
    def get_process_ids(self):
        """Retrieve the PIDs of all running processes."""
        return [p.pid for p in self.processes]

    def check_process_alive(self, pid:int):
        """Check if a process with a given PID is still running."""
        # Check the process status using the poll method of the Popen object
        for p in self.processes:
            if p.pid == pid:
                return p.poll() is None  # If the process hasn't terminated, poll() will return None
        return False
    
    def orchestrate_processes(self):

        print("⚡ Starting live processes.")

        # Start scans fetcher
        args_scans = (self.proc_params['scans']['wait_seconds'], self.config.live_mode, self.proc_params['scans']['ib_client_id'], 
                      self.config.paper_trading, self.config.remote_ib)
        self._launch_process(process_launcher.run_live_scans_fetcher, args=args_scans, log_path=self.proc_params['scans']['log_path'], pname='scans_fetcher')
        
        # Start L2 fecther
        args_L2 = (self.proc_params['L2']['wait_seconds'], self.config.live_mode, self.proc_params['L2']['ib_client_id'], 
                      self.config.paper_trading, self.config.remote_ib)
        self._launch_process(process_launcher.run_live_L2_fetcher, args=args_L2, log_path=self.proc_params['L2']['log_path'], pname='L2_fetcher')

        args_worker = (self.tmanager.strategy_name, self.config.stop, self.config.revised, self.config.step_duration, self.config.live_mode, 
                             self.config.sim_offset.total_seconds(), self.config.paper_trading, self.proc_params['fetch']['wait_seconds'])
        args_data_fetcher = ('fetch',) + args_worker

        # Start data fetcher        
        # self._launch_process(process_launcher.run_live_worker, args=args_data_fetcher, new_terminal=True, 
        #                      log_path=self.proc_params['fetch']['log_path'], tail_logs=True)
        
        # # Start data enricher
        # self._launch_process(process_launcher.run_live_worker, args=('enrich',) + args_worker, new_terminal=True, 
        #                      log_path=self.proc_params['fetch']['log_path'], tail_logs=True)
        
        # # Start init data fetcher
        # args_trade_manager_json = json.dumps({'strategy_name': self.tmanager.strategy_name, 'stop': self.tmanager.stop, 'target': self.tmanager.target, 
        #                       'revised': self.tmanager.revised, 'timeframe': self.tmanager.timeframe, 'look_backward': self.tmanager.look_backward, 
        #                       'step_duration': self.tmanager.step_duration})
        
        # args_logger_json = json.dumps({'mode': self.logger.mode, 'sim_offset': self.logger.sim_offset.total_seconds()})
        
        # args_data_fetcher = (self.paper_trading, self.proc_params['fetch']['wait_seconds'], args_trade_manager_json, args_logger_json, self.tickers_list, self.tmanager.tz)
        # self._launch_process(process_launcher.run_live_data_fetcher, args=args_data_fetcher, new_terminal=True, 
        #                      log_path=self.proc_params['fetch']['log_path'], tail_logs=True)

        # # Start init data enricher
        # args_data_enricher = (self.paper_trading, self.proc_params['enrich']['wait_seconds'], args_trade_manager_json, args_logger_json, self.tickers_list, self.tmanager.tz)
        # self._launch_process(process_launcher.run_live_data_enricher, args=args_data_enricher, new_terminal=True, 
        #                      log_path=self.proc_params['enrich']['log_path'], tail_logs=True)

        # # Start continuous data fetchers
        # for _ in range(self.max_fetchers - 1):
        #     self._launch_process(process_launcher.run_live_data_fetcher, args=args_data_fetcher + (True,), new_terminal=False, 
        #                          log_path=self.proc_params['fetch']['log_path'], tail_logs=True)
        #     # self._launch_process(process_launcher.run_live_data_fetcher, args=(self.shared_tickers, self.fetch_queue, self.enrich_queue))

        # # Start continuous data enrichers
        # for _ in range(self.max_enrichers - 1):
        #     self._launch_process(process_launcher.run_live_data_enricher, args=args_data_enricher + (True,), new_terminal=False, 
        #                          log_path=self.proc_params['enrich']['log_path'], tail_logs=True)

        # self.orchestrate()

        # Optional: Start executor here

        # # Monitor loop
        # try:
        #     while True:
        #         self.ib.sleep(5)
        #         print("🧠 Ticker status overview:")
        #         for symbol, state in self.shared_tickers.items():
        #             print(f"{symbol}: {state}")

        #         now = helpers.calculate_now(self.sim_offset, self.mode, self.tmanager.tz)
        #         if self.mode == 'sim' and now > self.sim_start + self.sim_max_time:
        #             print("Simulation time elapsed.")
        #             break

        #         symbols_scanner = self.tmanager.get_scanner_data(now)

        #         for scanned_symbol in symbols_scanner:
        #             if scanned_symbol in self.logger.symbols_list:
        #                 self.logger.update_symbols_list(scanned_symbol, 'active', True)
        #                 # self.symbols_list[scanned_symbol]['active'] = True
        #             else:
        #                 # self.symbols_list[scanned_symbol] = {'active': True, 'last_trig': None}
        #                 self.logger.update_symbols_list(scanned_symbol, 'active', True)
        #                 self.logger.update_symbols_list(scanned_symbol, 'last_trig', None)


        #         print("\n=================================")
        #         print("Current time: ", now)
        #         print("Active Symbols:")
        #         active_symbols_list = [s for s in self.logger.symbols_list if self.logger.symbols_list[s]['active']]
        #         for symbol in active_symbols_list:
        #             print(symbol)
        #         print("=================================\n")

        #         if not self.logger.last_scanner_check or (now - self.logger.last_scanner_check > self.scanner_interval):
        #             print("📡 Updating symbols from scanner...")
                    


        # except KeyboardInterrupt:
        #     print("💥 Shutting down processes...")
        #     for p in self.processes:
        #         p.terminate()


if __name__ == "__main__":

    args = sys.argv
    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    paper_trading = not 'live' in args
    remote_ib = 'remote' in args
    revised = 'revised' in args
    new_terminal = not 'no_terminal' in args
    tail_log = 'tail_log' in args
    seed = next((int(arg[5:]) for arg in args if arg.startswith('seed=')), None)
    strategy_name = next((arg[9:] for arg in args if arg.startswith('strategy=')), None)
    mode = next((arg[5:] for arg in args if arg.startswith('mode=')), 'live')

    ib, _ = helpers.IBKRConnect_any(IB(), paper=paper_trading, remote=remote_ib)

    orchestrator = LiveOrchestrator(ib, seed=seed, strategy_name=strategy_name, revised=revised, live_mode=mode, paper_trading=paper_trading, remote_ib=remote_ib, 
                                    new_terminal=new_terminal, tail_log=tail_log)
    orchestrator.orchestrate_processes()


    # def fetch_worker(self):
    #     while True:
    #         try:
    #             symbol = self.fetch_queue.get(timeout=2)
    #         except self.fetch_queue.Empty:
    #             continue

    #         try:
    #             # Perform fetch (warmup if not initialized, light otherwise)
    #             # Update shared_tickers with last_fetched timestamp
    #             print()
    #         finally:
    #             self._update_ticker(symbol, 'fetching', False)


    # def _assign_fetch(self, symbol, init: bool):
    #     self._update_ticker(symbol, 'fetching', True)

    #     args = (symbol, self.look_backward, self.step_duration, self.paper_trading, 0, False, init)
    #     self._launch_process(process_launcher.run_live_data_fetcher, args=args, new_terminal=False, tail_logs=False)

        

    # def _assign_enrich(self, symbol, init: bool):
    #     self._update_ticker(symbol, 'enriching', True)

    #     args = (symbol, self.look_backward, self.paper_trading, 0, False, init)
    #     self._launch_process(process_launcher.run_live_data_enricher, args=args, new_terminal=False, tail_logs=False)




















# class LiveOrchestrator2:
#     def __init__(self, ib, strategy_name, stop, target=None, timeframe=None, config=None, revised: bool=False, mode='live', timezone=None):
#         self.tmanager = trade_manager.TradeManager(ib, strategy_name, stop, target, revised, timeframe, config, timezone)
#         self.tz = timezone or CONSTANTS.TZ_WORK
#         self.mode = mode
#         self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
#         self.in_position = False
#         self.position_data = {}  # Track entry_time, price, etc.
#         self.trade_log = []
#         self.last_processed_minute = None
#         self.scanner_interval = timedelta(minutes=1)

#         # self.sim_speed = 60  # e.g. 60x faster → 1 simulated minute per real second
#         self.sim_start = self.tz.localize(datetime(2025, 9, 11, 9, 14, 0))
#         self.sim_max_time = timedelta(hours=4, minutes=0)
#         self.sim_offset = datetime.now(self.tz) - self.sim_start if self.mode == 'sim' else timedelta(0) if self.mode == 'live' else None

#         self.logger = live_data_logger.LiveDataLogger(mode, sim_offset=self.sim_offset, timezone=self.tz)
#         # self._create_paths()
#         # self.symbols_list = helpers.load_json(self.log_file_path)['symbols_list'] if os.path.exists(self.log_file_path) else {}
#         # self.last_scanner_check = pd.to_datetime(helpers.load_json(self.logger.log_file_path)['last_scanner_check']) if os.path.exists(self.logger.log_file_path) else None
    
#     def _assess_add_sr(self, symbol: str, trig_time: datetime) -> bool:
#         # Get the min refresh_rate from settings
#         min_refresh_str = min(CONSTANTS.SR_SETTINGS, key=lambda setting: helpers.get_ibkr_duration_as_timedelta(setting['refresh_rate']))['refresh_rate']
#         min_refresh_td = helpers.parse_timedelta(min_refresh_str)

#         last_trig = self.logger.symbols_list[symbol]['last_trig']
#         if not self.logger.symbols_list[symbol]['active']:
#             return True
#         if not last_trig:
#             return False # Case symbol not present yet or no last_trig yet

#         return not ((trig_time - last_trig) > min_refresh_td)

#     def _wait_for_new_minute(self, wait_time=1):
#         print("⏳ Waiting for new 1-min bar...")

#         now = helpers.calculate_now(self.sim_offset, self.tz)
#         current_minute = now.replace(second=0, microsecond=0)

#         if self.last_processed_minute is not None and current_minute > self.last_processed_minute:
#             # Already in a new minute → skip wait
#             print(f"⚡ New minute {current_minute.strftime('%H:%M')} already started. Skipping wait.")
#             return
        
#         current_second = now.second
#         remaining = 60 - current_second

#         for _ in tqdm.tqdm(range(remaining), desc="Time until next bar", ncols=70):
#             self.tmanager.ib.sleep(wait_time)

#         new_minute = datetime.now().replace(second=0, microsecond=0)
#         self.last_processed_minute = new_minute
#         print(f"✅ New minute started: {new_minute.strftime('%H:%M')}")

#     async def _process_symbol(self, symbol):
#         try:
#             # trig_time = datetime.now(self.tz)
#             trig_time = helpers.calculate_now(self.sim_offset, self.tz)

#             ####################################
#             # if not self.last_processed_minute: self.last_processed_minute = 16
#             # trig_time = trig_time.replace(day=9, hour=9, minute=self.last_processed_minute, second=0, microsecond=0) # Floor to minute
#             # self.last_processed_minute += 1
#             ####################################

#             block_add_sr = self._assess_add_sr(symbol, trig_time)
            
#             df = self.tmanager.load_data_live(symbol, trig_time=trig_time, look_backward='1 M', block_add_sr=block_add_sr)
#             # loop = asyncio.get_event_loop()
#             # df = await loop.run_in_executor(self.executor, self.tmanager.load_data_live, symbol, trig_time, None, '1M', 'parquet', block_add_sr)
            
#             if df.empty:
#                 return None

#             self.logger.update_symbols_list(symbol, 'last_trig', trig_time)
            
#             df = self.tmanager.apply_model_predictions(df)
#             # df = await loop.run_in_executor(self.executor, self.tmanager.apply_model_predictions, df)
            
#             latest_row = df.iloc[-1]

#             if self.tmanager.evaluate_entry_conditions(latest_row):
#                 return (symbol, latest_row)
            
#             # In case symbol doesn't meet trigger conditions, remove from active symbols
#             is_triggered = self.tmanager.strategy_instance.evaluate_trigger(latest_row)
#             # if not is_triggered:
#             #     self._update_symbols_list(symbol, 'active', False)
                

#             return None
#         except Exception as e:
#             print(f"⚠️ Error processing symbol {symbol}: {e}")
#             return None

#     async def _evaluate_all_symbols(self):
#         # loop = asyncio.get_event_loop()
#         # futures = [loop.run_in_executor(self.executor, self._process_symbol, symbol) for symbol in self.symbols_list]
#         # results = await asyncio.gather(*futures)

#         active_symbols_list = [s for s in self.logger.symbols_list if self.logger.symbols_list[s]['active']]
#         tasks = [self._process_symbol(symbol) for symbol in active_symbols_list]
#         results = await asyncio.gather(*tasks)

#         for result in results:
#             if result:
#                 symbol, latest_row = result
#                 print("\n\n================================================\n")
#                 print(f"🚀 Entry Signal: {symbol} at {latest_row['close']}")
#                 print(f"symbols_list[{symbol}] = {self.logger.symbols_list[symbol]}")
#                 print(f"Lastest row = {latest_row}")
#                 print("\n================================================\n\n")
#                 # TODO: Execute trade or log signal
#             else:
#                 print(f"No entry signal")

#         # futures = [self.executor.submit(self._process_symbol, symbol) for symbol in self.symbols_list]
#         # for future in futures:
#         #     result = future.result()
#         #     if result:
#         #         symbol, price = result
#         #         print(f"🚀 Entry Signal: {symbol} at {price}")
#         #         # TODO: Execute trade or log signal

#     async def _main_loop(self):

#         while True:
#             # self._wait_for_new_minute()
#             ib.sleep(10)

#             now = helpers.calculate_now(self.sim_offset, self.tz)
#             if self.mode == 'sim' and now > self.sim_start + self.sim_max_time:
#                 print("Simulation time elapsed.")
#                 break
#             print("\n=================================")
#             print("Current time: ", now)
#             print("Active Symbols:")
#             active_symbols_list = [s for s in self.logger.symbols_list if self.logger.symbols_list[s]['active']]
#             for symbol in active_symbols_list:
#                 print(symbol)
#             print("=================================\n")

#             if not self.logger.last_scanner_check or (now - self.logger.last_scanner_check > self.scanner_interval):
#                 print("📡 Updating symbols from scanner...")
#                 symbols_scanner = self.tmanager.get_scanner_data(now)

#                 for scanned_symbol in symbols_scanner:
#                     if scanned_symbol in self.logger.symbols_list:
#                         self.logger.update_symbols_list(scanned_symbol, 'active', True)
#                         # self.symbols_list[scanned_symbol]['active'] = True
#                     else:
#                         # self.symbols_list[scanned_symbol] = {'active': True, 'last_trig': None}
#                         self.logger.update_symbols_list(scanned_symbol, 'active', True)
#                         self.logger.update_symbols_list(scanned_symbol, 'last_trig', None)
                        
#                 self.logger.last_scanner_check = now
#                 self.logger.update_log()
#                 print(f"✅ {len(self.logger.symbols_list)} symbols loaded.")

#             await self._evaluate_all_symbols()

#     def start(self):
#         print(f"✅ LiveTradingOrchestrator started in {self.mode} mode...")
#         # Run async main loop inside ib.run()
#         self.tmanager.ib.run(self._main_loop())


# if __name2__ == "__main__":

#     args = sys.argv
#     pd.options.mode.chained_assignment = None # Disable Pandas warnings

#     paperTrading = 'live' not in args
#     revised = 'revised' in args
#     strategy_name = next((arg[9:] for arg in args if arg.startswith('strategy=')), '')
#     mode = next((arg[5:] for arg in args if arg.startswith('mode=')), 'live')

#     ib, _ = helpers.IBKRConnect_any(IB(), paper=paperTrading)

#     print("Loop:", util.getLoop())
#     print("Event loop running?", asyncio.get_event_loop().is_running())

#     stop = 'levels'

#     orchestrator = LiveOrchestrator(ib, strategy_name, stop=stop, revised=revised,  mode=mode)
#     orchestrator.start()


    # nest_asyncio.apply()

    # ib, _ = helpers.IBKRConnect_any(IB(), paper=True)
    # symbol = 'AAPL'
    # # util.startLoop()  # ✅ ensure event loop is created

    # print("Connected:", ib.isConnected())
    # print("Loop:", util.getLoop())
    # print("Event loop running?", asyncio.get_event_loop().is_running())
    # # print("Loop:", ib._loop)
    # # print("Running:", ib._loop.is_running())


    # # loop = asyncio.get_event_loop()
    # # loop.run_until_complete(test_loop(ib))  # ✅ This replaces ib.run(...)

    # ib.run(test_loop(ib))
    # print("Loop:", util.getLoop())

    # contract, symbol = helpers.get_symbol_contract(ib, symbol, currency="USD", exchange=None)

    # print(contract)
    # print()


    # def _get_terminal_command(self, target, args=()):
    #     """
    #     Builds a command to run fetcher_launcher.py in a new terminal with the specified function and arguments.
    #     """
    #     system = platform.system()
    #     command_str = f"python3 -u -c 'import {__name__}; {target.__name__}({', '.join(map(repr, args))})'"

    #     if system == "Linux":
    #         # Tries gnome-terminal, fallback to xterm
    #         terminal_cmds = [
    #             f"gnome-terminal -- bash -c \"{command_str}; exec bash\"",
    #             f"x-terminal-emulator -e bash -c \"{command_str}; exec bash\"",
    #             f"xterm -hold -e \"{command_str}\""
    #         ]
    #         return terminal_cmds[0]  # You can cycle through if needed

    #     elif system == "Darwin":
    #         # macOS - use osascript to tell Terminal to run the command
    #         apple_script = f'''
    #         tell application "Terminal"
    #             do script "{command_str}"
    #             activate
    #         end tell
    #         '''
    #         return f"osascript -e {shlex.quote(apple_script)}"

    #     elif system == "Windows":
    #         # Windows - use start cmd
    #         return f'start cmd /k "{command_str}"'

    #     else:
    #         return None
















     # p = multiprocessing.Process(target=target, args=args)
            # p.start()
            # self.processes.append(p)

            # # Tail logs in a new terminal
            # if tail_logs:
            #     # Look for the log path
            #     raw_log_path = next((arg for arg in args if isinstance(arg, str) and arg.endswith('.log')), None)
            #     if raw_log_path:
            #         base, ext = os.path.splitext(raw_log_path)
            #         glob_pattern = f"{base}_*{ext}"

            #         # Poll for the log file to appear
            #         max_wait_seconds = 1
            #         for _ in range(max_wait_seconds):
            #             matching_logs = sorted(glob.glob(glob_pattern), key=os.path.getmtime, reverse=True)
            #             if matching_logs:
            #                 newest_log = matching_logs[0]
            #                 if os.path.exists(newest_log) and os.path.getsize(newest_log) > 0:
            #                     break
            #             time.sleep(1) # wait a bit before retrying

            #         # Tail it if found
            #         if matching_logs:
            #             tail_command = helpers.get_tail_command(matching_logs[0])
            #             if tail_command:
            #                 subprocess.Popen(tail_command, shell=True)








# class LiveOrchestrator:
#     def __init__(self, ib, strategy_name, stop, target=None, timeframe=None, config=None, revised: bool=False, mode='live', timezone=None):
#         self.tmanager = trade_manager.TradeManager(ib, strategy_name, stop, target, revised, timeframe, config, timezone)
#         self.tz = timezone or CONSTANTS.TZ_WORK
#         self.mode = mode
#         self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
#         self.in_position = False
#         self.position_data = {}  # Track entry_time, price, etc.
#         self.trade_log = []
#         self.last_processed_minute = None
#         self.scanner_interval = timedelta(minutes=1)

#         # self.sim_speed = 60  # e.g. 60x faster → 1 simulated minute per real second
#         self.sim_start = self.tz.localize(datetime(2025, 9, 11, 9, 14, 0))
#         self.sim_max_time = timedelta(hours=4, minutes=0)
#         self.sim_offset = datetime.now(self.tz) - self.sim_start if self.mode == 'sim' else timedelta(0) if self.mode == 'live' else None

#         self.logger = live_data_logger.LiveDataLogger(mode, sim_offset=self.sim_offset, timezone=self.tz)
#         # self._create_paths()
#         # self.symbols_list = helpers.load_json(self.log_file_path)['symbols_list'] if os.path.exists(self.log_file_path) else {}
#         # self.last_scanner_check = pd.to_datetime(helpers.load_json(self.logger.log_file_path)['last_scanner_check']) if os.path.exists(self.logger.log_file_path) else None
    
#     def _assess_add_sr(self, symbol: str, trig_time: datetime) -> bool:
#         # Get the min refresh_rate from settings
#         min_refresh_str = min(CONSTANTS.SR_SETTINGS, key=lambda setting: helpers.get_ibkr_duration_as_timedelta(setting['refresh_rate']))['refresh_rate']
#         min_refresh_td = helpers.parse_timedelta(min_refresh_str)

#         last_trig = self.logger.symbols_list[symbol]['last_trig']
#         if not self.logger.symbols_list[symbol]['active']:
#             return True
#         if not last_trig:
#             return False # Case symbol not present yet or no last_trig yet

#         return not ((trig_time - last_trig) > min_refresh_td)

#     def _wait_for_new_minute(self, wait_time=1):
#         print("⏳ Waiting for new 1-min bar...")

#         now = helpers.calculate_now(self.sim_offset, self.mode, self.tz)
#         current_minute = now.replace(second=0, microsecond=0)

#         if self.last_processed_minute is not None and current_minute > self.last_processed_minute:
#             # Already in a new minute → skip wait
#             print(f"⚡ New minute {current_minute.strftime('%H:%M')} already started. Skipping wait.")
#             return
        
#         current_second = now.second
#         remaining = 60 - current_second

#         for _ in tqdm.tqdm(range(remaining), desc="Time until next bar", ncols=70):
#             self.tmanager.ib.sleep(wait_time)

#         new_minute = datetime.now().replace(second=0, microsecond=0)
#         self.last_processed_minute = new_minute
#         print(f"✅ New minute started: {new_minute.strftime('%H:%M')}")

#     async def _process_symbol(self, symbol):
#         try:
#             # trig_time = datetime.now(self.tz)
#             trig_time = helpers.calculate_now(self.sim_offset, self.mode, self.tz)

#             ####################################
#             # if not self.last_processed_minute: self.last_processed_minute = 16
#             # trig_time = trig_time.replace(day=9, hour=9, minute=self.last_processed_minute, second=0, microsecond=0) # Floor to minute
#             # self.last_processed_minute += 1
#             ####################################

#             block_add_sr = self._assess_add_sr(symbol, trig_time)
            
#             df = self.tmanager.load_data_live(symbol, trig_time=trig_time, look_backward='1 M', block_add_sr=block_add_sr)
#             # loop = asyncio.get_event_loop()
#             # df = await loop.run_in_executor(self.executor, self.tmanager.load_data_live, symbol, trig_time, None, '1M', 'parquet', block_add_sr)
            
#             if df.empty:
#                 return None

#             self.logger.update_symbols_list(symbol, 'last_trig', trig_time)
            
#             df = self.tmanager.apply_model_predictions(df)
#             # df = await loop.run_in_executor(self.executor, self.tmanager.apply_model_predictions, df)
            
#             latest_row = df.iloc[-1]

#             if self.tmanager.evaluate_entry_conditions(latest_row):
#                 return (symbol, latest_row)
            
#             # In case symbol doesn't meet trigger conditions, remove from active symbols
#             is_triggered = self.tmanager.strategy_instance.evaluate_trigger(latest_row)
#             # if not is_triggered:
#             #     self._update_symbols_list(symbol, 'active', False)
                

#             return None
#         except Exception as e:
#             print(f"⚠️ Error processing symbol {symbol}: {e}")
#             return None

#     async def _evaluate_all_symbols(self):
#         # loop = asyncio.get_event_loop()
#         # futures = [loop.run_in_executor(self.executor, self._process_symbol, symbol) for symbol in self.symbols_list]
#         # results = await asyncio.gather(*futures)

#         active_symbols_list = [s for s in self.logger.symbols_list if self.logger.symbols_list[s]['active']]
#         tasks = [self._process_symbol(symbol) for symbol in active_symbols_list]
#         results = await asyncio.gather(*tasks)

#         for result in results:
#             if result:
#                 symbol, latest_row = result
#                 print("\n\n================================================\n")
#                 print(f"🚀 Entry Signal: {symbol} at {latest_row['close']}")
#                 print(f"symbols_list[{symbol}] = {self.logger.symbols_list[symbol]}")
#                 print(f"Lastest row = {latest_row}")
#                 print("\n================================================\n\n")
#                 # TODO: Execute trade or log signal
#             else:
#                 print(f"No entry signal")

#         # futures = [self.executor.submit(self._process_symbol, symbol) for symbol in self.symbols_list]
#         # for future in futures:
#         #     result = future.result()
#         #     if result:
#         #         symbol, price = result
#         #         print(f"🚀 Entry Signal: {symbol} at {price}")
#         #         # TODO: Execute trade or log signal

#     async def _main_loop(self):

#         while True:
#             # self._wait_for_new_minute()
#             ib.sleep(10)

#             now = helpers.calculate_now(self.sim_offset, self.mode, self.tz)
#             if self.mode == 'sim' and now > self.sim_start + self.sim_max_time:
#                 print("Simulation time elapsed.")
#                 break
#             print("\n=================================")
#             print("Current time: ", now)
#             print("Active Symbols:")
#             active_symbols_list = [s for s in self.logger.symbols_list if self.logger.symbols_list[s]['active']]
#             for symbol in active_symbols_list:
#                 print(symbol)
#             print("=================================\n")

#             if not self.logger.last_scanner_check or (now - self.logger.last_scanner_check > self.scanner_interval):
#                 print("📡 Updating symbols from scanner...")
#                 symbols_scanner = self.tmanager.get_scanner_data(now)

#                 for scanned_symbol in symbols_scanner:
#                     if scanned_symbol in self.logger.symbols_list:
#                         self.logger.update_symbols_list(scanned_symbol, 'active', True)
#                         # self.symbols_list[scanned_symbol]['active'] = True
#                     else:
#                         # self.symbols_list[scanned_symbol] = {'active': True, 'last_trig': None}
#                         self.logger.update_symbols_list(scanned_symbol, 'active', True)
#                         self.logger.update_symbols_list(scanned_symbol, 'last_trig', None)
                        
#                 self.logger.last_scanner_check = now
#                 self.logger.update_log()
#                 print(f"✅ {len(self.logger.symbols_list)} symbols loaded.")

#             await self._evaluate_all_symbols()

#     def start(self):
#         print(f"✅ LiveTradingOrchestrator started in {self.mode} mode...")
#         # Run async main loop inside ib.run()
#         self.tmanager.ib.run(self._main_loop())




































# def _load_symbols_data(self):
    #     for symbol in self.symbols_list:
    #         df = trade_manager.TradeManager.load_data_live(symbol, trig_time=datetime.now(), look_backward='2 M', file_format='parquet')


    # def on_tick_update(self, tickers):
    #     for ticker in tickers:
    #         symbol = ticker.contract.symbol
    #         last_price = ticker.last

    #         # Run your strategy's entry condition
    #         if self.strategy.evaluate_live_entry(ticker):
    #             print(f"🚀 Entry signal for {symbol} at {last_price}")
    #             # → Execute trade / send alert here


    # def start(self):
    #     print("✅ LiveTradingOrchestrator started...")
    #     self.symbols_list = self._get_symbols_list()
    #     symbols_data = self._load_symbols_data()
    #     # self.load_tickers()
    #     self.ib.pendingTickersEvent += self.on_tick_update
    #     self.ib.run()

    #     last_scanner_check = None
    # scanner_interval = timedelta(minutes=5)

    # while True:
    #     wait_for_new_minute()

    #     # Check scanner update
    #     now = datetime.now()
    #     if not last_scanner_check or (now - last_scanner_check > scanner_interval):
    #         self.symbols_list = self._get_symbols_list()
    #         last_scanner_check = now
    #         print(f"📡 Scanner updated. {len(self.symbols_list)} symbols found.")

    #     # Evaluate all symbols in parallel
    #     self._evaluate_all_symbols()

    # def run(self):
    #     self.ib.reqMktData(self.contract, '', False, False)
    #     self.ib.pendingTickersEvent += self.on_tick
    #     print(f"🚀 Live trading started for: {self.contract.symbol}")
    #     self.ib.run()

    # def load_symbols(self):
    #     for _, row in self.ticker_df.iterrows():
    #         symbol = row['symbol']
    #         contract = Stock(symbol, 'SMART', 'USD')
    #         qualified = self.ib.qualifyContracts(contract)[0]
    #         ticker = self.ib.reqMktData(qualified, '', False, False)
    #         self.active_contracts[symbol] = qualified
    #         self.symbol_to_ticker[symbol] = ticker
    #         print(f"📡 Subscribed to {symbol}")






    # def on_tick(self, tickers):
    #     ticker = tickers[0]  # Assumes single ticker
    #     row = self.build_feature_row(ticker)
    #     row = self.apply_model(row)

    #     if not self.in_position:
    #         if self.should_enter(row):
    #             self.enter_position(row)
    #     else:
    #         if self.should_exit(row):
    #             self.exit_position(row)

    # def build_feature_row(self, ticker):
    #     # Build a feature row (like `build_row()` in backtrader wrapper)
    #     # Includes current price, indicators, etc.
    #     row = {
    #         'date': pd.Timestamp.utcnow(),
    #         'close': ticker.marketPrice(),
    #         # Add custom features or indicators here
    #     }

    #     # You may need a rolling buffer (e.g., deque) to store last N bars for indicators
    #     return row

    # def apply_model(self, row):
    #     row_df = pd.DataFrame([row])
    #     row_df = backtest_utils.BacktestManager.apply_model_predictions(row_df, self.model)
    #     return row_df.iloc[0]

    # def should_enter(self, row):
    #     is_triggered = self.strategy.evaluate_trigger(row)
    #     is_predicted = backtest_utils.BacktestManager.evaluate_prediction(row['model_prediction'], self.config.prediction_threshold)
    #     is_RRR = backtest_utils.BacktestManager.evaluate_RRR(row, self.target_handler, self.config.rrr_threshold)
    #     return is_triggered and is_predicted and is_RRR

    # def enter_position(self, row):
    #     price = row['close']
    #     quantity = self.determine_quantity(row['model_prediction'])
    #     order = MarketOrder('BUY', quantity)
    #     trade = self.ib.placeOrder(self.contract, order)

    #     self.in_position = True
    #     self.position_data = {
    #         'entry_time': row['date'],
    #         'entry_price': price,
    #         'quantity': quantity,
    #         'model_prediction': row['model_prediction'],
    #         'active_stop_price': self.resolve_stop_price(row),
    #         'order': trade
    #     }
    #     print(f"✅ Entered position: {quantity} @ {price:.2f}")

    # def should_exit(self, row):
    #     pd_entry = self.position_data
    #     reason = backtest_utils.BacktestManager.assess_reason2close(row, row, self.target_handler,
    #                                                  self.stop_handler,
    #                                                  pd_entry['active_stop_price'],
    #                                                  direction=1)  # hardcoded to long for now
    #     return reason

    # def exit_position(self, row):
    #     quantity = self.position_data['quantity']
    #     order = MarketOrder('SELL', quantity)
    #     trade = self.ib.placeOrder(self.contract, order)

    #     self.log_trade(row)
    #     self.in_position = False
    #     self.position_data = {}
    #     print(f"❌ Exited position")

    # def determine_quantity(self, prediction):
    #     return backtest_utils.BacktestManager.evaluate_quantity(prediction, self.config.size, 
    #                                              self.config.prediction_threshold,
    #                                              self.config.tier_max)

    # def resolve_stop_price(self, row):
    #     return backtest_utils.BacktestManager.resolve_stop_price(row, None, self.stop_handler, direction=1)

    # def log_trade(self, exit_row):
    #     entry = self.position_data
    #     trade = {
    #         'entry_time': entry['entry_time'],
    #         'exit_time': exit_row['date'],
    #         'entry_price': entry['entry_price'],
    #         'exit_price': exit_row['close'],
    #         'quantity': entry['quantity'],
    #         'model_prediction': entry['model_prediction']
    #     }
    #     self.trade_log.append(trade)
