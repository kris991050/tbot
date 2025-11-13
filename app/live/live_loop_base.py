import sys, os
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers, constants
import trading_config, live_data_logger


class LiveLoopBase:
    def __init__(self, wait_seconds=None, continuous=True, single_symbol=None, ib_disconnect=False, live_mode:str='live', config=None, paper_trading=None, timezone=None):
        self.live_mode = helpers.set_var_with_constraints(live_mode, constants.CONSTANTS.MODES['live'])
        self.config = self._resolve_config(config)
        self.wait_seconds = wait_seconds if wait_seconds else None
        self.continuous = continuous
        self.ib = IB()
        self._connect_ib()
        self.single_symbol = [single_symbol] if single_symbol else None
        self.ib_disconnect = ib_disconnect
        self.daily_data_folder = helpers.get_path_daily_data_folder()
        # self.paper_trading = paper_trading
        # self.tz = timezone or constants.CONSTANTS.TZ_WORK

    def _resolve_config(self, config):
        if isinstance(config, trading_config.TradingConfig):
            return config
        
        config_file_path = live_data_logger.LiveDataLogger(live_mode=self.live_mode).config_file_path
        if os.path.exists(config_file_path):
            return trading_config.TradingConfig(live_mode=self.live_mode).load_config(config_file_path)
            
        return trading_config.TradingConfig(live_mode=self.live_mode).set_config(locals())
    
    def _connect_ib(self):
        print("\n🔌 Connecting IB")
        self.ib, _ = helpers.IBKRConnect_any(self.ib, paper=self.config.paper_trading)

    def _disconnect_ib(self):
        if self.ib:
            print("📴 Disconnecting IB")
            self.ib.disconnect()

    def _sleep_wait(self):
        if sys.stdout.isatty():
            helpers.sleep_display(self.wait_seconds, self.ib)
        else:
            print("⌛ Waiting ", self.wait_seconds, " sec...")
            self.ib.sleep(self.wait_seconds)

    def _execute_main_task(self, action=None):
        """
        This should be overridden by each child class.
        """
        raise NotImplementedError("Subclasses must implement _execute_main_task()")
    
    def run(self):

        counter = 10000
        while counter > 0:
            self._connect_ib()

            self._execute_main_task()  # Call the subclass-specific method

            if self.continuous:
                counter -= 1
                if self.ib_disconnect: self._disconnect_ib()
                self._sleep_wait()
            else:
                counter = 0
            print(f"\n⌛ Countdown completed")
        

# class LiveLoopBaseData(LiveLoopBase):
#     def __init__(self, strategy_name:str, stop:str, revised:bool=False, look_backward:str='1 M', step_duration:str='1 W', mode:str='live', sim_offset_seconds:int=0, 
#                  file_format: str='parquet', tickers_list:dict={}, initialize:bool=True, paper_trading=True, wait_seconds=None, continuous=True, 
#                  single_symbol=None, ib_disconnect=False, timezone=None):
            
#             super().__init__(paper_trading, wait_seconds, continuous, single_symbol, ib_disconnect, timezone)
#             self.manager = trade_manager.TradeManager(self.ib, strategy_name, stop, revised=revised, look_backward=look_backward, 
#                                                       step_duration=step_duration, timezone=timezone)
#             # self.manager = self._build_manager(args_manager, self.ib, timezone)
#             # self.logger = self._build_logger(args_logger, timezone)
#             self.mode = helpers.set_var_with_constraints(mode, constants.CONSTANTS.MODES['live'])
#             self.logger = live_data_logger.LiveDataLogger(self.mode, datetime.timedelta(sim_offset_seconds), timezone)
#             self.file_format = file_format
#             self.initialize = initialize
#             # self.queue = queue or multiprocessing.Queue()
#             self.tickers_list = tickers_list or self.logger.load_tickers_list(lock=True)
    
    # @staticmethod
    # def _build_manager(args_manager, ib, timezone):
    #     # Deserialize the arguments if they are JSON strings
    #     if isinstance(args_manager, str):
    #         args_manager = json.loads(args_manager)  # Deserialize JSON to Python dict
        
    #     args_manager = args_manager or {'strategy_name': '', 'stop': '', 'target': None, 'revised': False, 'timeframe': None, 'look_backward': '1 M', 
    #                         'step_duration': '1 W', 'config': None}
    #     manager = trade_manager.TradeManager(ib, args_manager['strategy_name'], args_manager['stop'], args_manager['target'], 
    #                                                   args_manager['revised'], args_manager['timeframe'], args_manager['look_backward'], 
    #                                                   args_manager['step_duration'], timezone)

    #     return manager
    
    # @staticmethod
    # def _build_logger(args_logger, timezone):
    #     # Deserialize the arguments if they are JSON strings
    #     if isinstance(args_logger, str):
    #         args_logger = json.loads(args_logger)  # Deserialize JSON to Python dict
    #         args_logger['sim_offset'] = datetime.timedelta(seconds=args_logger['sim_offset'])

    #     args_logger = args_logger or {'mode':'live', 'sim_offset': datetime.timedelta(0)}
    #     logger = live_data_logger.LiveDataLogger(args_logger['mode'], args_logger['sim_offset'], timezone)
        
    #     return logger


    # def _execute_main_task_data(self, action):
    #     """
    #     This should be overridden by each child class.
    #     """
    #     raise NotImplementedError("Subclasses must implement _execute_main_task()")
    
    # def _execute_main_task(self, action=None):
    #     df = pd.DataFrame()

    #     symbol = self.queue.get(timeout=2) if self.queue and not self.queue.Empty else None
    #     if not symbol:
    #         return df
            
    #     info = self.tickers_list[symbol]
    #     init = self.initialize and not info['initialized']
    #     trig_time = helpers.calculate_now(self.logger.sim_offset, self.logger.mode, self.manager.tz)
    #     trig_time_min = trig_time.replace(second=0, microsecond=0) # Floor to minute
    #     end_time = helpers.substract_duration_from_time(trig_time_min, self.manager.look_backward)

    #     self._execute_main_task_data()
    #     if not info[f'{action}ing']:
    #         self.tickers_list = self.logger._update_ticker(symbol, f'{action}ing', True)
            
    #         if init:
    #             df = self.manager.fetch_df(symbol, trig_time, end_time, self.file_format, self.manager.step_duration)
    #         else:
    #             df = self.manager.complete_df(symbol, self.file_format, self.manager.step_duration)
        
    #         self.tickers_list = self.logger._update_ticker(symbol, f'{action}ing', False)
    #         self.tickers_list = self.logger._update_ticker(symbol, f'last_{action}ed', df['date'].iloc(-1))

    #     return df

# if __name__ == "__main__":

#     fetcher_type_list = ['scans', 'L2']

#     args = sys.argv
#     paper_trading = not 'live' in args
#     continuous = 'cont' in args
#     single_symbol = next(([arg[7:]] for arg in args if arg.startswith('symbol=')), None)
#     wait_minutes = next((int(float(arg[5:])) * 60 for arg in args if arg.startswith('wait=')), 5 * 60)
#     fetcher_type = next((arg[5:] for arg in args if arg.startswith('type=') and arg[5:] in fetcher_type_list), None)

#     if fetcher_type == 'L2':
#         fetcher = daily_L2_data_fetcher.DailyL2DataFetcher(paper_trading, wait_minutes, continuous, single_symbol, ib_disconnect=True)
#         fetcher.run()
#     elif fetcher_type == 'scans':
#         fetcher = daily_scans_fetcher.DailyScansFetcher(paper_trading, wait_minutes, continuous)
#         fetcher.run()
#     else:
#         print(f"Select fetcher_type from {fetcher_type_list}")