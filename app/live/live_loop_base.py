import sys, os
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers, logs
from utils.constants import CONSTANTS
from execution import trade_manager
import trading_config, live_data_logger


class LiveLoopBase:
    def __init__(self, worker_type:str=None, wait_seconds:int=None, continuous:bool=True, single_symbol:str=None, ib_disconnect:bool=False,
                 live_mode:str='live', ib_client_id:int=None, config=None, seed:int=None, paper_trading:bool=None, remote_ib:bool=None, timezone=None):
        self.worker_type = worker_type
        self.live_mode = helpers.set_var_with_constraints(live_mode, CONSTANTS.MODES['live'])
        self.ib_client_id = ib_client_id
        self.config = self._resolve_config(config, locals())

        # print("========================================")
        # print(f"remote_ib = {remote_ib}")
        # print(f"paper_trading = {paper_trading}")
        # print(f"self.config.remote_ib = {self.config.remote_ib}")
        # print(f"self.config.paper_trading = {self.config.paper_trading}")
        # print("========================================")
        IB.sleep(5)

        self.wait_seconds = wait_seconds if wait_seconds else None
        self.continuous = continuous
        self.ib = IB()
        self._connect_ib()
        self.single_symbol = [single_symbol] if single_symbol else None
        self.ib_disconnect = ib_disconnect
        self.daily_data_folder = helpers.get_path_daily_data_folder()
        self.tmanager = trade_manager.TradeManager(self.ib, config=self.config)
        self.logger = live_data_logger.LiveDataLogger(worker_type=self.worker_type, config=self.config)
        self.original_stdout = sys.stdout  # Save the original stdout reference, before redirecting stdout to logging class LogContext

    def _resolve_config(self, config, locals_main):
        if isinstance(config, trading_config.TradingConfig):
            return config

        config_file_path = live_data_logger.LiveDataLogger(live_mode=self.live_mode).config_file_path
        if os.path.exists(config_file_path):
            return trading_config.TradingConfig(live_mode=self.live_mode).load_config(config_file_path)

        return trading_config.TradingConfig(live_mode=self.live_mode).set_config(locals_main)

    def _connect_ib(self):
        if self.ib_client_id >= 0:
            print("\n🔌 Connecting IB")
            self.ib, _ = helpers.IBKRConnect_any(self.ib, paper=self.config.paper_trading, client_id=self.ib_client_id, remote=self.config.remote_ib)

    def _disconnect_ib(self):
        if self.ib and self.ib_disconnect:
            print("📴 Disconnecting IB")
            self.ib.disconnect()

    def _sleep_wait(self):
        if hasattr(self.original_stdout, 'isatty') and self.original_stdout.isatty():
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

        # current_time = helpers.calculate_now(sim_offset=self.config.sim_offset, tz=self.config.timezone)
        # start_time = pd.Timestamp.combine(current_time.date(), CONSTANTS.TH_TIMES['pre-market']).tz_localize(CONSTANTS.TZ_WORK)
        # end_time = pd.Timestamp.combine(current_time.date(), CONSTANTS.TH_TIMES['end_of_day']).tz_localize(CONSTANTS.TZ_WORK)

        # Create live log file path
        with logs.LogContext(self.logger.live_log_file_path, overwrite=True): # Logging starts here

            # while start_time <= current_time < end_time:
            now = helpers.calculate_now(sim_offset=self.config.sim_offset, tz=self.config.timezone)
            print(f"🕰️ Time now: {now}    |   Sim offset: {self.config.sim_offset}")
            while True:#helpers.is_between_market_times('pre-market', 'end_of_day', now=now, timezone=self.config.timezone):

                self._connect_ib()

                self._execute_main_task() # Call the subclass-specific method

                if self.continuous:
                    self._disconnect_ib()
                    self._sleep_wait()
                else:
                    break
                print(f"\n⌛ Countdown completed")
