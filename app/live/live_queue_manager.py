import os, sys, pandas as pd, datetime, subprocess, multiprocessing#, tqdm, asyncio, concurrent.futures
from ib_insync import *

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
import live_loop_base


class LiveQueueManager(live_loop_base.LiveLoopBase):
    def __init__(self, wait_seconds:int=None, continuous:bool=True, single_symbol=None, tickers_list:dict={}, config=None, live_mode:str=None,
                 ib_client_id:int=None, seed:int=None, paper_trading:bool=None, remote_ib:bool=None, timezone=None):

        super().__init__(wait_seconds=wait_seconds, continuous=continuous, single_symbol=single_symbol, ib_disconnect=False,
                         live_mode=live_mode, ib_client_id=ib_client_id, config=config, seed=seed, paper_trading=paper_trading, remote_ib=remote_ib, timezone=timezone)
        self.symbols_seed = helpers.get_symbol_seed_list(self.config.seed)
        self.scan_rate = self.tmanager.strategy_instance.timeframe.to_seconds
        self.tickers_list = tickers_list or self.logger.load_tickers_list(lock=True)
        self.fetch_queue = self.logger.get_queue('fetch', lock=True)
        self.enrich_queue = self.logger.get_queue('enrich', lock=True)
        self.execite_queue = self.logger.get_queue('execut', lock=True)

    def _organize_tickers_list(self, symbols_scanner:list):
        # Add new tickers
        for symbol in self.symbols_seed + symbols_scanner:
            self.tickers_list = self.logger.load_tickers_list(lock=True)
            if symbol not in self.tickers_list:
                self.tickers_list[symbol] = self.logger.initialize_ticker()
                self.logger.save_tickers_list(self.tickers_list, lock=True)
            else:
                if not self.tickers_list[symbol]['active']:
                    self.tickers_list = self.logger.update_ticker(symbol, 'active', True, lock=True, log=True)

        if self.live_mode:
            # Deactivate tickers not in scanner results anymore
            self.tickers_list = self.logger.load_tickers_list(lock=True)
            active_symbols = [symbol for symbol in self.tickers_list if self.tickers_list[symbol]['active']]
            for symbol in active_symbols:
                if symbol not in symbols_scanner and symbol not in self.symbols_seed:
                    self.tickers_list = self.logger.update_ticker(symbol, 'active', False, lock=True, log=True)

    def _manage_queues(self):
        now = helpers.calculate_now(self.config.sim_offset, self.tmanager.tz)
        print(f"\n⏱️ Current Time: {now}")
        symbols_scanner = self.tmanager.get_scanner_data(now, use_daily_data=self.live_mode=='sim')
        print(f"📡 Fetched symbols from scanner:\n{symbols_scanner}")

        # symbols_scanner = ['CPB']#, 'SHFS', 'UUUU']

        self._organize_tickers_list(symbols_scanner)

        self.tickers_list = self.logger.load_tickers_list(lock=True)
        for symbol, info in list(self.tickers_list.items()):
            if not info['active']:
                continue

            condition_handling = not (self.tickers_list[symbol]['fetching'] or self.tickers_list[symbol]['enriching'])
            # --- Initialization Phase ---
            if not self.tickers_list[symbol]['initialized']:
                if condition_handling and not self.tickers_list[symbol]['last_fetched']:
                    self.logger.put_queue(symbol, 'fetch', lock=True)
                    continue
                if self.tickers_list[symbol]['last_fetched'] and condition_handling and not self.tickers_list[symbol]['last_enriched']:
                    self.logger.put_queue(symbol, 'enrich', lock=True)
                    continue

                if self.tickers_list[symbol]['last_fetched'] and self.tickers_list[symbol]['last_enriched']:
                    self.logger.update_ticker(symbol, 'initialized', True)
                    self.tickers_list = self.logger.update_ticker(symbol, 'priority', 1, lock=True, log=True)
                    # self._update_ticker(symbol, 'status', 'ready')
                    # self._update_ticker(symbol, 'last_updated', now)
                    continue

            # --- Recurrent Phase ---
            if self.tickers_list[symbol]['initialized']:
                # time_since_update = (now - self.tickers_list[symbol]['last_updated']).total_seconds() if self.tickers_list[symbol]['last_updated'] else float('inf')
                time_since_fetched = (now - pd.to_datetime(self.tickers_list[symbol]['last_fetched'])).total_seconds() if self.tickers_list[symbol]['last_fetched'] else float('inf')
                time_since_enriched = (now - pd.to_datetime(self.tickers_list[symbol]['last_enriched'])).total_seconds() if self.tickers_list[symbol]['last_enriched'] else float('inf')
                time_since_executed = (now - pd.to_datetime(self.tickers_list[symbol]['last_executed'])).total_seconds() if self.tickers_list[symbol]['last_executed'] else float('inf')

                # p = self.tickers_list[symbol].get("priority", 2) if symbol in self.tickers_list else 2
                # if p > 0:
                #     print()

                if condition_handling and time_since_fetched > self.scan_rate:
                    self.logger.put_queue(symbol, 'fetch', lock=True)
                    continue

                if condition_handling and time_since_enriched > self.scan_rate and time_since_fetched < self.scan_rate:
                    self.logger.put_queue(symbol, 'enrich', lock=True)
                    continue

                if condition_handling and time_since_executed > self.scan_rate and time_since_enriched < self.scan_rate and time_since_fetched < self.scan_rate:
                    self.logger.put_queue(symbol, 'execut', lock=True)
                    continue

    def _execute_main_task(self):
        print("🧠 Starting queue manager...")
        self._manage_queues()

if __name__ == "__main__":

    args = sys.argv
    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    paper_trading = not 'live' in args
    remote_ib = 'remote' in args
    # revised = 'revised' in args
    # seed = next((int(arg[5:]) for arg in args if arg.startswith('seed=')), None)
    # strategy_name = next((arg[9:] for arg in args if arg.startswith('strategy=')), None)
    wait_seconds = next((int(float(arg[5:])) for arg in args if arg.startswith('wait=')), 20)
    mode = next((arg[5:] for arg in args if arg.startswith('mode=')), 'live')

    qmanager = LiveQueueManager(wait_seconds=wait_seconds, live_mode=mode, paper_trading=paper_trading, remote_ib=remote_ib)
    qmanager.run()