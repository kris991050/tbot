import os, sys, csv
from ib_insync import *

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
import trading_config


class LiveDataLogger:
    def __init__(self, worker_type:str=None, config=None, strategy_name:str=None, live_mode:str=None, sim_offset=None, timezone=None):
        self.worker_type = worker_type
        self.strategy_name = strategy_name
        self.config = config or trading_config.TradingConfig().set_config(locals())
        # self.mode = mode
        # self.sim_offset = sim_offset if self.config.mode == 'sim' else datetime.timedelta(0) if self.config.mode == 'live' else None
        # self.tz = timezone or constants.CONSTANTS.TZ_WORK
        self._create_paths()
        # self.tickers_list = tickers_list or helpers.load_json(self.tickers_log_file_path)['tickers_list'] if os.path.exists(self.tickers_log_file_path) else {}
        # self.last_scanner_check = pd.to_datetime(helpers.load_json(self.tickers_log_file_path)['last_scanner_check']) if os.path.exists(self.tickers_log_file_path) else None
    
    def _create_paths(self):
        date_now = helpers.calculate_now(self.config.sim_offset, self.config.timezone)
        sim_str = '_sim' if self.config.live_mode == 'sim' else '' if self.config.live_mode == 'live' else None
        config_file_name = f"config_live_{date_now.strftime("%Y%m%d")}{sim_str}.json"
        tickers_file_name = f"tickers_live_{date_now.strftime("%Y%m%d")}{sim_str}.json"
        fetch_queue_file_name = f"fetch_queue_live_{date_now.strftime("%Y%m%d")}{sim_str}.json"
        enrich_queue_file_name = f"enrich_queue_live_{date_now.strftime("%Y%m%d")}{sim_str}.json"
        execute_queue_file_name = f"execute_queue_live_{date_now.strftime("%Y%m%d")}{sim_str}.json"
        date_folder = helpers.get_path_date_folder(date_now, create_if_none=True)
        logs_folder = helpers.get_path_daily_logs_folder(date_now, create_if_none=True)
        strategy_folder = helpers.get_path_daily_strategy_folder(self.config.strategy_name, date_now, create_if_none=self.strategy_name is not None)
        self.config_file_path = os.path.join(strategy_folder, config_file_name)
        self.tickers_log_file_path = os.path.join(strategy_folder, tickers_file_name)
        self.queue_paths = {
            'fetch': os.path.join(strategy_folder, fetch_queue_file_name), 
            'enrich': os.path.join(strategy_folder, enrich_queue_file_name), 
            'execut': os.path.join(strategy_folder, execute_queue_file_name)
        }
        trade_log_file_name = f"trade_log_{date_now.strftime("%Y-%m-%d")}{sim_str}.csv"
        self.trade_log_file_path = os.path.join(date_folder, trade_log_file_name)
        live_log_file_name = f"live_log_{self.worker_type}_{date_now.strftime("%Y-%m-%d_%H-%M-%S")}{sim_str}.txt" if self.worker_type else None
        self.live_log_file_path = os.path.join(logs_folder, live_log_file_name) if live_log_file_name else None
        self.default_priority = 2

    def initialize_ticker(self):
        return {
            'active': True,
            'initialized': False,
            'fetching': False,
            'enriching': False,
            'executing': False,
            'trading': False,
            'last_fetched': None,
            'last_enriched': None,
            'last_executed': None,
            'priority': self.default_priority # 0: discarded, 1: initialized, 2: not initialized (default), 3: triggered, 4: executed
            # 'status': 'initializing',
            # 'last_updated': None
        }

    @staticmethod
    def _write_to_json_file(data, file_path, lock=True):
        """ Write data to JSON file with optional file locking """
        helpers.save_json(data, file_path, lock=lock)

    @staticmethod
    def _read_from_json_file(file_path, default_return=[], lock=True):
        """ Read data from JSON file with optional file locking """
        if os.path.exists(file_path):
            return helpers.load_json(file_path, lock=lock)
        else:
            return default_return
    
    # Tickers Methods
    def load_tickers_list(self, lock=True):
        """
        Load the tickers list from the JSON file with an optional lock.
        """
        return self._read_from_json_file(self.tickers_log_file_path, {}, lock)

    def save_tickers_list(self, tickers_list, lock=True):
        """
        Save the tickers list to the JSON file with an optional lock.
        """
        self._write_to_json_file(tickers_list, self.tickers_log_file_path, lock)

    def update_ticker(self, symbol, key, value, lock=True, log=True):
        tickers_list = self.load_tickers_list(lock)
        ticker = tickers_list.get(symbol, self.initialize_ticker())
        ticker[key] = value
        tickers_list[symbol] = ticker
        if log:
            self.save_tickers_list(tickers_list, lock)
            print(f"Set {symbol} {key} status to {value}")
        return tickers_list

    # Queue Methods
    def save_queue(self, queue, action, lock=True):
        queue_path = self.queue_paths[action] if action in self.queue_paths else None
        self._write_to_json_file(queue, queue_path, lock=lock)

    def get_queue(self, action, lock=True):
        queue_path = self.queue_paths[action] if action in self.queue_paths else None
        return self._read_from_json_file(queue_path, lock=lock)

    def put_queue(self, item, action, lock=True):
        queue = self.get_queue(action, lock=lock)
        tickers_list = self.load_tickers_list(lock=lock)
        item_priority = tickers_list[item].get("priority", self.default_priority) if item in tickers_list else self.default_priority
        
        if item not in queue:
            # Sort the queue based on priority, maintaining the items with the same priority in the order they were added
            queue_sorted = sorted(queue, key=lambda x: tickers_list[x].get('priority', self.default_priority), reverse=True)

            # Insert the item at the last position of its priority
            insert_position = len(queue_sorted)  # Default to last if no better position is found
            for idx, queued_item in enumerate(queue_sorted):
                if tickers_list[queued_item].get('priority', self.default_priority) < item_priority:
                    insert_position = idx
                    break
            
            # Insert the new item at the appropriate position
            queue_sorted.insert(insert_position, item)
            self.save_queue(queue_sorted, action, lock=lock)
            print(f"{item} placed in {action} queue at position {insert_position} with priority {item_priority}")
        else:
            print(f"{item} already in {action} queue")
    
    def pop_queue(self, queue, action, lock=True):
        try:
            # Pop the first element from the queue
            symbol = queue.pop(0)
            print(f"Ticker {symbol} popped from the queue.")
            
            # Save the remaining queue
            self.save_queue(queue, action, lock=lock)
            # if action == 'fetch':
            #     self.save_fetch_queue(queue, lock=lock)
            # elif action == 'enrich':
            #     self.save_enrich_queue(queue, lock=lock)
            # else:
            #     raise ValueError(f"Handler type must be 'fetch' or 'enrich'. Currently {self.handler_type}")
    
            return symbol  # Optionally return the popped symbol if needed

        except IndexError as e:
            print(f"⚠️ Error popping from the queue: {e}")
            return None
            
    def save_to_trade_log_csv(self, ib, time_order, symbol, row, quantity, target_price, stop_price, order_status, order_avg_fill, required_columns):
        required_columns_trimmed = [col for col in required_columns if col in row.index]

        helpers.initialize_log_csv_file(self.trade_log_file_path, ['Entry Time', 'Symbol', 'Close', 'Quantity', 'Target', 'Stop', 'Prediction', \
                                                                   'Order Status', 'Order Avg Fill', *required_columns_trimmed, 'Avg Volume', 'Rel Volume', 'Floats', \
                                                                    'Market Cap', 'Index', 'Last Earning', 'News'])

        with open(self.trade_log_file_path, mode='a') as file:
            infos = helpers.get_symbol_infos(ib, symbol)
            file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow([time_order, symbol, row['close'], quantity, target_price, stop_price, row['model_prediction'], \
                                  order_status, order_avg_fill, *row[required_columns_trimmed].to_list(), infos['avg_volume'], \
                                    infos['rel_volume'], infos['floats'], infos['market_cap'], infos['index'], \
                                        infos['last_earning_date'], infos['news']])
        print(f"💾 Logged trade with symbol {symbol}, quantity {quantity}, close price {row['close']}, target price {target_price}, stop price {stop_price}, prediction {row['model_prediction']} at {self.trade_log_file_path}")

        file.close()





# def save_fetch_queue(self, queue, lock=True):
    #     self._save_queue(queue, self.fetch_queue_log_file_path, lock=lock)

    # def save_enrich_queue(self, queue, lock=True):
    #     self._save_queue(queue, self.enrich_queue_log_file_path, lock=lock)
# def get_fetch_queue(self, lock=True):
    #     return self._get_queue(self.fetch_queue_log_file_path, lock=lock)
    
    # def get_enrich_queue(self, lock=True):
    #     return self._get_queue(self.enrich_queue_log_file_path, lock=lock)

# def put_fetch_queue(self, item, lock=True):
    #     self._put_queue(item, 'fetch', self.fetch_queue_log_file_path, lock=lock)

    # def put_enrich_queue(self, item, lock=True):
    #     self._put_queue(item, 'enrich', self.enrich_queue_log_file_path, lock=lock)

# class Ticker:
#     def __init__(self, symbol: str):
#         self.symbol = symbol
#         # self.last_updated = ''
#         # self.last_triggered = ''
#         self._initialize()
    
#     def _initialize(self):
#         self.status = {
#             'active': False, 
#             'fetching': False, 
#             'enriching': False, 
#             'initialized': False
#         }

#     def update_status(self, key: str, value: bool):
#         if not key in self.status.keys():
#             print(f"Failed to update status for {self.symbol}. Key must be one of {self.status.keys()}")
#         else:
#             self.status[key] = value
#         return self.status
    
    # def set_active(self):
    #     if self.live_active:
    #         print(f"Symbol {self.symbol} already active.")
    #     else:
    #         self.live_active = True
    #         print(f"Symbol {self.symbol} set to active.")

    # def set_inactive(self):
    #     if not self.live_active:
    #         print(f"Symbol {self.symbol} already inactive.")
    #     else:
    #         self.live_active = False
    #         print(f"Symbol {self.symbol} set to inactive.")








# def _set_ticker_status(self, symbol, tickers_list):
    #     ticker = tickers_list[symbol]

    #     # Determine if it should be marked 'ready'
    #     if ticker['initialized']:
    #         now = helpers.calculate_now(self.sim_offset, self.mode, self.tz)
    #         last_updated = ticker.get('last_updated')

    #         if last_updated and (now - last_updated).total_seconds() < 70:
    #             ticker['status'] = 'ready'
    #         else:
    #             ticker['status'] = 'stale'

    #     tickers_list[symbol] = ticker
    #     return tickers_list




    # def _log_tickers_list(self, tickers_list, verbose=False):
    #     helpers.save_json(tickers_list, self.tickers_log_file_path)
    #     if verbose: print(f"💾 Updated Tickers List at {self.tickers_log_file_path}")

    # def _log_fetch_queue(self, queue, verbose=False):
    #     queue_items = self._dump_queue(queue)
    #     helpers.save_json(queue_items, self.fetch_queue_log_file_path)
    #     if verbose:
    #         print(f"💾 Updated Fetch Queue at {self.fetch_queue_log_file_path}")

    # def _log_enrich_queue(self, queue, verbose=False):
    #     queue_items = self._dump_queue(queue)
    #     helpers.save_json(queue_items, self.enrich_queue_log_file_path)         
    #     if verbose: print(f"💾 Updated Enrich Queue at {self.enrich_queue_log_file_path}")

    # def update_log(self, verbose=False):
    #     log_json = {
    #         'last_scanner_check': self.last_scanner_check, 
    #         'tickers_list': self.tickers_list
    #         }
    #     helpers.save_json(log_json, self.tickers_log_file_path)         
    #     if verbose: print(f"💾 Updated Symbols_List at {self.tickers_log_file_path}")

    # def update_tickers_list(self, symbol, key, value):
    #     if symbol not in self.tickers_list:
    #         self.tickers_list[symbol] = {key: value}
    #     self.tickers_list[symbol][key] = value
    #     self.update_log()
        
  






# def put_fetch_queue(self, queue, item):
#         queue.put(item)
#         self._log_fetch_queue(queue)

#     def get_fetch_queue(self, queue, timeout=2):
#         try:
#             item = queue.get(timeout=timeout)
#             self._log_fetch_queue(queue)
#             return item
#         except Exception as e:
#             print(f"⚠️ Failed to get from fetch queue: {e}")
#             raise
    
#     def put_enrich_queue(self, queue, item):
#         queue.put(item)
#         self._log_enrich_queue(queue)
    
#     def get_enrich_queue(self, queue, timeout=2):
#         try:
#             item = queue.get(timeout=timeout)
#             self._log_enrich_queue(queue)
#             return item
#         except Exception as e:
#             print(f"⚠️ Failed to get from enrich queue: {e}")
#             raise

# @staticmethod
#     def _dump_queue(queue):
#         items = []
#         try:
#             while not queue.empty():
#                 items.append(queue.get_nowait())
#             # Put items back in the queue
#             for item in items:
#                 queue.put(item)
#         except Exception as e:
#             print(f"⚠️ Error while dumping queue: {e}")
#         return items