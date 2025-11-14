import sys, os
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from miscellaneous import level2 
import live_loop_base
from utils import helpers, constants


class LiveL2Fetcher(live_loop_base.LiveLoopBase):

    def _get_symbols(self):
        # Same as original get_symbols()
        folder = self.daily_data_folder
        paths = constants.PATHS.daily_csv_files

        symbols_gapper_up = [row[1] for i, row in enumerate(helpers.read_csv_file(os.path.join(folder, paths['gapper_up']))) if i > 0 and row and row[0] != '']
        symbols_gapper_down = [row[1] for i, row in enumerate(helpers.read_csv_file(os.path.join(folder, paths['gapper_down']))) if i > 0 and row and row[0] != '']
        symbols_earnings = [row[1] for i, row in enumerate(helpers.read_csv_file(os.path.join(folder, paths['earnings']))) if i > 0 and row and row[0] != '']
        symbols_rsi_reversal = [row[1] for i, row in enumerate(helpers.read_csv_file(os.path.join(folder, paths['bb_rsi_reversal']))) if i > 0 and row and row[0] != '']

        symbols_main = [row[0] for row in helpers.read_csv_file(constants.PATHS.csv_files['main']) if row and row[0] != '']
        symbols_index = [row[0] for row in helpers.read_csv_file(constants.PATHS.csv_files['index']) if row and row[0] != '']

        symbols = symbols_gapper_up + symbols_gapper_down + symbols_earnings + symbols_rsi_reversal + symbols_main + symbols_index

        unique_symbols = []
        [unique_symbols.append(s) for s in symbols if s not in unique_symbols]
        return unique_symbols

    def _transform_symbols(self, symbols, subset_size=2):
        symbol_remainder = len(symbols) % subset_size
        remainder_elements = symbols[-symbol_remainder:] if symbol_remainder else []
        grouped = [list(x) for x in zip(*[iter(symbols[:-symbol_remainder] if symbol_remainder else symbols)] * subset_size)]
        if remainder_elements:
            grouped.append(remainder_elements)
        return grouped
    
    def _fetch_L2_data(self):
        print('\n======== FETCHING LEVEL2 DATA ========\n')
        symbols_grouped = self._transform_symbols(self._get_symbols(), 2) if not self.single_symbol else [[self.single_symbol]]

        for group in symbols_grouped:
            print("Subset Symbols: ", group)

            for symbol in group:
                print(f"\nFetching Market Depth for symbol {symbol}")
                domL2 = level2.DomL2(self.ib, symbol, self.daily_data_folder)
                MD = domL2.get_dom()
                self.ib.sleep(constants.CONSTANTS.PROCESS_TIME['medium'])

                if MD.domBids and MD.domAsks:
                    domL2.assess_dom(full_dom=False, save_to_file=True)
                    domL2.print_dom(full_dom=False)

            self._disconnect_ib()
            self.ib.sleep(2)
            self._connect_ib()
    
    def _execute_main_task(self):
        self._fetch_L2_data()


if __name__ == "__main__":

    args = sys.argv
    paper_trading = not 'live' in args
    local_ib = 'local' in args
    continuous = not 'snapshot' in args
    single_symbol = next(([arg[7:]] for arg in args if arg.startswith('symbol=')), None)
    wait_seconds = next((int(float(arg[5:])) for arg in args if arg.startswith('wait=')), 5*60)

    fetcher = LiveL2Fetcher(wait_seconds=wait_seconds, continuous=continuous, single_symbol=single_symbol, ib_disconnect=True, 
                            paper_trading=paper_trading, remote_ib=not local_ib)
    fetcher.run()

# def get_symbols(folder=helpers.get_path_daily_data_folder()):

#     symbols_gapper_up_csv_path = os.path.join(folder, constants.PATHS.daily_csv_files['gapper_up'])
#     symbols_gapper_down_csv_path = os.path.join(folder, constants.PATHS.daily_csv_files['gapper_down'])
#     symbols_earnings_csv_path = os.path.join(folder, constants.PATHS.daily_csv_files['earnings'])
#     symbols_rsi_reversal_csv_path = os.path.join(folder, constants.PATHS.daily_csv_files['bb_rsi_reversal'])
#     symbols_main_csv_path = constants.PATHS.csv_files['main']
#     symbols_index_csv_path = constants.PATHS.csv_files['index']

#     symbols_gapper_up = [row[1] for i, row in enumerate(helpers.read_csv_file(symbols_gapper_up_csv_path)) if i > 0 and row != [] and row[0] != '']
#     symbols_gapper_down = [row[1] for i, row in enumerate(helpers.read_csv_file(symbols_gapper_down_csv_path)) if i > 0 and row != [] and row[0] != '']
#     symbols_earnings = [row[1] for i, row in enumerate(helpers.read_csv_file(symbols_earnings_csv_path)) if i > 0 and row != [] and row[0] != '']
#     symbols_rsi_reversal = [row[1] for i, row in enumerate(helpers.read_csv_file(symbols_rsi_reversal_csv_path)) if i > 0 and row != [] and row[0] != '']
#     symbols_main = [row[0] for row in helpers.read_csv_file(symbols_main_csv_path) if row != [] and row[0] != '']
#     symbols_index = [row[0] for row in helpers.read_csv_file(symbols_index_csv_path) if row != [] and row[0] != '']

#     symbols = symbols_gapper_up + symbols_gapper_down + symbols_earnings + symbols_rsi_reversal + symbols_main + symbols_index
#     unique_symbols = []
#     [unique_symbols.append(item) for item in symbols if item not in unique_symbols] # Remove duplicates

#     return symbols


# def transform_symbols(symbols, subset_size=2):
#     # Modify symbols shape from 1 x len to n x len/n
#     symbol_remainder = len(symbols) % subset_size
#     remainder_elements = symbols[-symbol_remainder:] if symbol_remainder else []
#     grouped = [list(x) for x in zip(*[iter(symbols[:-symbol_remainder] if symbol_remainder else symbols)] * subset_size)]
#     if remainder_elements:
#         grouped.append(remainder_elements)
#     return grouped


# def fetch_L2_data(ib, folder = helpers.get_path_daily_data_folder()):
#     print('\n======== FETCHING LEVEL2 DATA ========\n')
#     symbols = transform_symbols(get_symbols(folder), 2) if not single_symbol else [single_symbol]

#     for symbols_subset in symbols:
#         print("Subset Symbols: ", symbols_subset)

#         for symbol in symbols_subset:

#             # subprocess.run(['python', os.path.join(parent_folder, 'level2.py'), 'sym'+symbol])

#             print("\nFetching Market Depth for symbol ", symbol)
#             domL2 = level2.DomL2(ib, symbol, folder)
#             MD = domL2.get_dom()
#             ib.sleep(constants.CONSTANTS.PROCESS_TIME['medium'])

#             if MD.domBids and MD.domAsks:
#                 domL2.assess_dom(full_dom=False, save_to_file=True)
#                 domL2.print_dom(full_dom=False)

#         print("\nDisconnecting IB")
#         ib.disconnect()
#         ib.sleep(2)
#         print("\nReconnecting IB")
#         ib, _ = helpers.IBKRConnect_any(IB(), paper=paperTrading)


# if __name__ == "__main__":

#     # # Path Setup
#     # path = helpers.path_setup.path_current_setup(parent_folder)

#     args = sys.argv

#     paperTrading = not 'live' in args
#     continuous = 'cont' in args
#     single_symbol = next(([arg[7:]] for arg in args if arg.startswith('symbol=')), None)
#     time_wait = next((int(float(arg[5:])) * 60 for arg in args if arg.startswith('wait=')), 5 * 60)

#     # TWS Connection
#     paperTrading = False if len(args) > 1 and 'live' in args else True
#     # ib, ibConnection = helpers.IBKRConnect_any(IB(), paper=paperTrading)

#     # Paths Setup
#     daily_data_folder = helpers.get_path_daily_data_folder()


#     counter = 1000
#     while counter > 0:
#         print("\nConnecting IB")
#         ib, ibConnection = helpers.IBKRConnect_any(IB(), paper=paperTrading)

#         fetch_L2_data(ib, daily_data_folder)

#         if continuous:

#             print("\nDisconnecting IB")
#             ib.disconnect()

#             counter = counter - 1
#             helpers.sleep_display(time_wait, ib)

#         else: counter = 0

# print("\n\n")
