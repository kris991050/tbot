import os, sys, csv, pytz, prettytable, traceback, pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from ib_insync import *
from utils import helpers
from utils.constants import CONSTANTS


def get_symbols(csv_path):

    # Get symbols from CSV file
    with open(csv_path, mode ='r') as file:
        symbols = [lines[0] for lines in csv.reader(file) if lines]
    file.close()

    # Modify symbols shape from 1 x n to 2 x n/2
    symbol_odd = len(symbols) % 2 != 0
    if symbol_odd:
        last_el = symbols[-1]
        symbols = symbols[:-1]
    symbols = [list(x) for x in zip(*[iter(symbols)]*2)]
    if symbol_odd: symbols.append([last_el])

    return symbols


class TimeNSales:

    __slots__ = ('ib', 'symbol', 'data_folder', 'ts', 'ba', 'ts_aggregated', 'ts_aggregated_by_time_window', 'ts_sign_levels', 'ts_table', 'timezone')

    def __init__(self, ib, symbol, data_folder, timezone=pytz.timezone(CONSTANTS.TZ_WORK)):
        self.ib = ib
        self.symbol = symbol.upper()
        self.data_folder = data_folder
        self.ts = pd.DataFrame()#[]
        self.ba = pd.DataFrame()#[]
        self.ts_aggregated = pd.DataFrame()#[]
        self.ts_aggregated_by_time_window = []
        self.ts_sign_levels = []
        self.timezone = timezone
        self.ts_table = prettytable.PrettyTable()


    def reqHistTicks(self, type_data, time_periods, numTicks=1000, round_deg=4, currency=CONSTANTS.DEFAULT_CURRENCY, time_window=None):
        contract, mktData = helpers.get_symbol_mkt_data(self.ib, self.symbol, currency=currency)#, exchange="NASDAQ")
        if type_data == 'ticks': whatToShow = "TRADES" if contract.symbol not in helpers.get_forex_symbols_list() else "MIDPOINT"
        elif type_data == 'bid_ask': whatToShow = 'BID_ASK'
        else: whatToShow = ''

        for time_period in time_periods:
            start_time = time_period['start_time']
            end_time = time_period['end_time']
            try:
                condition_loop = start_time and end_time
                if condition_loop:
                    end_time_loop = end_time
                    end_time = ''
                else:
                    end_time_loop = start_time

                while start_time <= end_time_loop:
                    print("\rRemaining: {}".format(end_time_loop - start_time), end='')

                    data = ib.reqHistoricalTicks(contract=contract, startDateTime=start_time, endDateTime=end_time, numberOfTicks=numTicks, whatToShow=whatToShow, useRth=False)
                    ib.sleep(CONSTANTS.PROCESS_TIME['short'])

                    if type_data == 'ticks':
                        #self.ts += data
                        self.ts = pd.concat([self.ts, pd.DataFrame(data)], ignore_index=True)
                        time_end_df = self.ts.iloc[-1]['time']
                    if type_data == 'bid_ask':
                        # self.ba += data
                        self.ba = pd.concat([self.ba, pd.DataFrame(data)], ignore_index=True)
                        time_end_df = self.ba.iloc[-1]['time']

                    if not condition_loop: break


                    if time_end_df == start_time: start_time += timedelta(seconds=1)
                    else: start_time = time_end_df
                    # if self.ts[-1].time == start_time: start_time += timedelta(seconds=1)
                    # else: start_time = self.ts[-1].time

            except Exception as e: print("Could not get historical ticks. Error: ", e, "  |  Full error: ", traceback.format_exc())

        if type_data == 'ticks':
            self.__aggregate_ts(round_deg=round_deg)
            if time_window: self.__aggregate_ts_by_time_window(round_deg=round_deg, time_window=time_window)


    def __aggregate_ts_by_time_window(self, round_deg=4, time_window='s'):
        # Aggregate ticks by time window

        # Time window definition
        if time_window == 'm': time_expr = "%Y-%m-%d %H:%M"
        elif time_window == 'h': time_expr = "%Y-%m-%d %H"
        elif time_window == 'd': time_expr = "%Y-%m-%d"
        else: time_expr = "%Y-%m-%d %H:%M:%S"

        ts_grouped = defaultdict(lambda: defaultdict(int))
        # ts_grouped = defaultdict(lambda: defaultdict(lambda: {"price": 0, "size": 0})

        for index, tick in self.ts.iterrows():
            time = tick['time'].astimezone(tz=self.timezone)
            # ts_grouped[time.strftime(time_expr)][round(tick.price, round_deg)] += tick.size
            ts_grouped[time.strftime(time_expr)][round(tick['price'], round_deg)] += tick['size']

        self.ts_aggregated_by_time_window = []
        for time, price_data in ts_grouped.items():
            tick_list = sorted([{"price": price, "size": size} for price, size in price_data.items()], reverse=False, key=lambda x: x['price'])
            self.ts_aggregated_by_time_window.append({"time": time, "tick_list": tick_list})

        self.ts_aggregated_by_time_window = sorted(self.ts_aggregated_by_time_window, reverse=False, key=lambda x: x['time'])


    def __aggregate_ts(self, round_deg=4):
        # ts_grouped = defaultdict(float)

        # for tick in self.ts: ts_grouped[round(tick.price, round_deg)] += tick.size

        # self.ts_aggregated = sorted([{"price": price, "size": size} for price, size in ts_grouped.items()], reverse=False, key=lambda x: x['price'])

        self.ts['rounded_price'] = self.ts['price'].round(round_deg)
        self.ts_aggregated = self.ts.groupby('rounded_price', as_index=False)['size'].sum().sort_values(by='rounded_price', ascending=False)


    def get_sign_levels(self, factor_std=3):
        # ts_sizes = [t['size'] for t in self.ts_aggregated]
        # ts_sizes = [t['size'] for index, t in self.ts_aggregated.iterrows()]
        # ts_sizes_std = np.std(ts_sizes)
        # sign_levels = sorted(list(filter(lambda x: x['size'] >= factor_std * ts_sizes_std, self.ts_aggregated)), reverse=True, key=lambda x: x['size'])
        if not self.ts_aggregated.empty:
            sign_levels = self.ts_aggregated[self.ts_aggregated['size'] >= factor_std * self.ts_aggregated['size'].std()].sort_values(by='size', ascending=False)
        else: sign_levels = pd.DataFrame()

        return sign_levels


    def print_ticks(self, type='ticks'):
        try:
            if type == 'ticks':
                # for tick in self.ts:
                    # print(tick.time.astimezone(tz=self.timezone).strftime('%Y-%m-%d %H:%M:%S'), '  ', tick.price, '  ', tick.size)
                for index, tick in self.ts.iterrows():
                    print(tick['time'].astimezone(tz=self.timezone).strftime('%Y-%m-%d %H:%M:%S'), '  ', tick['price'], '  ', tick['size'])
            elif type == 'bid_ask':
                # for tick in self.ba:
                    # print(tick.time.astimezone(tz=self.timezone).strftime('%Y-%m-%d %H:%M:%S'), '  -  ', tick.priceBid, ' | ', tick.priceAsk, '  -  ', tick.sizeBid, ' | ', tick.sizeAsk, '  -  ', tick.tickAttribBidAsk.bidPastLow, ' | ', tick.tickAttribBidAsk.askPastHigh, '  -  ', round(tick.sizeBid *100 / (tick.sizeBid + tick.sizeAsk), 1), ' | ', round(tick.sizeAsk *100 / (tick.sizeBid + tick.sizeAsk), 1))
                for index, tick in self.ba.iterrows():
                    print(tick['time'].astimezone(tz=self.timezone).strftime('%Y-%m-%d %H:%M:%S'), '  -  ', tick['priceBid'], ' | ', tick['priceAsk'], '  -  ', tick['sizeBid'], ' | ', tick['sizeAsk'], '  -  ', tick['tickAttribBidAsk'].bidPastLow, ' | ', tick['tickAttribBidAsk'].askPastHigh, '  -  ', round(tick['sizeBid'] *100 / (tick['sizeBid'] + tick['sizeAsk']), 1), ' | ', round(tick['sizeAsk'] *100 / (tick['sizeBid'] + tick['sizeAsk']), 1))
            print()
        except Exception as e: print("Could not print ", type, ". Error: ", e, "  |  Full error: ", traceback.format_exc())


    def print_ticks_aggregated(self):
        try:
            # for t in self.ts_aggregated: print(t)
            print(self.ts_aggregated.to_string())
            # for index, t in self.ts_aggregated.iterrows(): print(t)
            print()
        except Exception as e: print("Could not print aggregated ticks. Error: ", e, "  |  Full error: ", traceback.format_exc())


    def print_ticks_aggregated_by_time_window(self):
        try:
            for ts in self.ts_aggregated_by_time_window:
                print(ts['time'])
                for t in ts['tick_list']:
                    print(t)
                print()
        except Exception as e: print("Could not print aggregated ticks. Error: ", e, "  |  Full error: ", traceback.format_exc())



if __name__ == "__main__":

    import helpers

    args = sys.argv

    journal_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Trading_Journal", "")
    # dataTS_folder = os.path.join(helpers.path_current_setup(journal_folder, print_path=False), helpers.get_path_date_folder(), 'dataTS')
    daily_data_folder = helpers.get_path_daily_data_folder()
    symbols_csv_path = os.path.join(daily_data_folder, "symbolsTS.csv")

    symbol = 'TSLA'
    stream = False
    time_window = None
    for arg in args:
        if 'sym' in arg: symbol = arg[3:]
        if 'stream' in arg: stream = True
        if 'window' in arg: time_window = arg[6:]

    paperTrading = False if len(args) > 1 and 'live' in args else True

    ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)

    if not stream:
        time_periods = [{'start_time': helpers.date_to_EST_aware(datetime(2025, 2, 28, 13, 56, 00)), 'end_time': helpers.date_to_EST_aware(datetime(2025, 2, 28, 13, 58, 00))},
                       {'start_time': helpers.date_to_EST_aware(datetime(2025, 2, 28, 13, 50, 00)), 'end_time': helpers.date_to_EST_aware(datetime(2025, 2, 28, 13, 53, 00))},
                       {'start_time': helpers.date_to_EST_aware(datetime(2025, 2, 27, 13, 1, 00)), 'end_time': helpers.date_to_EST_aware(datetime(2025, 2, 27, 13, 5, 00))}]

        time_periods = [{'start_time': helpers.date_to_EST_aware(datetime(2025, 4, 9, 4, 30, 00)), 'end_time': helpers.date_to_EST_aware(datetime(2025, 4, 9, 19, 30, 0))}]
        # end_time = helpers.date_to_EST_aware(datetime.now())

        time_now_start = datetime.now()

        ticks_list = TimeNSales(ib, symbol, data_folder=daily_data_folder)
        ticks_list.reqHistTicks('ticks', time_periods, numTicks=1000, round_deg=2, time_window=time_window)
        # ticks_list.reqHistTicks('bid_ask', time_periods, numTicks=1000, round_deg=2, time_window=time_window)
        # ib.reqTickByTickData()

        # ticks_list.print_ticks()
        # print()
        # input()
        # ticks_list.print_ticks('bid_ask')
        # print()
        # input()
        if not time_window: ticks_list.print_ticks_aggregated()
        else: ticks_list.print_ticks_aggregated_by_time_window()
        print()
        lev3 = ticks_list.get_sign_levels(3)
        print("\nSignificant Levels 3:")
        # for l in lev3: print(l)
        if not lev3.empty : print(lev3.to_string())
        print()

        time_now_end = datetime.now()
        print("\nTime elapsed: ", time_now_end - time_now_start)

    elif stream:
        print("symbol = ", symbol)
        contract, mktData = helpers.get_symbol_mkt_data(ib, symbol)#, exchange="NASDAQ")
        tick_list = ib.reqTickByTickData(contract, 'AllLast')
        print(tick_list)
        print()
        ib.sleep(1)
        print(tick_list)
        print()
        ib.sleep(1)
        print(tick_list)
        print()





    # counter = 60
    # while counter > 0:

    #     symbols = get_symbols(symbols_csv_path) if not symbol_single else [symbol_single]

    #     for symbols_subset in symbols:
    #         print("Subset Symbols: ", symbols_subset)

    #         for symbol in symbols_subset:
    #             if dummy:
    #                 MD = DummyMktDepth()
    #                 domL2 = DomL2(ib, symbol, dataL2_folder, MD.domBids, MD.domAsks)
    #             else:
    #                 print("\nFetching Market Depth for symbol ", symbol)
    #                 domL2 = DomL2(ib, symbol, dataL2_folder)
    #                 MD = domL2.get_dom()
    #                 ib.sleep(1)

    #             if MD.domBids and MD.domAsks:
    #                 # FD = ib.reqFundamentalData(contract, reportType='ReportsFinSummary', fundamentalDataOptions=[]) #https://ib-insync.readthedocs.io/api.html#ib_insync.ib.IB.reqMktDepth
    #                 # ib.sleep(1)

    #                 # for i, md in enumerate(domL2.bids):
    #                 #     print(md.price, "    ", md.size, "   |   ", domL2.asks[i].price, "    ", domL2.asks[i].size)
    #                 # print()

    #                 sumBidsPonderated, sumAsksPonderated = domL2.assess_dom()
    #                 # domL2.plot_dom()

    #             else:
    #                 print("Could not fetch Market Depth data.")

    #         if stats:
    #             print("\nDisconnecting IB")
    #             ib.disconnect()
    #             ib.sleep(4)
    #             print("\nReconnecting IB")
    #             ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)

    #     if stats:

    #         print("\nDisconnecting IB")
    #         ib.disconnect()

    #         counter = counter - 1
    #         helpers.sleep_display(time_wait)
    #         # # ib.sleep(time_wait)
    #         print("\nReconnecting IB")
    #         ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)

    #     else: counter = 0

print("\n\n")
