import os, pytz, csv, time, sys, datetime, re, shutil, pandas as pd, prettytable#, ezodf2
from utils import constants, indicators
# from datetime import datetime, timedelta, timezone
from ib_insync import *
# Doc ib_insync: https://ib-insync.readthedocs.io/api.html
# Doc IBKR API: https://interactivebrokers.github.io/tws-api/introduction.html
# util.startLoop()  # uncomment this line when in a notebook


# def date_local_to_EST(date):
#     local_timezone = [tz for tz in pytz.all_timezones if tz.find(time.tzname[0]) >= 0][0]
#     time_now_local = datetime.datetime.now(tz=pytz.timezone(local_timezone))
#     date = datetime.datetime.fromisoformat(str(date) + str(time_now_local)[-6:])#.replace(tzinfo=pytz.timezone(local_timezone))

#     return datetime.datetime.fromisoformat(str(date)).astimezone(tz=pytz.timezone(constants.CONSTANTS.TZ_WORK))

class Journal:

    def __init__(self, type):
        self.type = type
        self.__journal_entries = []
        self.__fills_entries = []


def findStopLoss_IBKR(fill, orders_list):
    order_selected = list(filter(lambda x: (x.parentPermId == fill.execution.permId and x.orderType=="STP"), orders_list))
    return order_selected[0].auxPrice if order_selected else "NA"


def findProfitTarget_IBKR(fill, orders_list):
    order_selected = list(filter(lambda x: (x.parentPermId == fill.execution.permId and x.orderType=="LMT"), orders_list))
    return order_selected[0].lmtPrice if order_selected else "NA"


def journalEntriesIBKR(fills_list, orders_list, number_days_back=0, single_symbol=None):

    date_now = datetime.datetime.fromisoformat(datetime.datetime.now().strftime('%Y-%m-%d'))
    symbols_in_play = list(set([fill.contract.symbol for fill in fills_list])) if not single_symbol else [single_symbol]
    print("Symbols in play:", symbols_in_play, "\n")

    # record_data_symbol_in_play(symbols_in_play, "1 min", local_hist_folder_rel)

    journal_entries = []
    fills_entries = []
    for symbol in symbols_in_play:

        # print("=======================\nSymbol = ", symbol, "\n=======================\n")

        # print("\nNo order found. Fills will be used instead of Trades.\n")
        # type_trade = "None"
        date_condition = lambda x: date_now - datetime.datetime.fromisoformat(x.replace(tzinfo=datetime.timezone.utc).astimezone(tz=pytz.timezone(constants.CONSTANTS.TZ_WORK)).strftime('%Y-%m-%d')) <= datetime.timedelta(days=number_days_back)
        option_factor = lambda fill: 100 if fill.contract.secType == 'OPT' else 1
        order_selected = [filter(lambda x: x.permId == fills_list[2].execution.permId, orders_list)]#list(filter(lambda x: x.permId == fills_list[2].execution.permId, orders_list))

        fills_ordered = [{"time": fill.execution.time.replace(tzinfo=datetime.timezone.utc).astimezone(tz=pytz.timezone(constants.CONSTANTS.TZ_WORK)),
                        # "type": list(filter(lambda x: x.permId == fill.execution.permId, orders_list))[0].orderType if list(filter(lambda x: x.permId == fill.execution.permId, orders_list)) else "None",
                        "action": fill.execution.side,
                        "quantity": fill.execution.shares * option_factor(fill),
                        "cum_quantity": fill.execution.shares * action_conversion_IBKR[fill.execution.side],
                        "price": fill.execution.price,
                        "profit_target": findProfitTarget_IBKR(fill, orders_list),
                        "stop_loss": findStopLoss_IBKR(fill, orders_list),
                        "commissions": fill.commissionReport.commission,
                        "currency": fill.contract.currency,
                        "ID": fill.execution.permId}
                        for fill in fills_list if (fill.contract.symbol == symbol and date_condition(fill.time))]


        fills_ordered.sort(key=lambda item:item["time"], reverse=False) # sort by time
        if fills_ordered: fills_entries.append({"symbol": symbol, "fills": fills_ordered})

        index_entry = 0
        for index in range(1, len(fills_ordered)):

            # Calculate cumulative quantities
            fills_ordered[index]["cum_quantity"] += fills_ordered[index-1]["cum_quantity"]

            # Create journal entries
            if fills_ordered[index]["cum_quantity"] == 0:
                journal_entry = fills_ordered[index_entry : index + 1]
                index_entry = index + 1

                direction = action_conversion_IBKR[journal_entry[0]["action"]]
                entry_direction_text = list(action_conversion_IBKR.keys())[list(action_conversion_IBKR.values()).index(direction)]
                close_direction_text = list(action_conversion_IBKR.keys())[list(action_conversion_IBKR.values()).index(-direction)]
                quantity = sum([el["quantity"] for el in journal_entry]) / 2
                entry_price = sum([el["quantity"] * el["price"] for el in journal_entry if el["action"] == entry_direction_text]) / quantity
                close_price = sum([el["quantity"] * el["price"] for el in journal_entry if el["action"] == close_direction_text]) / quantity
                commissions = sum(el["commissions"] for el in journal_entry)
                profit_target = journal_entry[0]["profit_target"]
                stop_loss = journal_entry[0]["stop_loss"]
                max_profit = round(quantity * direction * (float(profit_target) - entry_price), 5) if profit_target != "NA" else "NA"
                max_loss = round(quantity * direction * (float(stop_loss) - entry_price), 5) if stop_loss != "NA" else "NA"

                # print("\njournal_entry = ", journal_entry)
                # Formatting for Journal.ods
                journal_entries.append({"symbol": symbol,
                                        "entry_time": journal_entry[0]["time"].strftime("%Y-%m-%d %H:%M:%S"),
                                        "quantity": quantity,
                                        "currency": journal_entry[0]["currency"],
                                        "entry_price": str(round(entry_price, 5)),
                                        "direction": direction,
                                        "profit_target": profit_target,
                                        "max_profit": str(max_profit),
                                        "stop_loss": stop_loss,
                                        "max_loss": str(max_loss),
                                        "profit_ratio": str(round(abs(max_profit / max_loss), 5)) if (max_profit != "NA" and max_loss != "NA" and max_loss != 0) else "NA",
                                        "close_price": str(round(close_price, 5)),
                                        "close_time": journal_entry[-1]["time"].strftime("%Y-%m-%d %H:%M:%S"),
                                        "duration": journal_entry[-1]["time"] - journal_entry[0]["time"],
                                        "commissions": str(round(-commissions, 5)),
                                        "profit-loss": str(round((close_price - entry_price) * quantity * direction - commissions, 5))
                                        })

    # Sort by Entry Time
    journal_entries.sort(key=lambda item:item["entry_time"], reverse=False)

    return journal_entries, fills_entries


def findCSVHistoryFile(folder_path, delete_old=False):

    # Finding csv history file in date folder
    CSVFile_list = []
    for file in os.listdir(folder_path):
        if ".csv" in file and "trading-history" in file:
            date_str = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}', file).group()
            CSVFile_list.append({'path': os.path.join(folder_path, file), 'date': datetime.datetime.strptime(date_str, '%Y-%m-%dT%H_%M_%S')})

    # Selecting last csv history file
    CSVFiles_old = None
    if len(CSVFile_list) > 1:
        CSVFile = max(CSVFile_list, key=lambda x: x['date'])
        CSVFiles_old = [file for file in CSVFile_list if file['date'] != CSVFile['date']] + [file for file in CSVFile_list if file['date'] == CSVFile['date'] and file != CSVFile]
        CSVFile = CSVFile['path']
    elif len(CSVFile_list) == 1:
        CSVFile =  CSVFile_list[0]['path']
    else:
        CSVFile = ""
        print("\nNo .csv file found in ", folder_path)

    # Delete other CSV files
    if delete_old and CSVFiles_old:
        print("\nRemoving older history files")
        for file in CSVFiles_old:
            if os.path.exists(file['path']):
                os.remove(file['path'])
                print("Old history file removed: ", file['path'], "\n")

    return CSVFile


def moveCSVHistory(source_folder, destination_folder):

    CSVFile_path = findCSVHistoryFile(source_folder, delete_old=True)

    if CSVFile_path != "":
        # Remove existing other history files in destination
        print("\nRemoving existing other history files in destination...")
        old_CSVFile_path_destination = findCSVHistoryFile(destination_folder, delete_old=True)

        if os.path.exists(old_CSVFile_path_destination):
            os.remove(old_CSVFile_path_destination)
            print("Old history file removed: ", old_CSVFile_path_destination, "\n")

        print("\nMoving last history files to destination...")
        CSVFile_path_destination = os.path.join(destination_folder, os.path.basename(CSVFile_path))
        shutil.move(CSVFile_path, CSVFile_path_destination)
        print("Moved file ", CSVFile_path, "\nto ", CSVFile_path_destination, "\n")
    else:
        print("\nNo CSV hstory file found in ", source_folder, "\n")


def journalEntriesTV(date=datetime.datetime.now(), number_days_back=0, single_symbol=None):

    date_now = datetime.datetime.fromisoformat(datetime.datetime.now().strftime('%Y-%m-%d'))

    path_date_folder = helpers.get_path_date_folder(date)
    # dateOpenCol, dateCloseCol, symCol, quantityCol, priceCol, fill_priceCol, statusCol, orderDirectionCol, typeCol, commissionCol = 8, 9, 0, 3, 4, 5, 6, 1, 2, 7
    dateOpenCol, dateCloseCol, symCol, quantityCol, priceCol, fill_priceCol, statusCol, orderDirectionCol, typeCol, commissionCol = 11, 12, 0, 3, 4, 6, 7, 1, 2, 8

    CSVFile = findCSVHistoryFile(path_date_folder)
    if not CSVFile: return [], []

    with open(CSVFile) as csv_file: # Create list object from CSV file entries
        fills_list = [row for row in csv.reader(csv_file) if (row[symCol] != "Symbol" and row[symCol] != "" and row[symCol] != None)][::-1]

    for fill in fills_list:
        fill[symCol] = fill[symCol][fill[symCol].find(":")+1:] # Remove "NASDAQ" from symbol
        if fill[commissionCol] == '': fill[commissionCol] = 0.0
        else: fill[commissionCol] = float(fill[commissionCol]) # convert commission into float and replace '' by 0

    symbols_in_play = list(set([fill[0] for fill in fills_list])) if not single_symbol else [single_symbol]

    # record_data_symbol_in_play(symbols_in_play, "1 min", local_hist_folder_rel)


    journal_entries = []
    fills_entries = []
    for symbol in symbols_in_play:

        # date_condition = lambda x: date_now - datetime.datetime.fromisoformat(x) <= datetime.timedelta(days=number_days_back)
        date_condition = lambda x: date_now - datetime.datetime.fromisoformat(datetime.datetime.fromisoformat(x).strftime('%Y-%m-%d')) <= datetime.timedelta(days=number_days_back)

        time_calc = lambda p,t1,t2: helpers.date_local_to_EST(datetime.datetime.fromisoformat(t1)) if p == "" else helpers.date_local_to_EST(datetime.datetime.fromisoformat(t2))
        fills_ordered = [{"time": time_calc(fill[fill_priceCol], fill[dateOpenCol], fill[dateCloseCol]),
                        # "type": list(filter(lambda x: x.permId == fill.execution.permId, orders_list))[0].orderType if list(filter(lambda x: x.permId == fill.execution.permId, orders_list)) else "None",
                        "action": fill[orderDirectionCol],
                        "quantity": fill[quantityCol],
                        "cum_quantity": int(fill[quantityCol]) * action_conversion_TV[fill[orderDirectionCol]] if fill[fill_priceCol] != "" else 0,
                        "price": fill[fill_priceCol],
                        "status": fill[statusCol],
                        "profit_target": fill[priceCol] if fill[typeCol] == "Take Profit" else "NA",
                        "stop_loss": fill[priceCol] if fill[typeCol] == "Stop Loss" else "NA",
                        "commissions": fill[commissionCol],
                        "currency": "USD",
                        "ID": "NA"}
                        for fill in fills_list if (fill[symCol] == symbol and date_condition(fill[dateCloseCol]) and fill[statusCol] != 'Cancelled')]

        fills_ordered.sort(key=lambda item:item["time"], reverse=False) # sort by time
        # if symbol == 'SPY':
        #     for fill in fills_ordered:
        #         print(fill)
        #         print()
        #     input()
        if fills_ordered: fills_entries.append({"symbol": symbol, "fills": fills_ordered})

        index_entry = 0
        for index in range(1, len(fills_ordered)):

            # Calculate cumulative quantities
            if fills_ordered[index]["price"] != "":
                fills_ordered[index]["cum_quantity"] += fills_ordered[index-1]["cum_quantity"]
            else:
                fills_ordered[index]["cum_quantity"] = fills_ordered[index-1]["cum_quantity"]

            # Create journal entries
            if fills_ordered[index]["cum_quantity"] == 0 and fills_ordered[index]["price"] != '':
                journal_entry = fills_ordered[index_entry : index + 1]



                index_entry = index + 1

                direction = action_conversion_TV[journal_entry[0]["action"]]
                entry_direction_text = list(action_conversion_TV.keys())[list(action_conversion_TV.values()).index(direction)]
                close_direction_text = list(action_conversion_TV.keys())[list(action_conversion_TV.values()).index(-direction)]
                quantity = sum([int(el["quantity"]) for el in journal_entry if el["price"] != ""]) / 2
                entry_price = sum([int(el["quantity"]) * float(el["price"]) for el in journal_entry if (el["action"] == entry_direction_text and el["price"] != "")]) / quantity
                close_price = sum([int(el["quantity"]) * float(el["price"]) for el in journal_entry if (el["action"] == close_direction_text and el["price"] != "")]) / quantity
                commissions = sum(el["commissions"] for el in journal_entry)
                close_time = max([entry["time"] for entry in journal_entry if entry["price"] != ""])
                profit_target = [el["profit_target"] for el in journal_entry if el["profit_target"] != "NA"][0] if [el["profit_target"] for el in journal_entry if el["profit_target"] != "NA"] else "NA"
                stop_loss = [el["stop_loss"] for el in journal_entry if el["stop_loss"] != "NA"][0] if [el["stop_loss"] for el in journal_entry if (el["stop_loss"] != "NA" and el["stop_loss"] != "")] else "NA"
                max_profit = round(quantity * direction * (float(profit_target) - entry_price), 5) if profit_target != "NA" else "NA"
                max_loss = round(quantity * direction * (float(stop_loss) - entry_price), 5) if stop_loss != "NA" else "NA"

                journal_entries.append({"symbol": symbol,
                                    "entry_time": journal_entry[0]["time"].strftime("%Y-%m-%d %H:%M:%S"),
                                    "quantity": quantity,
                                    "currency": journal_entry[0]["currency"],
                                    "entry_price": str(round(entry_price, 5)),
                                    "direction": direction,
                                    "profit_target": profit_target,
                                    "max_profit": str(max_profit),
                                    "stop_loss": stop_loss,
                                    "max_loss": str(max_loss),
                                    "profit_ratio": str(round(abs(max_profit / max_loss), 5)) if (max_profit != "NA" and max_loss != "NA" and max_loss != 0) else "NA",
                                    "close_price": str(round(close_price, 5)),
                                    "close_time": close_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "duration": str(close_time - journal_entry[0]["time"]),
                                    "commissions": str(round(-commissions, 5)),
                                    "profit-loss": str(round((close_price - entry_price) * quantity * direction - commissions, 5))
                                    })

    # Sort by Entry Time
    journal_entries.sort(key=lambda item:item["entry_time"], reverse=False)

    return journal_entries, fills_entries


def journalToCSV(ib, entries, ibkr, paperTrading, local_hist_folder, minimal_display=False):

    date_folder = helpers.get_path_date_folder()
    date_str = date_folder[-8:]
    journal_type = "TV"
    paper_type = ""
    if ibkr:
        journal_type = "IBKR"
        if paperTrading: paper_type += "_paper"
    journal_CSV_local_File = date_folder + "/Entries_" + journal_type + paper_type + "_" + date_str + ".csv"

    if entries:
        if not minimal_display:
            with open(journal_CSV_local_File, mode='w') as file:
                journal_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                saved_data_list = []
                for entry in entries:

                    symbol = entry["symbol"]
                    contract, mktData = helpers.get_symbol_mkt_data(ib, symbol, currency='USD')
                    # contract = helpers.get_symbol_contract(ib, symbol)

                    query_time = helpers.date_to_EST_aware(datetime.datetime.fromisoformat(entry["entry_time"]))
                    query_date = query_time.strftime('%Y-%m-%d')

                    try:
                        df_1 = helpers.get_symbol_hist_data(ib, symbol, "1 min", query_time, duration="1 W", indicators_list=['rsi', 'atr_D'])
                        df_1['date'] = pd.to_datetime(df_1['date'], utc=True)
                        df_1['date'] = df_1['date'].dt.tz_convert(constants.CONSTANTS.TZ_WORK)
                        df_5 = df_1.resample('5min', on='date').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna().reset_index()
                        df_5 = indicators.add_indicator(df_5, ib, contract, indicators_list=["rsi"])
                        df_15 = df_1.resample('15min', on='date').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna().reset_index()
                        df_15 = indicators.add_indicator(df_15, ib, contract, indicators_list=["rsi"])
                        df_60 = df_1.resample('60min', on='date').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna().reset_index()
                        df_60 = indicators.add_indicator(df_60, ib, contract, indicators_list=["rsi"])

                        # df_5 = helpers.get_symbol_hist_data(ib, symbol, "5 mins", query_time, duration="1 W", indicators_list = ["rsi"])
                        # df_15 = helpers.get_symbol_hist_data(ib, symbol, ,"15 mins", query_time, duration="1 W", indicators_list = ["rsi"])
                        # df_60 = helpers.get_symbol_hist_data(ib, symbol, "1 hour", query_time, duration="1 W", indicators_list = ["rsi"])

                        rsi_5, rsi_15, rsi_60 = round(indicators.get_indicator(df_5, 'rsi', query_time), 2), round(indicators.get_indicator(df_15, 'rsi', query_time), 2), round(indicators.get_indicator(df_60, 'rsi', query_time), 2)

                        saved_data = list(filter(lambda x: x['symbol'] == symbol and x['date'] == query_date, saved_data_list))[0] if symbol in [data['symbol'] for data in saved_data_list] else None

                        if saved_data:
                            avg_volume, rel_volume, floats, index = saved_data['avg_volume'], saved_data['rel_volume'], saved_data['floats'], saved_data['index']
                            pivots, pivots_D = saved_data['pivots'], saved_data['pivots_D']

                            # indicators_list = ['rsi', 'atr_D']
                            # df_1 = helpers.get_symbol_hist_data(ib, symbol, "1 min", query_time, duration="1 W", indicators_list=indicators_list)

                        else:
                            # Get Avg Volume, Relative Volume, Floats and Index
                            avg_volume, rel_volume, floats = helpers.get_volumes_from_Finviz(symbol)
                            index = helpers.get_stock_info_from_Finviz(symbol, "Index")

                            # Get RSI, ATR, CPR size and level comparisons
                            # df_1 = helpers.get_symbol_hist_data(ib, symbol, "1 min", query_time, duration="1 W", indicators_list=['levels'])
                            df_1 = indicators.add_indicator(df_1, ib, contract, indicators_list=["levels"])

                            pivots = indicators.get_indicator(df_1, 'pivots', query_time)
                            pivots_D = indicators.get_indicator(df_1, 'pivots_D', query_time)

                            saved_data_list.append({'symbol': symbol, 'date': query_date, 'avg_volume': avg_volume, 'rel_volume': rel_volume, 'floats': floats, 'index': index, 'pivots': pivots, 'pivots_D': pivots_D})

                        rsi_1 = indicators.get_indicator(df_1, 'rsi', query_time)
                        rsi_1 = round(indicators.get_indicator(df_1, 'rsi', query_time), 2)

                    except Exception as e:
                        print("Could not fetch saved data for symbol ", symbol, ". Error: ", e)
                        avg_volume, rel_volume, floats, index = '', '', '', ''
                        rsi1, rsi5, rsi15, rsi60 = '', '', '', ''
                        cpr_size_to_yst_perc, cpr_midpoint_to_yst_perc, cpr_size_to_yst_perc_D, cpr_midpoint_to_yst_perc_D = '', '', '', ''

                    try: atr_reached = "Y" if indicators.get_indicator(df_1, 'close', query_time) > indicators.get_indicator(df_1, 'atr_band_high', query_time) or indicators.get_indicator(df_1, 'close', query_time) < indicators.get_indicator(df_1, 'atr_band_low', query_time) else "N"
                    except Exception as e:
                        print("Could not fetch ATR. Error: ", e)
                        atr_reached = ''

                    # Get CPR size and level comparisons
                    cpr_size_to_yst_perc, cpr_midpoint_to_yst_perc = indicators.get_CPR(df_1, query_time, pivots)
                    cpr_size_to_yst_perc_D, cpr_midpoint_to_yst_perc_D = indicators.get_CPR(df_1, query_time, pivots_D)

                    pivot_type_used = '' # 'Non Daily'

                    # Write to csv file
                    text_line = [str(e).center(0) for e in entry.values()]
                    isLive = "N" if paperTrading else "Y"
                    for item in ["", "", isLive, journal_type, "", avg_volume, rel_volume, floats, index, str(rsi_1), str(rsi_5), str(rsi_15), str(rsi_60), atr_reached, str(cpr_size_to_yst_perc), str(cpr_midpoint_to_yst_perc), str(cpr_size_to_yst_perc_D), str(cpr_midpoint_to_yst_perc_D), pivot_type_used]:
                        text_line.append(item)
                    print(text_line)
                    print()
                    journal_writer.writerow(text_line)
        else:
            sum_entries = []
            factor_adjusted = 0.1
            for symbol in list(set([entry['symbol'] for entry in entries])):
                profit_loss = round(sum([float(entry['profit-loss']) for entry in entries if entry['symbol'] == symbol]), 2)
                commission = round(sum([float(entry['commissions']) for entry in entries if entry['symbol'] == symbol]), 2)
                sum_entries.append({'symbol': symbol, 'profit-loss': profit_loss, 'commissions': commission,
                                   'profit-loss_adjusted': round(profit_loss - (1-factor_adjusted) * commission, 2)})
            sum_entries.append({'symbol': 'Total', 'profit-loss': round(sum([entry['profit-loss'] for entry in sum_entries]), 2),
                                   'commissions': round(sum([entry['commissions'] for entry in sum_entries]), 2),
                                   'profit-loss_adjusted': round(sum([entry['profit-loss_adjusted'] for entry in sum_entries]), 2)})

            minimal_display_table = prettytable.PrettyTable()
            minimal_display_table.title = "P & L  " + str(datetime.datetime.fromisoformat(entries[0]['entry_time']).strftime('%Y-%m-%d'))
            minimal_display_table.field_names = ['Symbol', 'P & L', 'Commissions', 'P & L Adjusted (0.1)']
            for entry in sum_entries:
                minimal_display_table.add_row([entry['symbol'], entry['profit-loss'], entry['commissions'], entry['profit-loss_adjusted']])
            print('\n', minimal_display_table, '\n')
    else:
        print("\nNo entries found")


def record_data_symbol_in_play(symbols_in_play, timeframe, local_hist_folder):

    # Check timeframe
    if 'sec' in timeframe or 'min' in timeframe or 'hour' in timeframe: duration = '1 W'
    elif 'day' in timeframe: duration = '1 M'
    elif 'week' in timeframe: duration = '1Y'
    elif 'month' in timeframe: duration = '5Y'
    else: duration = ''

    query_time = helpers.date_to_EST_aware(datetime.datetime.fromisoformat(str(datetime.datetime.now())))
    for symbol in symbols_in_play:
        data_path = helpers.construct_hist_data_path(local_hist_folder, symbol, timeframe, duration)
        df = helpers.get_symbol_hist_data(ib, symbol, timeframe, query_time, duration=duration, indicators_list = ['rsi', 'atr_D', 'levels'])
        helpers.save_df_to_file(df, data_path)


def combineJournalCSV(journal_CSVFile_path, date=datetime.datetime.now()):

    path_date_folder = helpers.get_path_date_folder(date)
    date_str = path_date_folder[-8:]

    CSVFile_list = []
    for file in os.listdir(path_date_folder):
        if date_str + ".csv" in file:
            CSVFile_list.append(os.path.join(path_date_folder, file))

    if not CSVFile_list:
        print("\nNo .csv Entries file found in ", path_date_folder)

    else:
        # Read from CSV files entries
        rows_entries = []
        for CSVFile in CSVFile_list:
            with open(CSVFile) as input_file:
                journal_reader = csv.reader(input_file)
                for row in journal_reader:
                    rows_entries.append(row)

        # Sort entries by date
        rows_entries.sort(key=lambda x: x[1])

        # Write to CSV output file entries
        with open(journal_CSVFile_path, mode='w') as output_file:
            journal_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in rows_entries:
                print(row)
                journal_writer.writerow(row)


# def fill_stocks_data(file):

#     # Doc ezodf2: https://pypi.org/project/ezodf2/

#     ods = ezodf2.opendoc(filename=file)
#     sheet = ods.sheets["Journal"]
#     symbol = sheet[5,0].value
#     date = datetime.datetime.fromisoformat(sheet[5,1].value).strftime('%Y-%m-%d')
#     time = datetime.datetime.fromisoformat(sheet[5,1].value).strftime('%H:%M:%S')
#     print("symbol = ", symbol)
#     print("date = ", date)
#     print("time = ", time)
#     # sheet['AI6'].set_value("test")
#     ods.save()


if __name__ == "__main__":

    from utils import helpers
    import chart

    args = sys.argv

    journal_folder = constants.PATHS.folders_path['journal']
    journal_CSVFile_path = os.path.join(journal_folder, "Trading_Journal_Entries.csv") #"/Volumes/untitled/Trading/Trading_Journal/Trading_Journal_Entries.csv"
    download_folder = constants.PATHS.folders_path['download'] #"/Users/user/Downloads"
    # journal_ODSFile = "/Volumes/untitled/Trading/Trading_Journal/Trading_Journal copy.ods"
    path_journal = helpers.path_current_setup(journal_CSVFile_path)
    local_hist_folder = os.path.join(helpers.get_path_date_folder(), "hist_data_temp")
    action_conversion_IBKR = {"BOT": 1, "SLD": -1}
    action_conversion_TV = {"Buy": 1, "Sell": -1}

    # Script Arguments
    number_days_back, single_symbol, ibkr, paperTrading, chart_override, combine, move_history_file, minimal_display = 0, None, False, True, True, False, True, False
    if len(args) > 1:
        if 'combine' in args: combine = True
        if 'no_move'  in args: move_history_file = False
        if 'no_chart_override' in args: chart_override = False
        if 'minimal' in args: minimal_display = True
        if 'live' in args: paperTrading = False
        if 'ibkr' in args: ibkr = True
        for arg in args:
            if 'sym' in arg: single_symbol = arg[3:]
            if 'days_back' in arg:
                try: number_days_back = int(arg[9:])
                except Exception as e:
                    print("Wrong format for argument 'days_back'. Error: ", e)

    if combine:
        combineJournalCSV(journal_CSVFile_path)
    else:

        # TWS Connection
        ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)
        if not ibConnection:
            paperTrading = not paperTrading
            ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)

        if ibkr:

            if ibConnection:
                positions_symbols = [position.contract.symbol for position in ib.positions()]
                print("Current open positions: ", positions_symbols, "\n")

                fills_list = ib.fills()
                orders_list = ib.orders()
                ib.sleep(constants.CONSTANTS.PROCESS_TIME['short'])

                # for fill in ib.fills():
                #     print(fill)
                #     print(fill.contract.secType)
                #     print()
                # for fill in ib.orders():
                #     print(fill)
                #     print()
                # input()

                # # fills_list = [Fill(contract=Stock(conId=76792991, symbol='TSLA', exchange='DRCTEDGE', currency='USD', localSymbol='TSLA'), execution=Execution(execId='00025b46.6667d117.01.01', time=datetime.datetime(2024, 6, 11, 13, 30, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='DRCTEDGE', side='BOT', shares=5.0, price=173.93, permId=1319655275, clientId=0, orderId=0, liquidation=0, cumQty=5.0, avgPrice=173.93, orderRef='MktDepth', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1), commissionReport=CommissionReport(execId='00025b46.6667d117.01.01', commission=1.0, currency='USD', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 13, 30, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=76792991, symbol='TSLA', exchange='IBKRATS', currency='USD', localSymbol='TSLA'), execution=Execution(execId='0000dc8f.6667cbf1.01.01', time=datetime.datetime(2024, 6, 11, 13, 30, 5, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='IBKRATS', side='BOT', shares=10.0, price=173.75, permId=1319655306, clientId=0, orderId=0, liquidation=0, cumQty=10.0, avgPrice=173.75, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1), commissionReport=CommissionReport(execId='0000dc8f.6667cbf1.01.01', commission=1.0, currency='USD', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 13, 30, 5, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=76792991, symbol='TSLA', exchange='PEARL', currency='USD', localSymbol='TSLA'), execution=Execution(execId='00025b46.6667fe4e.01.01', time=datetime.datetime(2024, 6, 11, 14, 30, 1, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='PEARL', side='SLD', shares=10.0, price=169.65, permId=34098571, clientId=0, orderId=0, liquidation=0, cumQty=10.0, avgPrice=169.65, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b46.6667fe4e.01.01', commission=1.048823, currency='USD', realizedPNL=-43.982156, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 14, 30, 1, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=76792991, symbol='TSLA', exchange='IBKRATS', currency='USD', localSymbol='TSLA'), execution=Execution(execId='0000dc8f.666825b4.01.01', time=datetime.datetime(2024, 6, 11, 14, 45, 16, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='IBKRATS', side='SLD', shares=5.0, price=169.77, permId=34098584, clientId=0, orderId=0, liquidation=0, cumQty=5.0, avgPrice=169.77, orderRef='MktDepth', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='0000dc8f.666825b4.01.01', commission=1.024428, currency='USD', realizedPNL=-21.891095, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 14, 45, 16, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=208813720, symbol='GOOG', exchange='IEX', currency='USD', localSymbol='GOOG'), execution=Execution(execId='00025b44.6668132b.01.01', time=datetime.datetime(2024, 6, 11, 15, 19, 14, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='IEX', side='BOT', shares=5.0, price=175.95, permId=34098631, clientId=0, orderId=0, liquidation=0, cumQty=5.0, avgPrice=175.95, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b44.6668132b.01.01', commission=1.0, currency='USD', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 15, 19, 14, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=76792991, symbol='TSLA', exchange='NASDAQ', currency='USD', localSymbol='TSLA'), execution=Execution(execId='00025b46.6668194a.01.01', time=datetime.datetime(2024, 6, 11, 15, 41, 34, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='NASDAQ', side='BOT', shares=1.0, price=168.77, permId=34098686, clientId=0, orderId=0, liquidation=0, cumQty=1.0, avgPrice=168.77, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b46.6668194a.01.01', commission=1.0, currency='USD', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 15, 41, 34, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=76792991, symbol='TSLA', exchange='NASDAQ', currency='USD', localSymbol='TSLA'), execution=Execution(execId='00025b46.66681b51.01.01', time=datetime.datetime(2024, 6, 11, 15, 49, 11, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='NASDAQ', side='SLD', shares=1.0, price=168.36, permId=34098732, clientId=0, orderId=0, liquidation=0, cumQty=1.0, avgPrice=168.36, orderRef='MktDepth', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1), commissionReport=CommissionReport(execId='00025b46.66681b51.01.01', commission=1.004846, currency='USD', realizedPNL=-2.414846, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 15, 49, 11, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=208813720, symbol='GOOG', exchange='EDGEA', currency='USD', localSymbol='GOOG'), execution=Execution(execId='00025b44.66681f8d.01.01', time=datetime.datetime(2024, 6, 11, 15, 50, 58, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='EDGEA', side='SLD', shares=5.0, price=175.81, permId=34098738, clientId=0, orderId=0, liquidation=0, cumQty=5.0, avgPrice=175.81, orderRef='MktDepth', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b44.66681f8d.01.01', commission=1.025268, currency='USD', realizedPNL=-2.725268, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 15, 50, 58, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=208813720, symbol='GOOG', exchange='IBKRATS', currency='USD', localSymbol='GOOG'), execution=Execution(execId='0000dc8f.66682f04.01.01', time=datetime.datetime(2024, 6, 11, 15, 58, 33, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='IBKRATS', side='BOT', shares=3.0, price=175.89, permId=34098759, clientId=0, orderId=0, liquidation=0, cumQty=3.0, avgPrice=175.89, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='', commission=0.0, currency='', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 15, 58, 33, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=208813720, symbol='GOOG', exchange='NASDAQ', currency='USD', localSymbol='GOOG'), execution=Execution(execId='00025b44.66683280.01.01', time=datetime.datetime(2024, 6, 11, 16, 48, 50, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='NASDAQ', side='SLD', shares=3.0, price=176.0, permId=34098761, clientId=0, orderId=0, liquidation=0, cumQty=3.0, avgPrice=176.0, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1), commissionReport=CommissionReport(execId='00025b44.66683280.01.01', commission=1.015176, currency='USD', realizedPNL=-1.685176, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 16, 48, 50, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=272800, symbol='ORCL', exchange='ARCA', currency='USD', localSymbol='ORCL'), execution=Execution(execId='00025b49.666868ec.01.01', time=datetime.datetime(2024, 6, 11, 20, 14, 33, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='ARCA', side='SLD', shares=10.0, price=135.56, permId=1849208199, clientId=0, orderId=0, liquidation=0, cumQty=10.0, avgPrice=135.56, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b49.666868ec.01.01', commission=1.039346, currency='USD', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 20, 14, 33, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=272800, symbol='ORCL', exchange='NASDAQ', currency='USD', localSymbol='ORCL'), execution=Execution(execId='00025b49.66686980.01.01', time=datetime.datetime(2024, 6, 11, 20, 23, 27, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='NASDAQ', side='BOT', shares=10.0, price=133.4, permId=1849208220, clientId=0, orderId=0, liquidation=0, cumQty=10.0, avgPrice=133.4, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b49.66686980.01.01', commission=1.0, currency='USD', realizedPNL=19.560654, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 20, 23, 27, tzinfo=datetime.timezone.utc))]
                # fills_list = [Fill(contract=Stock(conId=483492393, symbol='MRVL', right='?', exchange='SMART', currency='USD', localSymbol='MRVL', tradingClass='NMS'), execution=Execution(execId='00025b45.666ab296.01.01', time=datetime.datetime(2024, 6, 13, 13, 57, 7, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='NASDAQ', side='SLD', shares=100.0, price=73.12, permId=1057681533, clientId=0, orderId=0, liquidation=0, cumQty=100.0, avgPrice=73.12, orderRef='ChartTrader2060815905', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1), commissionReport=CommissionReport(execId='00025b45.666ab296.01.01', commission=1.219874, currency='USD', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 13, 13, 57, 7, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=483492393, symbol='MRVL', right='?', exchange='SMART', currency='USD', localSymbol='MRVL', tradingClass='NMS'), execution=Execution(execId='00025b45.666ae9a8.01.01', time=datetime.datetime(2024, 6, 13, 15, 0, 23, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='NASDAQ', side='BOT', shares=100.0, price=72.45, permId=1057681537, clientId=0, orderId=0, liquidation=0, cumQty=100.0, avgPrice=72.45, orderRef='MktDepth', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b45.666ae9a8.01.01', commission=1.0, currency='USD', realizedPNL=64.780126, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 13, 15, 0, 23, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=483492393, symbol='MRVL', right='?', exchange='SMART', currency='USD', localSymbol='MRVL', tradingClass='NMS'), execution=Execution(execId='00025b45.666b0331.01.01', time=datetime.datetime(2024, 6, 13, 15, 35, 43, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='ARCA', side='BOT', shares=10.0, price=72.41, permId=1057681538, clientId=0, orderId=0, liquidation=0, cumQty=10.0, avgPrice=72.41, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b45.666b0331.01.01', commission=1.0, currency='USD', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 13, 15, 35, 43, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=483492393, symbol='MRVL', right='?', exchange='SMART', currency='USD', localSymbol='MRVL', tradingClass='NMS'), execution=Execution(execId='00025b45.666b03bd.01.01', time=datetime.datetime(2024, 6, 13, 15, 36, 35, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='IEX', side='SLD', shares=10.0, price=72.33, permId=1057681798, clientId=0, orderId=0, liquidation=0, cumQty=10.0, avgPrice=72.33, orderRef='MktDepth', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b45.666b03bd.01.01', commission=1.021768, currency='USD', realizedPNL=-2.821767, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 13, 15, 36, 35, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=76792991, symbol='TSLA', exchange='DRCTEDGE', currency='USD', localSymbol='TSLA'), execution=Execution(execId='00025b46.6667d117.01.01', time=datetime.datetime(2024, 6, 11, 13, 30, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='DRCTEDGE', side='BOT', shares=5.0, price=173.93, permId=1319655275, clientId=0, orderId=0, liquidation=0, cumQty=5.0, avgPrice=173.93, orderRef='MktDepth', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1), commissionReport=CommissionReport(execId='00025b46.6667d117.01.01', commission=1.0, currency='USD', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 13, 30, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=76792991, symbol='TSLA', exchange='IBKRATS', currency='USD', localSymbol='TSLA'), execution=Execution(execId='0000dc8f.6667cbf1.01.01', time=datetime.datetime(2024, 6, 11, 13, 30, 5, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='IBKRATS', side='BOT', shares=10.0, price=173.75, permId=1319655306, clientId=0, orderId=0, liquidation=0, cumQty=10.0, avgPrice=173.75, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1), commissionReport=CommissionReport(execId='0000dc8f.6667cbf1.01.01', commission=1.0, currency='USD', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 13, 30, 5, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=76792991, symbol='TSLA', exchange='PEARL', currency='USD', localSymbol='TSLA'), execution=Execution(execId='00025b46.6667fe4e.01.01', time=datetime.datetime(2024, 6, 11, 14, 30, 1, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='PEARL', side='SLD', shares=10.0, price=169.65, permId=34098571, clientId=0, orderId=0, liquidation=0, cumQty=10.0, avgPrice=169.65, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b46.6667fe4e.01.01', commission=1.048823, currency='USD', realizedPNL=-43.982156, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 14, 30, 1, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=76792991, symbol='TSLA', exchange='IBKRATS', currency='USD', localSymbol='TSLA'), execution=Execution(execId='0000dc8f.666825b4.01.01', time=datetime.datetime(2024, 6, 11, 14, 45, 16, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='IBKRATS', side='SLD', shares=5.0, price=169.77, permId=34098584, clientId=0, orderId=0, liquidation=0, cumQty=5.0, avgPrice=169.77, orderRef='MktDepth', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='0000dc8f.666825b4.01.01', commission=1.024428, currency='USD', realizedPNL=-21.891095, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 14, 45, 16, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=208813720, symbol='GOOG', exchange='IEX', currency='USD', localSymbol='GOOG'), execution=Execution(execId='00025b44.6668132b.01.01', time=datetime.datetime(2024, 6, 11, 15, 19, 14, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='IEX', side='BOT', shares=5.0, price=175.95, permId=34098631, clientId=0, orderId=0, liquidation=0, cumQty=5.0, avgPrice=175.95, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b44.6668132b.01.01', commission=1.0, currency='USD', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 15, 19, 14, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=76792991, symbol='TSLA', exchange='NASDAQ', currency='USD', localSymbol='TSLA'), execution=Execution(execId='00025b46.6668194a.01.01', time=datetime.datetime(2024, 6, 11, 15, 41, 34, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='NASDAQ', side='BOT', shares=1.0, price=168.77, permId=34098686, clientId=0, orderId=0, liquidation=0, cumQty=1.0, avgPrice=168.77, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b46.6668194a.01.01', commission=1.0, currency='USD', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 15, 41, 34, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=76792991, symbol='TSLA', exchange='NASDAQ', currency='USD', localSymbol='TSLA'), execution=Execution(execId='00025b46.66681b51.01.01', time=datetime.datetime(2024, 6, 11, 15, 49, 11, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='NASDAQ', side='SLD', shares=1.0, price=168.36, permId=34098732, clientId=0, orderId=0, liquidation=0, cumQty=1.0, avgPrice=168.36, orderRef='MktDepth', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1), commissionReport=CommissionReport(execId='00025b46.66681b51.01.01', commission=1.004846, currency='USD', realizedPNL=-2.414846, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 15, 49, 11, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=208813720, symbol='GOOG', exchange='EDGEA', currency='USD', localSymbol='GOOG'), execution=Execution(execId='00025b44.66681f8d.01.01', time=datetime.datetime(2024, 6, 11, 15, 50, 58, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='EDGEA', side='SLD', shares=5.0, price=175.81, permId=34098738, clientId=0, orderId=0, liquidation=0, cumQty=5.0, avgPrice=175.81, orderRef='MktDepth', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b44.66681f8d.01.01', commission=1.025268, currency='USD', realizedPNL=-2.725268, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 15, 50, 58, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=208813720, symbol='GOOG', exchange='IBKRATS', currency='USD', localSymbol='GOOG'), execution=Execution(execId='0000dc8f.66682f04.01.01', time=datetime.datetime(2024, 6, 11, 15, 58, 33, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='IBKRATS', side='BOT', shares=3.0, price=175.89, permId=34098759, clientId=0, orderId=0, liquidation=0, cumQty=3.0, avgPrice=175.89, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='', commission=0.0, currency='', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 15, 58, 33, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=208813720, symbol='GOOG', exchange='NASDAQ', currency='USD', localSymbol='GOOG'), execution=Execution(execId='00025b44.66683280.01.01', time=datetime.datetime(2024, 6, 11, 16, 48, 50, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='NASDAQ', side='SLD', shares=3.0, price=176.0, permId=34098761, clientId=0, orderId=0, liquidation=0, cumQty=3.0, avgPrice=176.0, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1), commissionReport=CommissionReport(execId='00025b44.66683280.01.01', commission=1.015176, currency='USD', realizedPNL=-1.685176, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 16, 48, 50, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=272800, symbol='ORCL', exchange='ARCA', currency='USD', localSymbol='ORCL'), execution=Execution(execId='00025b49.666868ec.01.01', time=datetime.datetime(2024, 6, 11, 20, 14, 33, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='ARCA', side='SLD', shares=10.0, price=135.56, permId=1849208199, clientId=0, orderId=0, liquidation=0, cumQty=10.0, avgPrice=135.56, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b49.666868ec.01.01', commission=1.039346, currency='USD', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 20, 14, 33, tzinfo=datetime.timezone.utc)), Fill(contract=Stock(conId=272800, symbol='ORCL', exchange='NASDAQ', currency='USD', localSymbol='ORCL'), execution=Execution(execId='00025b49.66686980.01.01', time=datetime.datetime(2024, 6, 11, 20, 23, 27, tzinfo=datetime.timezone.utc), acctNumber='DU8758392', exchange='NASDAQ', side='BOT', shares=10.0, price=133.4, permId=1849208220, clientId=0, orderId=0, liquidation=0, cumQty=10.0, avgPrice=133.4, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=2), commissionReport=CommissionReport(execId='00025b49.66686980.01.01', commission=1.0, currency='USD', realizedPNL=19.560654, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2024, 6, 11, 20, 23, 27, tzinfo=datetime.timezone.utc))]
                # orders_list = [Order(permId=1057681798, action='SELL', orderType='LMT', lmtPrice=72.33, auxPrice=0.0, tif='DAY', ocaType=3, orderRef='MktDepth', displaySize=2147483647, trailStopPrice=71.33, openClose='', volatilityType=0, deltaNeutralOrderType='None', referencePriceType=0, account='DU8758392', clearingIntent='IB', cashQty=0.0, dontUseAutoPriceForHedge=True, filledQuantity=10.0, refFuturesConId=2147483647, shareholder='Not an insider or substantial shareholder', parentPermId=9223372036854775807), Order(permId=1057681538, action='BUY', orderType='MKT', lmtPrice=72.37, auxPrice=0.0, tif='DAY', ocaType=3, displaySize=2147483647, trailStopPrice=73.37, openClose='', volatilityType=0, deltaNeutralOrderType='None', referencePriceType=0, account='DU8758392', clearingIntent='IB', cashQty=0.0, dontUseAutoPriceForHedge=True, filledQuantity=10.0, refFuturesConId=2147483647, shareholder='Not an insider or substantial shareholder', minCompeteSize=100, competeAgainstBestOffset=0.02), Order(permId=1057681765, action='SELL', totalQuantity=10.0, orderType='LMT', lmtPrice=73.37, auxPrice=0.0, tif='DAY', ocaGroup='1057681538', ocaType=3, displaySize=2147483647, trailStopPrice=73.37, openClose='', volatilityType=0, deltaNeutralOrderType='None', referencePriceType=0, account='DU8758392', clearingIntent='IB', cashQty=0.0, dontUseAutoPriceForHedge=True, filledQuantity=0.0, refFuturesConId=2147483647, shareholder='Not an insider or substantial shareholder', parentPermId=1057681538), Order(permId=1057681764, action='SELL', totalQuantity=10.0, orderType='STP', lmtPrice=72.37, auxPrice=71.37, tif='DAY', ocaGroup='1057681538', ocaType=3, displaySize=2147483647, trailStopPrice=71.37, openClose='', volatilityType=0, deltaNeutralOrderType='None', referencePriceType=0, account='DU8758392', clearingIntent='IB', cashQty=0.0, dontUseAutoPriceForHedge=True, filledQuantity=0.0, refFuturesConId=2147483647, shareholder='Not an insider or substantial shareholder', parentPermId=1057681538)]
                journal_entries, fills_entries = journalEntriesIBKR(fills_list, orders_list, number_days_back=number_days_back, single_symbol=single_symbol)

            else:
                print("\n\nCould not load IBKR. No Journal Entry created.\n\n")
                journal_entries, fills_entries = [], []

        else:

            if move_history_file: moveCSVHistory(download_folder, helpers.get_path_date_folder())
            # trading_history_TV_CSVFile = "/Volumes/untitled/Trading/Trading Journal/trading_history_TV.csv"
            journal_entries, fills_entries = journalEntriesTV(number_days_back=number_days_back, single_symbol=single_symbol)

            # for entry in fills_entries:
            #     print(entry["symbol"])
            #     for fill in entry["fills"]:
            #         print(fill)
            #     print("\n----------------------------------------------------------\n")
        journalToCSV(ib, journal_entries, ibkr, paperTrading, local_hist_folder, minimal_display=minimal_display)

        # Create charts and screenshots
        if ibConnection and not minimal_display:

            symbols_list = [entry["symbol"] for entry in journal_entries]# if not single_symbol else [single_symbol]

            for symbol in list(set(symbols_list)): # Iterate over list of symbols in play, duplicates removed

                # duration = str(round((queryTime - startTime).total_seconds())) + " S"
                duration = "1 W"

                for timeframe in ["1 min"]:#, "5 mins"]:

                    tf = int(timeframe[:timeframe.index(" ")])
                    max_timedelta = 2 * tf
                    display_offset_time = 30 * tf

                    # Verify if trades are not too far appart, otherwise create different charts
                    symbol_entries_list = [entry for entry in journal_entries if entry["symbol"] == symbol]
                    start_time = [helpers.date_to_EST_aware(datetime.datetime.fromisoformat(symbol_entries_list[0]["entry_time"])) - datetime.timedelta(minutes=display_offset_time)]
                    end_time = []
                    for entry_index, entry in enumerate(symbol_entries_list):

                        entry_time = helpers.date_to_EST_aware(datetime.datetime.fromisoformat(entry["entry_time"]))
                        close_time = helpers.date_to_EST_aware(datetime.datetime.fromisoformat(symbol_entries_list[entry_index-1]["close_time"]))
                        if entry_index > 0 and close_time - entry_time > datetime.timedelta(hours=max_timedelta):
                            start_time.append(entry_time - datetime.timedelta(minutes=display_offset_time))
                            end_time.append(min(close_time + datetime.timedelta(minutes=display_offset_time), helpers.date_local_to_EST(datetime.datetime.now())))


                    end_time.append(min(helpers.date_to_EST_aware(datetime.datetime.fromisoformat(symbol_entries_list[len(symbol_entries_list)-1]["close_time"])) + datetime.timedelta(minutes=display_offset_time), helpers.date_local_to_EST(datetime.datetime.now())))

                    print("\nCreating charts for symbol ", symbol, "...")
                    screenshot_folder = helpers.get_path_date_folder()
                    for index_time, _time in enumerate(start_time):

                        # Create Marker list
                        marker_list = []
                        fills_list = [entry for entry in fills_entries if entry["symbol"] == symbol][0]["fills"]

                        for fill in fills_list:
                            if fill["price"] != "" and fill["time"] > start_time[index_time] and fill["time"] < end_time[index_time]:
                                # print(fill)
                                if fill["action"] == "Buy" or fill["action"] == "BOT":
                                    position, shape, color = "below", "arrow_up", "green"
                                elif fill["action"] == "Sell" or fill["action"] == "SLD":
                                    position, shape, color = "above", "arrow_down", "red"
                                text = str(fill["quantity"]) + "@" + str(fill["price"])
                                marker_list.append({"time":fill["time"], "position":position, "shape":shape, "color":color, "text": text})

                        # Create screenshot filename and handle case of several chart per symbol
                        additional_path_name = ""
                        date_now_str = datetime.datetime.now().strftime('%Y%m%d')
                        if paperTrading:
                            if ibkr: additional_path_name += "_IBKR"
                            else: additional_path_name += "_TV"
                            additional_path_name = "_paper"
                        if len(start_time) > 1: additional_path_name = "_" + str(index_time + 1)
                        screenshot_filename = date_now_str + " " + symbol + " " + timeframe + additional_path_name + ".jpg"
                        screenshot_path = os.path.join(screenshot_folder, screenshot_filename)

                        if chart_override or (not chart_override and not os.path.isfile(screenshot_path)):

                            df = helpers.get_symbol_hist_data(ib, symbol, timeframe=timeframe, query_time=end_time[index_time], duration=duration, indicators_list=["emas", "vwap", "macd", "rsi", "atr", "levels"])
                            chart.create_chart(df=df, symbol=symbol, timeframe=timeframe, start_time=start_time[index_time], end_time=end_time[index_time], marker_list=marker_list, screenshot_path=screenshot_path, show_time=1)

                        else:
                            print("\nChart already exists or override deactivated\n")
        else:
            print("\n\nCould not load IBKR. No chart created.\n\n")

        ib.disconnect()









    # bars = ib.reqHistoricalData(
    # 	contract, endDateTime='', durationStr='30 D',
    # 	barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)

    # # convert to pandas dataframe (pandas needs to be installed):
    # df = util.df(bars)
    # print(df)

    print("\n\n")
    # input("\nEnter anything to exit")