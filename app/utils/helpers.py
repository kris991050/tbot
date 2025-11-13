import os, sys, datetime, time, requests, csv, re, prettytable, numpy as np, json, speedtest, pandas_market_calendars
import xml.etree.ElementTree as ET, traceback, speedtest, typing, chardet, pickle, filelock, yfinance, polygon, re
import math, pandas as pd, psutil, gc#, dask.dataframe as dd
from datetime import datetime, timedelta
from ib_insync import *

current_folder = os.path.dirname(os.path.realpath(__file__))
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)
# sys.path.append(current_folder)
from miscellaneous import newsScraper
from utils.timeframe import Timeframe
from utils.constants import CONSTANTS, PATHS, FORMATS

# Doc ib_insync: https://ib-insync.readthedocs.io/api.html
# Doc IBKR API: https://interactivebrokers.github.io/tws-api/introduction.html
# util.startLoop()  # uncomment this line when in a notebook
# Doc TA library: https://technical-analysis-library-in-python.readthedocs.io/en/latest/
# Available timframes and duurations for historical data: https://interactivebrokers.github.io/tws-api/historical_bars.html



# ======================
# Miscellaneous
# ======================


def sleep_display(time_wait, ib=None):
    print("⌛ Waiting ", time_wait, " sec...")
    for i in range(time_wait):
        print("\r{} seconds.".format(time_wait - i), end='')
        if not ib: time.sleep(1)
        else: ib.sleep(1)

# def print_timer(msg, overwrite=False, log_intermediate=False):
#     if sys.stdout.isatty() and overwrite:
#         print(f"\r{msg}", end='', flush=True)
#     elif log_intermediate or not overwrite:
#         print(msg, flush=True)


def get_page(url):
    """Download a webpage and return a beautiful soup doc"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36'}
    response = requests.get(url, headers=headers)

    try:
        if not response.ok:
            print('Status code:', response.status_code)
            raise Exception('Failed to load page {}'.format(url))
        page_content = response.text
        doc = BeautifulSoup(page_content, 'html.parser')
    except Exception as e:
        print("Page did not load. Error: ", e, "  |  Full error: ", traceback.format_exc())
        doc = None
    return doc


def convert_large_numbers(number_text):

    factor_converter = {"K": 1000, "M": 1000000, "B": 1000000000}
    lenNum = len(number_text)-1
    for factor in factor_converter.keys():
        if number_text[lenNum] == factor:
            return float(number_text[:lenNum]) * factor_converter[factor]

    return float(number_text)


def IBKRConnect_any(ib:IB=IB(), paper:bool=True, client_id:int=None, remote:bool=False):
    # remote = True
    ib, ibConnection = IBKRConnect(ib, paper=paper, client_id=client_id, remote=remote)
    if not ibConnection:
        ib, ibConnection = IBKRConnect(IB(), paper=not paper, client_id=client_id, remote=remote)

    return ib, ib.isConnected()


def IBKRConnect(ib:IB=IB(), paper:bool=True, client_id:int=None, remote:bool=False):
    ib_ip = CONSTANTS.IB_IP_REMOTE if remote else CONSTANTS.IB_IP_LOCAL
    port_number = CONSTANTS.IB_PORT_PAPER if paper else CONSTANTS.IB_PORT_LIVE
    max_clientId = 10 if not client_id else 1
    clientId = 1 if not client_id else client_id
    
    try:
        while clientId <= max_clientId and not ib.isConnected():
            try:
                ib.connect(ib_ip, port_number, clientId=clientId, timeout=2)
            except Exception as e:
                print("\n\n📵 Could not connect to IB at port # ", port_number, " with client Id ", ib.client.clientId, ". Error: ", e, "\n\n")
                clientId += 1

        print("IBKR Connection Status: ", ib.isConnected(), ": ", ib, "\n")

    except Exception as e:
        print("\n\nCould not load IBKR at port # ", port_number, ". Error: ", e, "\n\n")

    return ib, ib.isConnected()


def test_internet_speed():
    try:
        st = speedtest.Speedtest()
        print("Testing internet speed...")

        # Perform the download and upload speed tests
        download_speed = st.download() / 1000000  # Convert to Mbps
        upload_speed = st.upload() / 1000000  # Convert to Mbps

        # Print the results
        print("Download Speed: {:.2f} Mbps".format(download_speed))
        print("Upload Speed: {:.2f} Mbps".format(upload_speed))

        return download_speed, upload_speed

    except Exception as e:
        print("An error occurred during the speed test:", str(e))
        return None, None


def handle_ib_error_factory(error_list):
    """
    Returns a reusable IB error handler that appends to error_list.
    """
    def handle_ib_error(reqId, errorCode, errorString, contract=None):
        error_list.append((reqId, errorCode, errorString, contract))

    return handle_ib_error


def set_var_with_constraints(var, allowed_list):
    if var not in allowed_list:
        raise ValueError(f"Invalid mode: {var}. Must be from {allowed_list}")
    else:
        return var


def get_stock_currency_yf(symbol:str):
    try:
        stock = yfinance.Ticker(symbol)
        info = stock.info
        exchange = info.get('exchange', 'Unknown')
        # Determine currency based on the exchange
        if exchange == 'NASDAQ' or exchange == 'NYSA':
            return 'USD', exchange
        elif exchange == 'LSE':
            return 'GBP', exchange
        elif exchange == 'FRA':
            return 'EUR', exchange
        elif exchange == 'TSX':
            return 'CAD', exchange
        else:
            print(f"Unknown currency for {symbol}")
            return None, exchange
    except Exception as e:
        print(f"Could not fetch currency from Yahoo Finace. Error: {e}" + "  |  Full error: ", traceback.format_exc())



# ======================
# Files and Folder paths
# ======================


def get_path_date_folder(date=None, create_if_none=True, local=False):

    date = date or datetime.now(CONSTANTS.TZ_WORK)
    if isinstance(date, str):
        dateISO = date[0:4] + "-" + date[4:6] + "-" + date[6:]
    else:
        dateISO = date.strftime('%Y-%m-%d')
    date = dateISO.replace("-", "")

    if not local: root_folder_path = PATHS.folders_path['journal']
    else: root_folder_path = os.getcwd()
    date_folder_path = os.path.join(root_folder_path, date)

    if not os.path.exists(date_folder_path) and create_if_none:
        os.mkdir(date_folder_path)

    return date_folder_path


def get_path_daily_data_folder(date=None, create_if_none=True, local=False):

    date = date or datetime.now(CONSTANTS.TZ_WORK)
    if not local: root_folder_path = get_path_date_folder(date, create_if_none, local)
    else: root_folder_path = os.getcwd()
    daily_data_folder_path = os.path.join(root_folder_path, 'daily_data')

    if not os.path.exists(daily_data_folder_path) and create_if_none:
        os.mkdir(daily_data_folder_path)

    return daily_data_folder_path


def path_current_setup(path_current_file, ch_dir=True, print_path=True):

    # Current Directory Path Setup
    #name = os.path.basename(__file__)
    path_current = os.path.dirname(path_current_file)
    print("path_current = ", path_current)
    if ch_dir: os.chdir(path_current)
    if print_path: print("\nCurrent Directory = ", os.getcwd())

    return path_current


# def construct_hist_data_path(local_hist_folder, symbol, timeframe, to_time, from_time, file_format='csv'):

#     # return str(local_hist_folder) + '/hist_data_' + symbol + '_' + timeframe + '_' + duration + '.csv'
#     to_time = pd.to_datetime(to_time, utc=True).tz_convert(CONSTANTS.TZ_WORK)
#     from_time = pd.to_datetime(from_time, utc=True).tz_convert(CONSTANTS.TZ_WORK)

#     start_str = to_time.strftime(FORMATS.MKT_DATA_FILENAME_DATETIME_FMT)
#     end_str = from_time.strftime(FORMATS.MKT_DATA_FILENAME_DATETIME_FMT)

#     filename = f"hist_data_{symbol}_{timeframe}_{end_str}_{start_str}.{file_format}"
#     return os.path.join(local_hist_folder, filename)

#     # return str(local_hist_folder) + '/hist_data_' + symbol + '_' + timeframe + '_' + pd.to_datetime(from_time).strftime('%Y-%m-%d-%H-%M-%S') + '_' + pd.to_datetime(to_time).strftime('%Y-%m-%d-%H-%M-%S') + '.csv'
#     # return str(local_hist_folder) + '/hist_data_' + symbol + '_' + timeframe + '_' + from_time + '_' + to_time + '.csv'

def construct_data_path(local_hist_folder, symbol, timeframe, to_time, from_time, file_format='csv', data_type='hist_data'):
    filename = FORMATS.construct_filename(symbol=symbol, timeframe=timeframe, to_time=to_time, from_time=from_time, file_format=file_format, data_type=data_type)
    return os.path.join(local_hist_folder, filename)


# def build_hist_data_filename_pattern(timeframe: str='1 min', file_format: str='csv') -> re.Pattern:
#     date_regex = FORMATS.MKT_DATA_FILENAME_DATETIME_REGEX
#     return re.compile(
#         rf"^hist_data_(?P<symbol>[A-Z]+)_{timeframe}_"
#         rf"(?P<end>{date_regex})_"
#         rf"(?P<start>{date_regex})\.{file_format}$"
#     )


def build_data_filename_pattern(timeframe:Timeframe=None, file_format:str=None, data_type:str=None) -> re.Pattern:
    timeframe = timeframe or Timeframe()
    file_format = file_format or FORMATS.DEFAULT_FILE_FORMAT

    date_regex = FORMATS.MKT_DATA_FILENAME_DATETIME_REGEX
    template = FORMATS.DATA_FILENAME_TEMPLATE

    pattern = template \
        .replace('.', r'\.') \
        .replace('{symbol}', r'(?P<symbol>[A-Z]+)') \
        .replace('{timeframe}', re.escape(timeframe.pandas)) \
        .replace('{from_str}', f"(?P<from_str>{date_regex})") \
        .replace('{to_str}', f"(?P<to_str>{date_regex})") \
        .replace('{file_format}', re.escape(file_format)) \
        .replace('{data_type}', re.escape(data_type))

    return re.compile(f"^{pattern}$")

# def build_hist_data_filename_pattern(timeframe:Timeframe=None, file_format:str=None) -> re.Pattern:
#     timeframe = timeframe or Timeframe()
#     file_format = file_format or FORMATS.DEFAULT_FILE_FORMAT

#     date_regex = FORMATS.MKT_DATA_FILENAME_DATETIME_REGEX
#     template = FORMATS.HIST_MKT_DATA_FILENAME_TEMPLATE

#     pattern = template \
#         .replace('.', r'\.') \
#         .replace('{symbol}', r'(?P<symbol>[A-Z]+)') \
#         .replace('{timeframe}', re.escape(timeframe.pandas)) \
#         .replace('{from_str}', f"(?P<from_str>{date_regex})") \
#         .replace('{to_str}', f"(?P<to_str>{date_regex})") \
#         .replace('{file_format}', re.escape(file_format))

#     return re.compile(f"^{pattern}$")


# def build_enriched_data_filename_pattern(timeframe:Timeframe=None, file_format:str=None) -> re.Pattern:
#     timeframe = timeframe or Timeframe()
#     file_format = file_format or FORMATS.DEFAULT_FILE_FORMAT

#     date_regex = FORMATS.MKT_DATA_FILENAME_DATETIME_REGEX
#     template = FORMATS.ENRICHED_MKT_DATA_FILENAME_TEMPLATE

#     pattern = template \
#         .replace('.', r'\.') \
#         .replace('{symbol}', r'(?P<symbol>[A-Z]+)') \
#         .replace('{timeframe}', re.escape(timeframe.pandas)) \
#         .replace('{from_str}', f"(?P<from_str>{date_regex})") \
#         .replace('{to_str}', f"(?P<to_str>{date_regex})") \
#         .replace('{file_format}', re.escape(file_format))

#     return re.compile(f"^{pattern}$")


def save_json(obj, path, lock=False):
    if lock:
        lock_file = filelock.FileLock(f"{path}.lock")
        with lock_file:  # Acquires the lock here
            with open(path, 'w') as f:
                json.dump(obj, f, indent=2, default=str)  # str conversion for datetimes
    else:
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2, default=str)  # str conversion for datetimes


# def load_json(path, lock=False):
#     if lock:
#         lock_file = filelock.FileLock(f"{path}.lock")
#         with lock_file:  # Acquires the lock here
#             with open(path, 'r') as f:
#                 return json.load(f)
#     else:
#         with open(path, 'r') as f:
#             return json.load(f)


def load_json(path, lock=False, lock_timeout=10):
    """
    Load JSON from a file with an optional lock mechanism.
    If `lock` is True, a file lock will be used.

    :param path: Path to the JSON file
    :param lock: Whether to lock the file during reading
    :param lock_timeout: Time in seconds to wait for the lock (default: 10)
    :return: Parsed JSON data
    """
    if lock:
        lock_file = filelock.FileLock(f"{path}.lock")
        to_time = time.time()

        # Attempt to acquire the lock with retry mechanism
        while True:
            try:
                with lock_file:  # Acquires the lock here
                    with open(path, 'r') as f:
                        return json.load(f)
            except filelock.Timeout:
                # Retry if the lock is not available
                if time.time() - to_time > lock_timeout:
                    raise TimeoutError(f"Could not acquire lock for {path} within {lock_timeout} seconds.")
                time.sleep(0.5)  # Wait a bit before retrying
    else:
        with open(path, 'r') as f:
            return json.load(f)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ======================
# CSV Files
# ======================


def read_csv_file(file_csv_path):

    rows_list = []
    if os.path.exists(file_csv_path):
        with open(file_csv_path) as file:
            file_reader = csv.reader(file)
            for row in file_reader: rows_list.append(row)
        file.close()

    return rows_list


# def read_data_file(file_path: str):

#     path = pathlib.Path(file_path)
#     ext = path.suffix.lower().lstrip('.')

#     if ext not in FORMATS.DATA_FILE_FORMATS_LIST:
#         raise ValueError(f"Unsupported file extension: .{ext}")

#     if ext in FORMATS.DATA_FILE_FORMATS['csv']:
#         return pd.read_csv(path)
#     elif ext in FORMATS.DATA_FILE_FORMATS['pickle']:
#         return pd.read_pickle(path)
#     elif ext in FORMATS.DATA_FILE_FORMATS['parquet']:
#         return pd.read_parquet(path)
#     else:
#         # This branch shouldn't be reached due to the check above
#         raise ValueError(f"Unhandled file extension: .{ext}")


def change_file_format(file_path: typing.Union[str, os.PathLike], format_to: str) -> str:
    """
    Change the file extension of the given file path to the desired format.
    """
    if not isinstance(file_path, (str, os.PathLike)): raise TypeError("file_path must be a string or os.PathLike")
    if not isinstance(format_to, str): raise TypeError("format_to must be a string")
    if not format_to.strip(): raise ValueError("format_to must not be empty")

    file_path = str(file_path)
    base, ext = os.path.splitext(file_path)

    if not ext: raise ValueError(f"The file path '{file_path}' does not have an extension to replace.")

    new_extension = format_to.lstrip('.')
    return f"{base}.{new_extension}"


def initialize_symbols_csv_file(file_csv_path, title_row):

    # If CSV file does not exist, create it and write first title row
    if not os.path.exists(file_csv_path):
        with open(file_csv_path, mode='w') as file:
            file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(title_row)
            print("\nRecorded title row at ", file_csv_path, "\n")
        file.close()


def detect_file_encoding(filepath, verbose=False):
    result = None
    with open(filepath, 'rb') as f:
        sample = f.read(10000)
        result = chardet.detect(sample)

    # Fallback to full read if confidence is too low
    if result['confidence'] < 0.5:
        with open(filepath, 'rb') as f:
            result = chardet.detect(f.read())

    if verbose: print(f"Detected encoding: {result['encoding']} (confidence: {result['confidence']:.2f})")
    return result


# def check_existing_mkt_data_file(symbol, timeframe, folder, delete_file=False, file_format='csv'):

#     df, to_time, from_time, file_path = pd.DataFrame(), None, None, None

#     if file_format not in FORMATS.DATA_FILE_FORMATS_LIST:
#         print(f"Format must be {FORMATS.DATA_FILE_FORMATS_LIST}.")
#         return df, to_time, from_time, file_path

#     if not os.path.isdir(folder):
#         print(f"{folder} is not a directory")
#         return df, to_time, from_time, file_path

#     file_ext = '.' + file_format
#     filename_pattern = build_hist_data_filename_pattern(timeframe, file_format=file_format)

#     for file in os.listdir(folder):
#         if not file.endswith(file_ext):
#             continue

#         match = filename_pattern.match(file)
#         if match and match.group("symbol") == symbol:
#             file_path = os.path.join(folder, file)

#             # to_time = pd.to_datetime(match.group("start")).strftime(FORMATS.MKT_DATA_FILENAME_DATETIME_FMT)
#             to_time = pd.to_datetime(match.group("start"),
#                                         format=FORMATS.MKT_DATA_FILENAME_DATETIME_FMT,
#                                         utc=True).tz_convert(CONSTANTS.TZ_WORK)
#             # from_time = pd.to_datetime(match.group("end")).strftime(FORMATS.MKT_DATA_FILENAME_DATETIME_FMT)
#             from_time = pd.to_datetime(match.group("end"),
#                                     format=FORMATS.MKT_DATA_FILENAME_DATETIME_FMT,
#                                     utc=True).tz_convert(CONSTANTS.TZ_WORK)

#             df = load_df_from_file(file_path)
#             df = format_df_date(df)

#             # df_matched_list.append({'df': df, 'to_time': to_time, 'from_time': from_time, 'file_path': file_path})

#             break  # Exit after first match


#         if delete_file:
#             os.remove(file_path)
#             print("Removed file", file_path)

#     return df, to_time, from_time, file_path


def check_existing_data_file(symbol:str, timeframe:Timeframe, folder:str, data_type:str, delete_file:bool=False, file_format:str=None):

    df, to_time, from_time, file_path = pd.DataFrame(), None, None, None

    if file_format and file_format not in FORMATS.DATA_FILE_FORMATS_LIST:
        print(f"Format must be {FORMATS.DATA_FILE_FORMATS_LIST}.")
        return df, to_time, from_time, file_path

    if not os.path.isdir(folder):
        print(f"{folder} is not a directory")
        return df, to_time, from_time, file_path

    matched_files = []
    file_formats = FORMATS.DATA_FILE_FORMATS_LIST if not file_format else [file_format]
    data_type_formats = FORMATS.DATA_TYPE_FORMATS
    # if data_type == 'hist':
    #     file_formats = FORMATS.DATA_FILE_FORMATS_LIST if not file_format else [file_format]
    # elif data_type == 'enriched':
    #     file_formats = FORMATS.ENRICHED_MKT_DATA_FILENAME_TEMPLATE if not file_format else [file_format]
    # else: file_formats = []

    for file_format in file_formats:
        file_ext = '.' + file_format

        if data_type in data_type_formats:
            filename_pattern = build_data_filename_pattern(timeframe, file_format=file_format, data_type=data_type)
        # if data_type == 'hist':
        #     filename_pattern = build_data_filename_pattern(timeframe, file_format=file_format)
        # elif data_type == 'enriched':
        #     filename_pattern = build_data_filename_pattern(timeframe, file_format=file_format)
        else:
            raise ValueError(f"Unsupported data type '{data_type}'. Supported: {data_type_formats}")
            # filename_pattern = None

        for file in os.listdir(folder):
            if not file.endswith(file_ext):
                continue

            match = filename_pattern.match(file)
            if match and match.group("symbol") == symbol:
                current_file_path = os.path.join(folder, file)

                current_to_time = pd.to_datetime(match.group("to_str"), format=FORMATS.MKT_DATA_FILENAME_DATETIME_FMT
                                                    ).tz_localize(CONSTANTS.TZ_WORK)

                current_from_time = pd.to_datetime(match.group("from_str"), format=FORMATS.MKT_DATA_FILENAME_DATETIME_FMT,
                                                  ).tz_localize(CONSTANTS.TZ_WORK)

                matched_files.append({
                    'file_path': current_file_path,
                    'to_time': current_to_time,
                    'from_time': current_from_time,
                    'file_format': file_format,
                    'time_range': current_to_time - current_from_time
                })

    if not matched_files:
        return df, to_time, from_time, file_path

    # Select file with the largest time range (most data)
    selected = max(matched_files, key=lambda x: x['time_range'])

    df = load_df_from_file(selected['file_path'])
    df = format_df_date(df)

    if delete_file:
        os.remove(selected['file_path'])
        print(f"Removed file {selected['file_path']}")

    return df, selected['to_time'], selected['from_time'], selected['file_path']


# def check_existing_enriched_data_file(symbol, timeframe, folder, file_format=None):
#     df, to_time, from_time, file_path = pd.DataFrame(), None, None, None
#     if file_format and file_format not in FORMATS.DATA_FILE_FORMATS_LIST:
#         print(f"Format must be {FORMATS.DATA_FILE_FORMATS_LIST}.")
#         return df, to_time, from_time, file_path

#     if not os.path.isdir(folder):
#         print(f"{folder} is not a directory")
#         return df, to_time, from_time, file_path

#     matched_files = []
#     file_formats = FORMATS.DATA_FILE_FORMATS_LIST if not file_format else [file_format]

#     for file_format in file_formats:
#         file_ext = '.' + file_format
#         filename_pattern = build_hist_data_filename_pattern(timeframe, file_format=file_format)

#         for file in os.listdir(folder):
#             if not file.endswith(file_ext):
#                 continue
#             match = filename_pattern.match(file)
#             if match and match.group("symbol") == symbol and 'enriched' in file:
#                 current_file_path = os.path.join(folder, file)
#                 current_to_time = pd.to_datetime(match.group("start"), format=FORMATS.MKT_DATA_FILENAME_DATETIME_FMT,
#                                                     utc=True).tz_convert(CONSTANTS.TZ_WORK)
#                 current_from_time = pd.to_datetime(match.group("end"), format=FORMATS.MKT_DATA_FILENAME_DATETIME_FMT,
#                                                   utc=True).tz_convert(CONSTANTS.TZ_WORK)
#                 matched_files.append({
#                     'file_path': current_file_path,
#                     'to_time': current_to_time,
#                     'from_time': current_from_time,
#                     'file_format': file_format,
#                     'time_range': current_to_time - current_from_time
#                 })

#     if not matched_files:
#         return df, to_time, from_time, file_path

#     selected = max(matched_files, key=lambda x: x['time_range'])

#     df = load_df_from_file(selected['file_path'])
#     df = format_df_date(df)

#     return df, selected['to_time'], selected['from_time'], selected['file_path']

    # if format in ['csv', 'pkl', 'parquet']:

    #     l = len(format)
    #     file_extension = '.' + format
    #     for file in os.listdir(folder):
    #         if (symbol + '_' + timeframe + '_') in file and file_extension in file:
    #             file_path = os.path.join(folder, file)
    #             to_time = date_to_EST_aware(datetime.fromisoformat(str(file[-(l+20):-(l+10)] + 'T' + file[-(l+9):-(l+7)] + ':' + file[-(l+6):-(l+4)] + ':' + file[-(l+3):-(l+1)]) + '.000000'))
    #             from_time = date_to_EST_aware(datetime.fromisoformat(str(file[-(l+40):-(l+30)] + 'T' + file[-(l+29):-(l+27)] + ':' + file[-(l+26):-(l+24)] + ':' + file[-(l+23):-(l+21)]) + '.000000'))
    #             # df = pd.read_csv(file_path)
    #             df = load_df_from_file(file_path)
    #             df = format_df_date(df)
    #             if delete_file:
    #                 os.remove(file_path)
    #                 print("Removed file ", file_path)

    # else: print("Format must be 'csv' or 'pkl'.")

    # return df, to_time, from_time, file_path


def save_df_to_file(df, data_path, file_format='csv'):

    if df.empty:
        print("Dataframe empty. Data not saved.")
    else:
        # Case local folder does not exist
        folder = os.path.dirname(data_path)
        if not os.path.exists(folder):
            try:
                os.mkdir(folder)
                print(f"Directory '{folder}' created successfully.")
            except Exception as e: print(f"Could not create directory. Error: {e}" + "  |  Full error: ", traceback.format_exc())

        # If local folder exists, create or append data
        if os.path.exists(folder):

            # # Case Data already exist, append Data
            if False: # os.path.exists(data_path):
                print()
                # see https://www.geeksforgeeks.org/how-to-append-pandas-dataframe-to-existing-csv-file/

            # Case Data do not yet exist, create Data file
            else:
                df.attrs = make_attrs_json_safe(df.attrs)

                if file_format == 'csv' and '.csv' in data_path and not '.pkl' in data_path and not '.parquet' in data_path:
                    df.to_csv(data_path, index=False)
                    print("\nData file created at " + data_path + "\n")
                elif file_format == 'pkl' and not '.csv' in data_path and '.pkl' in data_path and not '.parquet' in data_path:
                    df.to_pickle(data_path)
                    print("\nData file created at " + data_path + "\n")
                elif file_format == 'parquet' and not '.csv' in data_path and not '.pkl' in data_path and '.parquet' in data_path:
                    compression_type = 'brotli' # 'zstd'
                    df.to_parquet(data_path, compression=compression_type, index=False)

                    print("\nData file created at " + data_path + "\n")
                else:
                    print("\nWrong file format for ", file_format,  " and / or ", data_path + "\n")


# def load_df_from_file(filepath, **kwargs):
#     """
#     Load a DataFrame from a file by automatically detecting the format
#     based on the file extension (.csv, .pkl, .parquet).

#     Parameters:
#         filepath (str): Path to the input file.
#         **kwargs: Additional arguments passed to the read function.

#     Returns:
#         pd.DataFrame: Loaded DataFrame.
#     """
#     ext = os.path.splitext(filepath)[1].lower()

#     if ext == '.csv':
#         return pd.read_csv(filepath, low_memory=False, **kwargs)
#     elif ext in ['.pkl', '.pickle']:
#         return pd.read_pickle(filepath)
#     elif ext == '.parquet':
#         return pd.read_parquet(filepath, **kwargs)
#     else:
#         raise ValueError(f"Unsupported file extension '{ext}'. Supported: .csv, .pkl, .parquet")


def load_df_from_file(filepath, **kwargs):
    """
    Load a DataFrame from a file by automatically detecting the format
    based on the file extension using CONSTANTS.DATA_FILE_FORMATS.

    Parameters:
        filepath (str): Path to the input file.
        **kwargs: Additional arguments passed to the read function.

    Returns:
        pd.DataFrame: Loaded DataFrame with restored attrs if possible.
    """
    if not filepath or not os.path.exists(filepath):
        return pd.DataFrame()

    ext = os.path.splitext(filepath)[1].lower().lstrip('.')  # e.g. 'csv' not '.csv'
    format_map = FORMATS.DATA_FILE_FORMATS


    try:
        if ext in format_map['csv']:
            file_encoding = detect_file_encoding(filepath)
            df = pd.read_csv(filepath, encoding=file_encoding['encoding'], low_memory=False, **kwargs)
        elif ext in format_map['pickle']:
            df = pd.read_pickle(filepath)
        elif ext in format_map['parquet']:
            df = pd.read_parquet(filepath, **kwargs)
        else:
            supported = [e for sublist in format_map.values() for e in sublist]
            raise ValueError(f"Unsupported file extension '{ext}'. Supported: {supported}")
    except Exception as e:
        print(f"Could not read {filepath} using encoding {file_encoding}. Error: ", e)
        return pd.DataFrame()


    # Try to restore JSON-safe attrs (only if they exist)
    if hasattr(df, 'attrs') and df.attrs:
        df.attrs = restore_attrs_from_json_safe(df.attrs)

    return df


def save_to_daily_csv(ib, symbols, file_name):

    daily_data_folder = get_path_daily_data_folder()
    symbols_csv_path = os.path.join(daily_data_folder, file_name)

    if symbols:
        initialize_symbols_csv_file(symbols_csv_path, ['Time', 'Symbol', 'Avg Volume', 'Rel Volume', 'Floats', 'Market Cap', 'Index', 'Last Earning', 'News'])

        # Record existing symbols from file
        symbols_existing = []
        for i, row in enumerate(read_csv_file(symbols_csv_path)):
            if i > 0 and row != [] and row[0] != '': symbols_existing.append(row[1])

        date_now = date_local_to_EST(datetime.now())
        with open(symbols_csv_path, mode='a') as file:
            file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for symbol in symbols:
                if symbol not in symbols_existing:
                    avg_volume, rel_volume, floats, index, last_earning_date, news = '', '', '', '', '', []
                    try:
                        avg_volume, rel_volume, floats, market_cap = get_volumes_from_Finviz(symbol)

                        index = get_index_from_symbol(ib, symbol)
                        if not index: index = get_stock_info_from_Finviz(symbol, 'Index')

                        last_earning_date = get_stock_info_from_Finviz(symbol, 'Earnings')

                        newsList = newsScraper.getNewsFinviz(symbol, max_days=1, date_range=None)
                        news = newsList[0]['news'] if len(newsList) > 0 else ''

                    except Exception as e: print("Could not fetch infos from Finviz for symbol ", symbol, ". Error: ", e)
                    print("Adding symbol ", symbol, "\nAvg Vol: ", avg_volume, "\nRel Vol: ", rel_volume, "\nFloats: ", floats, "\nMarket Cap: ", market_cap, "\nIndex: ", index, "\nLast Earning Date: ", last_earning_date, "\nNews: ", news, "\nto: ", symbols_csv_path)
                    file_writer.writerow([date_now.strftime('%Y-%m-%dT%H:%M:%S'), symbol, avg_volume, rel_volume, floats, market_cap, index, last_earning_date, news])

        print("\nSaved symbols at ", symbols_csv_path, "\n")
        file.close()


# ======================
# Time and Dates
# ======================


def date_local_to_EST(date):
    timezone_EST = CONSTANTS.TZ_WORK
    timezone_local = CONSTANTS.TZ_LOCAL

    date_local = timezone_local.localize(date)
    date_EST = datetime.fromisoformat(str(date_local)).astimezone(tz=timezone_EST)

    # time_now_local = datetime.now(tz=timezone_local)
    # date = datetime.fromisoformat(str(date) + str(time_now_local)[-6:])#.replace(tzinfo=pytz.timezone(timezone_local))
    # date_EST = datetime.fromisoformat(str(date)).astimezone(tz=CONSTANTS.TZ_WORK)

    return date_EST


def date_to_EST_aware(date, reverse=False):
    timezone_EST = CONSTANTS.TZ_WORK

    # date_now = datetime.now().strftime('%Y-%m-%d')
    # date_start_DST = datetime(2024,3,10).strftime('%Y-%m-%d')
    # date_end_DST = datetime(2024,11,3).strftime('%Y-%m-%d')

    # DST_offset = 0 if (date_now > date_start_DST or date_now <= date_end_DST) else 1
    # local_offset = -5 + DST_offset
    # extra_zero = "0" if local_offset < 10 else ""
    # sign_offset = "-" if local_offset < 0 else ""
    # local_offset_str = sign_offset + extra_zero + str(abs(local_offset)) + ":00"

    if not reverse:
        # newDate = datetime.fromisoformat(str(date) + local_offset_str).astimezone(tz=CONSTANTS.TZ_WORK)
        newDate = timezone_EST.localize(date)
    else:
        newDate = datetime.fromisoformat(str(date)[:-6])

    return newDate


def date_to_unix(date):
    """
    Converts a variety of date formats to Unix timestamp (milliseconds).

    Accepts:
        - Strings: 'YYYY-MM-DD', 'MM/DD/YYYY', '2025-06-01T00:00:00', etc.
        - datetime objects: datetime, pandas.Timestamp
        - Timestamp: pandas.Timestamp or datetime

    Returns:
        - int: Unix timestamp in milliseconds
    """
    if isinstance(date, (pd.Timestamp, datetime)):
        date_obj = pd.to_datetime(date)
    else:
        date_obj = pd.to_datetime(date, errors='coerce')  # Handle invalid strings gracefully

    if pd.isna(date_obj):
        raise ValueError("Invalid date format or input")

    # Convert to Unix timestamp in milliseconds
    return int(date_obj.timestamp() * 1000)


def calculate_now(sim_offset=timedelta(0), tz=CONSTANTS.TZ_WORK):
        # offset = sim_offset if mode == 'sim' else timedelta(days=0) if mode == 'live' else None
        return datetime.now(tz) - sim_offset


def parse_timedelta(lookback_str):

    if lookback_str.endswith(('M', 'Y')):  # calendar duration
        end_len = 1#2 if lookback_str.endswith(('MS')) else 1
        n = int(lookback_str[:-end_len])
        unit = lookback_str[-end_len]

        if unit == 'M' or unit == 'MS': return pd.DateOffset(months=n)
        elif unit == 'Y': return pd.DateOffset(years=n)

    else:
        return pd.Timedelta(lookback_str)


def is_valid_timedelta(s):
    try:
        pd.to_timedelta(s)
        return True
    except (ValueError, TypeError):
        return False


def get_dates_list(start_date: datetime.date, end_date: datetime.date):
    """
    Returns a list of datetime objects from `start_date` to `end_date`, all set to midnight (00:00:00).

    :param start_date: The start date (inclusive) as a datetime.date.
    :param end_date: The end date (inclusive) as a datetime.date.
    :return: List of datetime objects at midnight.
    :raises ValueError: If start_date or end_date is in the future or start_date > end_date.
    """
    now = datetime.now()
    today = now.date()

    if start_date > today or end_date > today:
        raise ValueError("start_date and end_date must not be in the future.")
    if start_date > end_date:
        raise ValueError("start_date must be less than or equal to end_date.")

    date_range = [
        datetime.combine(date, datetime.time.min)
        for date in (start_date + timedelta(days=n)
                     for n in range((end_date - start_date).days + 1))
    ]
    return date_range


# def get_market_holidays(year, exchange='NYSE'):
#     # Get the market calendar for the exchange (default: NYSE)
#     market = pandas_market_calendars.get_calendar(exchange)

#     # Get the market's schedule for the year
#     schedule = market.sessions_in_range(pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31'))

#     # Convert the schedule to a list of holiday dates
#     holidays = pd.to_datetime(market.holidays().holidays)
#     return holidays

def get_market_holidays(start_date, end_date, exchange='NYSE'):
    # Get the market calendar for the specified exchange (default: NYSE)
    market = pandas_market_calendars.get_calendar(exchange)

    # Get the market's holiday schedule for the specified range
    holidays = market.holidays().holidays

    # Filter the holidays that fall within the given date range
    holidays_in_range = [holiday for holiday in holidays if start_date.date() <= holiday <= end_date.date()]

    # Return as a pandas datetime index
    return pd.to_datetime(holidays_in_range)




# ======================
# Market data and infos
# ======================


def get_forex_symbols_list():
    return ['USD', 'CAD', 'EUR', 'GBP', 'CHF', 'JPY', 'AUD']


def get_index_list():
    return [{'index': 'S&P 500', 'ETF': 'SPY', 'name': "Standard & Poor's 500"},
            {'index': 'NDX', 'ETF': 'QQQ', 'name': 'Nasdaq-100 Index'},
            {'index': 'RUT', 'ETF': 'IWM', 'name': 'Russell 2000'},
            {'index': 'Dow Industry', 'ETF': 'DIA', 'name': 'Dow Jones Industrial Average (DJIA)'},
            {'index': 'S&P 400 Mid Cap', 'ETF': 'MDY', 'name': 'S&P 400 Mid Cap'},
            {'index': 'S&P 600 Small Cap', 'ETF': 'IJR', 'name': 'S&P 600 Small Cap'}]


def get_index_etf(index):
    if not isinstance(index, (str, int)):
        return ''

    index_map = {item['index']: item['ETF'] for item in get_index_list()}
    return index_map.get(index, '')


def get_index_from_symbol(ib, symbol):

    contract = get_symbol_contract(ib, symbol)[0]

    fund = ib.reqFundamentalData(contract, reportType='ReportSnapshot', fundamentalDataOptions=[])

    if fund:
        report_xml = ET.fromstring(fund)
        indexes = report_xml.findall('.//Indexconstituet')
    else:
        indexes = []

    return [index.text for index in indexes]


def get_tick_value(price):

     return  10 ** -int(str(price)[::-1].find('.'))


def get_symbol_seed_list(seed: int, base_folder=PATHS.folders_path['market_data']):
    stock_list_file = os.path.join(base_folder, f"stock_list_seed{seed}.csv")
    if not os.path.exists(stock_list_file):
        return []

    df = load_df_from_file(stock_list_file)
    return sorted(df['symbol'].to_list())



def get_stock_info_from_Finviz(symbol, field):

    from miscellaneous import scanner

    symbol_info_page = scanner.get_page("https://finviz.com/quote.ashx?t=" + symbol + "&p=d")
    table_data_items = symbol_info_page.find('table', {'class': "screener_snapshot-table-body"})
    data_items = table_data_items.find_all('td', {'class': "snapshot-td2"})
    index_field = [data_item.text for data_item in data_items].index(field)

    return data_items[index_field + 1].text


def get_volumes_from_Finviz(symbol):

    # Get Avg Volume, Relative Volume and Floats
    if symbol[:3] not in get_forex_symbols_list():
        try:
            avg_volume = convert_large_numbers(get_stock_info_from_Finviz(symbol, "Avg Volume"))
        except Exception as e:
            print(f"Could not load Avg Volume value for {symbol}. Error: {e}  |  Full error: ", traceback.format_exc())
            avg_volume = ""
        try:
            rel_volume = convert_large_numbers(get_stock_info_from_Finviz(symbol, "Rel Volume"))
        except Exception as e:
            print(f"Could not load Relative Volume value for {symbol}. Error: {e}  |  Full error: ", traceback.format_exc())
            rel_volume = ""
        try:
            market_cap = convert_large_numbers(get_stock_info_from_Finviz(symbol, "Market Cap"))
        except Exception as e:
            print(f"Could not load Market Cap value for {symbol}. Error: {e}  |  Full error: ", traceback.format_exc())
            market_cap = ""
        try:
            floats = convert_large_numbers(get_stock_info_from_Finviz(symbol, "Shs Float"))
        except Exception as e:
            try:
                floats = convert_large_numbers(get_stock_info_from_Finviz(symbol, "Shs Outstand"))
            except Exception as e:
                print(f"Could not load Float value for {symbol}. Error: {e}  |  Full error: ", traceback.format_exc())
                floats = ""
    else:
        avg_volume, rel_volume, floats, market_cap = "", "", "", ""

    return avg_volume, rel_volume, floats, market_cap


def get_share_floats_from_polygon(symbol):
    pclient = polygon.RESTClient(api_key=CONSTANTS.POLYGON_API_KEY)
    return pclient.get_ticker_details(symbol).share_class_shares_outstanding


def get_symbol_contract(ib, symbol, currency="USD", exchange=None):

    if symbol[:3] in get_forex_symbols_list():
        exchange = "IDEALPRO" if not exchange else exchange
        secType="CASH"
        primaryExch = ""
        if len(symbol) > 3:
            currency = symbol[ len(symbol)-3:len(symbol)]
            symbol = symbol[0:3]
    else:
        exchange = "SMART" if not exchange else exchange
        primaryExch = "NASDAQ"
        primaryExch = ""
        secType="STK"

        # exchange = "CBOE" if not exchange else exchange
        # primaryExch = ""
        # secType="IND"

    # https://interactivebrokers.github.io/tws-api/basic_contracts.html
    # contract = Stock(symbol=symbol, exchange='SMART', currency=currency)
    # contract = Forex(pair="USDCAD")

    contract = Contract(symbol=symbol, exchange=exchange, secType=secType, currency=currency, primaryExchange=primaryExch)

    try:
        ib.qualifyContracts(contract)
    except Exception as e:
        print("Symbol not found as a Forex pair or Stock. Error: ", e, "  |  Full error: ", traceback.format_exc())
        contract = []
        print(" Contract could not be loaded.")

    return contract, symbol


def get_symbol_mkt_data(ib, symbol, currency="USD", exchange=None):

    # ib, ibConnection = IBKRConnect(ib)

    contract, symbol = get_symbol_contract(ib, symbol, currency, exchange=exchange)

    if contract:
        try:
            mktData = ib.reqMktData(contract=contract, genericTickList='', snapshot=False, regulatorySnapshot=False)
            ib.sleep(CONSTANTS.PROCESS_TIME['short'])
        except Exception as e:
            print("Could not fetch Market Data. Error: ", e, "  |  Full error: ", traceback.format_exc())
            mktData = []
    else:
        mktData = []

    return contract, mktData


def categorize_market_cap(market_cap: float):
    if market_cap is not None:
        for category, upper_bound in CONSTANTS.MARKET_CAP_CATEGORIES.items():
            if market_cap < upper_bound:
                return category
    return 'Unknown'  # Fallback (shouldn't be hit unless market_cap is NaN or invalid)



# ======================
# Dataframes
# ======================


# def check_data_from_df(df, symbol, timeframe, query_time, duration, indicators_list):

#     df_timeframe = datetime.fromisoformat(str(df['date'].iloc[-1])) - datetime.fromisoformat(str(df['date'].iloc[-2]))
#     df_duration = datetime.fromisoformat(str(df['date'].iloc[-1])) - datetime.fromisoformat(str(df['date'].iloc[0]))

#     # Check timeframe
#     tfww, tfdd, tfhh, tfmm, tfss = 0, 0, 0, 0, 0
#     if 'sec' in timeframe: tfss = int(timeframe.rsplit(' ')[0])
#     elif 'min' in timeframe: tfmm = int(timeframe.rsplit(' ')[0])
#     elif 'hour' in timeframe: tfhh = int(timeframe.rsplit(' ')[0])
#     elif 'day' in timeframe: tfdd = int(timeframe.rsplit(' ')[0])
#     elif 'week' in timeframe: tfww = int(timeframe.rsplit(' ')[0])
#     elif 'month' in timeframe: tfdd = int(timeframe.rsplit(' ')[0]) * 20
#     timeframe = timedelta(weeks=tfww, days=tfdd, hours=tfhh, minutes=tfmm, seconds=tfss)

#     # Check duration
#     dww, ddd, dhh, dmm, dss = 0, 0, 0, 0, 0
#     if 'S' in duration: dss = int(duration.rsplit(' ')[0])
#     elif 'D' in duration: ddd = int(duration.rsplit(' ')[0])
#     elif 'W' in duration: ddd = int(duration.rsplit(' ')[0]) * 5
#     elif 'M' in duration: ddd = int(duration.rsplit(' ')[0]) * 20
#     elif 'Y' in duration: ddd = int(duration.rsplit(' ')[0]) * 52 * 5
#     duration = timedelta(weeks=dww, days=ddd, hours=dhh, minutes=dmm, seconds=dss)

#     # Check query_time and indicators included
#     query_time_conform = query_time < datetime.fromisoformat(str(df['date'].iloc[-1])) + timeframe and query_time > datetime.fromisoformat(str(df['date'].iloc[0]))
#     indicators_conform = set(list(map(lambda x: 'pdc' if x == 'levels' else x, indicators_list))).issubset(df.columns)
#     df_conform = timeframe == df_timeframe and duration <= df_duration and query_time_conform and indicators_conform

#     return df_conform


def get_previous_date_from_df(df, query_time, day_offset):

    query_time_adjusted = adjust_time_to_df(df, query_time)
    query_time_offset = query_time_adjusted
    if query_time_adjusted:

        i = df.loc[df['date'] == str(query_time_adjusted)].index.item()
        day_count = 0
        while day_count <= day_offset:

            df_date = datetime.fromisoformat(str(df['date'].iloc[i]))
            if df_date.time() == query_time_adjusted.time(): day_count += 1
            i -= 1
        query_time_offset = df_date

    else:
        print("Could not find adjusted query time.")

    return query_time_offset


def adjust_time_to_df(df, query_time):

    query_time_adjusted = None
    if len(df) > 1:
        query_time_adjusted = datetime.fromisoformat(str(df['date'].iloc[-1]))
        for i in range (1, len(df['date'])):
            if query_time >= datetime.fromisoformat(str(df['date'].iloc[i-1])) and query_time < datetime.fromisoformat(str(df['date'].iloc[i])):
                query_time_adjusted = datetime.fromisoformat(str(df['date'].iloc[i-1]))
                break

    return query_time_adjusted


def format_df_date(df, col='date', set_index=False):

    if df.empty:
        print("df is empty, no fomatting performed")
        return df

    # Make sure date column is in datetime format
    df[col] = pd.to_datetime(df[col], utc=True)
    df[col] = df[col].dt.tz_convert(CONSTANTS.TZ_WORK)

    if set_index: df.index = df['date']

    return df


def df_to_table(df, title=None):

    table_df = prettytable.PrettyTable()
    table_df.title = title if title else 'Symbol: ' + df.attrs['symbol'] if 'symbol' in df.attrs else None
    # if 'symbol' in df.attrs: table_df.title = 'Symbol: ' + df.attrs['symbol']
    table_df.field_names = df.columns.tolist()
    for row in df.itertuples(index=False, name=None):
        table_df.add_row(row)

    return table_df


def display_df(df, columns_list=['date', 'close'], date_ranges=[], conditions_list=[]):

    # Filters a DataFrame by multiple date ranges and displays selected columns.
    # Example of date_ranges: date_ranges = [['2025-04-10T00:00:00', '2025-04-11T20:59:59'], ['2025-03-05T00:00:00', '2025-03-09T20:59:59']]
    # conditions_list contains column names to be True

    df = format_df_date(df)
    date_ranges = [[date_to_EST_aware(pd.to_datetime(date)) for date in date_range] for date_range in date_ranges]
    # Filter by date ranges if provided
    if date_ranges:
        date_mask = pd.Series(False, index=df.index)
        for date_range in date_ranges:
            date_mask |= (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
        df = df[date_mask]

    # Apply condition filters
    # for condition_col in conditions_list:
    #     if condition_col in df.columns:
    #         df = df[df[condition_col] == True]

    if conditions_list:
        condition_mask = pd.Series(False, index=df.index)
        for condition_col in conditions_list:
            if condition_col in df.columns:
                condition_mask |= df[condition_col] == True
        df = df[condition_mask]

    # Select only the requested columns (if they exist)
    available_columns = [col for col in columns_list if col in df.columns]

    print(df[available_columns].to_string())


def drop_df_columns(df:pd.DataFrame, str_drop:str) -> pd.DataFrame:
    columns_list = [col for col in df.columns if col.endswith(str_drop)]
    df.drop(columns=columns_list, axis=1, inplace=True)
    gc.collect()
    return df


def get_df_memory_usage(df:pd.DataFrame, verbose:bool=False) -> pd.DataFrame:
    # Get memory usage of each column (in MB)
    bytes_MB_conversion = 1024 ** 2
    size_row = df.memory_usage(deep=True) / bytes_MB_conversion
    df_memory_usage = size_row.sum()

    # Get dtype of each column
    dtype_row = df.dtypes

    # Combine size and dtype rows into a new DataFrame
    info_df = pd.DataFrame({
        'Names': size_row.index[1:],
        'Size (MB)': size_row.values[1:],
        'Type': dtype_row.values
    })

    # Sort the DataFrame by 'Size (MB)' in descending order
    info_df = info_df.sort_values(by='Size (MB)', ascending=False)

    if verbose:
        print(df_to_table(info_df, title="DF Sizes"))
        print(f"💾 Total df memory usage: {df_memory_usage:.2f} MB")

    return df_memory_usage, info_df


def get_system_memory_usage() -> int:
    """Returns the system memory usage in MB"""
    memory = psutil.virtual_memory()
    return memory.used / (1024 ** 2)  # Convert bytes to MB


def get_df_chunk_size(df:pd.DataFrame, memory_threshold_MB:int=None):
    memory_threshold_MB = memory_threshold_MB or CONSTANTS.MEMORY_THRESHOLD_MB
    df_memory_usage_MB, _ = get_df_memory_usage(df, verbose=False)

    # If dataframe size exceeds the threshold, split it into chunks
    if df_memory_usage_MB > memory_threshold_MB:
        print(f"💾 Dataframe total memory usage {df_memory_usage_MB:.2f} MB is above memory threshold {memory_threshold_MB} MB.")
        num_chunks = int(math.ceil(df_memory_usage_MB / memory_threshold_MB))
        chunk_size = len(df) // num_chunks
        print(f"✂ Splitting dataframe into {num_chunks} chunks with chunk size of {chunk_size} rows each.")
    else:
        print(f"💾 Dataframe total memory usage {df_memory_usage_MB:.2f} MB is less than memory threshold {memory_threshold_MB} MB. No splitting.")
        num_chunks = 1
        chunk_size = len(df)  # No splitting needed, process the whole dataframe
    return chunk_size, num_chunks


def apply_df_in_chunks(df:pd.DataFrame, evaluate_func=None, memory_threshold_MB:int=None) -> pd.DataFrame:
    """
    Process a DataFrame in chunks to avoid memory overflow based on its memory usage.
    """
    memory_threshold_MB = memory_threshold_MB or CONSTANTS.MEMORY_THRESHOLD_MB
    chunk_size, _ = get_df_chunk_size(df, memory_threshold_MB)
    result_chunks = []
    total_rows = len(df)

    # Split and process the dataframe in chunks
    chunk_count = 1
    for start in range(0, total_rows, chunk_size):
        print(f"🧮 Processing chunk {chunk_count} for applying operation...")
        end = min(start + chunk_size, total_rows)
        df_chunk = df.iloc[start:end]

        # Apply the evaluation function to the chunk
        df_chunk = df_chunk.apply(evaluate_func, axis=1)

        result_chunks.append(df_chunk)
        chunk_count += 1

    # Concatenate results from all chunks back into a single DataFrame
    df_result = pd.concat(result_chunks, ignore_index=True)
    return df_result


def apply_df_with_dask(df:pd.DataFrame, evaluate_func=None) -> pd.DataFrame:
    """
    Process a DataFrame in chunks to avoid memory overflow based on its memory usage.
    """
    chunk_size, num_chunks = get_df_chunk_size(df)

    # Convert to Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=num_chunks)

    # Ensure evaluate_func is callable
    if evaluate_func is None or not callable(evaluate_func):
        raise ValueError("evaluate_func must be a callable function.")

    # Apply the custom function to the Dask DataFrame
    result = ddf.apply(evaluate_func, axis=1, meta=('x', 'f8'))
    # # Apply a function (example: adding two columns)
    # result = ddf.apply(lambda row: row['col1'] + row['col2'], axis=1, meta=('x', 'f8'))

    # Compute result
    result.compute()

    return result


def merge_mtf_df_in_chunks(df:pd.DataFrame, df2:pd.DataFrame, on:str, memory_threshold_MB:int=None,
                            system_memory_threshold_MB:int=10000, chunk_size:int=None, verbose:bool=False) -> pd.DataFrame:
    """
    Merge a DataFrame in chunks to avoid memory overflow based on its memory usage.
    """
    gc.collect()
    how = 'left'
    memory_threshold_MB = memory_threshold_MB or CONSTANTS.MEMORY_THRESHOLD_MB

    if not chunk_size:
        biggest_df = df if get_df_memory_usage(df, verbose=False)[0] > get_df_memory_usage(df2, verbose=False)[0] else df2
        chunk_size, _ = get_df_chunk_size(biggest_df, memory_threshold_MB)

    merged_chunks = []  # List to store the merged chunks
    current_buffer = []  # Temporary buffer to hold chunks before concatenating
    current_buffer_memory = 0  # Track the memory usage of the current buffer

    # Loop over chunks of the left DataFrame (df)
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        df_chunk = df.iloc[start:end]

        # Merge with the corresponding chunk from df2
        # For each chunk, filter df2 to avoid unnecessary large memory usage
        df2_chunk = df2[df2[on].between(df_chunk[on].min(), df_chunk[on].max())] if len(df[on].apply(type).unique()) == 1 else df2

        # Perform the merge
        merged_chunk = pd.merge(df_chunk, df2_chunk, on=on, how=how)

        # Add the merged chunk to the current buffer
        current_buffer.append(merged_chunk)
        current_buffer_memory += get_df_memory_usage(merged_chunk, verbose=False)[0]
        system_memory_usage = get_system_memory_usage()

        print(f"Merged chunk {start // chunk_size + 1} ({end}/{len(df)}), current buffer memory: {current_buffer_memory:.2f} MB / {memory_threshold_MB:.2f} MB, current system memory usage: {system_memory_usage:.2f} MB / {system_memory_threshold_MB:.2f} MB")

        # Check for early concatenation
        # Periodically check system memory usage and decide whether to concatenate
        if system_memory_usage > system_memory_threshold_MB:
            if verbose: print(f"Warning: System memory usage is high ({system_memory_usage:.2f} MB). Force concatenation to free memory.")
            # Concatenate the buffer and reset it
            merged_chunks.append(pd.concat(current_buffer, ignore_index=True))
            current_buffer = []  # Reset the buffer
            current_buffer_memory = 0  # Reset the memory usage

        # Check if the buffer exceeds the memory threshold
        if current_buffer_memory >= memory_threshold_MB:
            if verbose: print(f"Warning: Current buffer memory {current_buffer_memory:.2f} MB exceeds limit ({memory_threshold_MB}). Force concatenation to free memory.")
            # Concatenate the buffer and reset it
            merged_chunks.append(pd.concat(current_buffer, ignore_index=True))
            current_buffer = []  # Reset the buffer
            current_buffer_memory = 0  # Reset the memory usage

    # After all chunks are processed, concatenate any remaining chunks in the buffer
    if current_buffer:
        merged_chunks.append(pd.concat(current_buffer, ignore_index=True))

    # Final concatenation of all chunks
    merged_df = pd.concat(merged_chunks, ignore_index=True)
    gc.collect()

    return merged_df


def get_optimal_num_partitions(memory_usage_MB:float, partition_size_MB:int=500):
    # memory_usage_MB, _ = get_df_memory_usage(df, verbose=False) # Get the memory usage in MB

    # Calculate the number of partitions required
    num_partitions = min(memory_usage_MB // partition_size_MB, CONSTANTS.MAX_CORE_PARTITIONS)
    if num_partitions < 1: num_partitions = 1  # Ensure at least one partition

    # Ensure we don't exceed the available memory
    available_partitions = CONSTANTS.MEMORY_THRESHOLD_MB // partition_size_MB
    return int(min(num_partitions, available_partitions))


def merge_df_with_dask(df:pd.DataFrame, df2:pd.DataFrame, on:str, how:str='left', partition_size_MB:int=200) -> pd.DataFrame:
    """
    Merge a DataFrame using dask
    """
    mu_df, _ = get_df_memory_usage(df, verbose=False)
    mu_df2, _ = get_df_memory_usage(df2, verbose=False)
    # biggest_df = df if mu_df > mu_df2 else df2
    num_partitions = get_optimal_num_partitions(memory_usage_MB=max(mu_df, mu_df2), partition_size_MB=partition_size_MB)

    print(f"🧮 Merging with dask using {num_partitions} partitions... Memory usage: {max(mu_df, mu_df2)} MB")

    # Load the data as Dask DataFrames
    df_dask = dd.from_pandas(df, npartitions=num_partitions)
    df2_dask = dd.from_pandas(df2, npartitions=num_partitions)

    # Merge the Dask DataFrames
    merged_df = dd.merge(df_dask, df2_dask, on=on, how=how)

    # Compute the result (convert it back to a Pandas DataFrame)
    merged_df = merged_df.compute()
    merged_df = merged_df.sort_values(by='date').reset_index(inplace=False, drop=True)
    return merged_df





# def get_df_timeframe(df):

#     if df.empty:
#         return None

#     # Make sure date column is in datetime format
#     df = format_df_date(df)

#     time_diffs = df['date'].diff().dropna()    # Get the time differences between consecutive rows
#     most_common_diff = time_diffs.mode()[0]    # Get the most common frequency (mode) of time differences

#     seconds = most_common_diff.total_seconds()

#     sec_to_tf = CONSTANTS.SECONDS_TO_INTRADAY_TIMEFRAME

#     if str(seconds) in sec_to_tf.keys():
#         timeframe = (next((value for key, value in sec_to_tf.items() if key == str(seconds)), None))
#     else:
#         if most_common_diff == timedelta(days=1): timeframe = '1 day'
#         elif most_common_diff == timedelta(weeks=1): timeframe = '1 week'
#         elif timedelta(weeks=4) <= most_common_diff < timedelta(weeks=5) : timeframe = '1 month'
#         else: timeframe = None

#     return timeframe


def get_df_timeframe(df):
    """
    Detects the most common frequency of a dataframe's datetime index or 'date' column
    and returns it as a Timeframe object.

    Returns:
        Timeframe instance (or None if unknown or df is empty)
    """
    if df.empty:
        return None

    # Ensure 'date' column is in datetime format
    df = format_df_date(df)

    # Compute time differences
    time_diffs = df['date'].diff().dropna()
    most_common_diff = time_diffs.mode()[0]

    seconds = most_common_diff.total_seconds()
    str_seconds = str(seconds)

    # Check if the interval matches known intraday timeframes
    if str_seconds in CONSTANTS.SECONDS_TO_INTRADAY_TIMEFRAME:
        ibkr_style = CONSTANTS.SECONDS_TO_INTRADAY_TIMEFRAME[str_seconds]
        return Timeframe(ibkr_style, style='ibkr')

    # Handle coarser resolutions (daily, weekly, monthly)
    if most_common_diff == timedelta(days=1):
        return Timeframe("1 day", style='ibkr')
    elif most_common_diff == timedelta(weeks=1):
        return Timeframe("1 week", style='ibkr')
    elif timedelta(weeks=4) <= most_common_diff < timedelta(weeks=5):
        return Timeframe("1 month", style='ibkr')

    # If no known match, return None
    return None


def get_daily_df(df, th='all', rename_volume=True):

    th_times = CONSTANTS.TH_TIMES

    df_D = pd.DataFrame()
    if th == 'rth':
        df_D = df[(df['date'].dt.time >= th_times['rth']) & (df['date'].dt.time <= th_times['post-market'])].resample('D', on='date').agg({'open':'first', 'close':'last', 'high':'max', 'low':'min', 'volume':'sum'}).dropna(subset=['close']).reset_index()
        if rename_volume: df_D.rename(columns={'volume': 'vol_D_rth'}, inplace=True)
    elif th == 'pre-market':
        df_D = df[(df['date'].dt.time >= th_times['pre-market']) & (df['date'].dt.time <= th_times['rth'])].resample('D', on='date').agg({'open':'first', 'close':'last', 'high':'max', 'low':'min', 'volume':'sum'}).dropna(subset=['close']).reset_index()
        if rename_volume: df_D.rename(columns={'volume': 'vol_D_preM'}, inplace=True)
    elif th == 'post-market':
        df_D = df[(df['date'].dt.time >= th_times['post-market']) & (df['date'].dt.time <= th_times['end_of_day'])].resample('D', on='date').agg({'open':'first', 'close':'last', 'high':'max', 'low':'min', 'volume':'sum'}).dropna(subset=['close']).reset_index()
        if rename_volume: df_D.rename(columns={'volume': 'vol_D_postM'}, inplace=True)
    elif th == 'all':
        df_D = df.resample('D', on='date').agg({'open':'first', 'close':'last', 'high':'max', 'low':'min', 'volume':'sum'}).dropna(subset=['close']).reset_index()
        if rename_volume: df_D.rename(columns={'volume': 'vol_D'}, inplace=True)

    # dropna on column 'close' is to remove weekends and holidays

    return df_D


def trim_df(df:pd.DataFrame, from_time:datetime=None, to_time:datetime=None, keep_last:bool=False) -> pd.DataFrame:

    if from_time > to_time:
        print(f"⚠️ to_time < from_time, returning empty.")
        return pd.DataFrame()
    
    df = format_df_date(df)
    df_to_time = pd.to_datetime(df['date'].iloc[-1]) if not df.empty else None
    df_from_time = pd.to_datetime(df['date'].iloc[0]) if not df.empty else None

    if not to_time: to_time = df_to_time
    if not from_time: from_time = df_from_time

    # Verify if both from_time and to_time are bigger than last df date:
    if keep_last and from_time > df_to_time:
        return df.iloc[[-1]]
    # Verify if both from_time and to_time fall in between the same two consecutive rows of the df, in which case, return the previous row
    df['next_date'] = df['date'].shift(-1)
    mask = (df['date'] <= from_time) & (df['next_date'] > from_time) & (df['date'] <= to_time) & (df['next_date'] > to_time)
    matching_row = df[mask]
    df.drop(columns=['next_date'])

    if not matching_row.empty:
        df = df[mask]
    else:
        df = df[pd.to_datetime(df["date"]).between(max(from_time, df_from_time), min(to_time, df_to_time))]

    df = df.reset_index(inplace=False, drop=True)

    return df


def trim_df_non_trading_days(df:pd.DataFrame, from_time:datetime, to_time:datetime, timeframe:Timeframe=None):
        '''Drop non trading days and hours'''

        timeframe = timeframe or get_df_timeframe(df)
        if timeframe.to_timedelta > timedelta(days=1):
            return df

        holidays = get_market_holidays(from_time, to_time)
        th_times = CONSTANTS.TH_TIMES

        # Create masks
        # if TimeframeHandler.timeframe_to_seconds(timeframe) < TimeframeHandler.timeframe_to_seconds('1 day'):
        if timeframe.to_timedelta < timedelta(days=1):
            active_hours_mask = (df.index.hour >= th_times['pre-market'].hour) & (df.index.hour < th_times['end_of_tday'].hour)  # Active hours mask
        else: active_hours_mask = pd.Series([True] * len(df), index=df.index)
        weekdays_mask = df.index.weekday < 5  # Weekdays mask (Mon-Fri)
        # is_trading_day_mask = ~pd.to_datetime(df.index.date).isin(holidays.date) & weekdays_mask  # Exclude holidays and weekends
        is_trading_day_mask = ~pd.to_datetime(df.index.date).isin(pd.to_datetime(holidays.date)) & weekdays_mask  # Exclude holidays and weekends

        active_trading_mask = active_hours_mask & is_trading_day_mask

        return df[active_trading_mask]


def make_attrs_json_safe(attrs):
    safe_attrs = {}
    for k, v in attrs.items():
        if isinstance(v, pd.Timestamp):
            safe_attrs[k + "__pd.Timestamp"] = v.isoformat()
        else:
            safe_attrs[k] = v
    return safe_attrs


def restore_attrs_from_json_safe(attrs):
    restored_attrs = {}
    for k, v in attrs.items():
        if k.endswith("__pd.Timestamp"):
            new_key = k.replace("__pd.Timestamp", "")
            restored_attrs[new_key] = pd.Timestamp(v)
        else:
            restored_attrs[k] = v
    return restored_attrs


def compare_dataframes(df1, df2, calculate_diffs=True, exclusion_list = [], display_diffs=False):
    # Find column sets
    cols_df1 = set(df1.columns)
    cols_df2 = set(df2.columns)

    # Identify common and unique columns
    common_cols = cols_df1 & cols_df2
    only_in_df1 = list(cols_df1 - cols_df2)
    only_in_df2 = list(cols_df2 - cols_df1)

    # Compare common columns
    mismatched_columns = []

    for col in common_cols:
        if not df1[col].equals(df2[col]):
            mismatched_columns.append(col)

    # Result dictionary
    result = {'mismatched_columns': mismatched_columns, 'only_in_df1': only_in_df1, 'only_in_df2': only_in_df2}

    if calculate_diffs:
        # exclusion_list = ['sr_1h', 'sr_1D', 'sr_1W', 'pivots', 'pivots_D', 'pivots_M', 'levels', 'levels_M', 'breakout_up_score', 'breakout_down_score']
        col_list = [item for item in result['mismatched_columns'] if item not in exclusion_list]
        result['diffs'] = []

        for col in col_list:
            # print(f"Assessing {col} column")
            series1 = df1[col]
            series2 = df2[col]

            # Create mask of unequal values, treating NaNs as equal
            unequal_mask = ~((series1 == series2) | (series1.isna() & series2.isna()))

            # Extract mismatched values
            val1 = series1[unequal_mask]
            val2 = series2[unequal_mask]

            if not val1.empty and pd.api.types.is_numeric_dtype(val1) and pd.api.types.is_numeric_dtype(val2) and not pd.api.types.is_bool_dtype(val1) and not pd.api.types.is_bool_dtype(val2):
                # Compute absolute differences and average values
                diffs = (val1 - val2).abs()
                avg_vals = (val1 + val2) / 2

                avg_diff = diffs.mean()
                avg_diff_pct = (100 * diffs / avg_vals.replace(0, np.nan)).mean()  # Avoid div-by-zero

                diff = {'column': col, 'avg_diff': avg_diff, 'avg_diff_pct': avg_diff_pct}
                result['diffs'].append(diff)
                # if display_diffs:
                #     print(df_to_table(pd.DataFrame(diff)))
            else:
                result['diffs'].append({'column': col, 'avg_diff': None, 'avg_diff_pct': None})

    if display_diffs:
        if result['diffs']: print(df_to_table(pd.DataFrame(result['diffs'])))
        else: print("No difference recorded.")
    # diff_dates = df1.index[(df1['breakout_up_since_last'] - df2['breakout_up_since_last']).abs() > epsilon].tolist()
    return result


def extract_timeframe_from_df_column(s):
    if s.endswith('_list'):
        # If it ends with '_list', take the part before '_list'
        return s.split('_list')[0].split('_')[-1]
    else:
        # Otherwise, take the part after the last '_'
        return s.split('_')[-1]


# # ======================
# # Indicators
# # ======================


# def get_indicator(df, indicator, query_time):

#     query_time_df = adjust_time_to_df(df, query_time)
#     ind = ''

#     try:
#         if indicator == 'pivots' or indicator == 'pivots_D' or indicator == 'pivots_M':
#             ind = {}
#             if indicator == 'pivots': pivots_list = ['s1', 's2', 's3', 's4', 's5', 's6', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6']
#             elif indicator == 'pivots_D': pivots_list = ['s1_D', 's2_D', 's3_D', 's4_D', 's5_D', 's6_D', 'r1_D', 'r2_D', 'r3_D', 'r4_D', 'r5_D', 'r6_D']
#             elif indicator == 'pivots_M': pivots_list = ['s1_M', 's2_M', 's3_M', 's4_M', 's5_M', 's6_M', 'r1_M', 'r2_M', 'r3_M', 'r4_M', 'r5_M', 'r6_M']
#             for p in pivots_list:
#                 ind[p] = df.loc[df['date'] == query_time_df, p].iloc[0]
#         else:
#             ind = df.loc[df['date'] == query_time_df, indicator].iloc[0]

#     except Exception as e: print("Could not fetch ", indicator, ". Error: ", e, "  |  Full error: ", traceback.format_exc())

#     return ind


# def get_CPR(df, query_time, pivots):

#     char = '_D' if '_D' in list(pivots.keys())[0] else ''

#     # Get CPR size and level comparisons
#     cpr_size_to_yst_perc, cpr_midpoint_to_yst_perc = '', ''
#     try:
#         query_time_yst = get_previous_date_from_df(df, query_time, day_offset=1)
#         r2_yst, s2_yst = float(get_indicator(df, 'r2' + char, query_time_yst)), float(get_indicator(df, 's2' + char, query_time_yst))
#         cpr_size_to_yst_perc = round(((pivots['r2' + char] - pivots['s2' + char]) * 100 / (r2_yst - s2_yst)) - 100, 1)
#         cpr_midpoint_to_yst_perc = round((0.5 * (pivots['s2' + char] + pivots['r2' + char]) * 100 / (0.5 * (s2_yst + r2_yst))) - 100, 1)
#     except Exception as e: print("Could not calculate CPR. Error: ", e, "  |  Full error: ", traceback.format_exc())

#     return cpr_size_to_yst_perc, cpr_midpoint_to_yst_perc


# def get_levels_by_resampling(df, levels, timeframe, df2=pd.DataFrame()):

#         if df2.empty: df2 = df
#         date_tf_col = 'date_' + timeframe

#         # Resample data by day
#         levels_aggregate = {}
#         for l in levels: levels_aggregate[l['col']] = l['funct']

#         df2_resampled = df2.resample(timeframe, on='date').agg(levels_aggregate).dropna().reset_index()
#         # df2_resampled = df2.resample('ME' if timeframe == 'M' else timeframe, on='date').agg(levels_aggregate).dropna().reset_index()
#         if timeframe == 'D': df2_resampled[date_tf_col] = df2_resampled['date'].dt.date
#         elif timeframe == 'M': df2_resampled[date_tf_col] = df2_resampled['date'].dt.strftime('%Y-%m')

#         # Shift to get previous day's data
#         for l in levels: df2_resampled[l['name']] = df2_resampled[l['col']].shift(l['shift'])

#         # Map the previous day's data to the original DataFrame
#         levels_merge = [date_tf_col]
#         for l in levels: levels_merge += [l['name']]

#         df = pd.merge(df, df2_resampled[levels_merge], on=date_tf_col, how='left')

#         return df


# def get_supp_res(df, display_levels=False):

#     # Make sure date column is in datetime format
#     # df = format_df_date(df)

#     df_levels = get_daily_df(df, th='rth', rename_volume=False)
#     # df_levels = df.resample('W', on='date').agg({'open':'first', 'close':'last', 'high':'max', 'low':'min', 'volume':'sum'}).dropna(subset=['close']).reset_index()

#     # Detect local peaks and troughs (resistance and support)
#     peaks, _ = scipy.signal.find_peaks(df_levels['high'], distance=10, prominence=1)
#     troughs, _ = scipy.signal.find_peaks(-df_levels['low'], distance=10, prominence=1)

#     df_levels['is_peak'] = False
#     df_levels['is_peak'].iloc[peaks] = True

#     df_levels['is_trough'] = False
#     df_levels['is_trough'].iloc[troughs] = True

#     df_levels['swing_price'] = numpy.nan
#     df_levels.loc[df_levels['is_peak'] == True, 'swing_price'] = df_levels['high']
#     df_levels.loc[df_levels['is_trough'] == True, 'swing_price'] = df_levels['low']

#     level_granularity = 5  # or a percentage if using crypto
#     df_levels['rounded_level'] = df_levels['swing_price'].round(-int(numpy.log10(level_granularity)))

#     # Only use peaks and troughs
#     levels_df = df_levels[df_levels['is_peak'] | df_levels['is_trough']]
#     level_counts = levels_df['rounded_level'].value_counts().sort_index()

#     # Basic scoring based on reaction count
#     significant_levels = level_counts[level_counts > 2 * level_counts.std()]

#     # Calculate reversal strength
#     look_forward_reaction = 10
#     reaction_strengths = []
#     for level in significant_levels.index:
#         swing_points = levels_df[levels_df['rounded_level'] == level]
#         for idx in swing_points.index:
#             # look forward or backward N bars and measure max % move
#             future_df_subset = df_levels.iloc[idx:idx+look_forward_reaction+1]
#             future_price_max = future_df_subset['close'].max() if future_df_subset['is_trough'].iloc[0] else future_df_subset['close'].min() if future_df_subset['is_peak'].iloc[0] else None
#             future_price_max_index = future_df_subset['close'].idxmax()
#             future_volume_cum = df_levels['volume'].iloc[idx:future_price_max_index+1].sum()
#             current_price = df_levels['close'].iloc[idx]
#             pct_change = 100 * (future_price_max - current_price) / current_price
#             date_change = df_levels['date'].iloc[idx].date()
#             reaction_strengths.append((date_change, level, future_volume_cum, pct_change))

#     # Aggregate strength
#     df_levels_reaction = pd.DataFrame(reaction_strengths, columns=['date', 'level', 'cum_volume', 'pct_move']).sort_values(by='date', ascending=True)
#     df_levels_ranking = df_levels_reaction.groupby('level').agg(level_count=('level', 'count'), last_date=('date', 'last'), mean_cum_volume=('cum_volume', 'mean'), mean_pct_move=('pct_move', 'mean')).sort_values(by='mean_pct_move', key=abs, ascending=False).reset_index()
#     df_levels_ranking['date_score'] = (pd.to_datetime(df_levels_ranking['last_date']) - pd.to_datetime(df_levels_reaction['date'].max())).dt.days
#     df_levels_ranking['mean_pct_move_abs'] = df_levels_ranking['mean_pct_move'].abs()

#     # Normalize
#     ranking_columns = ['level_count', 'date_score', 'mean_cum_volume', 'mean_pct_move_abs']
#     normalized = sklearn.preprocessing.MinMaxScaler().fit_transform(df_levels_ranking[ranking_columns])
#     df_normalized = pd.DataFrame(normalized, columns=[col + '_norm' for col in ranking_columns])
#     df_levels_ranking = pd.concat([df_levels_ranking, df_normalized], axis=1)

#     # df_levels_ranking['score'] = df_normalized.mean(axis=1) # Case equal weights
#     weights = {'level_count': 0.25, 'date_score': 0.25, 'mean_cum_volume': 0.25, 'mean_pct_move_abs': 0.25}
#     df_levels_ranking['weighted_score'] = sum(df_normalized[col+'_norm'] * weight for col, weight in weights.items())

#     df_levels_ranking = df_levels_ranking.sort_values(by='weighted_score', ascending=False)#.reset_index(drop=True)

#     print(df_levels_reaction)
#     print()
#     print(df_levels_ranking)

#     # Display levels
#     display_levels = True
#     if display_levels:
#         plt.figure(figsize=(14,6))
#         plt.plot(df_levels['close'], label='Price')
#         for level in df_levels_ranking['level'].head(10):
#             plt.axhline(y=level, color='orange', linestyle='--', alpha=0.5)
#         plt.legend()
#         plt.title("Strong Support & Resistance Levels")
#         plt.show()


# def add_indicator(df, ib, contract, indicators_list):

#     if df.shape[0] > 0:

#         # Make sure date column is in datetime format
#         # df = format_df_date(df)

#         # Add day and month info columns
#         df['date_D'] = df['date'].dt.date
#         df['date_M'] = df['date'].dt.strftime('%Y-%m')

#         # Add low and high of day
#         df['low_of_day'] = df.groupby('date_D')['low'].cummin()
#         df['high_of_day'] = df.groupby('date_D')['high'].cummax()

#         if "emas" in indicators_list:
#             indicator_ema9 = ta.trend.EMAIndicator(close=df["close"], window=9, fillna=False)
#             df["ema9"] = indicator_ema9.ema_indicator()
#             indicator_ema20 = ta.trend.EMAIndicator(close=df["close"], window=20, fillna=False)
#             df["ema20"] = indicator_ema20.ema_indicator()
#             indicator_sma50 = ta.trend.SMAIndicator(close=df["close"], window=50, fillna=False)
#             df["sma50"] = indicator_sma50.sma_indicator()
#             indicator_sma200 = ta.trend.SMAIndicator(close=df["close"], window=200, fillna=False)
#             df["sma200"] = indicator_sma200.sma_indicator()

#         if "vwap" in indicators_list:
#             # Calculate window parameter as the average candle per day in the whole df
#             window = int(numpy.floor((numpy.average([len(group) for date, group in df.groupby(df['date'].dt.date)]))))
#             indicator_vwap = ta.volume.VolumeWeightedAveragePrice(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=window, fillna=False)
#             df["vwap"] = indicator_vwap.volume_weighted_average_price()

#         if "macd" in indicators_list:
#             indicator_macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=False)
#             df["macd"] = indicator_macd.macd()
#             df['macd_signal'] = indicator_macd.macd_signal()
#             df['macd_diff'] = indicator_macd.macd_diff()

#         if "index" in indicators_list:

#             # Get symbol related Index and ETF
#             symbol_index = get_index_from_symbol(ib, contract.symbol)
#             if len(symbol_index) > 0: symbol_index = symbol_index[0]
#             symbol_index_ETF = get_index_etf(symbol_index)

#             if symbol_index_ETF:

#                 # Gathering Index historical data
#                 timeframe_index = get_df_timeframe(df)
#                 query_time = add_ibkr_timeframe_to_date(df['date'].iloc[-1], timeframe_index)
#                 from_time = df['date'].iloc[0]
#                 duration_index = get_ibkr_duration_from_time_diff(query_time, from_time)
#                 df_index = get_symbol_hist_data(ib, symbol_index_ETF, timeframe_index, query_time, duration=duration_index, indicators_list=[])

#                 # Assess Index trend
#                 indicator_macd_index = ta.trend.MACD(close=df_index['close'], window_slow=50, window_fast=20, window_sign=9, fillna=False)
#                 df_index['macd_diff'] = indicator_macd_index.macd_diff()
#                 condition_index_trend_up = (df_index['macd_diff'] > df_index['macd_diff'].shift()) & (df_index['macd_diff'].shift() > df_index['macd_diff'].shift(2))
#                 condition_index_trend_down = (df_index['macd_diff'] < df_index['macd_diff'].shift()) & (df_index['macd_diff'].shift() < df_index['macd_diff'].shift(2))
#                 df_index['index_trend'] = numpy.where(condition_index_trend_up, 1, numpy.where(condition_index_trend_down, -1, 0))

#                 # Adding Index trend values to main df dataframe
#                 df = pd.merge(df, df_index[['date', 'index_trend']], on='date', how='left')

#         if 'rsi' in indicators_list:
#             indicator_rsi = ta.momentum.RSIIndicator(close=df['close'], window=14, fillna=False)
#             df['rsi'] = indicator_rsi.rsi()

#         if 'bollinger_bands' in indicators_list:
#             indicator_bbands = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2, fillna=False)
#             df['bband_h'] = indicator_bbands.bollinger_hband()
#             df['bband_l'] = indicator_bbands.bollinger_lband()
#             df['bband_mavg'] = indicator_bbands.bollinger_mavg()
#             # df["bband_h_ind"] = indicator_bbands.bollinger_hband_indicator()
#             # df["bband_l_ind"] = indicator_bbands.bollinger_lband_indicator()

#         if 'vol_ratio' in indicators_list:
#             df['vol_ratio'] = df.volume / (df.volume - df.volume.diff())

#         if 'atr_D' in indicators_list or 'change' in indicators_list:
#             # Create daily df
#             df_day = get_daily_df(df)
#             df_day['date'] = df_day['date'].dt.date
#             query_time = df_day['date'].iloc[0]
#             whatToShow = 'TRADES' if contract.symbol not in get_forex_symbols_list() else 'MIDPOINT'
#             bars_day_additional = ib.reqHistoricalData(contract, endDateTime=query_time, durationStr='1 M', barSizeSetting='1 day', whatToShow=whatToShow, useRTH=False)
#             ib.sleep(CONSTANTS.PROCESS_TIME['long'])
#             df_day_additional = util.df(bars_day_additional)
#             df_day = pd.concat([df_day, df_day_additional], ignore_index=True).drop_duplicates(subset=['date'])
#             df_day.sort_values(by=['date'], ascending=True, inplace=True)
#             df_day.reset_index(inplace=True, drop=True)

#         if "atr" in indicators_list:
#             # Getting ATR values from daily candlesticks
#             indicator_ATR = ta.volatility.AverageTrueRange(high=df_day["high"], low=df_day["low"], close=df_day["close"], window=14, fillna=False)
#             df_day["atr"] = indicator_ATR.average_true_range()

#             # Adding ATR values to main df dataframe
#             df_day.rename(columns={'date': 'date_D'}, inplace=True)
#             df = pd.merge(df, df_day[['date_D', 'atr_D']], on='date_D', how='left')

#             df['atr_band_high'] = df['low_of_day'] + df['atr_D']
#             df['atr_band_low'] = df['high_of_day'] - df['atr_D']

#         if "change" in indicators_list:
#             pdc = df_day.close[len(df_day)-2]
#             df["change"] = (100 * df.close / pdc) - 100
#             df["change_diff"] = df.change.diff()

#         if "vpa" in indicators_list:
#             # Calculate VPA deviations
#             lookback_period_vpa = 20
#             df['vpa_ratio_h'] = df['volume'] / abs(df['close'] - df['open'])
#             df['vpa_ratio_l'] = abs(df['close'] - df['open']) / df['volume']
#             df['vpa_z_score_h'] = (df['vpa_ratio_h'] - df['vpa_ratio_h'].rolling(window=lookback_period_vpa, min_periods=1).mean()) / df['vpa_ratio_h'].rolling(window=lookback_period_vpa, min_periods=1).std()
#             df['vpa_z_score_l'] = (df['vpa_ratio_l'] - df['vpa_ratio_l'].rolling(window=lookback_period_vpa, min_periods=1).mean()) / df['vpa_ratio_l'].rolling(window=lookback_period_vpa, min_periods=1).std()

#         if "r_vol" in indicators_list:
#             lookback_period_r_vol = 3
#             df['avg_vol'] = df['volume'].rolling(window=lookback_period_r_vol).mean()
#             df['r_vol'] = df['volume'] / df['avg_vol']

#             timeframe_df = df['date'].diff().dropna().mode()[0]
#             df.set_index('date', inplace=True)

#             if timeframe_df < timedelta(days=1):
#                 df['avg_vol_at_time'] = df.groupby(df.index.time).apply(lambda d: d['volume'].rolling(lookback_period_r_vol, min_periods=1).mean()).reset_index(level=0, drop=True).sort_index()#.reset_index(drop=True)#.sort_values()#.reset_index(drop=True)
#             else:
#                 df['avg_vol_at_time'] = None

#             df = df.reset_index()

#             df['r_vol_at_time'] = df['volume'] / df['avg_vol_at_time']
#             # print(df[df['date'].dt.time == datetime.time(19, 45)][['date', 'volume', 'avg_vol', 'r_vol', 'avg_vol_at_time', 'r_vol_at_time']].to_string())
#             # input()

#         if "levels" in indicators_list:

#             # Prepare df for resampling
#             df['date_D'] = df['date'].dt.date
#             df['date_M'] = df['date'].dt.strftime('%Y-%m')

#             th_times = CONSTANTS.TH_TIMES
#             df['session'] = ['pre-market' if (t >= th_times['pre-market'] and t < th_times['rth']) else 'post-market' if (t >= th_times['post-market'] and t <= th_times['end_of_day']) else 'rth' for t in df['date'].dt.time]#strftime('%H-%M-%S')]
#             # df['session2'] = pd.cut(df.date.dt.hour + df.date.dt.minute / 60.0, bins=[-float('inf'), 4, 9.5, 16, 20, float('inf')], labels=['Pre-market', 'RTH', 'Post-market', 'RTH', 'Post-market'], right=False)

#             # Calculate pdc, pdh and pdl
#             levels_list = [{'name':'pdh', 'col':'high', 'funct':'max', 'shift':1},
#                            {'name':'pdl', 'col':'low', 'funct':'min', 'shift':1}]
#             df = get_levels_by_resampling(df, levels_list, 'D')

#             # Calculate pmh and pml
#             df_pre_market = df[df['session'] == 'pre-market']
#             levels_list = [{'name':'pmh', 'col':'high', 'funct':'max', 'shift':0},
#                            {'name':'pml', 'col':'low', 'funct':'min', 'shift':0}]
#             df = get_levels_by_resampling(df, levels_list, 'D', df2=df_pre_market)

#             # Calculate shift to apply to pmh and pml
#             time_interval = df['date'].iloc[1] - df['date'].iloc[0]
#             time_interval2 = datetime(1, 1, 1, 9, 30) - datetime(1, 1, 1, 4, 0)
#             shift_pm = int(round(time_interval2 / time_interval, 0))
#             df_shift = df[df["date"].between(pd.to_datetime("2024-12-30 04:00:00-05:00"), pd.to_datetime("2024-12-30 09:30:00-05:00"))]

#             df[['pmh', 'pml']] = df[['pmh', 'pml']].shift(shift_pm)

#             # Calculate pdc, do, pdh_D, pdl_D
#             df_rth = df[df['session'] == 'rth']
#             levels_list = [{'name':'pdc', 'col':'close', 'funct':'last', 'shift':1},
#                            {'name':'do', 'col':'open', 'funct':'first', 'shift':0},
#                            {'name':'pdh_D', 'col':'high', 'funct':'max', 'shift':1},
#                            {'name':'pdl_D', 'col':'low', 'funct':'min', 'shift':1}]
#             df = get_levels_by_resampling(df, levels_list, 'D', df2=df_rth)

#             # Calculate pMc, pMh, pMl
#             levels_list = [{'name':'pMh', 'col':'high', 'funct':'max', 'shift':1},
#                            {'name':'pMl', 'col':'low', 'funct':'min', 'shift':1},
#                            {'name':'pMc', 'col':'close', 'funct':'last', 'shift':1},
#                         #    {'name':'Mo', 'col':'open', 'funct':'first', 'shift':0},
#                            {'name':'Mo', 'col':'open', 'funct':'first', 'shift':0}]
#             df = get_levels_by_resampling(df, levels_list, 'M')

#             # Calculate Camarilla Pivots
#             piv_addon = ['', '_D', '_M']
#             for p in piv_addon:
#                 pc = 'pMc' if p == '_M' else 'pdc'
#                 ph = 'pMh' if p == '_M' else 'pdh_D' if p == '_D' else 'pdh'
#                 pl = 'pMl' if p == '_M' else 'pdl_D' if p == '_D' else 'pdl'
#                 df['pp'+p] = (df[ph] + df[pl] + df[pc]) / 3
#                 df['r1'+p] = df[pc] + 1.1 * (df[ph] - df[pl]) / 12
#                 df['r2'+p] = df[pc] + 1.1 * (df[ph] - df[pl]) / 6
#                 df['r3'+p] = df[pc] + 1.1 * (df[ph] - df[pl]) / 4
#                 df['r4'+p] = df[pc] + 1.1 * (df[ph] - df[pl]) / 2
#                 df['r5'+p] = df['r4'+p] + 1.168 * (df['r4'+p] - df['r3'+p])
#                 df['r6'+p] = df[pc] * df[ph] / df[pl]
#                 df['s1'+p] = df[pc] - 1.1 * (df[ph] - df[pl]) / 12
#                 df['s2'+p] = df[pc] - 1.1 * (df[ph] - df[pl]) / 6
#                 df['s3'+p] = df[pc] - 1.1 * (df[ph] - df[pl]) / 4
#                 df['s4'+p] = df[pc] - 1.1 * (df[ph] - df[pl]) / 2
#                 df['s5'+p] = df['s4'+p] - 1.168 * (df['s3'+p] - df['s4'+p])
#                 df['s6'+p] = df[pc] - (df['r6'+p] - df[pc])

#                 # Create Camarilla position column
#                 conditions_cam_positions = [
#                     (df['close'] < df['s6'+p]), (df['r6'+p] < df['close']), (df['s6'+p] < df['close']) & (df['close'] < df['s5'+p]), (df['s5'+p] < df['close']) & (df['close'] < df['s4'+p]),
#                     (df['s4'+p] < df['close']) & (df['close'] < df['s3'+p]), (df['s3'+p] < df['close']) & (df['close'] < df['s2'+p]), (df['s2'+p] < df['close']) & (df['close'] < df['s1'+p]),
#                     (df['s1'+p] < df['close']) & (df['close'] < df['r1'+p]), (df['r1'+p] < df['close']) & (df['close'] < df['r2'+p]), (df['r2'+p] < df['close']) & (df['close'] < df['r3'+p]),
#                     (df['r3'+p] < df['close']) & (df['close'] < df['r4'+p]), (df['r4'+p] < df['close']) & (df['close'] < df['r5'+p]), (df['r5'+p] < df['close']) & (df['close'] < df['r6'+p])]

#                 choices_cam_positions = [-6, 6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

#                 df['cam'+p+'_position'] = numpy.select(conditions_cam_positions, choices_cam_positions, default=numpy.nan)

            # print(df[['date', 'date_D', 'date_M', 'pdc', 'pdh', 'pdl', 'pmh', 'pml', 'pdh_D', 'pdl_D', 'pMc', 'pMh', 'pMl']].to_string())
            # print(df[['date', 'date_D', 'date_M', 'pMc', 'pMh', 'pMl', 'cam_position', 'cam_position_D', 'cam_position_M']].to_string())


        # if "atr_old" in indicators_list:
            # atr_list = [df_day.loc[df_day['date'] == datetime.strptime(pd.to_datetime(row['date']).strftime('%Y-%m-%d'), '%Y-%m-%d').date(), 'atr_D'].to_frame().iloc[-1]['atr_D'] for index, row in df.iterrows()]
            # df_atr = pd.DataFrame(atr_list, columns=["atr"])
            # df["atr"] = df_atr["atr"]

            # print(df[["date", "close", "atr"]].to_string())
            # print(df_day[["date", "close", "atr"]].to_string())

            # ATR bands
            # # Method 1
            # atr_band_high_list, atr_band_low_list = [], []
            # # date_prev = pd.to_datetime(df.loc[0, "date"]).strftime('%Y-%m-%d')
            # # day_low, day_high = df.loc[0, "low"], df.loc[0, "high"]
            # date_prev = pd.to_datetime(df['date'].iloc[0]).strftime('%Y-%m-%d')
            # day_low, day_high = df['low'].iloc[0], df['high'].iloc[0]
            # for i, d in enumerate(df["date"]):
            #     date = pd.to_datetime(df['date'].iloc[i]).strftime('%Y-%m-%d')

            #     if date != date_prev:
            #         date_prev = date
            #         # day_low = df.loc[i, "low"]
            #         # day_high = df.loc[i, "high"]
            #         day_low = df['low'].iloc[i]
            #         day_high = df['high'].iloc[i]
            #     else:
            #         day_low = min(df['low'].iloc[i], day_low)
            #         day_high = max(df['high'].iloc[i], day_high)

            #     atr_band_high_list.append(day_low + df['atr_D'].iloc[i])
            #     atr_band_low_list.append(day_high - df['atr_D'].iloc[i])

            # df_atr_band_high = pd.DataFrame(atr_band_high_list, columns=["atr_band_high"])
            # df_atr_band_low = pd.DataFrame(atr_band_low_list, columns=["atr_band_low"])
            # df["atr_band_high"] = df_atr_band_high["atr_band_high"]
            # df["atr_band_low"] = df_atr_band_low["atr_band_low"]

        # if "levels_old" in indicators_list:
        #     # iterate through df using and create list with previous days values
        #     pdc_list, pdh_list, pdl_list, pmh_list, pml_list, pdp_list, pdp_list_D, pdp_list_M = [], [], [], [], [], [], [], []
        #     date_prev = pd.to_datetime(df['date'].iloc[0]).strftime('%Y-%m-%d')
        #     month_prev = pd.to_datetime(df['date'].iloc[0]).month
        #     index_new_day, index_end_premarket, index_start_postmarket = 0, 0, 0
        #     pdc, pdh, pdl, pmh, pml, pdh_D, pdl_D, pMc, pMh, pMl = None, None, None, None, None, None, None, None, None, None
        #     pp, s1, s2, s3, s4, s5, s6, r1, r2, r3, r4, r5, r6 = None, None, None, None, None, None, None, None, None, None, None, None, None
        #     pp_D, s1_D, s2_D, s3_D, s4_D, s5_D, s6_D, r1_D, r2_D, r3_D, r4_D, r5_D, r6_D = None, None, None, None, None, None, None, None, None, None, None, None, None
        #     pp_M, s1_M, s2_M, s3_M, s4_M, s5_M, s6_M, r1_M, r2_M, r3_M, r4_M, r5_M, r6_M = None, None, None, None, None, None, None, None, None, None, None, None, None

        #     # Get monthly data for monthly pivots calculation

        #     date_init = pd.to_datetime(df['date'].iloc[0])#.strftime('%Y-%m-%d'))
        #     query_time = pd.to_datetime(df["date"].iloc[len(df["date"])-1])#.strftime('%Y-%m-%d'))
        #     num_months = (query_time.year - date_init.year) * 12 + (query_time.month - date_init.month) + 2
        #     whatToShow = "TRADES" if contract.symbol not in get_forex_symbols_list() else "MIDPOINT"
        #     if num_months >= 12: duration = str(numpy.floor(num_months / 12)) + ' Y'
        #     else: duration = str(num_months) + ' M'
        #     bars_M = ib.reqHistoricalData(contract, endDateTime=query_time, durationStr=duration, barSizeSetting='1 month', whatToShow=whatToShow, useRTH=False)
        #     ib.sleep(CONSTANTS.PROCESS_TIME['long'])
        #     df_M = util.df(bars_M)

        #     # pdc_triggered = Trues
        #     for i, d in enumerate(df["date"]):
        #         date = pd.to_datetime(df['date'].iloc[i]).strftime('%Y-%m-%d')
        #         time = pd.to_datetime(df['date'].iloc[i]).strftime('%H-%M-%S')
        #         month = pd.to_datetime(date).month
        #         if i > 0: time_prev = pd.to_datetime(df['date'].iloc[i-1]).strftime('%H-%M-%S')
        #         else: time_prev = time

        #         # Previous day high and low calculation
        #         if date != date_prev:
        #             # print(date)
        #             date_prev = date
        #             # pdh = max([df.loc[j, "high"] for j in range(index_end_premarket, index_start_postmarket)])
        #             # pdl = min([df.loc[j, "low"] for j in range(index_end_premarket, index_start_postmarket)])
        #             pdh = max([df['high'].iloc[j] for j in range(index_new_day, i)])
        #             pdl = min([df['low'].iloc[j] for j in range(index_new_day, i)])
        #             index_new_day = i

        #             # Camarilla Pivots (Non-Daily Based Values)
        #             if pdh and pdl and pdc:
        #                 base_pivots = (pdh - pdl) * 1.1
        #                 pp = (pdh + pdl + pdc) / 3
        #                 r1, r2, r3, r4 = pdc + base_pivots / 12, pdc + base_pivots / 6, pdc + base_pivots / 4, pdc + base_pivots / 2
        #                 r5, r6 = r4 + 1.168 * (r4 - r3), (pdh / pdl) * pdc
        #                 s1, s2, s3, s4 = pdc - base_pivots / 12, pdc - base_pivots / 6, pdc - base_pivots / 4, pdc - base_pivots / 2
        #                 s5, s6 = s4 - 1.168 * (s3 - s4), pdc - (r6 - pdc)

        #         # Previous day close calculation
        #         if time >= "16-00-00" and time_prev < "16-00-00":
        #             pdc = df['close'].iloc[i-1]
        #             index_start_postmarket = i

        #         # Premarket high and low calculation
        #         if time >= "09-30-00" and time_prev < "09-30-00":
        #             pmh = max([df['high'].iloc[j] for j in range(index_new_day, i)])
        #             pml = min([df['low'].iloc[j] for j in range(index_new_day, i)])
        #             # pdh_D = max([df.loc[j, "high"] for j in range(index_end_premarket, i)])
        #             # pdl_D = min([df.loc[j, "low"] for j in range(index_end_premarket, i)])
        #             pdh_D = max([df['high'].iloc[j] for j in range(index_end_premarket, index_start_postmarket)], default=None)
        #             pdl_D = min([df['low'].iloc[j] for j in range(index_end_premarket, index_start_postmarket)], default=None)
        #             index_end_premarket = i

        #             # Camarilla Pivots (Daily Based Values)
        #             if pdh_D and pdl_D and pdc:
        #                 base_pivots_D = (pdh_D - pdl_D) * 1.1
        #                 pp_D = (pdh_D + pdl_D + pdc) / 3
        #                 r1_D, r2_D, r3_D, r4_D = pdc + base_pivots_D / 12, pdc + base_pivots_D / 6, pdc + base_pivots_D / 4, pdc + base_pivots_D / 2
        #                 r5_D, r6_D = r4_D + 1.168 * (r4_D - r3_D), (pdh_D / pdl_D) * pdc
        #                 s1_D, s2_D, s3_D, s4_D = pdc - base_pivots_D / 12, pdc - base_pivots_D / 6, pdc - base_pivots_D / 4, pdc - base_pivots_D / 2
        #                 s5_D, s6_D = s4_D - 1.168 * (s3_D - s4_D), pdc - (r6_D - pdc)

        #         # Previous month high and low calculation
        #         prev_month = lambda d: 12 if pd.to_datetime(d).month == 1 else pd.to_datetime(d).month - 1
        #         pMc = df_M.loc[pd.to_datetime(df_M['date']).dt.month == prev_month(date), 'close'].iloc[0]
        #         pMh = df_M.loc[pd.to_datetime(df_M['date']).dt.month == prev_month(date), 'high'].iloc[0]
        #         pMl = df_M.loc[pd.to_datetime(df_M['date']).dt.month == prev_month(date), 'low'].iloc[0]

        #         if pMh and pMl and pMc:
        #             base_pivots_M = (pMh - pMl) * 1.1
        #             pp_M = (pMh + pMl + pMc) / 3
        #             r1_M, r2_M, r3_M, r4_M = pMc + base_pivots_M / 12, pMc + base_pivots_M / 6, pMc + base_pivots_M / 4, pMc + base_pivots_M / 2
        #             r5_M, r6_M = r4_M + 1.168 * (r4_M - r3_M), (pMh / pMl) * pMc
        #             s1_M, s2_M, s3_M, s4_M = pMc - base_pivots_M / 12, pMc - base_pivots_M / 6, pMc - base_pivots_M / 4, pMc - base_pivots_M / 2
        #             s5_M, s6_M = s4_M - 1.168 * (s3_M - s4_M), pMc - (r6_M - pMc)

        #             pdp_list_M.append([pp_M, s1_M, s2_M, s3_M, s4_M, s5_M, s6_M, r1_M, r2_M, r3_M, r4_M, r5_M, r6_M])
        #         else:
        #             pdp_list_M.append([None, None, None, None, None, None, None, None, None, None, None, None, None])


        #         if index_start_postmarket == i:
        #             pdc_list.append(None)
        #         else:
        #             pdc_list.append(pdc)

        #         if index_new_day == i:
        #             pdh_list.append(None)
        #             pdl_list.append(None)
        #             pdp_list.append([None, None, None, None, None, None, None, None, None, None, None, None, None])
        #         else:
        #             pdh_list.append(pdh)
        #             pdl_list.append(pdl)
        #             pdp_list.append([pp, s1, s2, s3, s4, s5, s6, r1, r2, r3, r4, r5, r6])

        #         if index_end_premarket == i:
        #             pmh_list.append(None)
        #             pml_list.append(None)
        #             pdp_list_D.append([None, None, None, None, None, None, None, None, None, None, None, None, None])
        #         else:
        #             pmh_list.append(pmh)
        #             pml_list.append(pml)
        #             pdp_list_D.append([pp_D, s1_D, s2_D, s3_D, s4_D, s5_D, s6_D, r1_D, r2_D, r3_D, r4_D, r5_D, r6_D])

        #     levels_list = [pdc_list, pdh_list, pdl_list, pmh_list, pml_list] + [list(x) for x in zip(*pdp_list)] + [list(x) for x in zip(*pdp_list_D)] + [list(x) for x in zip(*pdp_list_M)]
        #     levels_titles = ['pdc', 'pdh', 'pdl', 'pmh', 'pml', 'pp', 's1', 's2', 's3', 's4', 's5', 's6', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'pp_D', 's1_D', 's2_D', 's3_D', 's4_D', 's5_D', 's6_D', 'r1_D', 'r2_D', 'r3_D', 'r4_D', 'r5_D', 'r6_D', 'pp_M', 's1_M', 's2_M', 's3_M', 's4_M', 's5_M', 's6_M', 'r1_M', 'r2_M', 'r3_M', 'r4_M', 'r5_M', 'r6_M']

        #     for i, level in enumerate(levels_list):
        #         df_level = pd.DataFrame(level, columns=[levels_titles[i]])
        #         df[levels_titles[i]] = df_level[levels_titles[i]]


            # print(df.to_string())
            # print(df[["date", "close", "pdc", "pdh", "pdl", "pmh", "pml"]].to_string())
            # input()


        # df["diff"] = bars_df.close.diff()
        # df["delta"] = bars_df.close - close_previous_day

    # else: print("\nCould not add indicators, DataFrame is empty.\n")

    # return df


# def get_RSI(ib, symbol, timeframe, query_time=date_local_to_EST(datetime.now())):

#     duration = "1W"
#     if "min" in timeframe or "sec" in timeframe: duration = "1 W"
#     elif "hour" in timeframe or "day" in timeframe: duration = "1 M"
#     elif "month" in timeframe: duration = "1Y"

#     df = get_symbol_hist_data(ib, symbol, timeframe, query_time, duration, indicators_list = ["rsi"])
#     # RSI =

#     # print(df.to_string())
#     print(df[["date", "close", "rsi"]].to_string())
#     print(df['rsi'].iloc[-1])
#     input()

#     # return RSI



# def get_symbol_hist_data_recursive(ib, symbol, timeframe, time_brackets, step_duration="1 D", exisiting_df=pd.DataFrame(), indicators_list=None):

#     contract, symbol = get_symbol_contract(ib, symbol)

#     if step_duration == 'auto':
#         step_duration = get_ibkr_fetch_duration_from_timeframe(timeframe)

#     df = exisiting_df
#     try:
#         for time_bracket in time_brackets:
#             to_time , from_time = time_bracket['to_time'], time_bracket['from_time']
#             if to_time > from_time:
#                 to_time_local = to_time
#                 while to_time_local > from_time:
#                     print("\nCreating DataFrame for ", symbol, " ", timeframe, " from ", to_time_local, ", duration ", step_duration)
#                     df_local = get_symbol_hist_data(ib, symbol, timeframe, query_time=to_time_local, duration=step_duration, indicators_list=indicators_list)

#                     if pd.to_datetime(df_local['date'].iloc[0]).tzinfo is None: df_local['date'] = pd.to_datetime(df_local['date']).dt.tz_localize(CONSTANTS.TZ_WORK)
#                     df = pd.concat([df, df_local], ignore_index=True).drop_duplicates(subset=['date'])
#                     to_time_local = df_local['date'].iloc[0]

#             else: print("Could not create DataFrame, Start Time < End Time")

#         if not df.empty:
#             df.sort_values(by=['date'], ascending=True, inplace=True)
#             df.reset_index(inplace = True, drop=True)
#             # df = add_indicator(df, ib, contract, indicators_list)

#     except Exception as e: print("Could not create DataFrame for symbol ", symbol, ". Error: ", e, "  |  Full error: ", traceback.format_exc())

#     return df

# def get_symbol_hist_data(ib, symbol, timeframe, query_time=date_local_to_EST(datetime.now()), duration="1 D", indicators_list=['all']):

#     contract, symbol = get_symbol_contract(ib, symbol)
#     if contract:

#         end_query_time = substract_duration_from_time(query_time, duration)

#         # Determine if all or portion of time period requested is already saved
#         # Check if data are present locally first, including all required indicators
#         hist_folder_symbol = os.path.join(PATHS.folders_path['hist_market_data'], str(symbol))
#         if not os.path.exists(hist_folder_symbol): os.mkdir(hist_folder_symbol)
#         df_existing, to_time_exist, from_time_exist, file_path = check_existing_mkt_data_file(symbol, timeframe, hist_folder_symbol, delete_file=False)

#         time_brackets = []
#         if to_time_exist and from_time_exist:

#             if query_time < to_time_exist:
#                 to_time = query_time
#             else:
#                 to_time = to_time_exist
#                 time_brackets.append({'query_time': query_time, 'duration': duration})#get_ibkr_duration_from_time_diff(query_time, to_time_exist)})

#             if end_query_time > from_time_exist:
#                 from_time = end_query_time
#             else:
#                 from_time = from_time_exist
#                 time_brackets.append({'query_time': from_time_exist, 'duration': duration})#get_ibkr_duration_from_time_diff(from_time_exist, end_query_time)})

#             df = df_existing[(df_existing['date'] <= to_time) & (df_existing['date'] >= from_time)]

#         else:
#             df = pd.DataFrame()
#             time_brackets.append({'query_time': query_time, 'duration': duration})

#         # Get remaining of historical data from IBKR
#         for time_bracket in time_brackets:
#             query_time = time_bracket['query_time']
#             duration = time_bracket['duration']

#             whatToShow = "TRADES"
#             if symbol in get_forex_symbols_list():
#                 whatToShow = "MIDPOINT"
#                 if "levels" in indicators_list:
#                     indicators_list.remove("levels")

#             bars = ib.reqHistoricalData(
#                 # contract, endDateTime=query_time, durationStr='3600 S',
#                 # barSizeSetting='10 secs', whatToShow='TRADES', useRTH=False) # Available Settings: https://interactivebrokers.github.io/tws-api/historical_bars.html
#                 contract, endDateTime=query_time, durationStr=duration,
#                 barSizeSetting=timeframe, whatToShow=whatToShow, useRTH=False) # Available Settings: https://interactivebrokers.github.io/tws-api/historical_bars.html
#             ib.sleep(CONSTANTS.PROCESS_TIME['long'])

#             # Convert to pandas dataframe (pandas needs to be installed):
#             bars_df = util.df(bars)

#             if pd.to_datetime(bars_df['date'].iloc[0]).tzinfo is None: bars_df['date'] = pd.to_datetime(bars_df['date']).dt.tz_localize(CONSTANTS.TZ_WORK)
#             df = pd.concat([df, bars_df], ignore_index=True).drop_duplicates(subset=['date'])
#             if not df.empty:
#                 df.sort_values(by=['date'], ascending=True, inplace=True)
#                 df.reset_index(inplace = True, drop=True)

#         if indicators_list:
#             df = indicators.add_indicator(df, ib, contract, indicators_list)

#     return df

# def get_symbol_hist_data_recursive(ib, symbol, timeframe, to_time=None, from_time=None, step_duration="1 D", indicators_list=['all'], hist_folder=None, file_format=None) -> pd.DataFrame:

#     if indicators_list is None:
#         indicators_list = ['all']

#     if step_duration == 'auto':
#         step_duration = get_ibkr_fetch_duration_from_timeframe(timeframe)

#     # contract, symbol = get_symbol_contract(ib, symbol)
#     # if not contract:
#     #     return pd.DataFrame()

#     hist_folder = hist_folder if hist_folder else PATHS.folders_path['hist_market_data']
#     hist_folder_symbol = os.path.join(hist_folder, str(symbol))
#     os.makedirs(hist_folder_symbol, exist_ok=True)

#     # df_existing, existing_start, existing_end, existing_file_path = check_existing_mkt_data_file(
#     #     symbol, timeframe, hist_folder_symbol, delete_file=False)

#     # # Complete mode overrides time range
#     # if complete:
#     #     if existing_end is None:
#     #         print(f"[{symbol}] No existing data found for 'complete' mode.")
#     #         return pd.DataFrame()
#     #     to_time = date_local_to_EST(datetime.now())
#     #     from_time = existing_end

#     if not to_time or not from_time:
#         raise ValueError("Either provide both to_time and from_time")

#     # # Skip existing data
#     # if existing_start and existing_end:
#     #     if to_time >= existing_end:
#     #         print(f"[{symbol}] All data already fetched up to {existing_end}. Nothing new to fetch.")
#     #         return df_existing
#     #     # Skip existing data range by adjusting to_time to just after last existing record
#     #     to_time = min(to_time, existing_end + timedelta(seconds=1))

#     step_duration_td = get_ibkr_duration_as_timedelta(step_duration)
#     all_data = pd.DataFrame()
#     # latest_loaded_time = None
#     current_start = to_time

#     try:
#         # # # Split time range into gaps: before existing data, and after
#         # # missing_ranges = []

#         # # if existing_start and existing_end:
#         # #     if to_time < existing_start:
#         # #         missing_ranges.append((to_time, existing_start - timedelta(seconds=1)))
#         # #     if from_time > existing_end:
#         # #         missing_ranges.append((existing_end + timedelta(seconds=1), from_time))
#         # # else:
#         # #     missing_ranges.append((to_time, from_time))

#         # # Loop through missing ranges and fetch in chunks
#         # for range_start, range_end in missing_ranges:
#             # for chunk_start, chunk_end in generate_time_chunks(range_end, range_start, step_duration_td):
#         # for chunk_start, chunk_end in generate_time_chunks(to_time, from_time, step_duration_td):
#         while current_start > from_time:
#             chunk_start = current_start
#             chunk_end = chunk_start - step_duration_td
#             if chunk_end < from_time:
#                 chunk_end = from_time

#             print(f"⏳ Fetching data for {symbol}, {timeframe}, {step_duration} | {chunk_end} → {chunk_start}")
#             df_chunk = get_symbol_hist_data(ib, symbol, timeframe, query_time=chunk_start,
#                                             duration=step_duration, indicators_list=indicators_list, file_format=file_format)

#             if df_chunk.empty:
#                 print(f"⚠️ No data for chunk: {chunk_start} → {chunk_end}")
#                 current_start = chunk_end - timedelta(seconds=1)
#                 continue

#             df_chunk = format_df_date(df_chunk)
#             all_data = pd.concat([all_data, df_chunk], ignore_index=True).drop_duplicates(subset=['date'])
#             all_data.sort_values('date', inplace=True)

#             # # Update to_time to be the latest date minus a second (to go backward)
#             # latest_loaded_time = df_chunk['date'].min()
#             # if latest_loaded_time and latest_loaded_time <= from_time:
#             #     # We reached or passed the end of requested data
#             #     break
#             # to_time = latest_loaded_time - timedelta(seconds=1)

#             # Move the window backward based on the oldest date in this chunk
#             current_start = df_chunk['date'].min() - timedelta(seconds=1)

#         all_data.reset_index(drop=True, inplace=True)

#         # # Merge with existing data
#         # if not df_existing.empty:
#         #     all_data = pd.concat([df_existing, all_data], ignore_index=True).drop_duplicates(subset=['date'])
#         #     all_data.sort_values('date', inplace=True)
#         #     all_data.reset_index(drop=True, inplace=True)

#         return all_data

#     except Exception as e:
#         print(f"❌ Error fetching {symbol} from {to_time} to {from_time}: {e}")
#         print(traceback.format_exc())
#         return all_data



if __name__ == "__main__":

    # current_path = os.path.realpath(__file__)
    # path = path_current_setup(current_path)

    input("\nEnter anything to exit")
