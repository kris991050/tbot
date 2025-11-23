import os, sys, time, requests, csv, re, prettytable, numpy as np, json, speedtest, pandas_market_calendars, arch
import xml.etree.ElementTree as ET, traceback, speedtest, typing, chardet, pickle, filelock, yfinance, polygon, re
import math, pandas as pd, psutil, gc#, dask.dataframe as dd
from datetime import datetime, timedelta
from ib_insync import *

current_folder = os.path.dirname(os.path.realpath(__file__))
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)
from miscellaneous import newsScraper
from utils.timeframe import Timeframe
from utils.constants import CONSTANTS, PATHS, FORMATS
from utils import logs

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
        message = "\r{} seconds.".format(time_wait - i)
        if isinstance(sys.stdout, logs.DualLogger):
            sys.stdout.write(message, no_log=True)
            # print(message, end='', no_log=True)
        else:
            print(message, end='')

        if not ib: time.sleep(1)
        else: ib.sleep(1)


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


def IBKRConnect_any(ib:IB=IB(), paper:bool=True, client_id:int=None, remote:bool=True):
    ib, ibConnection = IBKRConnect(ib, paper=paper, client_id=client_id, remote=remote)
    if not ibConnection:
        ib, ibConnection = IBKRConnect(IB(), paper=not paper, client_id=client_id, remote=remote)

    return ib, ib.isConnected()


def IBKRConnect(ib:IB=IB(), paper:bool=True, client_id:int=None, remote:bool=True):
    ib_ip = CONSTANTS.IB_IP_REMOTE if remote else CONSTANTS.IB_IP_LOCAL
    port_number = CONSTANTS.IB_PORT_PAPER if paper else CONSTANTS.IB_PORT_LIVE
    # clientId = 1 if not client_id else client_id

    client_ids = [client_id] if client_id else range(1, CONSTANTS.IB_MAX_CLIENTS + 1)
    
    try:
        for clientId in client_ids:
            if ib.isConnected():
                break
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


def get_entry_delay_from_timeframe(timeframe:Timeframe):
    return 0 if timeframe.to_timedelta >= Timeframe(CONSTANTS.ENTRY_DELAY_CUTOFF_TIMEFRAME).to_timedelta else 1


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


def calculate_garch_volatility(price_data:pd.Series, p:int=1, q:int=1, forecast_horizon:int=1) -> float:
    """
    Calculate the volatility using a GARCH model.

    :param price_data: A pandas Series of historical price data.
    :param p: The order of the GARCH model (GARCH(p, q)).
    :param q: The order of the GARCH model (GARCH(p, q)).
    :param forecast_horizon: The number of steps ahead to forecast volatility.
    :return: Forecasted volatility.
    """
    # Calculate returns
    returns = price_data.pct_change().dropna() * 100  # Percentage returns
    returns_scaled = returns * 10  # Scaling factor recommended to avoid model convergence issues

    # Fit GARCH model
    model = arch.arch_model(returns_scaled, vol='Garch', p=p, q=q)
    model_fit = model.fit(disp="off")

    # Forecast volatility for the next forecast_horizon periods
    forecast = model_fit.forecast(horizon=forecast_horizon)
    pred_vlty = forecast.variance.values[-1, 0]  # Forecasted variance for the last period
    pred_vlty = np.sqrt(pred_vlty)  / 10  # Rescale the volatility  (Volatility is the square root of variance)

    return pred_vlty


def calculate_ewma_volatility(price_data:pd.Series, lambda_:float=0.94, prev_volatility:float=0.0) -> float:
    """
    Calculate the volatility using the EWMA method.

    :param price_data: A pandas Series of historical price data.
    :param lambda_: The smoothing factor (default is 0.94).
    :param prev_volatility: The previous volatility value (default is 0.0 for the first period).
    :return: Forecasted volatility.
    """
    # Calculate returns
    returns = price_data.pct_change().dropna() * 100  # Percentage returns
    squared_returns = returns ** 2

    # Calculate the EWMA volatility
    if prev_volatility == 0.0:  # If it's the first calculation, initialize volatility
        volatility = np.sqrt(squared_returns.mean()).values[0]  # Use the mean of squared returns as initial volatility
    else:
        volatility = np.sqrt(lambda_ * squared_returns.iloc[-1] + (1 - lambda_) * prev_volatility ** 2).values[0]

    return volatility


def calculate_weighted_atr(atr_series:pd.Series, lambda_:float=0.94):
    """
    Calculate a weighted ATR using an exponential moving average (EMA).
    
    :param atr_series: The ATR series.
    :param lambda_: The smoothing factor (default 0.94).
    :return: The weighted ATR series.
    """
    # Apply the EMA to the ATR series
    return atr_series.ewm(span=(2 / (1 - lambda_)) - 1, adjust=False).mean()




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


def get_path_daily_logs_folder(date=None, create_if_none=True, local=False):

    date = date or datetime.now(CONSTANTS.TZ_WORK)
    if not local: root_folder_path = get_path_date_folder(date, create_if_none, local)
    else: root_folder_path = os.getcwd()
    daily_logs_folder_path = os.path.join(root_folder_path, 'live_logs')

    if not os.path.exists(daily_logs_folder_path) and create_if_none:
        os.mkdir(daily_logs_folder_path)

    return daily_logs_folder_path


def path_current_setup(path_current_file, ch_dir=True, print_path=True):

    # Current Directory Path Setup
    #name = os.path.basename(__file__)
    path_current = os.path.dirname(path_current_file)
    print("path_current = ", path_current)
    if ch_dir: os.chdir(path_current)
    if print_path: print("\nCurrent Directory = ", os.getcwd())

    return path_current


def construct_data_path(local_hist_folder, symbol, timeframe, to_time, from_time, file_format='csv', data_type='hist_data'):
    filename = FORMATS.construct_filename(symbol=symbol, timeframe=timeframe, to_time=to_time, from_time=from_time, file_format=file_format, data_type=data_type)
    return os.path.join(local_hist_folder, filename)


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


def save_json(obj, path, lock=False):
    if lock:
        lock_file = filelock.FileLock(f"{path}.lock")
        with lock_file:  # Acquires the lock here
            with open(path, 'w') as f:
                json.dump(obj, f, indent=2, default=str)  # str conversion for datetimes
    else:
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2, default=str)  # str conversion for datetimes


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

    for file_format in file_formats:
        file_ext = '.' + file_format

        if data_type in data_type_formats:
            filename_pattern = build_data_filename_pattern(timeframe, file_format=file_format, data_type=data_type)
        else:
            raise ValueError(f"Unsupported data type '{data_type}'. Supported: {data_type_formats}")

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

        # date_now = date_local_to_EST(datetime.now())
        date_now = datetime.now(CONSTANTS.TZ_WORK)
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

    return date_EST


def date_to_EST_aware(date, reverse=False):
    timezone_EST = CONSTANTS.TZ_WORK

    if not reverse:
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


def calculate_now(sim_offset=timedelta(0), tz=CONSTANTS.TZ_WORK) -> datetime:
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


def get_market_holidays(start_date, end_date, exchange='NYSE'):
    # Get the market calendar for the specified exchange (default: NYSE)
    market = pandas_market_calendars.get_calendar(exchange)

    # Get the market's holiday schedule for the specified range
    holidays = market.holidays().holidays

    # Filter the holidays that fall within the given date range
    holidays_in_range = [holiday for holiday in holidays if start_date.date() <= holiday <= end_date.date()]

    # Return as a pandas datetime index
    return pd.to_datetime(holidays_in_range)


def is_between_market_times(start_label, end_label, now:datetime=None, timezone=CONSTANTS.TZ_WORK):
    now = now.time() if now else datetime.now(tz=timezone).time()
    return CONSTANTS.TH_TIMES[start_label] < now < CONSTANTS.TH_TIMES[end_label]




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


def get_symbol_seed_list(seed:int, base_folder=PATHS.folders_path['market_data']):
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
    '''
    Filters a DataFrame by multiple date ranges and displays selected columns.
    Example of date_ranges: date_ranges = [['2025-04-10T00:00:00', '2025-04-11T20:59:59'], ['2025-03-05T00:00:00', '2025-03-09T20:59:59']]
    conditions_list contains column names to be True
    '''

    df = format_df_date(df)
    date_ranges = [[date_to_EST_aware(pd.to_datetime(date)) for date in date_range] for date_range in date_ranges]
    # Filter by date ranges if provided
    if date_ranges:
        date_mask = pd.Series(False, index=df.index)
        for date_range in date_ranges:
            date_mask |= (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
        df = df[date_mask]

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



if __name__ == "__main__":

    input("\nEnter anything to exit")
