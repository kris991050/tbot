import os, platform, pytz, pandas as pd, tzlocal

def get_sys(display=False):

        # if display: print("\nSystem Info: ", platform.uname(), "\n")

        # if 'MacBook' in platform.uname().node: SYS = 'Macbook' #'MacBook-Pro.local'
        # elif 'XCR104' in platform.uname().node: SYS = 'Windows'
        # else: SYS = 'None'

        # return SYS

        SYS = platform.uname().system
        if display: print("\nSystem Type: ", SYS, "\n")
        return SYS


class CONSTANTS():

        SYS_LIST = {'macos': 'Darwin', 'windows': 'Windows', 'linux': 'Linux'}
        SYS = get_sys()

        PROCESS_TIME = {'short': 0.2, 'medium': 0.5, 'long': 1}
        MEMORY_THRESHOLD_MB = 5000
        MAX_CORE_PARTITIONS = 10

        IB_IP_LOCAL = '127.0.0.1'
        IB_IP_REMOTE = '18.116.186.191'
        IB_PORT_LIVE = 7496
        IB_PORT_PAPER = 4002#7497
        IB_MAX_CLIENTS = 10

        TZ_WORK_STR = 'US/Eastern'
        TZ_WORK = pytz.timezone(TZ_WORK_STR)
        TZ_LOCAL = pytz.timezone('US/Pacific')
        # TZ_LOCAL = pytz.timezone(tzlocal.get_localzone().key if hasattr(tzlocal.get_localzone(), 'key') else str(tzlocal.get_localzone()))

        DEFAULT_CURRENCY = 'USD'
        
        POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
        POLYGON_DATA_DELAY = '15min'
        POLYGON_DATA_MAX_BACKWARD = '5Y'

        TH_TIMES = {'start_of_day': pd.to_datetime('00-00-00', format='%H-%M-%S').time(),
                    'pre-market': pd.to_datetime('04-00-00', format='%H-%M-%S').time(),
                    'post-market': pd.to_datetime('16-00-00', format='%H-%M-%S').time(),
                    'rth': pd.to_datetime('09-30-00', format='%H-%M-%S').time(),
                    'end_of_tday': pd.to_datetime('20-00-00', format='%H-%M-%S').time(),
                    'end_of_day': pd.to_datetime('23-59-59', format='%H-%M-%S').time()}

        SECONDS_TO_INTRADAY_TIMEFRAME = {'1.0': '1 sec', '5.0': '5 secs', '10.0': '10 secs', '15.0': '15 secs', '30.0': '30 secs',
                                '60.0': '1 min', '120.0': '2 mins', '180.0': '3 mins', '300.0': '5 mins',
                                '600.0': '10 mins', '900.0': '15 mins', '1200.0': '20 mins', '1800.0': '30 mins',
                                '3600.0': '1 hour', '7200.0': '2 hours', '10800.0': '3 hours', '14400.0': '4 hours', '28800.0': '8 hours'}

        # Conversion table to seconds
        TIME_TO_SEC = {'sec': 1, 'secs': 1, 'second': 1, 'seconds': 1, 'min': 60, 'mins': 60, 'minute': 60, 'minutes': 60, 'hour': 3600, 'hours': 3600, 'h': 3600, 'H': 3600,
                       'day': 86400, 'days': 86400, 'd': 86400, 'D': 86400, 'week': 604800, 'weeks': 604800, 'w': 604800, 'W': 604800,
                       'month': 2592000, 'months': 2592000, 'm': 2592000, 'M': 2592000, 'year': 31536000, 'years': 31536000, 'y': 31536000, 'Y': 31536000}

        MARKET_CAP_CATEGORIES = {'Nano': 50e6, 'Micro': 300e6, 'Small': 2e9, 'Mid': 10e9, 'Large': 200e9, 'Mega': float('inf')}
        
        ENTRY_DELAY_CUTOFF_TIMEFRAME = '1h'

        WARMUP_MAP = {'1min':'1M', '5min':'3M', '15min':'4M', '1h':'6M', '4h':'1Y', '1D':'2Y', '1W':'5Y', '2min':'2M', '30min':'5M'}
        TIMEFRAMES_STD = ['1min', '5min', '15min', '1h', '4h', '1D', '1W']#list(WARMUP_MAP)
        FORBIDDEN_TIMEFRAMES_POLYGON = []#['1M']
        FORBIDDEN_SYMBOLS_POLYGON = ['QIPT', 'LIN']

        MODES = {'live': ['live', 'sim'], 'backtest': ['backtest', 'forward']}
        LIVE_ACTIONS = ['fetch', 'enrich', 'execut']
        FETCHER_TYPES = ['ibkr', 'polygon']
        pred_vlty_TYPES = ['garch', 'ewma']
        DEFAULT_TIMEFRAME = {'multiplier': 1, 'unit': 'minute'}
        DIRECTIONS = ['bull', 'bear']

        LEVELS_ROUNDING_PRECISION = 2

        # Define mappings for each indicator and its category
        INDICATOR_TYPES = [
                {'category': 'trend', 'names': ['ema', 'sma', 'adx', 'macd']},
                {'category': 'volume', 'names': ['vwap', 'r_vol', 'avg_vol', 'pm_vol']},
                {'category': 'volatility', 'names': ['bband', 'atr', 'day_range', 'volatility_ratio', 'volatility_change']},
                {'category': 'momentum', 'names': ['rsi', 'awesome']},
                {'category': 'price', 'names': ['gap', 'change']}, 
                {'category': 'vpa', 'names': ['vpa']}
                ]

        # Define mappings for candle patterns
        PATTERN_TYPES = [
                {'category': 'candle', 'names': ['hammer', 'engulfing', 'marubozu', 'doji', 'volume_spike', 'bullish_score', 'bearish_score', 
                                                 'score_bias', 'return', 'directional_bias', 'bias_trend', 'hybrid_bias', 'hybrid_direction']},
                {'category': 'divergence', 'names': ['divergence']},
                {'category': 'range', 'names': ['low_volume', 'bband_width_pct', 'atr_in_range', 'inside_bar', 'dbscan_cluster', 'consolidation', 'breakout']},
                {'category': 'breakout', 'names': ['breakout', 'price_buffer_pct']},
                {'category': 'trend', 'names': ['trend']},
                {'category': 'index_trend', 'names': ['index_trend']}
                ]
        
        # Define mappings for levels
        LEVEL_TYPES = [
                {'category': 'daily', 'names': ['levels', 'cam_position', 'cam_D_position']},
                {'category': 'monthly', 'names': ['levels_M', 'cam_M_position']},
                {'category': 'camarilla', 'names': ['pivots', 'cam_']}
                ]
        
        # Define support/resistance (SR) settings
        SR_SETTINGS = [{'timeframe': '1W', 'lookback': '5Y', 'refresh_rate': '1M', 'granularity': 1, 'count_threshold': 2, 'proximity_threshold': 0.5},
                      {'timeframe': '1D', 'lookback': '2Y', 'refresh_rate': '1W', 'granularity': 1, 'count_threshold': 2, 'proximity_threshold': 0.5},
                #       {'timeframe': '4h', 'lookback': '1Y', 'refresh_rate': '3D', 'granularity': 0.5, 'count_threshold': 2, 'proximity_threshold': 0.25},
                      {'timeframe': '1h', 'lookback': '3M', 'refresh_rate': '1D', 'granularity': 0.5, 'count_threshold': 2, 'proximity_threshold': 0.25}]
                #       {'timeframe': '5min', 'lookback': '2D', 'refresh_rate': '1h', 'granularity': 0.1, 'count_threshold': 2, 'proximity_threshold': 0.05}]

class PATHS():

        SYS = CONSTANTS.SYS
        SYS_LIST = CONSTANTS.SYS_LIST

        root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        folders_path = {'root': root_folder, 'app': os.path.join(root_folder, 'app')}
        folders_path['stats'] = os.path.join(folders_path['app'],'stats')
        folders_path['data'] = os.path.join(folders_path['app'],'data')

        if SYS == SYS_LIST['macos']:
                document_folder = "/Users/user/Documents/"
                folders_path['journal'] = os.path.join(root_folder, 'trading_journal')
                # folders_path['market_data'] = os.path.join(root_folder, 'market_data')
                folders_path['market_data'] = os.path.join(document_folder, 'market_data')
                folders_path['download'] =  '/Users/user/Downloads'
                IB_GATEWAY_PATH = ''

        elif SYS == SYS_LIST['windows']:
                folders_path['journal'] = os.path.join(root_folder, 'T_journal')
                folders_path['market_data'] = os.path.join(root_folder, 't_data')
                folders_path['download'] = 'C:/Users/ChristopheReis/Downloads'
                IB_GATEWAY_PATH = 'C:/Jts/ibgateway/1041/ibgateway.exe'
        
        elif SYS == SYS_LIST['linux']:
                folders_path['journal'] = os.path.join(root_folder, 't_journal')
                folders_path['market_data'] = os.path.join(root_folder, 't_data')
                folders_path['download'] = '/home/ubuntu/Downloads'
                IB_GATEWAY_PATH = ''

        folders_path['hist_market_data'] = os.path.join(folders_path['market_data'], 'hist_data')
        folders_path['live_data'] = os.path.join(folders_path['market_data'], 'live_data')
        folders_path['strategies_data'] = os.path.join(folders_path['market_data'], 'strategies_data')
        # folders_path['daily_data'] = os.path.join(folders_path['journal'], 'daily_data')

        csv_files = {'main': os.path.join(folders_path['market_data'], 'symbols_main.csv'),
                     'index': os.path.join(folders_path['market_data'], 'symbols_index.csv'),
                     'russell2000': os.path.join(folders_path['market_data'], 'symbols_russell2000.csv'),
                     'stock_list': os.path.join(folders_path['market_data'], 'stock_list.csv')}

        daily_csv_files = {#'rsi-reversal': os.path.join(folders_path['daily_data'], 'symbols_rsi-reversal.csv'),
                           'bb_rsi_reversal': 'symbols_rsi-reversal.csv',
                           'gapper_up': 'symbols_gapper_up.csv',
                           'gapper_down': 'symbols_gapper_down.csv',
                           'earnings': 'earnings.csv'}


class FORMATS():

        # Market Data format file types
        DATA_FILE_FORMATS = {'csv': ['csv'], 'pickle': ['pkl', 'pickle'], 'parquet':['pqt', 'parquet']}
        DATA_FILE_FORMATS_LIST = [ext for exts in DATA_FILE_FORMATS.values() for ext in exts]
        DEFAULT_FILE_FORMAT = 'parquet'
        DATA_TYPE_FORMATS = ['hist_data', 'enriched_data', 'sr', 'levels', 'pivots']

        # Timestamp format Regex-friendly pattern used in filenames
        MKT_DATA_FILENAME_DATETIME_FMT = "%Y-%m-%d-%H-%M-%S"  # Example: 2024-01-01-09-30-00
        MKT_DATA_FILENAME_DATETIME_REGEX = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
        # HIST_MKT_DATA_FILENAME_TEMPLATE = "hist_data_{symbol}_{timeframe}_{from_str}_{to_str}.{file_format}"
        # ENRICHED_MKT_DATA_FILENAME_TEMPLATE = "enriched_data_{symbol}_{timeframe}_{from_str}_{to_str}.{file_format}"
        # SR_FILENAME_TEMPLATE = "sr_{symbol}_{timeframe}_{from_str}_{to_str}.{file_format}"
        # LEVELS_FILENAME_TEMPLATE = "levels_{symbol}_{timeframe}_{from_str}_{to_str}.{file_format}"
        # PIVOTS_FILENAME_TEMPLATE = "pivots_{symbol}_{timeframe}_{from_str}_{to_str}.{file_format}"
        DATA_FILENAME_TEMPLATE = "{data_type}_{symbol}_{timeframe}_{from_str}_{to_str}.{file_format}"
        # RESULTS_STRATEGY_REGEX = re.compile(r'results_(?P<strategy>.+)_(?P<timeframe>[^_]+)_(?P<target>[^.]+)\.csv$')
        # RESULTS_STRATEGY_REGEX = re.compile(r'^results_(?P<strategy>.+)_(?P<timeframe>\d+ (?:min|hour|day))_(?P<target>.+)\.csv$')
        # RESULTS_STRATEGY_REGEX = re.compile(r'^results_(?P<strategy>[^_]+(?:_[^_]+){2})_(?P<timeframe>\d+\s*(?:min|h|D|W))_(?P<target>.+)\.csv$')
        # RESULTS_STRATEGY_REGEX = re.compile(r'^results_(?P<strategy>[^_]+(?:_[^_]+){2})_(?P<timeframe>\d+\s*(?:min|h|D|W))_(?P<target>.+?)\.csv$')
        # RESULTS_STRATEGY_REGEX = re.compile(r'^results_(?P<strategy>[^_]+(?:_[^_]+){2})_(?P<timeframe>\d+\s*(?:min|h|D|W))_(?P<target>.+)\.csv$')
        # RESULTS_STRATEGY_REGEX = re.compile(r'^results_(?P<strategy>[^_]+(?:_[^_]+){2})_(?P<timeframe>\d+\s*(?:min|h|D|W))_(?P<target>[^_]+(?:_\d+\s*(?:min|h|D|W))*)\.csv$')
        # RESULTS_STRATEGY_REGEX = re.compile(r'^results_(?P<strategy>[^_]+(?:_\d+\s*(?:min|h|D|W))_[^_]+)_(?P<timeframe>\d+\s*(?:min|h|D|W))_(?P<target>[^_]+(?:_\d+\s*(?:min|h|D|W))*)\.csv$')
        # RESULTS_STRATEGY_REGEX = re.compile(r'^results_(?P<strategy>[^_]+(?:_\d+\s*(?:min|h|D|W))_[^_]+)_(?P<timeframe>\d+\s*(?:min|h|D|W))_(?P<target>.+)\.csv$')
        # RESULTS_STRATEGY_REGEX = re.compile(r'^results_(?P<strategy>[^_]+(?:_\d+\s*(?:min|h|D|W))_[^_]+)_(?P<timeframe>\d+\s*(?:min|h|D|W))_(?P<target>.+)\.csv$')
        # RESULTS_STRATEGY_REGEX = re.compile(r'^results_(?P<strategy>[^_]+(?:_\d+(?:min|h|D|W))_[^_]+)_(?P<timeframe>\d+(?:min|h|D|W))_(?P<target>.+)\.csv$')











        @staticmethod
        def construct_filename(symbol, timeframe, to_time, from_time, file_format='csv', data_type='hist') -> str:
                """
                Returns a standardized market data filename, with formatted timestamps.
                """
                to_time = pd.to_datetime(to_time, utc=True).tz_convert(CONSTANTS.TZ_WORK)
                from_time = pd.to_datetime(from_time, utc=True).tz_convert(CONSTANTS.TZ_WORK)

                to_str = to_time.strftime(FORMATS.MKT_DATA_FILENAME_DATETIME_FMT)
                from_str = from_time.strftime(FORMATS.MKT_DATA_FILENAME_DATETIME_FMT)

                if any(item in data_type for item in FORMATS.DATA_TYPE_FORMATS):
                        return FORMATS.DATA_FILENAME_TEMPLATE.format(data_type=data_type, symbol=symbol, timeframe=timeframe, to_str=to_str,
                                                                        from_str=from_str, file_format=file_format)

                # if data_type == 'hist':
                #         return FORMATS.HIST_MKT_DATA_FILENAME_TEMPLATE.format(symbol=symbol, timeframe=timeframe, to_str=to_str,
                #                                                         from_str=from_str, file_format=file_format)
                # elif data_type == 'enriched':
                #         return FORMATS.ENRICHED_MKT_DATA_FILENAME_TEMPLATE.format(symbol=symbol, timeframe=timeframe, to_str=to_str,
                #                                                         from_str=from_str, file_format=file_format)
                # elif 'sr' in data_type:
                #         return FORMATS.SR_FILENAME_TEMPLATE.format(symbol=symbol, timeframe=timeframe, to_str=to_str,
                #                                                         from_str=from_str, file_format=file_format)
                # elif 'levels' in data_type:
                #         return FORMATS.LEVELS_FILENAME_TEMPLATE.format(symbol=symbol, timeframe=timeframe, to_str=to_str,
                #                                                         from_str=from_str, file_format=file_format)
                # elif 'pivots' in data_type:
                #         return FORMATS.PIVOTS_FILENAME_TEMPLATE.format(symbol=symbol, timeframe=timeframe, to_str=to_str,
                #                                                         from_str=from_str, file_format=file_format)
                else:
                        print("Type must be 'hist_data', 'enriched_data', 'sr', 'levels' or 'pivots'.")
                        return None