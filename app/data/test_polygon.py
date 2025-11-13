import sys, os, polygon, datetime, pandas as pd, pytz, dask.dataframe as dd, gc, cProfile, pstats, duckdb
# from dask.distributed import Client as dClient
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)
os.environ["MODIN_ENGINE"] = "dask"

from utils import constants, helpers
from utils.timeframe import Timeframe
from features import feature_builder
import data.market_data_fetcher as market_data_fetcher


def unix_to_date(unix:int):
    timestamp = datetime.datetime.fromtimestamp(unix / 1000, tz=pytz.utc)
    timestamp = pd.to_datetime(timestamp).tz_convert(tz=constants.CONSTANTS.TZ_WORK)
    return timestamp


def date_to_unix(date_str):
    return int(pd.to_datetime(date_str).timestamp() * 1000)
# return int(datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").timestamp())


def get_rsi_for_date(pclient, symbol, date_str):
    # Get RSI data for the specified symbol and date
    rsi_data = pclient.get_rsi(ticker=symbol, timestamp=date_str, timespan="minute", adjusted="true", window="14",
                               series_type="close", expand_underlying=True, order="desc", limit="1")

    # Check if we have any data and return the RSI value, otherwise return None
    if rsi_data:
        return rsi_data.values  # Return the RSI value of the first element
    return None


def get_stock_data(symbol):
    api_key = constants.CONSTANTS.POLYGON_API_KEY
    pclient = polygon.RESTClient(api_key=api_key)


    date_from = date_to_unix("2025-12-01T00:00:00")
    date_to = date_to_unix("2025-10-01T20:00:00")
    # date_from2 = date_to_unix2("2025-06-01T00:00:00Z")
    # date_to2 = date_to_unix2("2025-10-01T20:00:00Z")

    # try:
    print()

    timeframe = 'minute'
    timeframe_short = 'T'

    t_now = datetime.datetime.now()

    response = pclient.list_aggs(ticker=symbol, multiplier=1, timespan=timeframe, from_=date_from, to=date_to, limit=50000)
    # response = pclient.list_aggs(ticker=symbol, multiplier=1, timespan="minute", from_=date_from, limit=50000)

    df = pd.DataFrame(response)
    # df['date2'] = df['timestamp'].apply(unix_to_date)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(constants.CONSTANTS.TZ_WORK)
    # df_copy = df.copy()
    df.set_index('date', inplace=True)
    df_copy = df
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=timeframe_short)
    df = df.reindex(full_range, fill_value=None)
    # df[['volume', 'transactions']] = df[['volume', 'transactions']].where(df[['volume', 'transactions']].isna(), 0)
    df[['volume', 'transactions']] = df[['volume', 'transactions']].fillna(0)

    df.drop('timestamp', axis=1, inplace=True)
    columns_to_ffill = ['close', 'open', 'high', 'low', 'vwap']
    df[columns_to_ffill] = df[columns_to_ffill].ffill()
    df = df.ffill()
    # df['rsi'] = df['date'].apply(lambda date: get_rsi_for_date(pclient, symbol, date.strftime('%Y-%m-%d')))
    print(df)
    print(f"Time to load data: {datetime.datetime.now() - t_now}")
    # print(df_copy)
    print()


    # response = []
    # for r in pclient.list_aggs(ticker=symbol, multiplier=1, timespan="minute", from_=date_from, to=date_to, limit=50000):
    #     response.append(r)
    # for r in response:
    #     print(r)

    # trades = []
    # for t in pclient.list_trades(ticker="AAPL",order="asc",limit=10,sort="timestamp"):
    #     trades.append(t)


    # rsi = pclient.get_rsi(ticker=symbol,timestamp="2025-10-01", timespan="day", adjusted="true",window="14",series_type="close",order="desc",limit="10")
    # print(rsi)

    # timestamp = datetime.fromtimestamp(1759340100000 / 1000, tz=timezone.utc)

    # ticker_details = pclient.get_ticker_details(symbol)
    # pshares = ticker_details.share_class_shares_outstanding
    # fshares = helpers.get_volumes_from_Finviz(symbol)
    # print(pshares)
    # print(fshares[2])
    # print(pshares-fshares[2])
    print()

    # snapshot = pclient.get_snapshot_ticker("stocks",symbol)
    # print(snapshot)

    # movers = pclient.get_snapshot_direction("stocks", direction="gainers")
    # print(movers)
    print()


    # tickers = pclient.list_tickers(market="stocks", active="true", order="asc", limit=5, sort="ticker")
    # print(tickers)
    # for t in tickers:
    #     print(t)
    #     print()

    # tickers = []
    # for t in pclient.list_tickers(market="stocks", active="true", order="asc", limit=100, sort="ticker"):
    #     tickers.append(t)

    # print(tickers)




    # # Check if the response contains data
    # if response:
    #     print(f"Market Data for {symbol}:")
    #     print(json.dumps(response, indent=4))  # Pretty print the data
    # else:
    #     print(f"No data found for {symbol}.")
    # # except Exception as e:
    # #     print(f"Error fetching data: {str(e)}")

def profile_merge_function(df1, df2):
    df = pd.merge(df1, df2, on='date', how='left')
    return df

if __name__ == "__main__":

    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    # Args Setup
    args = sys.argv
    paperTrading = not 'live' in args
    single_symbol = next((arg[7:] for arg in args if arg.startswith('symbol=')), None)

    ib, ibConnection = helpers.IBKRConnect_any(IB(), paper=paperTrading)
    print()
    # # Example usage: Get market data for AAPL
    # get_stock_data('ACIW')
    # get_stock_data('D')
    # get_stock_data('AAPL')

    # date_from = date_to_unix("2025-12-01T00:00:00Z")
    # date_to = date_to_unix("2025-10-01T20:00:00Z")

    # ray.init(num_cpus=4, ignore_reinit_error=True)
    # print(ray.cluster_resources())

    # path_df1 = '/Users/user/Documents/market_data/hist_data/PGNY/enriched_PGNY_df.parquet'
    # path_df2 = '/Users/user/Documents/market_data/hist_data/PGNY/enriched_PGNY_df_tf.parquet'
    path_df1 = "C:/Users/ChristopheReis/Documents/T/t_data/hist_data/PGNY/enriched_PGNY_df.parquet"
    path_df2 = "C:/Users/ChristopheReis/Documents/T/t_data/hist_data/PGNY/enriched_PGNY_df_tf.parquet"

    path_df1 = "C:/Users/ChristopheReis/Documents/T/t_data/hist_data/ACIW/hist_data_ACIW_1D_2017-06-05-00-00-00_2025-10-08-00-00-00.parquet"
    # path_df2 = "C:/Users/ChristopheReis/Documents/T/t_data/hist_data/ACIW/enriched_data_ACIW_1min_2020-01-02-09-30-00_2025-10-08-16-00-00_merge_iteratively.parquet"
    path_df2 = "C:/Users/ChristopheReis/Documents/T/t_data/hist_data/ACIW/enriched_data_ACIW_15min_2020-01-02-09-30-00_2025-10-08-12-00-00.parquet"

    # path_df1 = "C:/Users/ChristopheReis/Documents/T/t_data/hist_data/ACIW/enriched_data_ACIW_15min_2020-01-02-09-30-00_2025-10-08-16-00-00_merge_compact.parquet"
    # path_df2 = "C:/Users/ChristopheReis/Documents/T/t_data/hist_data/ACIW/enriched_data_ACIW_15min_2020-01-02-09-30-00_2025-10-08-16-00-00_merge_iteratively.parquet"

    gc.collect()

    print("Loading df2 pandas")
    t_now = datetime.datetime.now()
    df2 = helpers.load_df_from_file(path_df2)
    # df2 = pd.read_parquet(path_df2)
    print(f"Elapsed time for loading df2: {datetime.datetime.now() - t_now}\n")

    print("Loading df1 pandas")
    t_now = datetime.datetime.now()
    df1 = helpers.load_df_from_file(path_df1)
    # df1 = pd.read_parquet(path_df1)
    print(f"Elapsed time for loading df1: {datetime.datetime.now() - t_now}\n")

    col_list = [col for col in df1.columns if '_list' in col]
    df1 = df1.drop(columns=col_list)
    df2 = df2.drop(columns=col_list)

    helpers.compare_dataframes(df1, df2, True, col_list, True)

    common_cols = [col for col in df1.columns if col in df2.columns and col not in ['date']]
    df1.drop(columns=common_cols, inplace=True)

    # print(df1._mgr.is_single_block)

    print("Merging pandas")
    t_now = datetime.datetime.now()
    df = pd.merge(df1, df2, on='date', how='left')
    print(f"Elapsed time for merging pandas: {datetime.datetime.now() - t_now}\n")

    print("Merging pandas in chunks")
    t_now = datetime.datetime.now()
    df_chunk = helpers.merge_mtf_df_in_chunks(df1, df2, on='date')#, memory_threshold_MB=10000)
    print(f"Elapsed time for merging pandas in chunks: {datetime.datetime.now() - t_now}\n")

    # print(f"df equals df_chunk: {df.equals(df_chunk)}")

    print("Registering df with duckDB")
    t_now = datetime.datetime.now()
    con = duckdb.connect()
    con.register('df1', df1)
    con.register('df2', df2)
    print(f"Elapsed time for merging duckDB: {datetime.datetime.now() - t_now}\n")

    print("Merging duckDB")
    t_now = datetime.datetime.now()
    query = f"""
    SELECT
        a.*, 
        b.*
    FROM df1 a
    LEFT JOIN df2 b ON a.date = b.date
    """
    df_con = con.execute(query).fetchdf()
    print(f"Elapsed time for merging duckDB: {datetime.datetime.now() - t_now}\n")

    print(f"df equals df_con: {df.equals(df_con)}")

    print()

    print("Merging pandas")
    t_now = datetime.datetime.now()
    pr = cProfile.Profile()
    pr.enable()  # Start profiling
    df = profile_merge_function(df1, df2)  # Call the function
    pr.disable()  # Stop profiling
    print(f"Elapsed time for merging pandas: {datetime.datetime.now() - t_now}\n")

    # Analyzing the profiling data with pstats
    print("Profiling results:")
    stats = pstats.Stats(pr)

    # Sort by cumulative time (time spent in a function + sub-functions)
    stats.sort_stats('cumtime')

    # Print the top 10 most time-consuming functions
    stats.print_stats(10)

    t_now = datetime.datetime.now()
    print("Merging pandas")
    df = cProfile.run('profile_merge_function(df1, df2)')
    # df = pd.merge(df1, df2, on='date', how='left')
    print(f"Elapsed time for merging pandas: {datetime.datetime.now() - t_now}\n")

    t_now = datetime.datetime.now()
    print("Setting indexes")
    df1_index = df1.set_index('date')
    df2_index = df2.set_index('date')
    print(f"Elapsed time for setting indexes on dfs: {datetime.datetime.now() - t_now}\n")

    t_now = datetime.datetime.now()
    print("Merging pandas over indexes")
    df_index = pd.merge(df1_index, df2_index, left_index=True, right_index=True, how='left')
    print(f"Elapsed time for merging pandas over indexes: {datetime.datetime.now() - t_now}\n")



    print()

    # client = dClient()
    # print(f'client: {client}')

    # print("Loading ddf2 dask")
    # t_now = datetime.datetime.now()
    # # df22 = helpers.load_df_from_file(path_df2)
    # # ddf2 = dd.read_parquet(path_df2).compute()
    # ddf2 = dd.from_pandas(df2)
    # print(f"Elapsed time for loading ddf2: {datetime.datetime.now() - t_now}\n")

    # print("Loading ddf1 dask")
    # t_now = datetime.datetime.now()
    # # df1 = helpers.load_df_from_file(path_df1)
    # # ddf1 = dd.read_parquet(path_df1).compute()
    # ddf1 = dd.from_pandas(df1)
    # # df1 = pd.read_parquet(path_df1)
    # print(f"Elapsed time for loading ddf1: {datetime.datetime.now() - t_now}\n")

    # t_now = datetime.datetime.now()
    # print("Merging dask")
    # # df = pd.merge(df1, df2, on='date', how='left')
    # ddf = dd.merge(ddf1, ddf2, on='date', how='left').compute()
    # print(f"Elapsed time for merging dask: {datetime.datetime.now() - t_now}\n")

    print()



    from_time = pd.to_datetime("2020-01-01T04:00:00").tz_localize(constants.CONSTANTS.TZ_WORK)
    to_time = pd.to_datetime("2025-10-15T20:00:00").tz_localize(constants.CONSTANTS.TZ_WORK)
    timeframe = Timeframe('1min')

    fetcher = market_data_fetcher.MarketDataFetcher(single_symbol, 'polygon', from_time, to_time, ib, timeframe=timeframe, step_duration='1W')

    df = fetcher.run()
    ind_list = ['all']
    mtf = ['1D', '1sec', '2min', '2min', '5min', '15min', '30min', '1h', '4h', '1D', '1D']

    # mtf = ['1D']
    fb = feature_builder.FeatureBuilder(df=df, ib=ib, symbol=single_symbol, ind_types=ind_list, mtf=mtf, save_to_file=True)
    dfe = fb.add_features(False, False, False, True)
    # dfe2 = helpers.format_df_date(dfe2, set_index=True)
    print(dfe)
    # print(dfe2)
    print()

    pfetcher = market_data_fetcher.MarketDataFetcher(single_symbol, 'polygon', from_time, to_time, ib, timeframe=timeframe, step_duration='1W')
    ibfetcher = market_data_fetcher.MarketDataFetcher(single_symbol, 'ibkr', from_time, to_time, ib, timeframe=timeframe, step_duration='1W')

    # to_time = pd.to_datetime("2025-09-30T06:00:00").tz_localize(constants.CONSTANTS.TZ_WORK)
    # from_time = pd.to_datetime("2025-09-26T17:00:00").tz_localize(constants.CONSTANTS.TZ_WORK)

    # pfetcher = market_data_fetcher.MarketDataFetcher(single_symbol, 'polygon', from_time, to_time, ib, timeframe='1 min', step_duration='1 W')
    # ibfetcher = market_data_fetcher.MarketDataFetcher(single_symbol, 'ibkr', from_time, to_time, ib, timeframe='1 min', step_duration='3600 S')

    pdf = pfetcher.run()
    ibdf = ibfetcher.run()
    print(f"pdf\n{pdf}")
    print(f"ibdf\n{ibdf}")

    # contract, symbol = helpers.get_symbol_contract(ib, single_symbol)
    # ind_list = ['vwap', 'emas']
    # pfb = feature_builder.FeatureBuilder(df=pdf, ib=ib, contract=contract, types=ind_list, save_to_file=False)
    # ibfb = feature_builder.FeatureBuilder(df=ibdf, ib=ib, contract=contract, types=ind_list, save_to_file=False)
    # pdfe = pfb.add_features(add_indicators=True, add_levels=False, add_patterns=False, add_sr=False)
    # ibdfe = ibfb.add_features(add_indicators=True, add_levels=False, add_patterns=False, add_sr=False)
    # pdfev = pdfe[['date', 'close', 'pvwap', 'vwap', 'ema9', 'ema20', 'sma50', 'sma200']]
    # ibdfev = ibdfe[['date', 'close', 'vwap', 'ema9', 'ema20', 'sma50', 'sma200']]
    # print(f"pdfev\n{pdfev}")
    # print(f"ibdfev\n{ibdfev}")
    print()