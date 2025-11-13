import sys, os, pandas as pd, polygon, traceback
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.constants import PATHS, CONSTANTS, FORMATS
from utils.timeframe import Timeframe, TimeframeHandler


class MarketDataFetcher:
    def __init__(self, symbol, ftype, from_time, to_time, ib=None, timeframe:Timeframe=None, step_duration:str='auto', file_format:str=None,
                 timezone=None, hist_folder=None):
        self.symbol = symbol
        self.fetcher_type = ftype
        self.fetcher = helpers.set_var_with_constraints(ftype, CONSTANTS.FETCHER_TYPES)
        self.pclient = polygon.RESTClient(api_key=CONSTANTS.POLYGON_API_KEY)
        self.ib = ib
        self.timeframe = timeframe or Timeframe()
        self.step_duration = self._match_ibkr_duration_with_timeframe(self.timeframe.ibkr) if step_duration == 'auto' else step_duration
        self.file_format = file_format or FORMATS.DEFAULT_FILE_FORMAT
        self.timezone = timezone or CONSTANTS.TZ_WORK
        self.hist_folder = hist_folder or PATHS.folders_path['hist_market_data']
        self.df_existing, self.existing_to, self.existing_from, self.existing_file_path = self._get_existing_data()
        self.from_time_orig, self.to_time_orig = from_time, to_time
        self.from_time, self.to_time = self._resolve_fetch_intervals(from_time, to_time)

    def _get_existing_data(self):
        # Load local cache
        hist_folder_symbol = os.path.join(self.hist_folder, str(self.symbol))
        df_existing, existing_start, existing_end, existing_file_path = \
            helpers.check_existing_data_file(symbol=self.symbol, timeframe=self.timeframe, folder=hist_folder_symbol, data_type='hist_data', file_format=self.file_format)
        return df_existing, existing_start, existing_end, existing_file_path

    def _validate_fetch_intervals(self, from_time, to_time):
        if not from_time or not to_time:
            raise ValueError("Both to_time and from_time must be defined")

        if to_time < from_time:
            raise ValueError("Start time must be > end time")

    def _resolve_fetch_intervals(self, from_time, to_time):
        self._validate_fetch_intervals(from_time, to_time)

        t_now = pd.Timestamp.now(tz=self.timezone).replace(microsecond=0, second=0)
        if to_time > t_now: to_time = t_now
        if from_time > t_now: from_time = t_now

        offset_now = t_now - to_time
        offset_ibkr = min(self.timeframe.to_timedelta, offset_now) if self.fetcher_type == 'ibkr' else timedelta()

        # Case no existing data
        if self.df_existing.empty or not self.existing_to or not self.existing_from or not self.existing_file_path:
            return from_time, to_time + offset_ibkr

        # Case data already fully exists
        if from_time >= self.existing_from and to_time <= self.existing_to:
            print(f"⏳ Hist data already present for {self.symbol}")
            return to_time, to_time

        # Case just need completion or new data after existing data
        if from_time >= self.existing_from and to_time > self.existing_to:
            return max(self.existing_to, from_time), to_time + offset_ibkr

        # Case new data before existing data or data partially existing
        if from_time < self.existing_from and to_time <= self.existing_to:
            return from_time, min(self.existing_from, to_time) + offset_ibkr

        return from_time, to_time + offset_ibkr

    def _fetch_polygon(self, match_ibkr=True):

        df = pd.DataFrame()
        date_from = helpers.date_to_unix(self.from_time)
        date_to = helpers.date_to_unix(self.to_time)

        # try:
        # print(f"⏳ Fetching data with Polygon for {self.symbol}, {self.timeframe} | {self.from_time} → {self.to_time}")
        multiplier, timespan = self.timeframe.polygon
        response = self.pclient.list_aggs(ticker=self.symbol, multiplier=multiplier, timespan=timespan, from_=date_from, to=date_to, limit=50000)
        df = pd.DataFrame(response)
        if df.empty:
            print(f"⚠️ No data from {self.from_time} → {self.to_time} with Polygon")
            return self.df_existing

        # Convert unix timestamps to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(CONSTANTS.TZ_WORK)
        df = df.set_index('date', drop=True, inplace=False)

        # Forward fill missing values, excluding 'volume' and 'transactions' columns
        # full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=self.timeframe_unit['pandas'])
        if self.timeframe.to_timedelta < timedelta(days=1):
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=self.timeframe.pandas)
            if df.index.duplicated().sum() > 0:
                df = df.loc[~df.index.duplicated()] # Drop duplicates index in case there are any
            df = df.reindex(full_range, fill_value=None)
            df[['volume', 'transactions']] = df[['volume', 'transactions']].fillna(0)
        df.drop('timestamp', axis=1, inplace=True)
        columns_to_ffill = ['close', 'open', 'high', 'low', 'vwap']
        df[columns_to_ffill] = df[columns_to_ffill].ffill()

        df = helpers.trim_df_non_trading_days(df, self.from_time, self.to_time, self.timeframe)
        df = df.reset_index(drop=False).rename(columns={'index': 'date'})

        if match_ibkr:

            #######################################################
            df['pvwap'] = df['vwap']
            #######################################################

            df.drop(['transactions', 'vwap', 'otc'], axis=1, inplace=True)
            for ibkr_extra_col in ['average', 'barCount']:
                if ibkr_extra_col in self.df_existing.columns: self.df_existing.drop(ibkr_extra_col, axis=1, inplace=True)

        all_data = pd.concat([self.df_existing, df]).drop_duplicates(subset=['date']).sort_values('date', inplace=False)
        # all_data = all_data.set_index('date', drop=False, inplace=False)

        # except Exception as e:
        #     print(f"Error fetching data with Polygon: {str(e)}")

        return all_data

    def _get_symbol_hist_data_ibkr(self, query_time, from_time, match_polygon=True, contract=None):
        if contract:
            symbol = contract.symbol
        else:
            contract, symbol = helpers.get_symbol_contract(self.ib, self.symbol)
            if not contract:
                print(f"No contract found for {symbol}")
                return pd.DataFrame()

        # Transform duration into ibkr format in case not already
        duration = self._get_ibkr_duration_from_time_diff(query_time, from_time)

        # # Define requested interval
        # to_time = query_time
        # from_time = helpers.substract_duration_from_time(to_time, duration)

        # # Case data already fully exists
        # if from_time >= self.existing_from and to_time <= self.existing_to:
        #     print(f"⏳ Hist data already present for {symbol}")#, {timeframe}#, {duration} | {from_time} → {to_time}")
        #     return self.df_existing

        # # Case new data to fetch is outside of existing data
        # if from_time >= self.existing_to or to_time <= self.existing_from:
        #     from_time = max(from_time, )
        #     df_existing = pd.DataFrame()

        # # Case just need completion
        # elif from_time > self.existing_from and to_time > self.existing_to:
        #     duration = helpers.get_ibkr_duration_from_time_diff(to_time, self.existing_to)

        # Choose whatToShow
        whatToShow = "TRADES"
        if symbol in helpers.get_forex_symbols_list():
            whatToShow = "MIDPOINT"

        # Fetch from IBKR with error capture
        errors = []
        error_handler = helpers.handle_ib_error_factory(errors)

        # Subscribe to error event
        self.ib.errorEvent += error_handler

        # Fetch from IBKR
        print(f"⏳ Fetching with IBKR data for {symbol}, {self.timeframe}, {duration} | {from_time} → {query_time}")
        bars = self.ib.reqHistoricalData(contract, endDateTime=query_time, durationStr=duration, barSizeSetting=self.timeframe.ibkr,
                                         whatToShow=whatToShow, useRTH=False)
        self.ib.sleep(CONSTANTS.PROCESS_TIME['long'])

        # Unsubscribe from error events after the request
        self.ib.errorEvent -= error_handler

        # if not bars:
        #     print("bars empty")
        # Check if an error 162 (no data) occurred
        if any(errorCode == 162 for _, errorCode, _, _ in errors):
            raise ValueError(f"Not able to fetch market data for {symbol} from {from_time} to {query_time}. Error 162 encountered.")
            # return df_existing if df_existing is not None else pd.DataFrame()

        bars_df = util.df(bars)

        if bars_df is None or bars_df.empty:
            return self.df_existing

        bars_df['date'] = pd.to_datetime(bars_df['date'])
        if bars_df['date'].dt.tz is None:
            bars_df['date'] = bars_df['date'].dt.tz_localize(CONSTANTS.TZ_WORK)

        # Merge with local data
        # combined_df = pd.concat([df_existing, bars_df], ignore_index=True).drop_duplicates(subset=['date'])
        bars_df.sort_values('date', inplace=True)
        bars_df.reset_index(drop=True, inplace=True)

        if match_polygon:
            for ibkr_extra_col in ['average', 'barCount']:
                if ibkr_extra_col in self.df_existing.columns: self.df_existing.drop(ibkr_extra_col, axis=1, inplace=True)
                if ibkr_extra_col in bars_df.columns: bars_df.drop(ibkr_extra_col, axis=1, inplace=True)

        return bars_df

    def _get_symbol_hist_data_recursive_ibkr(self, match_polygon=True) -> pd.DataFrame:

        step_duration_td = Timeframe(self.step_duration).to_timedelta
        all_data = self.df_existing
        current_start = self.to_time
        df_chunk = pd.DataFrame()
        contract, _ = helpers.get_symbol_contract(self.ib, self.symbol)

        try:
            while current_start > self.from_time:
                chunk_start = current_start
                chunk_end = chunk_start - step_duration_td
                if chunk_end < self.from_time:
                    chunk_end = self.from_time

                # print(f"⏳ Fetching data for {symbol}, {timeframe}, {step_duration} | {chunk_end} → {chunk_start}")
                df_chunk_prev = df_chunk.copy()
                df_chunk = self._get_symbol_hist_data_ibkr(query_time=chunk_start, from_time=chunk_end, match_polygon=match_polygon, contract=contract)

                if df_chunk.empty:
                    print(f"⚠️ No data for chunk: {chunk_start} → {chunk_end}")
                    current_start = chunk_end
                    # chunk_end -= timedelta(seconds=1)
                    continue

                df_chunk_formatted = helpers.format_df_date(df_chunk)
                all_data = pd.concat([all_data, df_chunk_formatted], ignore_index=True).drop_duplicates(subset=['date'])
                all_data.sort_values('date', inplace=True)

                if (not df_chunk_prev.empty and df_chunk['date'].iloc[0] == df_chunk_prev['date'].iloc[0] and df_chunk['date'].iloc[-1] == df_chunk_prev['date'].iloc[-1]):
                    print(f"⚠️ Data chunk inside non market hours: {chunk_start} → {chunk_end}")
                    current_start = chunk_end
                    # chunk_end -= timedelta(seconds=1)
                    continue

                # Move the window backward based on the oldest date in this chunk
                current_start = df_chunk['date'].min()# - timedelta(seconds=1)

            return all_data

        except Exception as e:
            print(f"⚠️ Error fetching {self.symbol} from {self.from_time_orig} to {self.to_time_orig}: {e}")
            print(traceback.format_exc())
            return all_data

    def run(self, display_time=True):

        if self.from_time >= self.to_time:
            print(f"No data gathered: from_time {self.from_time} is >= to_time {self.to_time}")
            return self.df_existing

        t_start = datetime.now()
        if self.fetcher_type == 'polygon':
            df = self._fetch_polygon(match_ibkr=True)
        elif self.fetcher_type == 'ibkr':
            df = self._get_symbol_hist_data_recursive_ibkr(match_polygon=True)
        else:
            print(f"Unrecognized fetcher type {self.fetcher_type}. Must be 'ibkr' or 'polygon'")

        # Trim df to requested times
        if not df.empty:
            df = helpers.trim_df(df, self.from_time_orig, self.to_time_orig).reset_index(drop=True, inplace=False)

        if display_time: print(f"Time to load data using {self.fetcher_type}: {datetime.now() - t_start}")

        return df

    @staticmethod
    def _get_ibkr_duration_from_time_diff(to_time:datetime, from_time:datetime) -> str:
        """
        Calculate an IBKR-style duration string (e.g., '1 D', '2 W', '3 M') from two datetimes.
        """
        if not (isinstance(to_time, datetime) and isinstance(from_time, datetime)):
            raise ValueError("Both inputs must be datetime objects.")

        if to_time < from_time:
            raise ValueError("to_time must be after from_time.")

        delta = relativedelta(to_time, from_time)
        total_seconds = int((to_time - from_time).total_seconds())

        if delta.years >= 1: return f"{delta.years} Y" # Years
        if delta.months >= 1: return f"{delta.months} M" # Months
        total_days = (to_time - from_time).days
        if total_days >= 7 and total_days % 7 == 0: return f"{total_days // 7} W" # Weeks
        if total_days >= 1: return f"{total_days} D" # Days
        # if total_seconds >= 3600: return f"{total_seconds // 3600} H" # Hours
        return f"{total_seconds} S" # Seconds

    @staticmethod
    def _match_ibkr_duration_with_timeframe(timeframe_ibkr:str):

        if timeframe_ibkr in ['1 sec', '5 secs', '10 secs', '15 secs', '30 secs']: duration = '1 D'
        elif timeframe_ibkr in ['1 min', '2 mins', '3 mins']: duration = '1 W'
        elif timeframe_ibkr in ['5 mins', '10 mins', '15 mins', '20 mins', '30 mins']: duration = '2 W'
        elif timeframe_ibkr in ['1 hour', '2 hours', '3 hours', '4 hours', '8 hours']: duration = '1 M'
        elif timeframe_ibkr in ['1 day']: duration = '3 M'
        elif timeframe_ibkr in ['1 week']: duration = '6 M'
        elif timeframe_ibkr in ['1 month']: duration = '1 Y'
        else: duration = None
        return duration
