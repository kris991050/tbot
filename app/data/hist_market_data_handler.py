import sys, os, pandas as pd, numpy as np, exchange_calendars, gc
from datetime import datetime, timedelta
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from data import market_data_fetcher
from utils import helpers
from utils.constants import CONSTANTS, FORMATS, PATHS
from utils.timeframe import Timeframe
from features import feature_builder, indicators


class HistMarketDataHandler:
    def __init__(self, timeframe:Timeframe=None, file_format:str=None, drop_levels:bool=False, base_folder=None, timezone=None):
        self.timeframe = timeframe or Timeframe()
        self.file_format = file_format or FORMATS.DEFAULT_FILE_FORMAT
        self.drop_levels = drop_levels
        self.base_folder = base_folder or PATHS.folders_path['hist_market_data']
        self.timezone = timezone or CONSTANTS.TZ_WORK
        self.pdelay = Timeframe(CONSTANTS.POLYGON_DATA_DELAY).to_timedelta + timedelta(minutes=5)
        self.pmax_backward = Timeframe(CONSTANTS.POLYGON_DATA_MAX_BACKWARD).to_timedelta - 24 * timedelta(days=1)

    def get_attrs(self, df, ib, symbol):
        print(f"Building df attributes for {symbol}")
        market_cap = None
        try:
            market_cap = helpers.convert_large_numbers(helpers.get_stock_info_from_Finviz(symbol, 'Market Cap'))
            share_floats = helpers.get_share_floats_from_polygon(symbol) or ''
        except Exception as e:
                print(f"Could not fetch market cap for {symbol}. Error: {e}")
        timeframe = helpers.get_df_timeframe(df)
        df.attrs =  {'symbol': symbol,
                'timeframe': timeframe.pandas if timeframe else None,
                'index': helpers.get_index_from_symbol(ib, symbol),
                'market_cap': market_cap,
                'float': share_floats}
        df = self._update_attrs_times(df)
        return df

    def _update_attrs_times(self, df):
        if hasattr(df, 'attrs') and df.attrs:
            df.attrs.update({
                'data_to_time': df['date'].iloc[-1] if not df.empty else None,
                'data_from_time': df['date'].iloc[0] if not df.empty else None})

        return df

    def parse_filename(self, filename:str, data_type:str='hist_data'):
        pattern = helpers.build_data_filename_pattern(timeframe=self.timeframe, file_format=self.file_format, data_type=data_type)
        match = pattern.match(filename)
        return match.groupdict() if match else None

    def _parse_dt(self, dt_str, parse_format='%Y-%m-%d-%H-%M-%S'):
        dt_str = pd.to_datetime(dt_str, utc=True, format=parse_format)
        return dt_str.tz_convert(self.timezone)

    def load_file(self, path):
        df = helpers.load_df_from_file(path).reset_index(drop=True)#, parse_dates=["date"])
        # if not df.empty: df['date'] = self._parse_dt(df['date'])
        if not df.empty: df = helpers.format_df_date(df)
        return df

    def list_symbol_files(self, symbol):
        if not symbol.startswith('.'):
            folder = os.path.join(self.base_folder, symbol)
            if not os.path.isdir(folder):
                return []

            return [os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.endswith(f".{self.file_format}") and self.timeframe.pandas in f]
        else:
            return []


class HistMarketDataFetcher:
    def __init__(self, ib, ftype:str='auto', timeframe:Timeframe=None, file_format:str=None, symbols_csv_path:str=None,
                 validate:bool=True, delete_existing:bool=True, save_to_file:bool=True, base_folder:str=None, timezone=None):
        self.ib = ib
        self.fetcher_type = ftype
        # self.fetcher_type = 'ibkr'
        self.validate = validate
        self.delete_existing = delete_existing
        self.save_to_file = save_to_file
        self.symbols_csv_path = symbols_csv_path or os.path.join(PATHS.folders_path['market_data'], "symbols_market_data.csv")
        self.handler = HistMarketDataHandler(timeframe=timeframe, file_format=file_format, base_folder=base_folder, timezone=timezone)
        self.validator = HistMarketDataValidator(timeframe=timeframe, file_format=file_format, base_folder=base_folder, timezone=timezone)

    # def _resolve_fetcher(self, from_time, to_time):
    #     t_now_delay = pd.Timestamp.now(tz=self.handler.timezone) - pdelay
    #     t_now_max_backward = pd.Timestamp.now(tz=self.handler.timezone) - pmax_backward

    #     if to_time > t_now_delay or from_time < t_now_max_backward or self.fetcher_type == 'ibkr':
    #         return 'ibkr'

    #     if self.fetcher_type in ['polygon', 'auto']:
    #         return 'polygon'

    #     return None

    def _fetch_data(self, symbol, fetcher_type, from_time, to_time, step_duration, text):
        print(f"\n📥 Fetching {symbol} {self.handler.timeframe} with {fetcher_type} fetcher from {from_time} to {to_time}" + text)
        timeframe = Timeframe() if fetcher_type == 'polygon' else self.handler.timeframe # Fetch data at base timeframe with Polygon and resample after, otherwise get uneven results compared to IBKR
        fetcher = market_data_fetcher.MarketDataFetcher(symbol, fetcher_type, from_time=from_time, to_time=to_time, ib=self.ib,
                                                        timeframe=timeframe, step_duration=step_duration,
                                                        file_format=self.handler.file_format)
        df = fetcher.run()
        if fetcher_type == 'polygon' and not df.empty:
            df = df.resample(self.handler.timeframe.pandas, on='date').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
        return df

    def _fetch_symbol(self, symbol, from_time, to_time, step_duration='auto'):

        hist_folder_symbol = os.path.join(self.handler.base_folder, symbol)
        os.makedirs(hist_folder_symbol, exist_ok=True)
        t0 = datetime.now()

        condition_ibkr = self.fetcher_type == 'ibkr' or self.handler.timeframe.pandas in CONSTANTS.FORBIDDEN_TIMEFRAMES_POLYGON or symbol in CONSTANTS.FORBIDDEN_SYMBOLS_POLYGON
        fetcher_type = 'ibkr' if condition_ibkr else 'polygon' if self.fetcher_type in ['polygon', 'auto'] else None

        t_now_delay = pd.Timestamp.now(tz=self.handler.timezone) - self.handler.pdelay
        t_now_max_backward = pd.Timestamp.now(tz=self.handler.timezone) - self.handler.pmax_backward
        ibdf_delay, ibdf_max_backward = pd.DataFrame(), pd.DataFrame()

        if t_now_delay < to_time and fetcher_type == 'polygon':
            ibdf_delay = self._fetch_data(symbol=symbol, fetcher_type='ibkr', from_time=max(t_now_delay, from_time),
                                          to_time=to_time, step_duration=step_duration,
                                          text=' for Polygon delayed window.')
            if not ibdf_delay.empty:
                to_time = from_time if from_time > t_now_delay else max(ibdf_delay['date'].min(), from_time)

        if from_time < t_now_max_backward and fetcher_type == 'polygon':
            ibdf_max_backward = self._fetch_data(symbol=symbol, fetcher_type='ibkr', from_time=from_time,
                                                 to_time=min(t_now_max_backward, to_time), step_duration=step_duration,
                                                 text=' for Polygon max backward window.')
            if not ibdf_max_backward.empty:
                from_time = to_time if to_time < t_now_max_backward else min(ibdf_max_backward['date'].max(), to_time)

        pdf = self._fetch_data(symbol, fetcher_type, from_time, to_time, step_duration, '.')
        df = pd.concat([pdf, ibdf_delay, ibdf_max_backward], axis=0)

        # t_now_delayed = pd.Timestamp.now(tz=self.handler.timezone) - self.pdelay
        # t_now_max_backward = pd.Timestamp.now(tz=self.handler.timezone) - self.pmax_backward
        # to_time_adjusted = to_time#t_now_delayed if (self.fetcher_type == 'auto' and to_time > t_now_delayed) else to_time
        # from_time_adjusted = from_time#t_now_max_backward if (self.fetcher_type == 'auto' and from_time < t_now_max_backward) else from_time
        # fetcher_type = 'ibkr' if self.fetcher_type == 'ibkr' else 'polygon' if self.fetcher_type in ['polygon', 'auto'] else None

        # df = self._fetch_data(symbol, fetcher_type, from_time, to_time, step_duration, '.')

        # # Case requested data falls into Polygon data delay time range
        # from_time_max = df['date'].max() + self.handler.timeframe.to_timedelta if not df.empty else from_time
        # from_time = df['date'].max() if not df.empty else from_time
        # if from_time_max < to_time:
        #     ibdf_delay = self._fetch_data(symbol=symbol, fetcher_type='ibkr', from_time=from_time, to_time=to_time,
        #                                   step_duration=step_duration, text=' for Polygon delayed window.')
        #     df = pd.concat([df, ibdf_delay], axis=0)

        # # Case requested data falls into Polygon max historical data limit
        # to_time_min = df['date'].min() - self.handler.timeframe.to_timedelta if not df.empty else to_time
        # to_time = df['date'].min() if not df.empty else to_time
        # if to_time_min > from_time:
        #     ibdf_max_backward = self._fetch_data(symbol=symbol, fetcher_type='ibkr', from_time=from_time, to_time=to_time,
        #                                          step_duration=step_duration, text=' for Polygon max backward window.')
        #     df = pd.concat([df, ibdf_max_backward], axis=0)

        if df.empty:
            print(f"⚠️ No data returned for {symbol}")
            return None, None
        # df = pd.concat([df, ibdf_max_backward], axis=0)
        df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)#.copy()

        # Campare and concatenate with existing data
        df_existing, _, _, existing_file_path = helpers.check_existing_data_file(symbol, self.handler.timeframe, hist_folder_symbol, data_type='hist_data',
                                                                                     delete_file=False, file_format=self.handler.file_format)
        df = pd.concat([df_existing, df]).drop_duplicates(subset=['date']).sort_values('date', inplace=False)
        are_df_equals = df_existing.equals(df)

        path = helpers.construct_data_path(hist_folder_symbol, symbol, self.handler.timeframe, to_time=df['date'].iloc[-1],
                                                from_time=df['date'].iloc[0], file_format=self.handler.file_format, data_type='hist_data')

        if not are_df_equals:
            if self.delete_existing and existing_file_path and os.path.exists(existing_file_path):
                os.remove(existing_file_path)

            if self.save_to_file:
                helpers.save_df_to_file(df, path, file_format=self.handler.file_format)

            if self.validate:
                print(f"✅ Data fetched for {symbol}. Validating...")
                self.validator.run(symbol=symbol, print_result=True)
        else:
            print(f"ℹ️ New fetched data identical to existing data for {symbol}")

        print(f"⏱️ Time for fetching {symbol}: {datetime.now() - t0}")

        return df, path

    def run(self, params=None):

        results = {}
        if params:
            # try:
            self.handler.timeframe = params['timeframe']
            self.validator.handler.timeframe = params['timeframe']
            df, result_path = self._fetch_symbol(params['symbol'], params['from_time'], params['to_time'], params['step_duration'])
            # except Exception as e:
            #     print(f"Could not fetch data using params {params}. Error: {e}")
            results[params['symbol']] = {'df': df, 'path': result_path}

        elif self.symbols_csv_path:
            df_symbols = pd.read_csv(self.symbols_csv_path).dropna()
            # df_symbols = helpers.format_df_date(df_symbols, col='to_time')
            # df_symbols = helpers.format_df_date(df_symbols, col='from_time')

            for _, params in df_symbols.iterrows():
                # try:
                from_time = pd.to_datetime(params['from_time']).tz_localize(CONSTANTS.TZ_WORK)
                to_time = pd.to_datetime(params['to_time']).tz_localize(CONSTANTS.TZ_WORK)
                self.handler.timeframe = Timeframe(params['timeframe'])
                self.validator.handler.timeframe = Timeframe(params['timeframe'])
                df, result_path = self._fetch_symbol(params['symbol'], from_time, to_time, params.get('step_duration', default='auto'))
                results[params['symbol']] = {'df': df, 'path': result_path}
                # except Exception as e:
                #     print(f"Could not fetch data using params. Error: {e}")

        else:
            print("No symbols_csv_path or fetching parameters provided")

        return results


class HistMarketDataValidator:
    def __init__(self, timeframe:Timeframe=None, file_format:str=None, symbols=None,
                 calendar_name:str="XNYS", base_folder=None, timezone=None):
        self.symbols = symbols or []
        self.calendar = exchange_calendars.get_calendar(calendar_name)
        self.handler = HistMarketDataHandler(timeframe=timeframe, file_format=file_format,
                                             base_folder=base_folder, timezone=timezone)

    def _validate_file(self, path):

        filename = os.path.basename(path)
        meta = self.handler.parse_filename(filename)

        result = {'symbol': meta['symbol'] if meta else None,
                  'result': 'passed',
                  'error_type': None,
                  'error_value': None,
                  'text': f"✅ {filename} passed\n"}

        if not meta:
            result.update({'result': 'failed',
                           'text': f"❌ Invalid filename format: {filename}",
                           'error_type': 'format'})
            return result

        try:
            start_dt = self.handler._parse_dt(meta['to_str'])
            end_dt = self.handler._parse_dt(meta['from_str'])
        except ValueError:
            result.update({'result': 'failed',
                           'text': f"❌ Invalid date in filename: {filename}",
                           'error_type': 'format'})
            return result

        if start_dt <= end_dt:
            result.update({'result': 'failed',
                           'text': f"❌ Start date is before end date: {filename}",
                           'error_type': 'date order'})
            return result

        try:
            df = self.handler.load_file(path)
        except Exception as e:
            result.update({'result': 'failed',
                           'text': f"❌ Failed to load {filename} -> {e}",
                           'error_type': 'load'})
            return result

        if self.handler.timeframe.to_seconds / 60 == 1:
            # expected_minutes = self.calendar.minutes_in_range(df["date"].min().date(), df["date"].max().date()).tz_convert(self.handler.timezone)
            expected_minutes = self.calendar.minutes_in_range(df["date"].min(), df["date"].max()).tz_convert(self.handler.timezone)
            actual_minutes = pd.Series(df["date"].sort_values().drop_duplicates())
            missing = expected_minutes.difference(actual_minutes)

            if not missing.empty:
                result.update({'result': 'warning',
                            'text': f"⚠️ Missing {len(missing)} minutes in {filename}",
                            'error_type': 'missing',
                            'error_value': missing})

        return result

    def _check_single_file_per_symbol(self, symbol):

        paths = self.handler.list_symbol_files(symbol)

        # Filter files matching the timeframe and file_format
        # matching_files = [f for f in paths if f.endswith(f".{self.handler.file_format}") and self.handler.timeframe in f and 'hist' in f]

        matching_files = []
        valid_paths = []
        for path in paths:
            filename = os.path.basename(path)

            # pattern = helpers.build_hist_data_filename_pattern(timeframe=self.handler.timeframe, file_format=self.handler.file_format)
            # match = pattern.match(filename)
            match = self.handler.parse_filename(filename)

            if match:
                matching_files.append(match)
                valid_paths.append(path)

        result = {'symbol': symbol,
                  'result': 'passed' if len(matching_files) == 1 else 'failed',
                  'error_type': None if len(matching_files) == 1 else 'files',
                  'error_value': None if len(matching_files) == 1 else len(matching_files),
                  'text': f"✅ Exactly one file found for {symbol}" if len(matching_files) == 1 else
                        f"❌ Found {len(matching_files)} files for {symbol} with timeframe {self.handler.timeframe}"}

        return result, valid_paths[0] if valid_paths else None

    def run(self, symbol=None, print_result=True):
        symbols = [symbol] if symbol else self.symbols or os.listdir(self.handler.base_folder)
        results = []

        for symbol in symbols:
            print(f"🔍 Validating data for {symbol}")
            result, path = self._check_single_file_per_symbol(symbol)

            if result['result'] == 'passed':
                # for path in self.handler.list_symbol_files(symbol):
                result = self._validate_file(path)
                if print_result:
                    print(result['text'])

            results.append(result)

        return results


class HistMarketDataCompleter:
    def __init__(self, ib, timeframe:Timeframe=None, file_format:str=None, symbols=None, step_duration:str='auto',
                 validate:bool=True, base_folder=None, timezone=None):
        self.ib = ib
        self.symbols = symbols
        self.step_duration = step_duration
        # self.handler = HistMarketDataHandler(timeframe=timeframe, file_format=file_format,
        #                                      base_folder=base_folder, timezone=timezone)
        self.fetcher = HistMarketDataFetcher(ib, timeframe=timeframe, file_format=file_format, validate=validate,
                                             base_folder=base_folder, timezone=timezone)

    def _complete_symbol(self, symbol):
        folder = os.path.join(self.fetcher.handler.base_folder, symbol)
        os.makedirs(folder, exist_ok=True)

        df_existing, existing_start, existing_end, _ = \
            helpers.check_existing_data_file(symbol=symbol, timeframe=self.fetcher.handler.timeframe, folder=folder,
                                             data_type='hist_data', delete_file=False, file_format=self.fetcher.handler.file_format)

        if existing_end is None:
            print(f"[{symbol}] No existing data found. Skipping.")
            return

        # now = pd.to_datetime(datetime.now(), utc=True).tz_convert(CONSTANTS.TZ_WORK)
        now = datetime.now(CONSTANTS.TZ_WORK)

        params = {'symbol': symbol, 'to_time': now, 'timeframe': self.fetcher.handler.timeframe,
                  'from_time': existing_end, 'step_duration': self.step_duration}
        results = self.fetcher.run(params=params)

        return results

    def run(self):
        all_symbols = self.symbols or os.listdir(self.fetcher.handler.base_folder)
        # all_symbols = self.symbols or [name for name in os.listdir(self.handler.base_folder) if not name.startswith('.')]

        results = {}
        for symbol in all_symbols:
            if not symbol.startswith('.'):
                print(f"🔄 Completing hist data for {symbol}")
                result = self._complete_symbol(symbol)
                # results_path.append(result_path)
                results[symbol] = result[symbol]

        return results


class HistMarketDataEnricher:
    def __init__(self, ib, timeframe:Timeframe=None, base_timeframe:Timeframe=None, look_backward:str=None, feature_types:list=['all'], mtf:list=None,
                 file_format:str=None, symbols=None, base_folder=None, delete_existing=True, save_to_file=True, keep_results:bool=True, 
                 drop_levels:bool=False, block_add_sr=False, validate=True, timezone=None, separation_text:str='', override_existing:bool=False):
        self.ib = ib
        self.symbols = symbols or []
        self.delete_existing = delete_existing
        self.save_to_file = save_to_file
        self.keep_results = keep_results
        self.feature_types = feature_types
        # self.timeframe = helpers.pandas_to_ibkr_timeframe(timeframe)
        # self.base_timeframe = helpers.pandas_to_ibkr_timeframe(base_timeframe)
        self.mtf = mtf
        self.validate = validate
        self.block_add_sr = block_add_sr
        self.handler = HistMarketDataHandler(timeframe=timeframe, file_format=file_format, drop_levels=drop_levels,
                                             base_folder=base_folder, timezone=timezone)
        self.base_handler = HistMarketDataHandler(timeframe=base_timeframe, file_format=file_format, drop_levels=drop_levels,
                                             base_folder=base_folder, timezone=timezone)
        self.validator = HistMarketDataValidator(timeframe=base_timeframe, file_format=file_format,
                                                 base_folder=base_folder, timezone=timezone)
        self.look_backward = look_backward or CONSTANTS.WARMUP_MAP[self.handler.timeframe.pandas]
        self.separation_text = separation_text
        self.override_existing = override_existing

    def _validate_dates_alignment(self, df_original, df_enriched, filename):
        """Ensure dates match exactly and no gaps are introduced."""
        orig_dates = df_original['date'].sort_values().reset_index(drop=True)
        enrich_dates = df_enriched['date'].sort_values().reset_index(drop=True)

        if not orig_dates.equals(enrich_dates):
            raise ValueError(f"❌ Date mismatch or gaps found in enriched data: {filename}")
        else:
            print(f"✅ Date alignment validation successful for {filename}")

    def _resample_if_needed(self, df:pd.DataFrame, timeframe:Timeframe):
        if self.handler.timeframe != self.base_handler.timeframe:
            # timeframe_resample = helpers.ibkr_to_pandas_timeframe(timeframe)
            df_resampled = df.resample(timeframe.pandas, on='date').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
            if hasattr(df, 'attrs') and df.attrs:
                timeframe_resampled = timeframe#helpers.get_df_timeframe(df_resampled)
                df_resampled.attrs = df.attrs
                df_resampled.attrs.update({'timeframe': timeframe_resampled.pandas if timeframe_resampled else None})
            return df_resampled
        return df

    def _add_features(self, df, symbol, save_to_file, add_indicators=True, add_levels=True, add_patterns=True, add_sr=True):
        fb = feature_builder.FeatureBuilder(df=df, ib=self.ib, symbol=symbol, timeframe=self.handler.timeframe, feature_types=self.feature_types, 
                                            mtf=self.mtf, save_to_file=save_to_file, file_format=self.handler.file_format, 
                                            hist_folder=self.handler.base_folder, drop_levels=self.handler.drop_levels)
        df_enriched = fb.add_features(add_indicators=add_indicators, add_levels=add_levels, add_patterns=add_patterns, add_sr=add_sr)
        gc.collect()
        return df_enriched, fb

    def _remove_path(self, path):
        if self.delete_existing and path and os.path.exists(path):
            os.remove(path)
            print(f"File Deleted: {path}")
    
    def _build_enriched_path(self, df:pd.DataFrame, symbol:str):
        return helpers.construct_data_path(local_hist_folder=self.handler.base_folder, symbol=symbol, 
                                           timeframe=self.handler.timeframe.pandas, to_time=df['date'].iloc[-1], 
                                           from_time=df['date'].iloc[0], file_format=self.handler.file_format, data_type='enriched_data')

    def _enrich_file(self, path:str, from_time:datetime=None, to_time:datetime=None):
        filename = os.path.basename(path)
        meta = self.base_handler.parse_filename(filename)
        if not meta:
            print(f"⚠️ Skipping invalid file: {filename}")
            return None, None

        symbol = meta['symbol']
        # timeframe = Timeframe(meta.get('timeframe', self.handler.timeframe.pandas))
        to_time = to_time or self.handler._parse_dt(meta['to_str'])
        from_time = from_time or self.handler._parse_dt(meta['from_str'])
        symbol_folder = os.path.join(self.handler.base_folder, symbol)
        df_base = self.base_handler.load_file(path)
        df_base = self.base_handler.get_attrs(df_base, self.ib, symbol)

        if from_time and to_time:
            df_base = helpers.trim_df(df_base, from_time, to_time)

        print(f"\n📊 Enriching data for {symbol}: {filename}")
        t0 = datetime.now()

        # if self._enriched_file_exists_and_is_sufficient(path, enriched_path):
        df_existing_hist, existing_start_hist, existing_end_hist, _ = \
            helpers.check_existing_data_file(symbol, self.base_handler.timeframe, symbol_folder, data_type='hist_data', delete_file=False,
                                                 file_format=self.handler.file_format)
        if not self.override_existing:
            df_existing_enriched, existing_start_enriched, existing_end_enriched, path_existing_enriched = \
                helpers.check_existing_data_file(symbol, self.handler.timeframe, symbol_folder, data_type='enriched_data', delete_file=False,
                                                    file_format=self.handler.file_format)
        else:
            df_existing_enriched, existing_start_enriched, existing_end_enriched, path_existing_enriched = pd.DataFrame(), None, None, None

        if not df_existing_enriched.empty and not df_existing_hist.empty and existing_end_enriched <= existing_end_hist \
            and existing_start_enriched > existing_end_hist:

            if existing_start_enriched >= existing_start_hist:
                # Case existing enriched data is sufficient

                print(f"⏩ Skipping enrichment, existing enriched file is sufficient: {path_existing_enriched}")
                df_existing_enriched = self.handler._update_attrs_times(df_existing_enriched)
                return df_existing_enriched, path_existing_enriched

            else:
                # Case need to complete enriched data (extend existing enriched df)

                look_backward_td = Timeframe(self.look_backward).to_timedelta * 0.95 # 0.95 to account for imprecision in df_base fetching
                span_df_base = df_base['date'].iloc[-1] - df_base['date'].iloc[0]
                if span_df_base >= look_backward_td: # If this is not satisfied -> df_base too small -> fall back to full enrichment
                    print(f"🔄 Completing enrichment for existing file: {path_existing_enriched}")
                    df_to_enrich = df_base[df_base['date'] > existing_start_enriched]#.copy()

                    if df_to_enrich.empty or len(df_to_enrich) < int(self.handler.timeframe.to_seconds / self.base_handler.timeframe.to_seconds):
                        print("⏹️ No new data to enrich.")
                        return df_existing_enriched, path_existing_enriched

                    df_to_enrich.attrs = df_base.attrs  # Make sure we keep meta info
                    df_to_enrich = self._resample_if_needed(df_to_enrich, self.handler.timeframe)

                    # Check if S/R recalculation is needed
                    add_sr = False
                    if not self.block_add_sr:
                        offset_to_enrich = df_to_enrich['date'].max() - df_to_enrich['date'].min()
                        add_sr_dist = indicators.IndicatorsUtils.detect_last_sr_change(df_existing_enriched, offset=offset_to_enrich)
                        add_sr = any(add_sr_dist.values())

                    # Rebuild features on only this window
                    print("🛠️ Recomputing indicators and patterns on recent data...")
                    df_enriched_recent, fb = self._add_features(df_to_enrich, symbol, save_to_file=False, add_sr=add_sr)

                    # In case S/R have not been recalculated, forward fill previous values
                    if not add_sr:
                        sr_cols = [f"sr_{setting['timeframe']}_list" for setting in CONSTANTS.SR_SETTINGS]
                        for col in sr_cols:
                            if col in df_existing_enriched.columns:
                                last_val = df_existing_enriched[col].iloc[-1]
                                last_val_copy = last_val.copy() if last_val is not None else last_val
                                df_enriched_recent[col] = df_enriched_recent.apply(lambda _: last_val_copy, axis=1)
                                df_enriched_recent = indicators.IndicatorsUtils.calculate_next_levels(df=df_enriched_recent, 
                                                                                                      levels_col=col, show_next_level=False)

                    # Remove overlap and merge
                    df_enriched_recent_trimmed = df_enriched_recent[df_enriched_recent['date'] > df_existing_enriched['date'].max()]
                    df_enriched = pd.concat([df_existing_enriched, df_enriched_recent_trimmed], axis=0).sort_values('date').reset_index(drop=True)#.copy()

                    df_enriched = self.handler._update_attrs_times(df_enriched)
                    self._remove_path(path_existing_enriched)
                    self._validate_dates_alignment(self._resample_if_needed(df_base, self.handler.timeframe), df_enriched, filename)
                    fb.save(df_enriched)
                    enriched_path = self._build_enriched_path(df_enriched, symbol)
                    
                    print(f"✅ Enrichment updated with {len(df_enriched_recent)} new bars.")
                    print(f"⏱️ Time enrichment completion of {symbol}: {datetime.now() - t0}")
                    return df_enriched, enriched_path

        # Case not enough existing enriched data -> full enrichment needed

        # Resample if necessary
        df_resampled = self._resample_if_needed(df_base, self.handler.timeframe)
        df_enriched, fb = self._add_features(df_resampled, symbol, self.save_to_file)
        df_enriched = self.handler._update_attrs_times(df_enriched)
        self._remove_path(path_existing_enriched)
        self._validate_dates_alignment(df_resampled, df_enriched, filename)
        enriched_path = self._build_enriched_path(df_enriched, symbol)

        print(f"✅ Enrichment completed and validated for {symbol}")
        print(f"⏱️ Time for enrichment of {symbol}: {datetime.now() - t0}")

        return df_enriched, enriched_path

        # except Exception as e:
        #     print(f"❌ Error enriching {path}: {e}")
        #     return None, None

    def run(self, symbol:str=None, from_time:datetime=None, to_time:datetime=None):
        symbols = [symbol] if symbol else self.symbols or sorted(os.listdir(self.handler.base_folder))

        results = []
        for symbol in symbols:
            if symbol.startswith('.'):
                continue

            # Validate raw file before enriching
            if self.validate:
                validation_result = self.validator.run(symbol=symbol)[0]
                if validation_result['result'] == 'failed':
                    # if not validation_result: validation_result = {"text"}
                    print(f"❌ Skipping enrichment due to validation failure: {validation_result['text']}")
                    return

            t1 = datetime.now()
            for path in self.base_handler.list_symbol_files(symbol):
                # try:
                df_enriched, enriched_path = self._enrich_file(path, from_time, to_time)

                ############################################################################################################################
                # if enriched_path:
                #     if 'high_of_day' not in df_enriched.columns or 'low_of_day' not in df_enriched.columns:
                #         df_enriched = indicators.Indicators(df_enriched, self.ib, symbol, types=[]).df
                #         print(f"\n\nSAVING NEW DF_ENRICHED FOR {symbol}...\n\n")
                #         helpers.save_df_to_file(df_enriched, enriched_path, self.handler.file_format)
                ############################################################################################################################

                if self.keep_results: # Gives the option to prevent overloading memory if to many tickers to enrich
                    results.append({'df': df_enriched, 'path': enriched_path})
                # except Exception as e:
                #     print(f"❌ Error processing {path}: {e}")

            print(f"⏱️ Elapsed time for full enrichment of symbol {symbol}: {datetime.now() - t1}\n\n")
            print(self.separation_text)

        return results

    @staticmethod
    def get_valid_result(results):
        # Find the first valid result upfront
        return next((r for r in results if r.get('path') and r.get('df') is not None and not r['df'].empty), None)


class HistMarketDataLoader:
    def __init__(self, ib, symbol:str, timeframe:Timeframe=None, symbol_hist_folder:str=None, file_format:str=None,
                 data_type:str='hist_data', drop_levels:bool=True, base_folder:str=None, timezone=None):
        self.ib = ib
        self.symbol = symbol
        self.data_type = data_type if data_type in FORMATS.DATA_TYPE_FORMATS else None
        self.handler = HistMarketDataHandler(timeframe=timeframe, file_format=file_format, drop_levels=drop_levels,
                                             base_folder=base_folder, timezone=timezone)
        self.symbol_hist_folder = symbol_hist_folder or os.path.join(self.handler.base_folder, self.symbol)

    def _get_hist_file(self):
        for file in os.listdir(self.symbol_hist_folder):
            filename = os.path.basename(file)
            meta = self.handler.parse_filename(filename, self.data_type)
            if meta:
                return os.path.join(self.symbol_hist_folder, file)
            # if f"{self.data_type}_{self.symbol}_{self.handler.timeframe}_" in file and file.endswith(f".{self.handler.file_format}"):
            #     return os.path.join(self.symbol_hist_folder, file)
        raise FileNotFoundError(f"No file found for {self.symbol} at {self.handler.timeframe} in {self.symbol_hist_folder}")

    def load_and_trim(self, from_time=None, to_time=None) -> pd.DataFrame:
        file_path = self._get_hist_file()

        print(f"🔄 Loading data for {self.symbol}...")
        df = helpers.load_df_from_file(file_path)
        if self.handler.drop_levels: df = helpers.drop_df_columns(df, '_list')
        df = self.handler.get_attrs(df, self.ib, self.symbol)

        from_time = from_time or df['date'].iloc[0]
        to_time = to_time or df['date'].iloc[-1]

        return helpers.trim_df(df, from_time, to_time), file_path


class HistMarketDataConverter:
    def __init__(self, convert_from:str='parquet', convert_to:str='csv', timeframe:Timeframe=None, symbols=None, data_type:str='hist_data',
                 file_format:str=None, keep_file:bool=True, optimize:bool=False, base_folder:str=None, timezone=None):
        self.symbols = symbols or []
        self.data_type = data_type
        self.keep_file = keep_file
        self.convert_from = convert_from
        self.convert_to = convert_to
        self.optimize = optimize
        self.handler = HistMarketDataHandler(timeframe=timeframe, file_format=file_format, base_folder=base_folder, timezone=timezone)

    def run(self):

        symbols = self.symbols or os.listdir(self.handler.base_folder)

        for symbol in symbols:

            hist_folder_symbol = os.path.join(self.handler.base_folder, symbol)
            print(f"🔄 Converting data for {symbol}")

            df, _, _, symbol_file_path_from = helpers.check_existing_data_file(symbol, self.handler.timeframe, hist_folder_symbol, data_type=self.data_type,
                                                                                   delete_file=not self.keep_file, file_format=self.convert_from)
            # symbol_file_path_to = f"{symbol_file_path_from.rsplit('.', 1)[0]}.{convert_to.lstrip('.')}"

            ##############################################################################################
            # helpers.get_df_memory_usage(df, verbose=True)
            ##############################################################################################


            if df.empty or not symbol_file_path_from:
                print(f"No existing data found for symbol {symbol} with format {self.convert_from}")
                continue
            else:
                symbol_file_path_to = helpers.change_file_format(symbol_file_path_from, self.convert_to)

                # Optimization
                if self.optimize:
                    df = self._optimize_dataframe(df, verbose=True)

                # Conversion
                helpers.save_df_to_file(df, symbol_file_path_to, file_format=self.convert_to)

                if not self.keep_file:
                    os.remove(symbol_file_path_from)
                    print("Removed file ", symbol_file_path_from, "\n")

    def _optimize_dataframe(self, df: pd.DataFrame, verbose=True):
        optimized_df = df.copy()

        for col in optimized_df.columns:
            col_data = optimized_df[col]
            col_dtype = col_data.dtype

            if pd.api.types.is_object_dtype(col_data):
                num_unique = col_data.nunique(dropna=False)
                num_total = len(col_data)
                if num_unique / num_total < 0.5:
                    optimized_df[col] = col_data.astype('category')
                    if verbose:
                        print(f"Converted {col} to category")
                else:
                    # Try date parsing
                    try:
                        optimized_df[col] = pd.to_datetime(col_data)
                        if verbose:
                            print(f"Converted {col} to datetime")
                    except (ValueError, TypeError):
                        pass  # Not convertible to datetime

            elif pd.api.types.is_float_dtype(col_data):
                if (col_data.fillna(0) == col_data.fillna(0).astype(np.float32)).all():
                    optimized_df[col] = col_data.astype(np.float32)
                    if verbose:
                        print(f"Downcasted {col} from float64 to float32")

            elif pd.api.types.is_integer_dtype(col_data):
                optimized_df[col] = pd.to_numeric(col_data, downcast='integer')
                if verbose:
                    print(f"Downcasted {col} to {optimized_df[col].dtype}")

            elif pd.api.types.is_bool_dtype(col_data):
                optimized_df[col] = col_data.astype(bool)
                if verbose:
                    print(f"Converted {col} to bool")

        return optimized_df


if __name__ == "__main__":

    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    # Args Setup
    args = sys.argv
    action_list = ['fetch', 'complete', 'validate', 'load', 'enrich', 'convert']
    paperTrading = not 'live' in args
    local_ib = 'local' in args
    override = 'override' in args
    delete_file = not 'nodelete' in args
    keep_results = 'keep' in args
    fetcher_type = next((arg[13:] for arg in args if arg.startswith('fetcher_type=') and arg[13:] in CONSTANTS.FETCHER_TYPES), 'auto')
    file_format = next((arg[7:] for arg in args if arg.startswith('format=') and arg[7:] in FORMATS.DATA_FILE_FORMATS_LIST), 'parquet')
    seed = next((int(arg[5:]) for arg in args if arg.startswith('seed=')), None)
    action = next((arg[7:] for arg in args if arg.startswith('action=') and arg[7:] in action_list), None)
    single_symbol = next((arg[7:] for arg in args if arg.startswith('symbol=')), None)
    convert_to = next((arg[11:] for arg in args if arg.startswith('convert_to=') and arg[11:] in FORMATS.DATA_FILE_FORMATS_LIST), 'csv')
    convert_from = next((arg[13:] for arg in args if arg.startswith('convert_from=') and arg[13:] in FORMATS.DATA_FILE_FORMATS_LIST), 'parquet')
    data_type = next((arg[10:] for arg in args if arg.startswith('data_type=') and arg[10:] in FORMATS.DATA_TYPE_FORMATS), 'hist_data')
    timeframe = next((arg[3:] for arg in args if arg.startswith('tf=')), '1min')
    # optimize = 'optimize' in args

    symbols = [single_symbol] if single_symbol else []

    # TWS Connection
    ib, ibConnection = helpers.IBKRConnect_any(IB(), paper=paperTrading, remote=not local_ib)
    # ib=IB()

    if action == 'fetch':
        fetcher = HistMarketDataFetcher(ib, ftype=fetcher_type, file_format=file_format, delete_existing=delete_file)
        fetcher.run()

    elif action == 'validate':
        validator = HistMarketDataValidator(timeframe=Timeframe(timeframe), file_format=file_format, symbols=symbols)
        result = validator.run(print_result=True)

    elif action == 'complete':
        if seed: symbols = helpers.get_symbol_seed_list(seed)
        hist_completer = HistMarketDataCompleter(ib, timeframe=Timeframe(timeframe), file_format=file_format, symbols=symbols, step_duration='auto')
        hist_completer.run()

    elif action == 'enrich':
        mtf = None#['1D', '4h']
        if seed and not single_symbol: symbols = helpers.get_symbol_seed_list(seed)
        # symbols = ['QIPT','SPGI']
        separation_text = "*" * 75 + "\n\n"
        enricher = HistMarketDataEnricher(ib, timeframe=Timeframe(timeframe), mtf=mtf, file_format=file_format, symbols=symbols, 
                                          keep_results=keep_results, delete_existing=delete_file, separation_text=separation_text, 
                                          override_existing=override)
        results = enricher.run()

    elif action == 'convert':
        converter = HistMarketDataConverter(convert_from=convert_from, convert_to=convert_to, timeframe=Timeframe(timeframe), symbols=symbols, data_type=data_type)
        converter.run()

    else:
        print("\nAction definition is needed.\n")


# if __name__ == "__main__":

#     args = sys.argv

#     symbols = []
#     format_file = 'parquet'
#     action = 'validate'
#     if len(args) > 1:
#         for arg in args:
#             if 'sym' in arg: symbols = [arg[3:]]
#             if 'action' in arg:
#                 action = arg[6:] if arg[6:] in ['validate', 'enrich'] else action
#             if 'format' in arg:
#                 format_file = arg[6:] if arg[6:] in FORMATS.DATA_FILE_FORMATS_LIST else format_file

#     # Setup
#     hist_folder = PATHS.folders_path['hist_market_data']
#     market_data_folder = PATHS.folders_path['market_data']
#     print("\nHist_Data folder: ", hist_folder, "\n")

#     if action == 'validate':
#         validator = HistMarketDataValidator(timeframe='1 min', file_format=format_file, symbols=symbols)
#         results = validator.validate_all(print_result=True)
#     elif action == 'enrich':
#         enricher = HistMarketDataEnricher(timeframe='1 min', file_format=format_file, symbols=symbols)
#         results = enricher.enrich_all(print_result=True)



# class HistMarketDataValidator:
#     def __init__(self, timeframe='1 min', file_format='csv', symbols: list=[], calendar_name="XNYS",
#                  base_folder=PATHS.folders_path['hist_market_data'], timezone=CONSTANTS.TZ_WORK):
#         self.base_folder = base_folder
#         self.calendar = exchange_calendars.get_calendar(calendar_name)
#         self.timezone = pytz.timezone(timezone)
#         self.file_format = file_format
#         self.timeframe = timeframe
#         self.symbols = symbols

#     def parse_filename(self, filename):
#         match = helpers.build_hist_data_filename_pattern(timeframe=self.timeframe, file_format=self.file_format).match(filename)
#         return match.groupdict() if match else None

#     def validate_file(self, path):
#         filename = os.path.basename(path)
#         meta = self.parse_filename(filename)

#         result = {'symbol': meta['symbol'] if meta else None,
#                   'result': 'passed',
#                   'error_type': None,
#                   'error_value': None,
#                   'text': f"✅ {filename} passed\n"}

#         if not meta:
#             result['result'] = 'failed'
#             result['text'] = f"❌ Invalid filename format: {filename}"
#             result['error_type'] = 'format'
#             return result

#         try:
#             start_dt = self._parse_dt(meta['start'])
#             end_dt = self._parse_dt(meta['end'])
#         except ValueError:
#             result['result'] = 'failed'
#             result['text'] = f"❌ Invalid date in filename: {filename}"
#             result['error_type'] = 'format'
#             return result

#         if start_dt <= end_dt:
#             result['result'] = 'failed'
#             result['text'] = f"❌ Start date is before end date: {filename}"
#             result['error_type'] = 'date order'
#             return result

#         read_funct = pd.read_csv if self.file_format == 'csv' else pd.read_pickle if self.file_format in ['pkl', 'pickle'] else pd.read_parquet if self.file_format in ['pqt', 'parquet'] else None
#         try:
#             if self.file_format == 'csv':
#                 df = read_funct(path, parse_dates=["date"])
#             else:
#                 df = read_funct(path)
#         except Exception as e:
#             result['result'] = 'failed'
#             result['text'] = f"❌ Failed to load {filename} -> {e}"
#             result['error_type'] = 'load'
#             return result


#         # df["date"] = df["date"].dt.tz_localize(self.timezone, ambiguous="NaT", nonexistent="shift_forward")

#         # if abs((df["date"].min() - start_dt).total_seconds()) > 60:
#         #     result['result'] = 'failed'
#         #     result['text'] = f"❌ Start date mismatch in {filename}"
#         #     result['error_type'] = 'start mismatch'
#         # if abs((df["date"].max() - end_dt).total_seconds()) > 60:
#         #     result['result'] = 'failed'
#         #     result['text'] = f"❌ End date mismatch in {filename}"
#         #     result['error_type'] = 'end mismatch'

#         # expected_minutes = self.calendar.minutes_in_range(start_dt.date(), end_dt.date())
#         expected_minutes = self.calendar.minutes_in_range(df["date"].min().date(), df["date"].max().date())
#         expected_minutes = expected_minutes.tz_convert(self.timezone)

#         actual_minutes = pd.Series(df["date"].sort_values().drop_duplicates())
#         missing = expected_minutes.difference(actual_minutes)

#         if not missing.empty:
#             result['result'] = 'warning'
#             result['text'] = f"⚠️ Missing {len(missing)} minutes in {filename}"
#             result['error_type'] = 'missing'
#             result['error_value'] = missing
#             return result

#         return result

#     def _parse_dt(self, dt_str):
#         # fmt = FORMATS.MKT_DATA_FILENAME_DATETIME_FMT
#         dt_str = pd.to_datetime(dt_str, utc=True)
#         dt_str = dt_str.tz_convert(CONSTANTS.TZ_WORK)
#         return dt_str
#         # return datetime.strptime(dt_str, fmt).replace(tzinfo=self.timezone)

#     def validate_all(self, print_result=True):
#         symbols_list = self.symbols if self.symbols else os.listdir(self.base_folder)

#         results = []
#         for symbol in symbols_list:
#             print(f"Assessing data for {symbol}")
#             folder = os.path.join(self.base_folder, symbol)
#             if not os.path.isdir(folder):
#                 continue
#             for file in os.listdir(folder):
#                 if file.endswith("." + self.file_format) and self.timeframe in file:
#                     path = os.path.join(folder, file)
#                     result = self.validate_file(path)
#                     results.append(result)
#                     if print_result: print(result['text'])

#         return results