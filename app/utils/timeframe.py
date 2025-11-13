import os, sys, re, typing
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

current_folder = os.path.dirname(os.path.realpath(__file__))
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)
from utils.constants import CONSTANTS


class Timeframe:
    def __init__(self, value:typing.Union[str,tuple[int,str]]=None, style:str=None):
        """
        Create a Timeframe instance from any style (pandas, ibkr, polygon).

        Args:
            value: str or (int, str)
            style: 'pandas', 'ibkr', 'polygon' (optional, auto-detects if None)
        """
        if style:
            style = style.lower()

        if isinstance(value, tuple) or style == 'polygon':
            self._from_polygon(*value)
        elif isinstance(value, str):
            if style == 'pandas':
                self._from_pandas(value)
            elif style == 'ibkr':
                self._from_ibkr(value)
            else:
                self._auto_detect(value)
        elif not value:
            self._from_default()
        else:
            raise ValueError(f"Unsupported input format: {value}")

    def _auto_detect(self, value: str):
        value = value.strip()
        if ' ' in value:
            self._from_ibkr(value)
        else:
            self._from_pandas(value)

    def _from_pandas(self, pandas_tf: str):
        match = re.match(r'^(\d+)\s*([a-zA-Z]+)$', pandas_tf.strip())
        if not match:
            raise ValueError(f"Invalid pandas timeframe: {pandas_tf}")
        amount, unit = match.groups()
        self.multiplier = int(amount)
        self.unit = self._normalize_pandas_unit(unit)

    def _from_ibkr(self, ibkr_tf: str):
        match = re.match(r'^(\d+)\s*([a-zA-Z]+)$', ibkr_tf.strip().lower())
        if not match:
            raise ValueError(f"Invalid IBKR timeframe: {ibkr_tf}")
        amount, unit = match.groups()
        self.multiplier = int(amount)
        self.unit = self._normalize_ibkr_unit(unit)

    def _from_polygon(self, multiplier: int, timespan: str):
        self.multiplier = int(multiplier)
        self.unit = self._normalize_polygon_unit(timespan)

    def _from_default(self):
        self.multiplier = CONSTANTS.DEFAULT_TIMEFRAME['multiplier']
        self.unit = CONSTANTS.DEFAULT_TIMEFRAME['unit']
        print(f"Timeframe value is None, fall back to default timeframe: {self.multiplier} {self.unit}")

    def _normalize_pandas_unit(self, unit: str) -> str:
        unit = unit.lower()
        map = {'s': 'second', 'sec': 'second', 't': 'minute', 'min': 'minute', 'h': 'hour', 'd': 'day',
               'w': 'week', 'm': 'month', 'q': 'quarter', 'y': 'year', 'a': 'year'}
        if unit not in map:
            raise ValueError(f"Unsupported pandas unit: {unit}")
        return map[unit]

    def _normalize_ibkr_unit(self, unit: str) -> str:
        unit = unit.lower()
        map = {
            's': 'second', 'sec': 'second', 'secs': 'second', 'seconds': 'second',
            'min': 'minute', 'mins': 'minute', 'minute': 'minute', 'minutes': 'minute',
            'h': 'hour', 'hour': 'hour', 'hours': 'hour',
            'd': 'day', 'day': 'day', 'days': 'day',
            'w': 'week', 'week': 'week', 'weeks': 'week',
            'm': 'month', 'month': 'month', 'months': 'month',
            'q': 'quarter', 'quarter': 'quarter',
            'y': 'year', 'a': 'year', 'year': 'year', 'years': 'year',
        }

        if unit not in map:
            raise ValueError(f"Unsupported IBKR unit: {unit}")
        return map[unit]

    def _normalize_polygon_unit(self, unit: str) -> str:
        unit = unit.lower()
        valid = ['second', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year']
        if unit not in valid:
            raise ValueError(f"Unsupported Polygon unit: {unit}")
        return unit

    # ---------------------------
    # Properties for each format
    # ---------------------------

    @property
    def pandas(self) -> str:
        map = {'second': 'S', 'minute': 'min', 'hour': 'h', 'day': 'D', 'week': 'W', 'month': 'M', 'quarter': 'Q', 'year': 'Y'}
        return f"{self.multiplier}{map[self.unit]}"

    # @property
    # def ibkr(self) -> str:
    #     plural = self.unit + 's' if self.multiplier > 1 else self.unit
    #     return f"{self.multiplier} {plural}"

    @property
    def ibkr(self) -> str:
        """
        Returns the IBKR format for the timeframe, which is expected in the format of
        "<multiplier> <unit>" (e.g., "1 min", "2 hours", "5 days").

        For `quarter` and `year`, we treat them as months.
        """
        unit_map = {'second': 'sec', 'minute': 'min', 'hour': 'hour', 'day': 'day', 'week': 'week', 'month': 'month', 'year': 'year'}
        if self.unit == 'quarter':
            self.unit = 'month'
            self.multiplier *= 3  # 1 quarter = 3 months
        elif self.unit == 'year':
            self.unit = 'month'
            self.multiplier *= 12  # 1 year = 12 months

        ibkr_unit = unit_map[self.unit] + 's' if self.multiplier > 1 else unit_map[self.unit]
        return f"{self.multiplier} {ibkr_unit}"

    @property
    def polygon(self) -> tuple[int, str]:
        return self.multiplier, self.unit

    def __repr__(self):
        return f"<Timeframe {self.multiplier} {self.unit}>"

    def __str__(self):
        return self.pandas  # default string representation

    @property
    def to_seconds(self) -> int:
        """
        Convert this timeframe to its equivalent duration in seconds.
        """
        unit = self.unit.lower()
        if unit not in CONSTANTS.TIME_TO_SEC:
            raise ValueError(f"Unsupported unit for seconds conversion: {unit}")

        return self.multiplier * CONSTANTS.TIME_TO_SEC[unit]

    @property
    def to_timedelta(self) -> timedelta:
        """
        Return the timeframe as a `timedelta` object.
        """
        return timedelta(seconds=self.to_seconds)

    def _apply_to_date(self, date:datetime, direction:int=1) -> datetime:
        """
        Internal method to add or subtract this timeframe to/from a datetime.

        Args:
            date: A datetime object.
            direction: +1 to add, -1 to subtract.
        """
        if not isinstance(date, datetime):
            raise ValueError("`date` must be a datetime object.")

        unit_map = {'second': 'seconds', 'minute': 'minutes', 'hour': 'hours', 'day': 'days',
                    'week': 'weeks', 'month': 'months', 'quarter': 'months', 'year': 'years',}

        if self.unit not in unit_map:
            raise ValueError(f"Unsupported timeframe unit: '{self.unit}'")

        normalized_unit = unit_map[self.unit]
        value = self.multiplier * direction

        if self.unit == 'quarter':
            value *= 3  # Convert quarters to months

        if normalized_unit in ['seconds', 'minutes', 'hours', 'days', 'weeks']:
            delta = timedelta(**{normalized_unit: value})
        elif normalized_unit in ['months', 'years']:
            delta = relativedelta(**{normalized_unit: value})
        else:
            raise ValueError(f"Unhandled normalized unit: '{normalized_unit}'")

        return date + delta

    def add_to_date(self, date: datetime) -> datetime:
        """Public method to add timeframe to a datetime."""
        return self._apply_to_date(date, direction=1)

    def subtract_from_date(self, date: datetime) -> datetime:
        """Public method to subtract timeframe from a datetime."""
        return self._apply_to_date(date, direction=-1)
    
    def next_timeframe(self, level:int=1):
        """Public method to find next timeframe from standard timeframes.
        """
        current_seconds = self.to_seconds
        valid_timeframes_seconds = sorted([Timeframe(tf).to_seconds for tf in CONSTANTS.TIMEFRAMES_STD])
        upper_timeframes = [sec for sec in valid_timeframes_seconds if sec > current_seconds]
        lower_timeframes = [sec for sec in valid_timeframes_seconds if sec < current_seconds]

        if current_seconds not in valid_timeframes_seconds:
            if level > 0:
                current_seconds = max(lower_timeframes) if lower_timeframes else min(valid_timeframes_seconds)
            elif level < 0:
                current_seconds = min(upper_timeframes) if upper_timeframes else max(valid_timeframes_seconds)
            else:
                dist = [abs(current_seconds - tf_secs) for tf_secs in valid_timeframes_seconds]
                dist_min_index = dist.index(min(dist))
                current_seconds = valid_timeframes_seconds[dist_min_index]
                
                # current_seconds = filter(valid_timeframes_seconds, lambda tf_secs: abs(current_seconds - tf_secs))
                # current_seconds = min(range(0, len(valid_timeframes_seconds)))
        
        current_index = valid_timeframes_seconds.index(current_seconds)# - int(level/abs(level))
        next_index = min(current_index + level, len(valid_timeframes_seconds) - 1) if level >= 0 else max(current_index + level, 0)

        # Return a new Timeframe instance with the next valid timeframe
        next_timeframe = CONSTANTS.TIMEFRAMES_STD[next_index]
        return Timeframe(next_timeframe)
    

class TimeframeHandler():

    @staticmethod
    def timeframe_to_seconds(tf: str) -> int:
        """
        Convert timeframe string like '1W', '5 min', or '30H' into seconds.
        Handles both spaced and non-spaced formats.
        """
        tf = tf.strip()
        match = re.match(r'^(\d+)\s*([a-zA-Z]+)$', tf)
        if not match:
            raise ValueError(f"Invalid timeframe format: {tf}")

        num, unit = match.groups()
        unit = unit.lower()

        if unit not in CONSTANTS.TIME_TO_SEC:
            raise ValueError(f"Unsupported unit: {unit}")

        return int(num) * CONSTANTS.TIME_TO_SEC[unit]

    @staticmethod
    def ibkr_to_pandas_timeframe(ibkr_tf: str) -> str:
        """
        Convert an IBKR-style timeframe (e.g. '1 min', '5 days') to a pandas frequency string (e.g. '1min', '5D').
        """
        if not isinstance(ibkr_tf, str):
            raise ValueError("Timeframe must be a string.")

        tf = ibkr_tf.strip().lower()

        match = re.match(r"(\d+)\s*([a-zA-Z]+)", tf)
        if not match:
            raise ValueError(f"Invalid IBKR timeframe format: '{ibkr_tf}'")

        amount, unit = match.groups()

        unit_map = {
            'sec': 's', 'secs': 's', 'second': 's', 'seconds': 's',
            'min': 'min', 'mins': 'min', 'minute': 'min', 'minutes': 'min',
            'hour': 'h', 'hours': 'h',
            'day': 'D', 'days': 'D',
            'week': 'W', 'weeks': 'W',
            'month': 'M', 'months': 'M',
            'year': 'Y', 'years': 'Y'
        }

        if unit not in unit_map:
            raise ValueError(f"Unsupported IBKR unit: '{unit}'")

        return f"{amount}{unit_map[unit]}"

    @staticmethod
    def pandas_to_ibkr_timeframe(pandas_timeframe: str) -> str:
        """
        Convert a pandas frequency string (e.g. '1min', '5H', '1D') to an IBKR-compatible timeframe
        string (e.g. '1 min', '5 hours', '1 day').
        """
        if not isinstance(pandas_timeframe, str):
            raise ValueError("Timeframe must be a string.")

        tf = pandas_timeframe.strip().lower()

        match = re.match(r"(\d+)\s*([a-zA-Z]+)", tf)
        if not match:
            raise ValueError(f"Invalid pandas timeframe format: '{pandas_timeframe}'")

        amount, unit = match.groups()

        # Normalize pandas unit aliases
        unit_aliases = {
            't': 'min',
            'min': 'min',
            'h': 'hour',
            'd': 'day',
            'b': 'day',     # business day → closest approximation
            'w': 'week',
            'm': 'month',   # month-end/start treated the same
            'ms': None,     # unsupported in IBKR
            'us': None,
            'ns': None,
            's': 'second',
            'q': None,      # quarter unsupported directly
            'a': 'year',
            'y': 'year',
            'as': 'year',
            'ys': 'year',
            'qs': None,
            'bm': 'month',
            'bms': 'month',
        }

        unit = unit.lower()
        if unit not in unit_aliases:
            raise ValueError(f"Unrecognized or unsupported pandas unit: '{unit}'")

        ibkr_unit = unit_aliases[unit]
        if ibkr_unit is None:
            raise ValueError(f"Pandas unit '{unit}' is not supported by IBKR.")

        # Handle singular/plural properly
        if amount == '1':
            return f"{amount} {ibkr_unit}"
        else:
            # pluralize: IBKR expects "mins", "hours", etc.
            if ibkr_unit == "min":
                return f"{amount} mins"
            else:
                return f"{amount} {ibkr_unit}s"

    @staticmethod
    def pandas_to_ibkr_duration(pandas_duration: str) -> str:
        """
        Convert a pandas-style duration string (e.g. '2Y', '5D', '3M', '1Q')
        to an IBKR-compatible duration string (e.g. '2 Y', '5 D').
        """
        if not isinstance(pandas_duration, str):
            raise ValueError("Duration must be a string.")

        duration = pandas_duration.strip().lower()

        match = re.match(r"^(\d+)\s*([a-z]+)$", duration)
        if not match:
            raise ValueError(f"Invalid pandas duration format: '{pandas_duration}'")

        amount_str, unit = match.groups()
        amount = int(amount_str)

        # Mapping from pandas unit to IBKR durationStr unit
        unit_map = {
            's': 'S',
            'd': 'D',
            'w': 'W',
            'm': 'M',
            'y': 'Y',
            'q': 'M',  # 1 quarter = 3 months
        }

        if unit not in unit_map:
            raise ValueError(f"Unsupported duration unit for IBKR: '{unit}'")

        ibkr_unit = unit_map[unit]

        # Convert quarters to months
        if unit == 'q':
            amount *= 3

        return f"{amount} {ibkr_unit}"

    @staticmethod
    def ibkr_to_polygon_timeframe(ibkr_timeframe: str) -> tuple[int, str]:
        """
        Convert IBKR-style timeframe (e.g., '5 mins', '1 day') to
        Polygon.io's (multiplier, timespan) format.
        """
        if not isinstance(ibkr_timeframe, str):
            raise ValueError("Timeframe must be a string.")

        tf = ibkr_timeframe.strip().lower()

        match = re.match(r"(\d+)\s*([a-z]+)", tf)
        if not match:
            raise ValueError(f"Invalid IBKR timeframe format: '{ibkr_timeframe}'")

        amount_str, unit = match.groups()
        multiplier = int(amount_str)

        # Normalize unit aliases
        unit_map = {
            's': 'second', 'sec': 'second', 'secs': 'second', 'second': 'second', 'seconds': 'second',
            'min': 'minute', 'mins': 'minute', 'minute': 'minute', 'minutes': 'minute',
            'h': 'hour', 'hr': 'hour', 'hour': 'hour', 'hours': 'hour',
            'd': 'day', 'day': 'day', 'days': 'day',
            'w': 'week', 'wk': 'week', 'week': 'week', 'weeks': 'week',
            'm': 'month', 'mo': 'month', 'month': 'month', 'months': 'month',
            'q': 'quarter', 'quarter': 'quarter', 'quarters': 'quarter',
            'y': 'year', 'yr': 'year', 'year': 'year', 'years': 'year',
            'q': 'quarter', 'quarter': 'quarter', 'quarters': 'quarter'
        }

        if unit not in unit_map:
            raise ValueError(f"Unsupported IBKR unit: '{unit}'")

        timespan = unit_map[unit]

        return multiplier, timespan

    @staticmethod
    def polygon_to_ibkr_timeframe(multiplier: int, timespan: str) -> str:
        """
        Convert Polygon.io-style timeframe (multiplier, timespan)
        to an IBKR-style timeframe string.
        """
        if not isinstance(multiplier, int) or multiplier < 1:
            raise ValueError("Multiplier must be a positive integer.")

        timespan = timespan.strip().lower()

        # Normalized timespan → IBKR unit (singular/plural logic)
        supported = {
            'second': 'second',
            'minute': 'min',
            'hour': 'hour',
            'day': 'day',
            'week': 'week',
            'month': 'month',
            'quarter': 'quarter',  # optional; can be mapped to months
            'year': 'year'
        }

        if timespan not in supported:
            raise ValueError(f"Unsupported Polygon timespan: '{timespan}'")

        unit = supported[timespan]

        # Special case: pluralization
        if multiplier == 1:
            return f"{multiplier} {unit}"
        else:
            if unit == "min":
                return f"{multiplier} mins"
            elif unit == "hour":
                return f"{multiplier} hours"
            else:
                return f"{multiplier} {unit}s"

    def polygon_to_pandas(multiplier: int, timespan: str) -> str:
        """
        Convert a Polygon.io (multiplier, timespan) pair to a pandas-style frequency string.
        """
        if not isinstance(multiplier, int) or multiplier < 1:
            raise ValueError("Multiplier must be a positive integer.")

        if not isinstance(timespan, str):
            raise ValueError("Timespan must be a string.")

        timespan = timespan.strip().lower()

        # Map Polygon timespans to pandas frequency codes
        timespan_map = {
            'second': 'S',
            'minute': 'min',
            'hour': 'h',
            'day': 'D',
            'week': 'W',
            'month': 'M',
            'quarter': 'Q',
            'year': 'Y'
        }

        if timespan not in timespan_map:
            raise ValueError(f"Unsupported Polygon timespan: '{timespan}'")

        pandas_unit = timespan_map[timespan]

        return f"{multiplier}{pandas_unit}"

    @staticmethod
    def pandas_to_polygon(pandas_freq: str) -> tuple[int, str]:
        """
        Convert a pandas-style frequency string (e.g. '5min', '1h', '1D') to Polygon.io format:
        (multiplier: int, timespan: str)
        """
        if not isinstance(pandas_freq, str):
            raise ValueError("pandas_freq must be a string.")

        pandas_freq = pandas_freq.strip().lower()

        # Match number + unit (e.g. 5min, 1h, 2d)
        match = re.match(r'^(\d+)\s*([a-z]+)$', pandas_freq)
        if not match:
            raise ValueError(f"Invalid pandas frequency format: '{pandas_freq}'")

        value_str, unit = match.groups()
        multiplier = int(value_str)

        # Normalize pandas units to polygon-compatible timespans
        unit_map = {
            's': 'second', 'sec': 'second', 'second': 'second', 'seconds': 'second',
            'min': 'minute', 't': 'minute', 'minute': 'minute', 'minutes': 'minute',
            'h': 'hour', 'hr': 'hour', 'hour': 'hour', 'hours': 'hour',
            'd': 'day', 'day': 'day', 'days': 'day',
            'w': 'week', 'week': 'week', 'weeks': 'week',
            'm': 'month', 'month': 'month', 'months': 'month',
            'y': 'year', 'a': 'year', 'year': 'year', 'years': 'year'
        }

        if unit not in unit_map:
            raise ValueError(f"Unsupported pandas frequency unit: '{unit}'")

        timespan = unit_map[unit]
        return multiplier, timespan

    @staticmethod
    def get_ibkr_duration_as_timedelta(duration_str: str) -> timedelta:
        """
        Convert an IBKR-style or pandas-style duration string to a timedelta object.

        Assumptions:
            - 1M = 30 days (approx)
            - 1Y = 365 days (approx)
        """
        if not isinstance(duration_str, str):
            raise ValueError("Duration must be a string.")

        # Normalize input (remove spaces, make lowercase)
        duration_str = duration_str.strip().replace(" ", "").upper()

        # Match pattern: integer + single unit letter
        match = re.match(r"^(\d+)([SMWDY])$", duration_str)
        if not match:
            raise ValueError(f"Invalid duration format: '{duration_str}'")

        num_str, unit = match.groups()
        num = int(num_str)

        if unit == 'S':
            return timedelta(seconds=num)
        elif unit == 'M':
            return timedelta(days=30 * num)
        elif unit == 'Y':
            return timedelta(days=365 * num)
        elif unit == 'W':
            return timedelta(weeks=num)
        elif unit == 'D':
            return timedelta(days=num)
        else:
            raise ValueError(f"Unsupported duration unit: '{unit}'")

    @staticmethod
    def subtract_ibkr_duration_from_time(to_time: datetime, duration: str) -> datetime:
        """
        Subtract an IBKR-style or pandas-style duration string (e.g. '2 D', '3M') from a datetime.
        """
        if not isinstance(to_time, datetime):
            raise ValueError("to_time must be a datetime object.")

        # Normalize string: remove spaces, make uppercase
        duration = duration.strip().replace(" ", "").upper()

        # Match integer + single letter unit
        match = re.match(r"^(\d+)([SMWDY])$", duration)
        if not match:
            raise ValueError(f"Invalid duration format: '{duration}'")

        qty_str, unit = match.groups()
        qty = int(qty_str)

        # Perform subtraction based on unit
        if unit == 'S':
            return to_time - timedelta(seconds=qty)
        elif unit == 'D':
            return to_time - timedelta(days=qty)  # ✅ Use calendar days, not business
        elif unit == 'W':
            return to_time - timedelta(weeks=qty)
        elif unit == 'M':
            return to_time - relativedelta(months=qty)
        elif unit == 'Y':
            return to_time - relativedelta(years=qty)
        else:
            raise ValueError(f"Unsupported duration unit: '{unit}'")

    @staticmethod
    def add_ibkr_timeframe_to_date(date: datetime, timeframe: str) -> datetime:
        """
        Add a timeframe (e.g. '1 min', '2 days', '3 M') to a datetime.
        """
        if not isinstance(date, datetime):
            raise ValueError("date must be a datetime object.")

        if not isinstance(timeframe, str):
            raise ValueError("timeframe must be a string.")

        # Normalize and extract value/unit
        timeframe = timeframe.strip().lower()
        match = re.match(r"^(\d+)\s*([a-z]+)$", timeframe)
        if not match:
            raise ValueError(f"Invalid timeframe format: '{timeframe}'")

        value_str, unit = match.groups()
        value = int(value_str)

        # Unit normalization
        unit_map = {
            's': 'seconds', 'sec': 'seconds', 'second': 'seconds', 'seconds': 'seconds',
            'min': 'minutes', 'mins': 'minutes', 'minute': 'minutes', 'minutes': 'minutes',
            'h': 'hours', 'hr': 'hours', 'hour': 'hours', 'hours': 'hours',
            'd': 'days', 'day': 'days', 'days': 'days',
            'w': 'weeks', 'week': 'weeks', 'weeks': 'weeks',
            'm': 'months', 'mo': 'months', 'month': 'months', 'months': 'months',
            'y': 'years', 'yr': 'years', 'year': 'years', 'years': 'years'
        }

        if unit not in unit_map:
            raise ValueError(f"Unsupported timeframe unit: '{unit}'")

        normalized_unit = unit_map[unit]

        # Apply the appropriate delta
        if normalized_unit in ['seconds', 'minutes', 'hours', 'days', 'weeks']:
            delta = timedelta(**{normalized_unit: value})
            return date + delta
        elif normalized_unit in ['months', 'years']:
            delta = relativedelta(**{normalized_unit: value})
            return date + delta
        else:
            raise ValueError(f"Unhandled timeframe unit after normalization: '{normalized_unit}'")

    @staticmethod
    def add_pandas_timeframe_to_date(date: datetime, timeframe: str) -> datetime:
        """
        Add a pandas-style timeframe (e.g. '5min', '2D', '3M') to a datetime.
        """
        if not isinstance(date, datetime):
            raise ValueError("date must be a datetime object.")
        if not isinstance(timeframe, str):
            raise ValueError("timeframe must be a string.")

        timeframe = timeframe.strip().lower()

        # Regex to parse amount and unit, e.g. '5min', '1h', '2d', '3M'
        match = re.match(r"^(\d+)\s*([a-z]+)$", timeframe)
        if not match:
            raise ValueError(f"Invalid pandas-style timeframe format: '{timeframe}'")

        value_str, unit = match.groups()
        value = int(value_str)

        # Normalize pandas units
        unit_map = {
            's': 'seconds', 'sec': 'seconds',
            'min': 'minutes', 't': 'minutes',
            'h': 'hours',
            'd': 'days',
            'w': 'weeks',
            'm': 'months',
            'q': 'months',  # Treat 1Q as 3M
            'y': 'years', 'a': 'years'
        }

        if unit not in unit_map:
            raise ValueError(f"Unsupported pandas timeframe unit: '{unit}'")

        normalized_unit = unit_map[unit]

        if normalized_unit == 'months':
            # Handle quarters separately
            if unit == 'q':
                value *= 3
            return date + relativdelta(months=value)
        elif normalized_unit == 'years':
            return date + relativdelta(years=value)
        elif normalized_unit in ['seconds', 'minutes', 'hours', 'days', 'weeks']:
            return date + timedelta(**{normalized_unit: value})
        else:
            raise ValueError(f"Unhandled normalized unit: '{normalized_unit}'")










    # def get_ibkr_duration_from_time_diff(to_time, from_time):

    #     delta = relativdelta(to_time, from_time)

    #     # Calculate business days
    #     business_days = pd.tseries.offsets.BDay().rollforward(from_time) - from_time
    #     total_business_days = len(pd.date_range(from_time, to_time, freq='B'))

    #     # Determine the best unit based on the difference
    #     if delta.years > 0: duration = f"{int(delta.years)} Y"
    #     elif delta.months > 0: duration = f"{int(delta.months)} M"
    #     # elif total_business_days >= 1: duration = f"{total_business_days} D"
    #     elif total_business_days >= 1 and delta.days >= 1: duration = f"{int(total_business_days)} D"
    #     elif delta.days >= 7: duration = f"{int(delta.days // 7)} W"
    #     elif delta.days >= 1: duration = f"{int((to_time - from_time).total_seconds() // 3600)} H"
    #     else: duration = f"{int((to_time - from_time).total_seconds())} S"

    #     return duration


# def ibkr_to_polygon_timeframe(timeframe_ibkr):
#     '''Convert IBKR timeframe to Polygon.io timespan'''
#     if "sec" in timeframe_ibkr: return "second"
#     elif "min" in timeframe_ibkr: return "minute"
#     elif "hour" in timeframe_ibkr: return "hour"
#     elif "day" in timeframe_ibkr: return "day"
#     elif "week" in timeframe_ibkr: return "day"
#     elif "month" in timeframe_ibkr: return "day"
#     else:
#         raise ValueError(f"Invalid IBKR timeframe for polygon timeframe translation: {timeframe_ibkr}")


# def pandas_duration_to_ibkr_duration_str(pandas_duration: str) -> str:
#     """
#     Convert a pandas-style duration like '2Y', '5D', '3M' to an IBKR-compatible duration string.
#     """
#     if not isinstance(pandas_duration, str):
#         raise ValueError("Duration must be a string.")

#     duration = pandas_duration.strip().upper()

#     match = re.match(r"(\d+)\s*([A-Z]+)", duration)
#     if not match:
#         raise ValueError(f"Invalid pandas duration format: '{pandas_duration}'")

#     amount, unit = match.groups()

#     unit_map = {
#         'S': 'S',      # Seconds
#         'M': 'M',      # Months
#         'H': 'H',      # Hours
#         'D': 'D',      # Days
#         'W': 'W',      # Weeks
#         'Y': 'Y',      # Years
#         'Q': 'M',      # Quarters -> Months (approximate as 3M)
#         'MIN': 'D',    # Treat "minutes" as days if needed
#     }

#     if unit not in unit_map:
#         raise ValueError(f"Unsupported duration unit for IBKR: '{unit}'")

#     ibkr_unit = unit_map[unit]

#     # Special case for quarters
#     if unit == 'Q':
#         amount = str(int(amount) * 3)  # 1Q = 3M

#     return f"{amount} {ibkr_unit}"

# def convert_ibkr_to_pandas_freq(timeframe_ibkr):
#     if "sec" in timeframe_ibkr: return "S"  # Seconds in Pandas
#     elif "min" in timeframe_ibkr: return "T"  # Minutes in Pandas
#     elif "hour" in timeframe_ibkr: return "H"  # Hours in Pandas
#     elif "day" in timeframe_ibkr: return "D"  # Days in Pandas
#     elif "week" in timeframe_ibkr: return "W"  # Weeks in Pandas
#     elif "month" in timeframe_ibkr: return "M"  # Months in Pandas
#     else:
#         raise ValueError(f"Invalid IBKR timeframe for pandas frequency translation: {timeframe_ibkr}")

# def ibkr_to_pandas_timeframe(ibkr_tf: str) -> str:
#     """
#     Convert an IBKR-style timeframe (e.g. '1 min', '5 D') to a pandas resample frequency string (e.g. '1min', '5D').
#     """
#     ibkr_tf = ibkr_tf.strip().lower()
#     parts = ibkr_tf.split()

#     if len(parts) != 2:
#         raise ValueError(f"Invalid IBKR timeframe format: '{ibkr_tf}'")

#     num, unit = parts
#     num = int(num)

#     # Mapping IBKR unit to pandas frequency codes (preferred format)
#     unit_map = {'sec': 's', 'secs': 's', 'second': 's', 'seconds': 's', 'min': 'min', 'mins': 'min', 'minute': 'min',
#                 'minutes': 'min', 'hour': 'H', 'hours': 'H', 'day': 'D', 'days': 'D', 'week': 'W', 'weeks': 'W',
#                 'month': 'M', 'months': 'M', 'year': 'Y', 'years': 'Y'}

#     if unit not in unit_map:
#         raise ValueError(f"Unsupported unit: '{unit}'")

#     return f"{num}{unit_map[unit]}"

# def pandas_to_ibkr_timeframe(pandas_timeframe: str) -> str:
#     """
#     Convert a pandas resample-friendly timeframe to an IBKR-style timeframe string.

#     Parameters:
#         pandas_timeframe (str): Timeframe like '1min', '5H', '1D', etc.

#     Returns:
#         str: IBKR-style string like '1 min', '5 hours', '1 day'.
#     """
#     if not isinstance(pandas_timeframe, str):
#         raise ValueError("Timeframe must be a string.")

#     tf = pandas_timeframe.strip().lower()

#     # Extract number and unit
#     match = re.match(r"(\d+)\s*([a-zA-Z]+)", tf)
#     if not match:
#         raise ValueError(f"Invalid pandas timeframe format: '{pandas_timeframe}'")

#     amount, unit = match.groups()

#     # Mapping from pandas to human/IBKR-readable
#     reverse_map = {
#         's': 'second', 'ms': 'millisecond', 'us': 'microsecond', 'ns': 'nanosecond', 'min': 'min', 'h': 'hour', 'd': 'day',
#         'w': 'week', 'm': 'month', 'q': 'quarter', 'y': 'year', 'a': 'year'}

#     # Normalize unit to lowercase for matching
#     unit = unit.lower()

#     # Handle special cases like "min" and "h"
#     if unit in ['minute', 'minutes']:
#         unit = 'min'

#     if unit not in reverse_map:
#         raise ValueError(f"Unrecognized pandas time unit: '{unit}'")

#     ibkr_unit = reverse_map[unit]

#     # Pluralize when appropriate
#     if amount == '1':
#         return f"{amount} {ibkr_unit}"
#     else:
#         return f"{amount} {ibkr_unit}s"