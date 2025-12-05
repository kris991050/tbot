import sys, os, pytz
from datetime import datetime, timedelta

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.constants import CONSTANTS


class TradingConfig:
    def __init__(self, live_mode:str='live'):
        self.capital: float = 10000
        self.risk_pct: float = 0.01    # risk per trade (1%)
        self.sl_pct: float = 0.05               # stop loss: 1%
        self.tp_pct: float = 0.1                # take profit: 2%
        self.max_hold_minuties: int = 120       # max holding period
        self.seed = None

        # Predictions config
        self.use_model_prediction: bool = True  # use prediction for entry filter or position sizing
        self.pred_th: float = 0.5

        # Risk and stop config
        self.rrr_threshold: float = 1
        self.tier_max: int = 5
        self.size = 'exponential'
        self.offset_targets: int = 10
        self.profit_ratio: int = 2
        self.perc_gain: int = 10
        self.max_time_factor: int = 12
        self.max_loss_perc: int = 1
        self.partial_perc: int = 10
        self.currency: str = CONSTANTS.DEFAULT_CURRENCY
        self.pred_vlty_type: str = 'garch' # 'ewma'
        self.volatility_factor: int = 1

        # ML model config
        self.model_type: str = 'xgboost'
        self.model_dd_type: str = 'xgboost'
        self.selector_type: str = 'rf'

        # Strategy config
        self.strategy_name = 'bb_rsi_reversal_1min_bear'
        self.revised = False
        self.stop = 'pred_vlty'
        self.paper_trading = True
        self.remote_ib = True

        # Features thresholds
        self.rsi_threshold: int = 75
        self.cam_M_threshold: int = 3

        # Data handling config
        # self.look_backward = '1M'
        self.step_duration = 'auto'
        self.file_format = 'parquet'
        self.timezone = CONSTANTS.TZ_WORK

        # Simualtion config
        self.live_mode = helpers.set_var_with_constraints(live_mode, CONSTANTS.MODES['live']) or 'live'
        self.sim_start = self.timezone.localize(datetime(2025, 12, 4, 9, 30, 0))
        self.sim_max_time = timedelta(hours=4, minutes=0)
        self.set_sim_offset()

    def set_sim_offset(self):
        self.sim_offset = datetime.now(self.timezone) - self.sim_start if self.live_mode == 'sim' else timedelta(0) if self.live_mode == 'live' else None

    def save_config(self, config_path):
        """ Save the config parameters to a JSON/YAML file """
        # Convert complex types to serializable types
        config_dict = self.__dict__.copy()  # Copy the original config

        # Serialize timezone and datetime objects
        if isinstance(self.timezone, pytz.tzinfo.BaseTzInfo):
            config_dict['timezone'] = str(self.timezone)  # Save the timezone name (string)
        if isinstance(self.sim_start, datetime):
            config_dict['sim_start'] = self.sim_start.isoformat()  # Save datetime as string
        if isinstance(self.sim_max_time, timedelta):
            config_dict['sim_max_time'] = str(self.sim_max_time)  # Save timedelta as string

        helpers.save_json(config_dict, config_path, lock=False)
        print(f"💾 Saved config at {config_path}")

    def load_config(self, config_path):
        """ Load the configuration from the file """
        # date_folder = helpers.get_path_date_folder(date)
        # config_path = os.path.join(date_folder, config_filename)
        if os.path.exists(config_path):
            loaded_config = helpers.load_json(config_path, lock=False)

            # Convert complex types back to their original form
            if 'timezone' in loaded_config:
                loaded_config['timezone'] = pytz.timezone(loaded_config['timezone'])  # Convert string back to timezone
            if 'sim_start' in loaded_config:
                loaded_config['sim_start'] = datetime.fromisoformat(loaded_config['sim_start'])  # Convert string back to datetime
            if 'sim_max_time' in loaded_config and isinstance(loaded_config['sim_max_time'], str):
                # The string is in the form 'H:MM:SS' (e.g., '4:00:00')
                time_str = loaded_config['sim_max_time']
                hours, minutes, seconds = map(int, time_str.split(':'))
                loaded_config['sim_max_time'] = timedelta(hours=hours, minutes=minutes, seconds=seconds)
                # loaded_config['sim_max_time'] = timedelta(seconds=int(loaded_config['sim_max_time'].split()[0]))  # Convert string back to timedelta
                # Handle sim_offset (timedelta) from string 'H:MM:SS'
            # if 'sim_offset' in loaded_config:
            #     # Convert time string to timedelta
            #     hours, minutes, seconds = map(int, loaded_config['sim_offset'].split(':'))
            #     loaded_config['sim_offset'] = timedelta(hours=hours, minutes=minutes, seconds=seconds)

            # if 'sim_offset' in loaded_config and isinstance(loaded_config['sim_offset'], str):
            #     # Split the time string into hours, minutes, and seconds
            #     time_parts = loaded_config['sim_offset'].split(':')

            #     # Parse hours and minutes as integers
            #     hours = int(time_parts[0])
            #     minutes = int(time_parts[1])

            #     # The seconds part can be split into whole seconds and fractional seconds
            #     seconds_and_fraction = time_parts[2].split('.')
            #     seconds = int(seconds_and_fraction[0])  # Whole seconds
            #     fractional_seconds = float(f"0.{seconds_and_fraction[1]}") if len(seconds_and_fraction) > 1 else 0.0  # Fractional part as float

            #     # Create a timedelta with fractional seconds
            #     loaded_config['sim_offset'] = timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=fractional_seconds * 1e6)

            if 'sim_offset' in loaded_config and isinstance(loaded_config['sim_offset'], str):
                # Check if the string contains 'day' or 'days'
                if 'day' in loaded_config['sim_offset']:
                    # Extract days and time
                    days_part, time_part = loaded_config['sim_offset'].split(',')  # Split '1 day, 1:52:40.859785'
                    days = int(days_part.split()[0])  # Get the number of days (e.g., 1)
                else:
                    days = 0
                    time_part = loaded_config['sim_offset']  # If no days part, just take the whole string as time

                # Now handle the time part (which should be in the form H:MM:SS.mmmmmm)
                time_parts = time_part.strip().split(':')  # Split time into hours, minutes, and seconds

                # Parse hours, minutes, and seconds (handle cases where hours may not exist)
                hours = int(time_parts[0]) if len(time_parts) > 2 else 0
                minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
                seconds_and_fraction = time_parts[2].split('.')  # Separate seconds and fractional part
                seconds = int(seconds_and_fraction[0])  # Whole seconds
                fractional_seconds = float(f"0.{seconds_and_fraction[1]}") if len(seconds_and_fraction) > 1 else 0.0  # Fractional seconds

                # Create a timedelta including days, hours, minutes, seconds, and fractional seconds
                loaded_config['sim_offset'] = timedelta(days=days, hours=hours, minutes=minutes,
                                                                seconds=seconds, microseconds=fractional_seconds * 1e6)

            return self.set_config(loaded_config)
        else:
            return None

    def set_config(self, locals):
        # Replace config parameters if provided locally
        for param, value in locals.items():
            if param in vars(self) and value is not None:
                setattr(self, param, value)
        self.set_sim_offset()
        return self

    # @staticmethod
    # def load_config_param(param_name, default_value, locals, config_file=None):
    #     # First, check if the parameter is explicitly provided
    #     if param_name in locals:
    #         return locals[param_name]

    #     # Second, check if the parameter is in the daily config file
    #     if config_file and param_name in config_file:
    #         return config_file[param_name]

    #     # Last, fall back to the default value
    #     return default_value