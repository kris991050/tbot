import sys, os, pandas as pd
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from miscellaneous import scanner
import live_loop_base
from utils import helpers, constants


class LiveScansFetcher(live_loop_base.LiveLoopBase):

    def _scan_gappers(self):
        now = helpers.calculate_now(sim_offset=self.config.sim_offset, tz=self.config.timezone)
        if not helpers.is_between_market_times('pre-market', 'rth', now=now, timezone=self.config.timezone):
            return
        print('\n======== FETCHING GAPPERS UP ========\n')
        symbols_up, df_up = scanner.scannerTradingView("GapperUp")
        print(helpers.df_to_table(df_up.round(2)))
        helpers.save_to_daily_csv(self.ib, symbols_up, constants.PATHS.daily_csv_files['gapper_up'])

        print('\n======== FETCHING GAPPERS DOWN ========\n')
        symbols_down, df_down = scanner.scannerTradingView("GapperDown")
        print(helpers.df_to_table(df_down.round(2)))
        helpers.save_to_daily_csv(self.ib, symbols_down, constants.PATHS.daily_csv_files['gapper_down'])

    def _scan_earnings(self):
        now = helpers.calculate_now(sim_offset=self.config.sim_offset, tz=self.config.timezone)
        if not helpers.is_between_market_times('pre-market', 'rth', now=now, timezone=self.config.timezone):
            return
        print('\n======== FETCHING RECENT EARNINGS ========\n')
        symbols, _ = scanner.scannerFinviz("RE")
        print(symbols)
        helpers.save_to_daily_csv(self.ib, symbols, constants.PATHS.daily_csv_files['earnings'])

    def _scan_bb_rsi_reversal(self):
        symbols = self.manager.scan_bb_rsi_reversal()
        print(symbols)
        helpers.save_to_daily_csv(self.ib, symbols, constants.PATHS.daily_csv_files['bb_rsi_reversal'])
    
    def _execute_main_task(self):
        self._scan_gappers()
        self._scan_earnings()
        self._scan_bb_rsi_reversal()


if __name__ == "__main__":

    args = sys.argv
    paper_trading = not 'live' in args
    local_ib = 'local' in args
    continuous = not 'snapshot' in args
    wait_seconds = next((int(float(arg[5:])) for arg in args if arg.startswith('wait=')), 3*60)

    fetcher = LiveScansFetcher(worker_type='scans_fetcher', wait_seconds=wait_seconds, continuous=continuous, paper_trading=paper_trading, remote_ib=not local_ib)
    fetcher.run()





# def is_between_market_times(start_label, end_label):
#     now = pd.Timestamp.now(tz=constants.CONSTANTS.TZ_WORK).time()
#     return constants.CONSTANTS.TH_TIMES[start_label] < now < constants.CONSTANTS.TH_TIMES[end_label]


# # === SCANNER TASKS === #

# def scan_gappers():
#     if not is_between_market_times('pre-market', 'rth'):
#         return
#     print('\n======== FETCHING GAPPERS UP ========\n')
#     symbols_up, df_up = scanner.scannerTradingView("GapperUp")
#     print("\n", helpers.df_to_table(df_up.round(2)), "\n")
#     print("\nSymbols from TradingView scanner GapperUp=\n", symbols_up)
#     helpers.save_to_daily_csv(ib, symbols_up, constants.PATHS.daily_csv_files['gapper_up'])

#     print('\n======== FETCHING GAPPERS DOWN ========\n')
#     symbols_down, df_down = scanner.scannerTradingView("GapperDown")
#     print("\n", helpers.df_to_table(df_down.round(2)), "\n")
#     print("\nSymbols from TradingView scanner GapperDown=\n", symbols_down)
#     helpers.save_to_daily_csv(ib, symbols_down, constants.PATHS.daily_csv_files['gapper_down'])


# def scan_earnings():
#     if not is_between_market_times('pre-market', 'rth'):
#         return
#     print('\n======== FETCHING RECENT EARNINGS ========\n')
#     symbols_earnings = scanner.scannerFinviz("RE")
#     print("\nSymbols from Finviz scanner Recent Earnings=\n", symbols_earnings)
#     helpers.save_to_daily_csv(ib, symbols_earnings, constants.PATHS.daily_csv_files['earnings'])


# def scan_bb_rsi_reversal():
#     if not is_between_market_times('pre-market', 'post-market'):
#         return
#     print('\n======== FETCHING RSI REVERSALS ========\n')
#     symbols_rsi, df_rsi = scanner.scannerTradingView("RSI-Reversal")
#     print("\nSymbols from TradingView scanner RSI_Reversal=\n", symbols_rsi)
#     helpers.save_to_daily_csv(ib, symbols_rsi, constants.PATHS.daily_csv_files['bb_rsi_reversal'])


# if __name__ == "__main__":

#     # # Path Setup
#     # path = helpers.path_setup.path_current_setup(parent_folder)

#     args = sys.argv
#     paperTrading = not 'live' in args
#     continuous = 'cont' in args
#     time_wait = next((int(float(arg[5:])) * 60 for arg in args if arg.startswith('wait=')), 5 * 60)
    

#     # TWS Connection
#     paperTrading = False if len(args) > 1 and 'live' in args else True
#     ib, ibConnection = helpers.IBKRConnect_any(IB(), paper=paperTrading)

#     # Setup
#     daily_data_folder = helpers.get_path_daily_data_folder()

#     counter = 1000
#     while counter > 0:

#         time_now = pd.Timestamp.now(tz=constants.CONSTANTS.TZ_WORK)

#         # Scanner Gappers
#         scan_gappers()
#         scan_earnings()
#         scan_bb_rsi_reversal()

#         if continuous:

#             counter = counter - 1
#             helpers.sleep_display(time_wait, ib)

#         else: counter = 0

# print("\n\n")


# from apscheduler.schedulers.blocking import BlockingScheduler
# from apscheduler.triggers.interval import IntervalTrigger

# def display_next_runs():
#     now = datetime.datetime.now(datetime.timezone.utc)  # Use UTC for consistency with APScheduler for delta calcuation
#     print("\n📅 Scheduled Jobs:")
#     for job in scheduler.get_jobs():
#         if job.id != 'job_monitor':
#             if job.next_run_time and job.id != 'job_monitor':
#                 delta = (job.next_run_time - now).total_seconds()
#                 seconds_remaining = int(delta) if delta > 0 else 0
#                 print(f"  🔔 Job ID: {job.id:<15} | Runs in: {seconds_remaining:>4} seconds")
#             else:
#                 print(f"  ⚠️  Job ID: {job.id:<15} | No next run scheduled.")

# if __name__ == "__main__":

#     # # Path Setup
#     # path = helpers.path_setup.path_current_setup(parent_folder)

#     args = sys.argv
#     paperTrading = not 'live' in args

#     # TWS Connection
#     paperTrading = False if len(args) > 1 and 'live' in args else True
#     ib, ibConnection = helpers.IBKRConnect_any(IB(), paper=paperTrading)

#     # Setup
#     daily_data_folder = helpers.get_path_daily_data_folder() + '_2'
#     scheduler = BlockingScheduler(timezone=constants.CONSTANTS.TZ_WORK)
#     gappers_delay = 5
#     earnings_delay = 10
#     bb_rsi_reversal_delay = 3

#     # === JOB SCHEDULES === #

#     time_now = pd.Timestamp.now(tz=constants.CONSTANTS.TZ_WORK)

#     scheduler.add_job(display_next_runs, IntervalTrigger(seconds=10), id='job_monitor', next_run_time=time_now)
#     # scheduler.add_job(scan_earnings, IntervalTrigger(minutes=earnings_delay), id='earnings', next_run_time=time_now + datetime.timedelta(seconds=5))
#     # scheduler.add_job(scan_gappers, IntervalTrigger(minutes=gappers_delay), id='gappers', next_run_time=time_now + datetime.timedelta(seconds=35))
#     # scheduler.add_job(scan_bb_rsi_reversal, IntervalTrigger(minutes=bb_rsi_reversal_delay), id='bb_rsi_reversal', next_run_time=time_now + datetime.timedelta(seconds=65))
#     scheduler.add_job(run_all_scans, IntervalTrigger(minutes=bb_rsi_reversal_delay), id='all_scans', next_run_time=time_now + datetime.timedelta(seconds=5))

#     # === START SCHEDULER === #
#     print("Starting scheduler. Press Ctrl+C to exit.\n")
#     try:
#         scheduler.start()
#     except (KeyboardInterrupt, SystemExit):
#         print("Shutting down scheduler...")
#         ib.disconnect()

# print("\n\n")

