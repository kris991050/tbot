import sys, os, pandas as pd
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.constants import CONSTANTS, FORMATS, PATHS
from execution import trade_manager
from strategy_analyzer import StrategyAnalyzer
from strategies import rs3_strategy, breakouts_strategy
import strategy_summary


def build_symbols_list(hist_folder, to_time, from_time, seed=None, file_format='parquet', timeframe='1min', remove_indexes=True):

    if seed:
        symbols_list = helpers.get_symbol_seed_list(seed)
    else:
        symbols_list = []
        for symbol in os.listdir(hist_folder):

            hist_folder_symbol = os.path.join(hist_folder, str(symbol))
            df, to_time_exist, from_time_exist, symbol_file_path = \
                helpers.check_existing_data_file(symbol, timeframe, folder=hist_folder_symbol, data_type='hist_data', delete_file=False, file_format=file_format)

            if df.empty or not symbol_file_path or to_time > to_time_exist or from_time < from_time_exist:
                continue
            else:
                symbols_list.append(symbol)

    # Remove Index ETFs from symbols list
    if remove_indexes:
        index_etf_list = [item['ETF'] for item in helpers.get_index_list()]
        symbols_list = [s for s in symbols_list if s not in index_etf_list]

    return sorted(symbols_list)


if __name__ == "__main__":

    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    # Arguments management
    args = sys.argv
    paperTrading = not 'live' in args
    display_res = 'display' in args
    revised = 'revised' in args
    seed = next((int(arg[5:]) for arg in args if arg.startswith('seed=')), None)
    file_format = next((arg[7:] for arg in args if arg.startswith('format=') and arg[7:] in FORMATS.DATA_FILE_FORMATS_LIST), 'parquet')
    strategy = next((arg[9:] for arg in args if arg.startswith('strategy=')), '')
    single_symbol = next((arg[7:] for arg in args if arg.startswith('symbol=')), None)
    entry_delay = next((arg[10:] for arg in args if arg.startswith('entry_delay=')), 1)

    # helpers.test_internet_speed()


    # TWS Connection
    ib, ibConnection = helpers.IBKRConnect_any(IB(), paper=paperTrading)

    # ib = IB()
    # contract = Stock(symbol=symbol, exchange='SMART', currency='USD')

    # Setup
    rev = '' if not revised else '_R' if revised else None
    strategy += rev
    hist_folder = PATHS.folders_path['hist_market_data']
    strategies_folder = PATHS.folders_path['strategies_data']
    strategy_folder = os.path.join(strategies_folder, strategy)
    # strategy_data_file_csv = os.path.join(strategy_folder, strategy + '_results.csv')
    # strategy_data_file_parquet = os.path.join(strategy_folder, strategy + '_results.parquet')
    strategy_results_summary_file_csv = os.path.join(strategy_folder, strategy + '_summary.csv')
    strategy_results_summary_by_mktcap_file_csv = os.path.join(strategy_folder, strategy + '_summary_mktcap.csv')

    timeframe_1m = "1 min"
    to_time = pd.to_datetime('2025-10-01T19:59:59').tz_localize(CONSTANTS.TZ_WORK)#pd.to_datetime('2025-05-01T19:59:59').tz_localize(CONSTANTS.TZ_WORK)
    from_time = pd.to_datetime('2020-01-01T19:59:59').tz_localize(CONSTANTS.TZ_WORK)#pd.to_datetime('2020-01-01T04:00:00').tz_localize(CONSTANTS.TZ_WORK)
    symbols = build_symbols_list(hist_folder, to_time, from_time, seed, file_format) if not single_symbol else [single_symbol]
    if 'bb_rsi_reversal' in strategy:

        mtf = None
        strategy_func = trade_manager.get_strategy_instance(strategy_name=strategy, revised=revised, rsi_threshold=75, cam_M_threshold=4)
        config = [{'timeframe': strategy_func.timeframe, 'targets': [strategy_func.target_handler], 'mtf': mtf}]
    
    if 'sr_bounce' in strategy:

        mtf = None
        strategy_func = trade_manager.get_strategy_instance(strategy_name=strategy, revised=revised, rsi_threshold=75, cam_M_threshold=4)
        config = [{'timeframe': strategy_func.timeframe, 'targets': [strategy_func.target_handler], 'mtf': mtf}]

    elif 'breakouts' in strategy:

        # bb_rsi_tf_list = []
        mtf = None
        # config = [{'timeframe': '60min', 'targets': ['120 min', '240 min', 'eod', 'eod_rth']},
        #                         {'timeframe': '1D', 'targets': ['1 D', '5 D']}]
        config = [{'timeframe': '5min', 'targets': ['15min', '60min', 'eod', 'eod_rth']},
                                {'timeframe': '60min', 'targets': ['120min', '240min', 'eod', 'eod_rth']},
                                {'timeframe': '1D', 'targets': ['1D', '5D']}]
        # config_breakouts = [{'timeframe': '1min', 'targets': ['5 min', '60 min', 'eod', 'eod_rth']},
        #                         {'timeframe': '5min', 'targets': ['15 min', '60 min', 'eod', 'eod_rth']},
        #                         {'timeframe': '60min', 'targets': ['120 min', '240 min', 'eod', 'eod_rth']},
        #                         {'timeframe': '1D', 'targets': ['1 D', '5 D']}]

        if strategy == 'breakouts_bull':
            strategy_func_bull = breakouts_strategy.BreakoutsStrategy()
        elif strategy == 'breakouts_bear':
            strategy_func_bear = breakouts_strategy.BreakoutsStrategy()


    elif strategy == 'rs3_bull':

        # bb_rsi_tf_list = []
        mtf = None
        # config = [{'timeframe': '60min', 'targets': ['120 min', '240 min', 'eod', 'eod_rth']},
        #                   {'timeframe': '1D', 'targets': ['1 D', '5 D']}]
        config = [{'timeframe': '5min', 'targets': ['15min', '60min', 'eod', 'eod_rth']},
                          {'timeframe': '60min', 'targets': ['120min', '240min', 'eod', 'eod_rth']},
                          {'timeframe': '1D', 'targets': ['1D', '5D']}]
        # confid = [{'timeframe': '1min', 'targets': ['5 min', '60 min', 'eod', 'eod_rth']},
        #                         {'timeframe': '5min', 'targets': ['15 min', '60 min', 'eod', 'eod_rth']},
        #                         {'timeframe': '60min', 'targets': ['120 min', '240 min', 'eod', 'eod_rth']},
        #                         {'timeframe': '1D', 'targets': ['1 D', '5 D']}]

        strategy_func = rs3_strategy.RS3StrategyBull(daily_pivots=True)


    df_results_list = StrategyAnalyzer(ib, symbols, strategy_func, config, mtf, from_time=from_time, to_time=to_time, entry_delay=entry_delay, 
                                       file_format=file_format).assess()


    # Display results analysis
    if strategy:

        if not df_results_list:
            print("No result returned.")
        else:
            if display_res:
                strategy_summary.StrategyResultsVisualizer(from_time, to_time).display_results(df_results_list)

            df_results_summary = strategy_summary.StrategyResultsSummary.summarize(df_results_list)
            df_results_summary_by_mktcap = strategy_summary.StrategyResultsSummary.summarize(df_results_list, group_by_feature='market_cap_cat')
            print(df_results_summary)
            print(df_results_summary_by_mktcap)

            # Concatenate results accross timeframes and targets
            df_all_results = pd.concat(df_results_list, ignore_index=True)

            # Group by timeframe and strategy
            grouped = df_all_results.groupby(['timeframe', 'strategy', 'target'])
            df_results_list_grouped = []
            for (timeframe, strategy_name, target), df_group in grouped:
                # Reorder data per trigger time
                df_group_sorted = df_group.sort_values("trig_time").reset_index(drop=True)

                df_results_list_grouped.append(df_group_sorted)

                # Save to file
                # filename = f"results_{strategy_name}_{timeframe.replace('/', '_')}_{target.replace('/', '_')}.csv"
                filename = f"results_{strategy_name}_{target.replace('/', '_')}.csv"
                filepath = os.path.join(strategy_folder, filename)
                os.makedirs(strategies_folder, exist_ok=True)
                helpers.save_df_to_file(df_group_sorted, str(filepath), file_format='csv')


            # Save all_results and summary to file
            # helpers.save_df_to_file(df_all_results, strategy_data_file_csv, format='csv')
            # helpers.save_df_to_file(df_all_results, strategy_data_file_parquet, format='parquet')
            helpers.save_df_to_file(df_results_summary, strategy_results_summary_file_csv, file_format='csv')
            helpers.save_df_to_file(df_results_summary_by_mktcap, strategy_results_summary_by_mktcap_file_csv, file_format='csv')

            # Save full dataframe with full features to file
            df_indicators_file_csv = os.path.join(strategy_folder, 'df_indicators.csv')
            # print()
            # helpers.save_df_to_file(df, df_indicators_file_csv, format='csv')
