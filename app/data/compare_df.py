import sys, os, pandas as pd, pandas as pd
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers



if __name__ == "__main__":

    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    # Args Setup
    args = sys.argv
    display_diffs = not 'nodisplay' in args

    # # TWS Connection
    # ib, ibConnection = helpers.IBKRConnect_any(IB(), paper=paperTrading)

    path_complete_1M = 'C:\\Users\\ChristopheReis\\Documents\\T\\t_data\\hist_data\\ACIW\\complete-1M-enriched_data_ACIW_1 min_2024-07-31-08-52-00_2025-08-25-19-59-00.csv'
    path_complete_2M = 'C:\\Users\\ChristopheReis\\Documents\\T\\t_data\\hist_data\\ACIW\\complete-2M-enriched_data_ACIW_1 min_2024-07-31-08-52-00_2025-08-25-19-59-00.csv'
    path_full = 'C:\\Users\\ChristopheReis\\Documents\\T\\t_data\\hist_data\\ACIW\\enriched_data_ACIW_1 min_2024-07-31-08-52-00_2025-08-25-19-59-00.csv'
    # path_complete_1M_b = 'C:\\Users\\ChristopheReis\\Documents\\T\\t_data\\hist_data\\ACIW\\complete-1M_1-enriched_data_ACIW_1 min_2024-07-31-08-52-00_2025-08-25-19-59-00.csv'
    # path_complete_2M_b = 'C:\\Users\\ChristopheReis\\Documents\\T\\t_data\\hist_data\\ACIW\\complete-2M_1-enriched_data_ACIW_1 min_2024-07-31-08-52-00_2025-08-25-19-59-00.csv'
    path_complete_1M_b = 'C:\\Users\\ChristopheReis\\Documents\\T\\t_data\\hist_data\\ACIW_backup_20250827\\complete-1M-enriched_data_ACIW_1 min_2024-07-31-08-52-00_2025-08-25-19-59-00.csv'
    path_complete_2M_b = 'C:\\Users\\ChristopheReis\\Documents\\T\\t_data\\hist_data\\ACIW_backup_20250827\\complete-2M-enriched_data_ACIW_1 min_2024-07-31-08-52-00_2025-08-25-19-59-00.csv'
    path_full_b = 'C:\\Users\\ChristopheReis\\Documents\\T\\t_data\\hist_data\\ACIW\\full--enriched_data_ACIW_1 min_2024-07-31-08-52-00_2025-08-25-19-59-00.csv'
    # path_wrong = 'C:\\Users\\ChristopheReis\\Documents\\T\\t_data\\hist_data\\AAPL\\-wrong_code--enriched_data_AAPL_1 min_2019-11-20-04-00-00_2025-08-21-12-13-00.parquet'

    print("Loading Dataframes...")
    df_complete_1M = helpers.format_df_date(helpers.load_df_from_file(path_complete_1M))
    # df_complete_2M = helpers.format_df_date(helpers.load_df_from_file(path_complete_2M))
    # df_full = helpers.format_df_date(helpers.load_df_from_file(path_full))
    # df_complete_1M_b = helpers.format_df_date(helpers.load_df_from_file(path_complete_1M_b))
    # df_complete_2M_b = helpers.format_df_date(helpers.load_df_from_file(path_complete_2M_b))
    df_full_b = helpers.format_df_date(helpers.load_df_from_file(path_full_b))
    # df_wrong = helpers.format_df_date(helpers.load_df_from_file(path_wrong))

    print("Comparing Dataframes...")
    exclusion_list = []#['sr_1h', 'sr_1D', 'sr_1W', 'pivots', 'pivots_D', 'pivots_M', 'levels', 'levels_M', 'breakout_up_score', 'breakout_down_score']
    # comp_correct_full = helpers.compare_dataframes(df_complete_1M, df_full, exclusion_list=exclusion_list, display_diffs=display_diffs)
    # # print("Comparing df_full, df_wrong")
    # # comp_full_wrong = helpers.compare_dataframes(df_full, df_wrong)
    # # print("Comparing df_correct, df_wrong")
    # # comp_correct_wrong = helpers.compare_dataframes(df_correct, df_wrong)

    # comp_correct_full = helpers.compare_dataframes(df_complete_1M, df_complete_1M_b, exclusion_list=exclusion_list, display_diffs=display_diffs)
    # comp_correct_full = helpers.compare_dataframes(df_complete_2M, df_complete_2M_b, exclusion_list=exclusion_list, display_diffs=display_diffs)
    # comp_correct_full = helpers.compare_dataframes(df_full, df_full_b, exclusion_list=exclusion_list, display_diffs=display_diffs)
    comp_correct_full = helpers.compare_dataframes(df_full_b, df_complete_1M, exclusion_list=exclusion_list, display_diffs=display_diffs)

