import sys, os, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers


class StrategyResultsSummary:

    @staticmethod
    def summarize(df_results_list, group_by_feature=None):
        """
        Summarize key metrics for each strategy run. Optionally, break down metrics by a categorical feature.

        Args:
            df_results_list (list of pd.DataFrame): Each dataframe should be a result set for one strategy config.
            group_by_feature (str): Optional column name (e.g., 'market_cap_cat') to group and compare performance on.

        Returns:
            pd.DataFrame: Summary dataframe, optionally grouped by the given feature.
        """
        summaries = []

        for df in df_results_list:
            if df.empty:
                continue

            base_group_cols = ['strategy', 'timeframe', 'target', 'entry_delay']
            if group_by_feature and group_by_feature in df.columns:
                group_cols = base_group_cols + [group_by_feature]
                grouped = df.groupby(group_cols)
            else:
                grouped = [(tuple(df[col].iloc[0] for col in base_group_cols), df)]

            for group_keys, group_df in grouped:
                if group_df.empty:
                    continue

                # Reconstruct dictionary of grouping keys
                if isinstance(group_keys, tuple):
                    group_dict = dict(zip(group_cols, group_keys)) if group_by_feature else dict(
                        zip(base_group_cols, group_keys)
                    )
                else:
                    group_dict = {group_by_feature: group_keys}

                # group_dict = dict(zip(group_cols, group_keys)) if not isinstance(grouped, list) else {
                #     'strategy': group_keys[0],
                #     'timeframe': group_keys[1],
                #     'target': group_keys[2],
                # }

                win_loss_ratio = (
                    group_df[group_df['is_profit']]['end_profit_per_min'].mean() /
                    abs(group_df[~group_df['is_profit']]['end_profit_per_min'].mean())
                    if not group_df[~group_df['is_profit']].empty else float('inf')
                )

                label_binary_dist = group_df['label_binary'].value_counts(normalize=True).round(4).to_dict()
                label_multi_dist = group_df['label_multi'].value_counts(normalize=True).round(4).to_dict()

                sharpe_ratio_bar = group_df['sharpe_ratio_bar'].mean()
                sharpe_ratio_yearly = group_df['sharpe_ratio_yearly'].mean()
                sortino_ratio_bar = group_df['sortino_ratio_bar'].mean()
                sortino_ratio_yearly = group_df['sortino_ratio_yearly'].mean()
                summary = {
                    'trigger_count': len(group_df),
                    'mean_end_profit_per_min': group_df['end_profit_per_min'].mean().round(4),
                    'median_end_profit_per_min': group_df['end_profit_per_min'].median().round(4),
                    'profit_%': round(group_df['is_profit'].mean() * 100, 2),
                    'mean_sharpe_ratio_bar': sharpe_ratio_bar.round(2) if not pd.isna(sharpe_ratio_bar) else None,
                    'mean_sharpe_ratio_yearly': sharpe_ratio_yearly.round(2) if not pd.isna(sharpe_ratio_yearly) else None,
                    'mean_sortino_ratio_bar': sortino_ratio_bar.round(2) if not pd.isna(sortino_ratio_bar) else None,
                    'mean_sortino_ratio_yearly': sortino_ratio_yearly.round(2) if not pd.isna(sortino_ratio_yearly) else None,
                    'win_loss_ratio': win_loss_ratio.round(2) if win_loss_ratio != float('inf') else win_loss_ratio,
                    'mean_even_duration': group_df['event_duration'].mean().round(2),
                    'max_drawdown_per_min': group_df['max_drawdown_per_min'].min().round(4),
                    'mean_drawdown_duration_ratio': group_df['drawdown_duration_ratio'].mean().round(2),
                    'label_binary_dist': label_binary_dist,
                    'label_multi_dist': label_multi_dist
                }

                summary.update(group_dict)
                summaries.append(summary)

        summary_df = pd.DataFrame(summaries)

        # Reorder columns: move group keys to front
        group_keys = base_group_cols
        if group_by_feature:
            group_keys.append(group_by_feature)

        # Only keep keys that actually exist (safe in edge cases)
        group_keys = [col for col in group_keys if col in summary_df.columns]
        other_cols = [col for col in summary_df.columns if col not in group_keys]

        # Reorder
        summary_df = summary_df[group_keys + other_cols]
        return summary_df

    # @staticmethod
    # def summarize(df_results_list):
    #     """
    #     Generate a summary of key metrics for each DataFrame in the results list.
    #     Each df in df_results_list should already contain the necessary fields from the post-trigger analysis.
    #     """
    #     summaries = []
    #     for df in df_results_list:
    #         if df.empty:
    #             continue

    #         # Compute summary metrics
    #         win_loss_ratio = df[df['is_profit']]['end_profit_per_min'].mean() / abs(df[~df['is_profit']]['end_profit_per_min'].mean()) if not df[~df['is_profit']].empty else float('inf')
    #         label_binary_dist = df['label_binary'].value_counts(normalize=True).round(4).to_dict()
    #         label_multi_dist = df['label_multi'].value_counts(normalize=True).round(4).to_dict()

    #         summaries.append({
    #             'strategy': df['strategy'].iloc[0],
    #             'timeframe': df['timeframe'].iloc[0],
    #             'target': df['target'].iloc[0],
    #             'trigger_count': len(df),
    #             'mean_end_profit_per_min': df['end_profit_per_min'].mean().round(4),
    #             'median_end_profit_per_min': df['end_profit_per_min'].median().round(4),
    #             'profit_%': round(df['is_profit'].mean() * 100, 2),
    #             'mean_sharpe_ratio_bar': df['sharpe_ratio_bar'].mean().round(2),
    #             'mean_sharpe_ratio_yearly': df['sharpe_ratio_yearly'].mean().round(2),
    #             'mean_sortino_ratio_bar': df['sortino_ratio_bar'].mean().round(2),
    #             'mean_sortino_ratio_yearly': df['sortino_ratio_yearly'].mean().round(2),
    #             'win_loss_ratio': win_loss_ratio.round(2) if win_loss_ratio != float('inf') else win_loss_ratio,
    #             'mean_even_duration': df['event_duration'].mean().round(2),
    #             'max_drawdown_per_min': df['max_drawdown_per_min'].min().round(4),
    #             'mean_drawdown_duration_ratio': df['drawdown_duration_ratio'].mean().round(2),
    #             'label_binary_dist': label_binary_dist,
    #             'label_multi_dist': label_multi_dist
    #         })

    #     return pd.DataFrame(summaries)



class StrategyResultsVisualizer:

    def __init__(self, end_time, start_time):
        self.start_time = start_time
        self.end_time = end_time


    @staticmethod
    def plot_distribution(df, column, title_suffix=''):
        plt.figure(figsize=(10, 5))
        sns.histplot(df[column], bins=50, kde=True)
        plt.title(f'Distribution of {column} {title_suffix}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


    def display_results(self, df_results_list, display_table=False, column_list=['end_profit', 'max_drawdown', 'sharpe_ratio_bar', 'close_rtn_pct'],
                    label_list=['label_regression', 'label_binary', 'label_multi', 'label_R-R']):

        for df_results in df_results_list:

            symbol = df_results['symbol'].iloc[0]
            timeframe = df_results['timeframe'].iloc[0]

            # print(df_results)
            for column in column_list:
                self.plot_distribution(df_results, column, timeframe + ' timeframe, for ' + df_results['strategy'] + ' - ' + df_results['target'] + ' target')

            for label in label_list:
                print("Normalized counts and average returns for ", label)
                print(df_results[label].value_counts(normalize=True))

            if not df_results.empty:
                table_title = df_results['strategy'].iloc[0] + " " + timeframe + ", target " + \
                    df_results['target'].iloc[0] + " " + symbol + " from " + \
                    str(self.end_time.strftime('%Y-%m-%d')) + " to " + str(self.start_time.strftime('%Y-%m-%d'))

                table = helpers.df_to_table(df_results.round(2))
                if table.rowcount != 0:
                    table.title = table_title + " " + symbol + " from " + str(self.end_time.strftime('%Y-%m-%d')) + " to " + str(self.start_time.strftime('%Y-%m-%d'))

                if display_table:
                    print('\n', table, '\n')
