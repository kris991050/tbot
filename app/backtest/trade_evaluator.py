import sys, os, numpy as np, pandas as pd

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.timeframe import Timeframe


class TradeEvaluator:

    @staticmethod
    def _normalize_trade_logs(trade_logs):
        """
        Converts a list of dicts or DataFrames into a single DataFrame.
        """
        if not trade_logs:
            return pd.DataFrame()

        df_list = []
        for item in trade_logs:
            if isinstance(item, dict):
                df_list.append(pd.DataFrame([item]))
            elif isinstance(item, pd.DataFrame):
                df_list.append(item)
            else:
                raise ValueError("Trade log must be a dict or DataFrame.")

        return pd.concat(df_list, ignore_index=True)

    @staticmethod
    def summarize_results(df, symbol, rounding=2):
        if df.empty:
            return {}

        total_pnl = df['pnl'].sum()
        total_commission = df['total_commission'].sum()
        total_net_pnl = df['net_pnl'].sum()
        win_rate = (df['pnl'] > 0).mean()
        avg_return = df['return_pct'].mean()
        std_return = df['return_pct'].std()
        sharpe = avg_return / std_return * np.sqrt(252) if std_return else 0

        downside_returns = df[df['return_pct'] < 0]['return_pct']
        sortino = avg_return / (downside_returns.std() + 1e-9) * np.sqrt(252) if not downside_returns.empty else 0

        max_drawdown_avg = df['max_drawdown'].mean()
        reward_to_risk_avg = df['reward_to_risk'].replace([np.inf, -np.inf], np.nan).dropna().mean()

        expectancy = df['pnl'].mean()
        payoff_ratio = df[df['pnl'] > 0]['pnl'].mean() / abs(df[df['pnl'] < 0]['pnl'].mean()) if not df[df['pnl'] < 0].empty else np.nan


        summary =  {
            'Symbol': symbol,
            'Total Trades': len(df),
            'Total PnL': round(total_pnl, rounding),
            'Total Commission': round(total_commission, rounding),
            'Total Net PnL': round(total_net_pnl, rounding),
            'Win Rate (%)': round(win_rate * 100, rounding),
            'Avg Return (%)': round(avg_return, rounding),
            'Sharpe Ratio': round(sharpe, rounding),
            'Sortino Ratio': round(sortino, rounding),
            'Expectancy': round(expectancy, rounding),
            'Payoff Ratio': round(payoff_ratio, rounding) if payoff_ratio else None,
            'Avg Max Drawdown': round(max_drawdown_avg, rounding),
            'Avg Reward-to-Risk': round(reward_to_risk_avg, rounding),
            'Avg Trade Duration (min)': round(df['duration_min'].mean(), rounding),
            'Median PnL': round(df['pnl'].median(), rounding),
            '90th Percentile PnL': round(np.percentile(df['pnl'], 90), rounding),
            '10th Percentile PnL': round(np.percentile(df['pnl'], 10), rounding),
            'Exit Reason Breakdown': df['exit_reason'].value_counts().to_dict()
        }

        return pd.DataFrame([summary])

    @staticmethod
    def summarize_all_trades(trade_logs) -> pd.DataFrame:
        df = TradeEvaluator._normalize_trade_logs(trade_logs)

        if df.empty:
            return pd.DataFrame()

        return TradeEvaluator.summarize_results(df, symbol='All')

    @staticmethod
    def summarize_by_group(trade_logs, group_col: str) -> pd.DataFrame:
        df = TradeEvaluator._normalize_trade_logs(trade_logs)

        if df.empty:
            return pd.DataFrame()

        if group_col not in df.columns:
            raise KeyError(f"'{group_col}' column not found in trade logs.")

        all_group_summaries = []

        for group_value, group_df in df.groupby(group_col):
            summary_df = TradeEvaluator.summarize_results(group_df, symbol=group_value)
            summary_df.insert(0, 'Group Value', group_value)
            summary_df.insert(0, 'Group', group_col)
            summary_df.drop(columns='Symbol', inplace=True)
            all_group_summaries.append(summary_df)

        return pd.concat(all_group_summaries, ignore_index=True)

    @staticmethod
    def log_trade(direction, trade_log, entry_idx, exit_idx, df, entry_price, exit_price, quantity, entry_prediction, stop_price, reason2close, 
                  symbol, entry_exec_price, exit_exec_price, entry_commission, exit_commission, rounding=2):
        trade_df = df.iloc[entry_idx:exit_idx + 1]
        close_prices = trade_df['close']
        highs = trade_df['high']
        lows = trade_df['low']

        market_cap = df.attrs.get('market_cap', None)
        market_cap_cat = helpers.categorize_market_cap(market_cap) if market_cap is not None else 'Unknown'

        max_price = highs.max() if direction == 1 else lows.min()
        min_price = lows.min() if direction == 1 else highs.max()
        max_profit = direction * (max_price - entry_price) * quantity
        max_drawdown = -direction * (min_price - entry_price) * quantity
        predicted_drawdown = df.iloc[entry_idx].get('predicted_drawdown', np.nan)
        expected_reward = abs(df.iloc[entry_idx].get('vwap', np.nan) - entry_price)
        rrr = expected_reward / predicted_drawdown if predicted_drawdown else np.nan

        pnl = direction * (exit_exec_price - entry_exec_price) * quantity
        return_pct = pnl / (entry_exec_price * quantity) * 100 if entry_exec_price else np.nan
        total_commission = entry_commission + exit_commission
        net_pnl = pnl - total_commission
        slippage_entry = entry_exec_price - entry_price if entry_exec_price else None
        slippage_exit = exit_price - exit_exec_price if exit_exec_price else None

        duration_bars = exit_idx - entry_idx
        timeframe_min = Timeframe(df.attrs['timeframe']).to_seconds / 60#helpers.timeframe_to_seconds(df.attrs['timeframe']) / 60
        duration_min = duration_bars * timeframe_min

        trade_log.append({
            'symbol': symbol,
            'market_cap_cat': market_cap_cat,
            'entry_time': pd.to_datetime(df.iloc[entry_idx]['date']),
            'exit_time': pd.to_datetime(df.iloc[exit_idx]['date']),
            'entry_exec_price': round(entry_exec_price, rounding) if entry_exec_price is not None else None,
            'exit_exec_price': round(exit_exec_price, rounding) if exit_exec_price is not None else None,
            'entry_price': round(entry_price, rounding) if entry_price is not None else None,
            'exit_price': round(exit_price, rounding) if exit_price is not None else None,
            'quantity': quantity,
            'prediction': round(entry_prediction, rounding) if entry_prediction is not None else None,
            'predicted_drawdown': round(predicted_drawdown, rounding) if predicted_drawdown is not None else None,
            'estimated_reward': round(expected_reward, rounding) if expected_reward is not None else None,
            'estimated_rrr': round(rrr, rounding) if rrr is not None else None,
            'stop_price': round(stop_price, rounding) if stop_price is not None else None,
            'pnl': round(pnl, rounding),
            'net_pnl': round(net_pnl, rounding),
            'return_pct': round(return_pct, rounding),
            'duration_bars': duration_bars,
            'duration_min': round(duration_min, rounding),
            'exit_reason': reason2close,
            'max_profit': round(max_profit, rounding),
            'max_drawdown': round(max_drawdown, rounding),
            'reward_to_risk': round(pnl / abs(max_drawdown), rounding) if max_drawdown != 0 else np.nan,
            'is_win': pnl > 0,
            'entry_commission': round(entry_commission, rounding),
            'exit_commission': round(exit_commission, rounding),
            'total_commission': round(total_commission, rounding),
            'slippage_entry': round(slippage_entry, rounding) if slippage_entry is not None else None,
            'slippage_exit': round(slippage_exit, rounding) if slippage_exit is not None else None,
            'slippage_entry_pct': round((entry_exec_price - entry_price) / entry_price * 100, rounding)
                if entry_exec_price and entry_price else None,
            'slippage_exit_pct': round((exit_price - exit_exec_price) / exit_price * 100, rounding)
                if exit_exec_price and exit_price else None
        })
