import os, sys, backtrader as bt, pandas as pd, datetime
from ib_insync import *

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

import backtrader_wrapper
from execution import trade_manager


class BacktraderBacktestEngine:
    def __init__(self, df, symbol, manager):
        self.df = df
        self.symbol = symbol
        self.manager = manager
        self.trades = []
        self.all_required_columns = self._build_all_required_columns()
        self.dropped_required_columns = []

    def _build_all_required_columns(self):
        required_columns = self.manager.get_strategy_required_columns()
        bt_required_columns = ['open', 'high', 'low', 'close', 'volume', 'model_prediction']
        if self.manager.trainer_dd is not None: bt_required_columns.append('predicted_drawdown')

        all_required_columns = list(set(required_columns + bt_required_columns))

        # Identify missing columns
        missing_columns = [col for col in all_required_columns if col not in self.df.columns]

        # Warn about missing columns
        if missing_columns:
            print(f"⚠️ Warning: The following required columns are missing from the DataFrame: {missing_columns}")

        # Return only existing required columns
        existing_required_columns = [col for col in all_required_columns if col in self.df.columns]

        return existing_required_columns

    def _create_data_feed(self):
        df = self.df.copy()

        # Reduce columns to required
        df = df[self.all_required_columns]
        
        # Convert datetime.date columns to str, for Backtrader use
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df.set_index('date', inplace=True)
        # df = df.applymap(lambda x: x.isoformat() if isinstance(x, datetime.date) and not isinstance(x, datetime.datetime) else x)

        # Ensure Backtrader-friendly datetime index
        df.index = df.index.to_pydatetime()
        df = df.fillna(0)

        # Drop only non-required object columns and keep object required_columns that have been dropped
        object_cols = df.select_dtypes(include='object').columns
        self.dropped_required_columns = [col for col in object_cols if col in self.all_required_columns]
        df = df.drop(columns=df.select_dtypes(include='object').columns)

        cols = list(df.columns)
        class CustomData(bt.feeds.PandasData):
            lines = tuple(cols)
            params = {col: col for col in cols}

        # self.df = df.copy()

        return CustomData(dataname=df)

    def run(self):

        data_feed = self._create_data_feed()
        ib_comm = trade_manager.IBKRCanadaCommission()

        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed, name=self.symbol)#, preload=False)

        cerebro.addstrategy(
            backtrader_wrapper.BacktraderStrategyWrapper,
            manager=self.manager,
            total_bars=len(self.df),
            df=self.df,
            dropped_required_columns = self.dropped_required_columns
        )
        cerebro.broker.setcash(10000)
        cerebro.broker.addcommissioninfo(ib_comm)
        cerebro.broker.set_slippage_perc(perc=0.001) # 0.1%
        t_now = datetime.datetime.now()
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown') # Add the DrawDown analyzer to track the drawdown
        results = cerebro.run()#(runonce=False)
        print(f"\nElapsed time for running Cerebro: {datetime.datetime.now() - t_now}\n")
        strat = results[0]
        self.trades = getattr(strat, 'trades', [])
        # Plot the results (equity curve, drawdown, and price chart)
        # cerebro.plot(style='candlestick', iplot=True)
        # cerebro.plot(style='line', iplot=False)
        return self.trades





# class BacktraderBacktestEngine:
#     def __init__(self, df, symbol, strategy_logic, stop_handler, target_handler,
#                  size='auto', entry_delay=1, prediction_threshold=0.7, rrr_threshold=1.5):
#         self.df = df.copy()
#         self.symbol = symbol
#         self.strategy_logic = strategy_logic
#         self.stop_handler = stop_handler
#         self.target_handler = target_handler
#         self.pred_th = prediction_threshold
#         self.rrr_th = rrr_threshold
#         self.size = size
#         self.entry_delay = entry_delay
#         self.trades = []
#         self.required_columns = self._build_required_columns()
#         self.dropped_required_columns = []

#     def _build_required_columns(self):
#         strategy_required_columns = trade_manager.TradingManager.get_required_columns(self.strategy_logic, self.target_handler, self.stop_handler)
#         bt_required_columns = ['open', 'high', 'low', 'close', 'volume', 'model_prediction', 'predicted_drawdown']

#         all_required_columns = list(set(strategy_required_columns + bt_required_columns))

#         # Identify missing columns
#         missing_columns = [col for col in all_required_columns if col not in self.df.columns]

#         # Warn about missing columns
#         if missing_columns:
#             print(f"⚠️ Warning: The following required columns are missing from the DataFrame: {missing_columns}")

#         # Return only existing required columns
#         existing_required_columns = [col for col in all_required_columns if col in self.df.columns]

#         return existing_required_columns

#     def _create_data_feed(self):
#         df = self.df.copy()

#         # Convert datetime.date columns to str, for Backtrader use
#         df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
#         df.set_index('date', inplace=True)
#         # df = df.applymap(lambda x: x.isoformat() if isinstance(x, datetime.date) and not isinstance(x, datetime.datetime) else x)

#         # Ensure Backtrader-friendly datetime index
#         df.index = df.index.to_pydatetime()

#         # Reduce columns to requuired
#         df = df[self.required_columns]
#         df = df.fillna(0)

#         # Drop only non-required object columns and keep object required_columns that have been dropped
#         object_cols = df.select_dtypes(include='object').columns
#         self.dropped_required_columns = [col for col in object_cols if col in self.required_columns]
#         df = df.drop(columns=df.select_dtypes(include='object').columns)

#         cols = list(df.columns)
#         class CustomData(bt.feeds.PandasData):
#             lines = tuple(cols)
#             params = {col: col for col in cols}

#         # self.df = df.copy()

#         return CustomData(dataname=df)

#     def run(self):

#         # wrapper = backtrader_wrapper.BacktraderStrategyWrapper,
#         #     wrapped_strategy=self.strategy_logic,
#         #     stop_handler=self.stop_handler,
#         #     target_handler=self.target_handler,
#         #     prediction_threshold=self.pred_th,
#         #     rrr_threshold=self.rrr_th

#         data_feed = self._create_data_feed()
#         ib_comm = trade_manager.IBKRCanadaCommission.IBKRCanadaCommission()

#         cerebro = bt.Cerebro()
#         cerebro.adddata(data_feed, name=self.symbol)#, preload=False)

#         cerebro.addstrategy(
#             backtrader_wrapper.BacktraderStrategyWrapper,
#             wrapped_strategy=self.strategy_logic,
#             stop_handler=self.stop_handler,
#             target_handler=self.target_handler,
#             prediction_threshold=self.pred_th,
#             rrr_threshold=self.rrr_th,
#             size=self.size,
#             entry_delay=self.entry_delay,
#             total_bars=len(self.df),
#             df=self.df,
#             dropped_required_columns = self.dropped_required_columns
#         )
#         cerebro.broker.setcash(100000)
#         cerebro.broker.addcommissioninfo(ib_comm)
#         cerebro.broker.set_slippage_perc(perc=0.001) # 0.1%
#         t_now = datetime.datetime.now()
#         results = cerebro.run()#(runonce=False)
#         print(f"\nElapsed time for running Cerebro: {datetime.datetime.now() - t_now}\n")
#         strat = results[0]
#         self.trades = getattr(strat, 'trades', [])
#         return self.trades




# class MLStrategy(bt.Strategy):
#     params = dict(
#         prediction_threshold=0.7,
#         rrr_threshold=1.5,
#         stop_col='predicted_drawdown',  # Column in the dataframe
#         target_col='vwap',               # Column for target (could be a future prediction)
#         direction=1                      # 1 = long, -1 = short
#     )

#     def __init__(self):
#         self.order = None
#         self.entry_price = None
#         self.stop_price = None
#         self.target_price = None
#         self.quantity = None

#     def next(self):
#         if self.order:
#             return  # Pending order

#         curr = self.data
#         prediction = curr.model_prediction[0]
#         price = curr.close[0]
#         vwap = curr.vwap[0]
#         dd_pred = curr.predicted_drawdown[0]

#         # If in trade, check for exit
#         if self.position:
#             if self.p.exit_price:  # Optional target price
#                 if (self.p.direction == 1 and curr.close[0] >= self.target_price) or \
#                    (self.p.direction == -1 and curr.close[0] <= self.target_price):
#                     self.close()
#                     return

#             if self.stop_price:
#                 if (self.p.direction == 1 and curr.close[0] <= self.stop_price) or \
#                    (self.p.direction == -1 and curr.close[0] >= self.stop_price):
#                     self.close()
#                     return

#         # If not in trade, evaluate signal
#         if not self.position:
#             is_trigger = curr.trigger[0]
#             if is_trigger and prediction >= self.p.prediction_threshold:
#                 # Compute RRR
#                 if dd_pred and vwap:
#                     expected_reward = abs(vwap - price)
#                     rrr = expected_reward / dd_pred if dd_pred > 0 else float('inf')
#                     if rrr < self.p.rrr_threshold:
#                         return  # Skip trade

#                 self.entry_price = price
#                 self.stop_price = price - dd_pred if self.p.direction == 1 else price + dd_pred
#                 self.target_price = vwap
#                 self.quantity = self.broker.getcash() / price  # Can use custom sizing logic

#                 self.order = self.buy(size=self.quantity) if self.p.direction == 1 else self.sell(size=self.quantity)

#     def notify_order(self, order):
#         if order.status in [order.Completed, order.Canceled, order.Rejected]:
#             self.order = None


class MLData(bt.feeds.PandasData):
    lines = ('model_prediction', 'trigger', 'vwap', 'predicted_drawdown', 'score_bias')
    params = (
        ('model_prediction', -1),
        ('trigger', -1),
        ('vwap', -1),
        ('predicted_drawdown', -1),
        ('score_bias', -1),
    )
