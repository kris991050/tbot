import sys, os, backtrader as bt, traceback, pandas as pd
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import constants
import trade_evaluator


# class BBRSIReversalData(bt.feeds.PandasData):
#     lines = (
#         'rsi_5', 'rsi_15', 'rsi_60', 'rsi',
#         'bband_l_5', 'bband_l_15', 'bband_l_60',
#         'cam_M_position',
#         'rsi_slope_1D', 'levels_dist_to_next_up_pct',
#         'bband_h_1D_dist_pct_atr', 'bband_l_1D_dist_pct_atr',
#         'market_cap_cat'
#     )
#     params = {line: -1 for line in lines}


# class VWAPPredictionData(bt.feeds.PandasData):
#     lines = ('model_prediction', 'predicted_drawdown', 'vwap')
    # params = {line: -1 for line in lines}


class BacktraderStrategyWrapper(bt.Strategy):
    params = dict(
        manager=None,
        entry_delay=1,
        total_bars=None,
        df = pd.DataFrame(),
        dropped_required_columns=[]
    )

    def __init__(self):
        self.manager = self.p.manager
        self.active_stop_price = None  # stores fixed stop-loss value for current trade
        self.reset_active_stop_price = False
        self.total_bars = self.p.total_bars
        self.df = self.p.df
        self.dropped_required_columns = self.p.dropped_required_columns
        self.status_counter = 0
        self.prev_row = None
        self.in_position = None
        self.order = None
        self.entry_price = None
        self.entry_idx = None
        self.exit_idx = None
        self.entry_time = None
        self.decision_prediction = None
        self.entry_prediction = None
        self.quantity = None

        # Determine which columns the strategy requires
        self.required_columns = self.manager.get_required_columns()

        # Initialize trade log
        self.trades = []
        self.entry_exec_price = None
        self.exit_exec_price = None
        self.entry_commission = 0.0
        self.exit_commission = 0.0
        self.exit_reason = None

    def build_row(self, ago=0):
         # Load columns from Backtrader data (numeric ones)
        row = {}
        for col in self.required_columns:
            val = getattr(self.datas[0], col, None)
            row[col] = val[-ago] if val else 0  # ago = 0 is current bar, ago = 1 is previous, etc.

        # Special fields
        row['model_prediction'] = getattr(self.datas[0], 'model_prediction', [None])[-ago]
        row['predicted_drawdown'] = getattr(self.datas[0], 'predicted_drawdown', [None])[-ago] if self.manager.trainer_dd is not None else None
        row['date'] = pd.Timestamp(self.datas[0].datetime.datetime(-ago), tz=constants.CONSTANTS.TZ_WORK)
        row['close'] = self.datas[0].close[-ago] if self.datas[0].close else None
    
        # Inject dropped object columns back from self.df (source DataFrame)
        if self.dropped_required_columns:
            curr_idx = len(self) - ago - 1  # aligns with actual index
            for col in self.dropped_required_columns:
                try:
                    row[col] = self.df.iloc[curr_idx][col]
                except (IndexError, KeyError):
                    row[col] = None  # Or some sensible default if required
        
        return row

    def display_status_progress(self):
        # Status progress
        if self.total_bars:
            self.status_counter += 1
            progress = (self.status_counter / self.total_bars) * 100
            if self.status_counter % max(1, self.total_bars // 100) == 0 or self.status_counter == self.total_bars:
                print(f"\rBacktest progress for {self.datas[0]._name}: {progress:.1f}%", end='', flush=True)
    
    def _log_trade_if_complete(self):
        # Defensive: only log if all data is set
        if None in [self.entry_exec_price, self.exit_exec_price]:
            print("⚠️ Trade not fully executed, skipping log.")
            return
        
        symbol = self.datas[0]._name
        trade_evaluator.TradeEvaluator.log_trade(self.manager.direction, self.trades, self.entry_idx, self.exit_idx, self.df, 
                                                 self.entry_price, self.exit_price, self.quantity, self.decision_prediction, 
                                                 self.entry_prediction, self.active_stop_price, self.exit_reason, symbol, 
                                                 self.entry_exec_price, self.exit_exec_price, self.entry_commission, self.exit_commission)
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            is_entry = not self.in_position
            is_exit = self.in_position

            if order.isbuy():
                if self.manager.direction == 1 and is_entry:
                    self.entry_exec_price = order.executed.price
                    self.entry_commission = order.executed.comm
                    self.in_position = True
                    print(f"\n[LONG ENTRY] BUY executed @ {self.entry_exec_price:.2f}")
                elif self.manager.direction == -1 and is_exit:
                    self.exit_exec_price = order.executed.price
                    self.exit_commission = order.executed.comm
                    self.in_position = False
                    print(f"\n[SHORT EXIT] BUY executed @ {self.exit_exec_price:.2f}")
                    self._log_trade_if_complete()

            elif order.issell():
                if self.manager.direction == -1 and is_entry:
                    self.entry_exec_price = order.executed.price
                    self.entry_commission = order.executed.comm
                    self.in_position = True
                    print(f"\n[SHORT ENTRY] SELL executed @ {self.entry_exec_price:.2f}")
                elif self.manager.direction == 1 and is_exit:
                    self.exit_exec_price = order.executed.price
                    self.exit_commission = order.executed.comm
                    self.in_position = False
                    print(f"\n[LONG EXIT] SELL executed @ {self.exit_exec_price:.2f}")
                    self._log_trade_if_complete()

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"❌ Order {order.ref} {order.getstatusname()}")

        self.order = None

    def execute_order(self, action):
        if action == 'entry':
            if self.manager.direction == 1:
                self.order = self.buy()
            elif self.manager.direction == -1:
                self.order = self.sell()
            else: raise ValueError(f"Invalid direction: {self.manager.direction}")
            
        elif action == 'exit':
            if self.position.size > 0:
                self.order = self.sell()  # Closing long
            elif self.position.size < 0:
                self.order = self.buy()   # Closing short
            else:
                self.close()  # Fallback
        
        else: raise ValueError(f"Invalid action '{action}'. Must be 'entry' or 'exit'.")

    def next(self):
        # try:
        self.display_status_progress()

        curr_idx = len(self) - 1
        decision_idx = curr_idx - self.manager.entry_delay
        prev_decision_idx = decision_idx - 1

        if decision_idx < 0:
            return  # Not enough bars yet
        
        if self.reset_active_stop_price:
            self.active_stop_price = None
            self.reset_active_stop_price = False

        # decision_row = self.build_row(decision_idx)
        # curr_row = self.build_row(curr_idx)
        # prev_decision_row = self.build_row(decision_idx - 1) if decision_idx - 1 >= 0 else decision_row
        curr_row = self.build_row(0)
        decision_row = self.build_row(self.manager.entry_delay)
        prev_decision_row = self.build_row(self.manager.entry_delay + 1)

        # time_test = pd.Timestamp('2024-11-06 11:43:00-0500', tz='US/Eastern')
        # if curr_row['date'] == time_test:
        #     print()
        
        # Entry logic
        if not self.in_position:#self.position:
            if self.manager.evaluate_entry_conditions(decision_row, self.active_stop_price, display_triggers=False):
                self.entry_price = curr_row['close']
                self.entry_idx = curr_idx
                self.entry_time = curr_row['date']
                self.decision_prediction = decision_row['model_prediction']
                self.entry_prediction = curr_row['model_prediction']
                self.quantity = self.manager.evaluate_quantity(self.entry_prediction)
                self.order = self.execute_order('entry')

                # Resolve stop once here
                self.active_stop_price = self.manager.resolve_stop_price(curr_row, self.active_stop_price)

                # Set target entry time and price
                if hasattr(self.manager.strategy_instance.target_handler, 'set_entry_time'):
                    self.manager.strategy_instance.target_handler.set_entry_time(curr_row['date'])
                if hasattr(self.manager.strategy_instance.target_handler, 'set_target_price'):
                    self.manager.strategy_instance.target_handler.set_target_price(row=decision_row, stop_price=self.active_stop_price)

                reason2close = self.manager.assess_reason2close(decision_row, prev_decision_row, self.active_stop_price)
                if reason2close:
                    self.reset_active_stop_price = True # Don't reset active_stop_price directly or log_trade won't get the qctive_stop_price value
        else:
            # Exit logic
            reason2close = self.manager.assess_reason2close(decision_row, prev_decision_row, self.active_stop_price)
            if reason2close:
                self.exit_idx = curr_idx
                self.exit_price = curr_row['close']
                self.exit_reason = reason2close
                self.order = self.execute_order('exit')
                # self.active_stop_price = None  # Reset for next trade
                self.reset_active_stop_price = True # Don't reset active_stop_price directly or log_trade won't get the qctive_stop_price value

        self.prev_row = curr_row



# class BacktraderStrategyWrapper(bt.Strategy):
#     params = dict(
#         wrapped_strategy=None,          # Instance of your BaseStrategy subclass
#         prediction_threshold=0.7,
#         rrr_threshold=1.5,
#         tier_max=5,
#         size='auto',
#         entry_delay=1,
#         stop_handler=None,
#         target_handler=None,
#         total_bars=None,
#         df = pd.DataFrame(),
#         dropped_required_columns=[]
#     )

#     def __init__(self):
#         self.strategy = self.p.wrapped_strategy
#         self.prediction_threshold = self.p.prediction_threshold
#         self.rrr_threshold = self.p.rrr_threshold
#         self.tier_max = self.p.tier_max
#         self.size = self.p.size
#         self.entry_delay = self.p.entry_delay
#         self.stop_handler = self.p.stop_handler
#         self.target_handler = self.p.target_handler
#         self.active_stop_price = None  # stores fixed stop-loss value for current trade
#         self.total_bars = self.p.total_bars
#         self.df = self.p.df
#         self.dropped_required_columns = self.p.dropped_required_columns
#         self.status_counter = 0
#         self.prev_row = None
#         self.direction = trade_manager.TradeManager.resolve_direction(self.strategy)
#         # self.position = None
#         self.in_position = None
#         self.order = None
#         self.entry_price = None
#         self.entry_idx = None
#         self.exit_idx = None
#         self.entry_time = None
#         self.entry_prediction = None
#         self.quantity = None

#         # Determine which columns the strategy requires
#         self.required_columns = trade_manager.TradingManager.get_required_columns(self.strategy, self.target_handler, self.stop_handler)

#         # Initialize trade log
#         self.trades = []
#         self.entry_exec_price = None
#         self.exit_exec_price = None
#         self.entry_commission = 0.0
#         self.exit_commission = 0.0
#         self.exit_reason = None

#     def build_row(self, ago=0):
#         # Load columns from Backtrader data (numeric ones)
#         row = {}
#         for col in self.required_columns:
#             val = getattr(self.datas[0], col, None)
#             row[col] = val[-ago] if val else 0  # ago = 0 is current bar, ago = 1 is previous, etc.

#         # Special fields
#         row['model_prediction'] = getattr(self.datas[0], 'model_prediction', [None])[-ago]
#         row['predicted_drawdown'] = getattr(self.datas[0], 'predicted_drawdown', [None])[-ago]
#         row['date'] = pd.Timestamp(self.datas[0].datetime.datetime(-ago), tz=constants.CONSTANTS.TZ_WORK)
#         row['close'] = self.datas[0].close[-ago] if self.datas[0].close else None
    
#         # Inject dropped object columns back from self.df (source DataFrame)
#         if self.dropped_required_columns:
#             curr_idx = len(self) - ago - 1  # aligns with actual index
#             for col in self.dropped_required_columns:
#                 try:
#                     row[col] = self.df.iloc[curr_idx][col]
#                 except (IndexError, KeyError):
#                     row[col] = None  # Or some sensible default if required
        
#         return row

#     def display_status_progress(self):
#         # Status progress
#         if self.total_bars:
#             self.status_counter += 1
#             progress = (self.status_counter / self.total_bars) * 100
#             if self.status_counter % max(1, self.total_bars // 100) == 0 or self.status_counter == self.total_bars:
#                 print(f"\rBacktest progress for {self.datas[0]._name}: {progress:.1f}%", end='', flush=True)
    
#     def _log_trade_if_complete(self):
#         # Defensive: only log if all data is set
#         if None in [self.entry_exec_price, self.exit_exec_price]:
#             print("⚠️ Trade not fully executed, skipping log.")
#             return
        
#         symbol = self.datas[0]._name
#         trade_evaluator.TradeEvaluator.log_trade(self.direction, self.trades, self.entry_idx, self.exit_idx, self.df, self.entry_price, self.exit_price, 
#                                     self.quantity, self.entry_prediction, self.active_stop_price, self.exit_reason, symbol, self.entry_exec_price, 
#                                     self.exit_exec_price, self.entry_commission, self.exit_commission)
    
#     def notify_order(self, order):
#         if order.status in [order.Submitted, order.Accepted]:
#             return

#         if order.status == order.Completed:
#             is_entry = not self.in_position
#             is_exit = self.in_position

#             if order.isbuy():
#                 if self.direction == 1 and is_entry:
#                     self.entry_exec_price = order.executed.price
#                     self.entry_commission = order.executed.comm
#                     self.in_position = True
#                     print(f"\n[LONG ENTRY] BUY executed @ {self.entry_exec_price:.2f}")
#                 elif self.direction == -1 and is_exit:
#                     self.exit_exec_price = order.executed.price
#                     self.exit_commission = order.executed.comm
#                     self.in_position = False
#                     print(f"\n[SHORT EXIT] BUY executed @ {self.exit_exec_price:.2f}")
#                     self._log_trade_if_complete()

#             elif order.issell():
#                 if self.direction == -1 and is_entry:
#                     self.entry_exec_price = order.executed.price
#                     self.entry_commission = order.executed.comm
#                     self.in_position = True
#                     print(f"\n[SHORT ENTRY] SELL executed @ {self.entry_exec_price:.2f}")
#                 elif self.direction == 1 and is_exit:
#                     self.exit_exec_price = order.executed.price
#                     self.exit_commission = order.executed.comm
#                     self.in_position = False
#                     print(f"\n[LONG EXIT] SELL executed @ {self.exit_exec_price:.2f}")
#                     self._log_trade_if_complete()

#         elif order.status in [order.Canceled, order.Margin, order.Rejected]:
#             print(f"❌ Order {order.ref} {order.getstatusname()}")

#         self.order = None

#     def execute_order(self, action):
#         if action == 'entry':
#             if self.direction == 1:
#                 self.order = self.buy()
#             elif self.direction == -1:
#                 self.order = self.sell()
#             else: raise ValueError(f"Invalid direction: {self.direction}")
            
#         elif action == 'exit':
#             if self.position.size > 0:
#                 self.order = self.sell()  # Closing long
#             elif self.position.size < 0:
#                 self.order = self.buy()   # Closing short
#             else:
#                 self.close()  # Fallback
        
#         else: raise ValueError(f"Invalid action '{action}'. Must be 'entry' or 'exit'.")

#     def next(self):
#         # try:
#         self.display_status_progress()

#         curr_idx = len(self) - 1
#         decision_idx = curr_idx - self.entry_delay
#         prev_decision_idx = decision_idx - 1

#         if decision_idx < 0:
#             return  # Not enough bars yet

#         # decision_row = self.build_row(decision_idx)
#         # curr_row = self.build_row(curr_idx)
#         # prev_decision_row = self.build_row(decision_idx - 1) if decision_idx - 1 >= 0 else decision_row
#         curr_row = self.build_row(0)
#         decision_row = self.build_row(self.entry_delay)
#         prev_decision_row = self.build_row(self.entry_delay + 1)

#         # time_test = pd.Timestamp('2024-11-06 11:43:00-0500', tz='US/Eastern')
#         # if curr_row['date'] == time_test:
#         #     print()

#         # Entry logic
#         if not self.in_position:#self.position:
#             if trade_manager.TradeManager.evaluate_entry_conditions(decision_row, self.strategy, self.target_handler,
#                                                             self.prediction_threshold, self.rrr_threshold):
#                 self.entry_price = curr_row['close']
#                 self.entry_idx = curr_idx
#                 self.entry_time = curr_row['date']
#                 self.entry_prediction = curr_row['model_prediction']
#                 self.quantity = trade_manager.TradeManager.evaluate_quantity(self.entry_prediction, self.size,
#                                                                     self.prediction_threshold, self.tier_max)
#                 self.order = self.execute_order('entry')
#                 if hasattr(self.target_handler, 'set_entry_time'):
#                     self.target_handler.set_entry_time(curr_row['date'])

#                 # Resolve stop once here
#                 self.active_stop_price = trade_manager.TradeManager.resolve_stop_price(curr_row, self.active_stop_price,
#                                                                             self.stop_handler, self.direction)

#                 reason2close = trade_manager.TradeManager.assess_reason2close(decision_row, prev_decision_row, self.target_handler,
#                                                                     self.stop_handler, self.active_stop_price,
#                                                                     self.direction)
#                 if reason2close:
#                     self.active_stop_price = None
#         else:
#             # Exit logic
#             reason2close = trade_manager.TradeManager.assess_reason2close(decision_row, prev_decision_row, self.target_handler,
#                                                                 self.stop_handler, self.active_stop_price,
#                                                                 self.direction)
#             if reason2close:
#                 self.exit_idx = curr_idx
#                 self.exit_price = curr_row['close']
#                 self.exit_reason = reason2close
#                 self.order = self.execute_order('exit')
#                 self.active_stop_price = None  # Reset for next trade

#         self.prev_row = curr_row

        # except Exception as e:
        #     print("Exception in strategy `next`:", e)
        #     traceback.print_exc()
        #     self.env.runstop()  # Stop cerebro so it doesn’t keep running

    # def _evaluate_rrr(self, row):
    #     pred_dd = row.get('predicted_drawdown')
    #     vwap = row.get('vwap')
    #     if pred_dd and vwap:
    #         reward = abs(vwap - row['close'])
    #         return reward / pred_dd >= self.rrr_th
    #     return True

    # def _check_stop_hit(self, row):
    #     if self.stop_handler and self.entry_price is not None:
    #         return self.stop_handler.check_stop_loss(row, self.entry_price, direction=1)
    #     return False

    # def _check_target_hit(self, row):
    #     if self.target_handler:
    #         # Assuming target_handler.get_target_event(prev, curr)
    #         return self.target_handler.get_target_event(self.prev_row, row)
    #     return False

    # def _log_trade(self, row):
    #     exit_price = row['close']
    #     pnl = exit_price - self.entry_price
    #     return_pct = (pnl / self.entry_price) * 100 if self.entry_price else 0

    #     # Calculate max drawdown from entry to current bar
    #     if hasattr(self, 'entry_bar') and self.entry_idx is not None:
    #         lows = self.datas[0].low.get(size=len(self) - self.entry_idx)
    #         max_drawdown = min(lows) - self.entry_price  # most negative movement
    #     else:
    #         max_drawdown = None

    #     # Duration in minutes
    #     entry_time = getattr(self, 'entry_time', None)
    #     exit_time = row.get('date', None)
    #     if entry_time and exit_time:
    #         duration_min = (exit_time - entry_time).total_seconds() / 60
    #     else:
    #         duration_min = None

    #     self.trades.append({
    #         'symbol': self.datas[0]._name,
    #         'market_cap_cat': row.get('market_cap_cat'),
    #         'entry_price': self.entry_price,
    #         'exit_price': exit_price,
    #         'pnl': pnl,
    #         'return_pct': return_pct,
    #         'max_drawdown': max_drawdown,
    #         'duration_min': duration_min,
    #         'prediction': row.get('model_prediction'),
    #         'predicted_drawdown': row.get('predicted_drawdown'),
    #         'reward_to_risk': None,  # optionally compute
    #         'exit_reason': 'target' if self._check_target_hit(row) else 'stop'
    #     })
