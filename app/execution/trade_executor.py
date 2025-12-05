import sys, os, datetime, math
import xml.etree.ElementTree as ET
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from execution import orders
from live import trading_config
from utils import helpers, constants


class OOrder():
    def __init__(self, symbol, stop_loss=None, take_profit=None, quantity:int=None, type:str='Fast', config=None):
        self.symbol = symbol
        self.config = config or trading_config.TradingConfig().set_config(locals())
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.quantity = quantity
        self.order_type = type
        self.currency = helpers.get_stock_currency_yf(symbol) or self.config.currency


class TradeExecutor():
    def __init__(self, ib=IB(), config=None):
        self.ib = ib
        self.config = config or trading_config.TradingConfig().set_config(locals())
        self.equity = self._get_equity()

    def _get_equity(self):
        account_values = self.ib.accountValues()
        self.ib.sleep(constants.CONSTANTS.PROCESS_TIME['long'])
        if account_values:
            return float([accountValue.value for accountValue in self.ib.accountValues() if accountValue.tag == 'CashBalance' and accountValue.currency == self.config.currency][0])
        else:
            return self.config.capital

    def execute_order(self, direction:int, oorder:OOrder, ideal_price:float=None, price_diff_threshold:float=None):
        contract, mktData = helpers.get_symbol_mkt_data(self.ib, oorder.symbol, currency=oorder.currency)

        if mktData:

            price = mktData.ask if direction == 1 else (mktData.bid if direction == -1 else None)
            price_bis = mktData.bid if direction == 1 else (mktData.ask if direction == -1 else None)

            # Check if price used for evaluation not too different from real price
            if ideal_price and price_diff_threshold and abs(price - ideal_price) > price_diff_threshold:
                print(f"🚧 Price {price} - ideal price {ideal_price} > max threshold {price_diff_threshold}. Trade not executed for {oorder.symbol}.")
                return None, None

            # Check if real price not past the stop price
            if oorder.stop_loss and (price - oorder.stop_loss) * direction <= 0:
                print(f"🚧 Price {price} beyond stop_loss {oorder.stop_loss} with direction {direction}. Trade not executed for {oorder.symbol}.")
                return None, None

            # Place order
            order, TPSL_order = orders.autoOrder(self.ib, contract, direction, oorder.quantity, mktData, self.config.offset_targets, oorder.take_profit, oorder.stop_loss, oorder.order_type)#, TP_qty=delta_TP, SL_qty=delta_SL)#, ocaGroup=ocaGroup, ocaType=2)
            action = 'BUY' if direction == 1 else ('SELL' if direction == -1 else None)
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' - ' + action.lower() + ' order for ' + oorder.symbol + ': ' + str(direction * oorder.quantity) + ' shares @ ' + str(price) + '\n')
            self.ib.sleep(constants.CONSTANTS.PROCESS_TIME['medium'])

            return order, TPSL_order

        else:
            print(f"📵 Could not execute order, no Market Data found for {oorder.symbol}.")
            return None, None

    def close_position(self, oorder:OOrder, partial:int=100):

        contract, mktData = helpers.get_symbol_mkt_data(self.ib, oorder.symbol, currency=oorder.currency)
        positions = orders.get_positions_by_symbol(self.ib, oorder.symbol)
        if positions:
            oorder_copy = oorder.copy()
            BO_list, TP_orders, SL_orders = orders.record_bracket(self.ib, contract)
            for position in positions:
                print('\nPosition found: ', position)
                oorder_copy.symbol = position.contract.localSymbol if position.contract.localSymbol else position.contract.symbol
                oorder_copy.quantity = round(position.position * partial / 100)
                direction = 1 if oorder_copy.quantity < 0 else -1
                text_close = 'Closing' if partial == 100 else 'Partialling'
                print(text_close, ' position for symbol ', oorder_copy.symbol)
                self.execute_order(self.ib, direction, oorder_copy)
                self.ib.sleep(constants.CONSTANTS.PROCESS_TIME['short'])
        else:
            print('\nNo position found for this symbol')

        # Adjust bracket orders
        orders.adjust_bracket_orders(self.ib, oorder.symbol, oorder.currency)

    def set_max_quantity(self, stop_loss, price, currency, direction, max_loss_percentage):
        equity = float([accountValue.value for accountValue in self.accountValues() if accountValue.tag == 'CashBalance' and accountValue.currency == currency][0])
        # equity = 1500
        quantity = abs(math.floor((float(equity) * max_loss_percentage / 100.0) / (direction * (price - float(stop_loss)))))
        if quantity * price > equity: quantity = math.floor(equity / price)

        print('equity = ', equity)
        print('quantity = ', quantity)

        return quantity

    def get_account_balance(self, currency, max_loss_percentage):
        if currency == 'CAD': currency = 'BASE'
        try:
            values_list = self.ib.accountValues()
            cash_balance_BASE = [value.value for value in values_list if value.tag == 'CashBalance' and value.currency == 'BASE'][0]
            cash_balance_CURR = [value.value for value in values_list if value.tag == 'CashBalance' and value.currency == currency][0]
            # AccountValue(account='U14264367', tag='CashBalance', value='1297.3277', currency='BASE', modelCode='')
            max_loss = str(round(float(cash_balance_CURR) * float(max_loss_percentage) / 100, 2))
        except Exception as e:
            print(f"Couldn't fetch account balance and max loss. Error: {e}")
            cash_balance_CURR = '-1'
            max_loss = '-1'

        return cash_balance_CURR, max_loss


    if __name__ == "__main__":
        print()
