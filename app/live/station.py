import os, sys, pickle, prettytable, math, datetime
import PySimpleGUI as sg
# from bs4 import BeautifulSoup
# from matplotlib import pyplot
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from execution import trade_executor, orders
from utils import helpers
from utils.constants import CONSTANTS


def execute_order(ib, direction, values, simple_order=False):
    contract, mktData = helpers.get_symbol_mkt_data(ib, values['-symbol-'], currency=values['-currency-'])

    if mktData:

        price = mktData.ask if direction == 1 else (mktData.bid if direction == -1 else None)
        price_bis = mktData.bid if direction == 1 else (mktData.ask if direction == -1 else None)

        # Check if trying to place stop loss and take profit when reducing existing position
        open_position = orders.get_positions_by_symbol(ib, values['-symbol-'])
        if open_position and (open_position[0].position >= 0 and direction == -1 or open_position[0].position < 0 and direction == 1):
            simple_order = True

        # Calculate Stop Loss
        try:
            if values['-auto_stop_loss-'] and not simple_order: stop_loss = set_auto_stop_loss(price_bis, direction, values)
            else: stop_loss = abs(float(values['-stop_loss-'])) if (not simple_order and values['-stop_loss-'] != '') else ''
        except Exception as e:
                print('Could not calculate stop loss.\n Error: ', e)
                stop_loss = ''

        # Calculate Take Profit
        try:
            if values['-auto_take_profit-'] and not simple_order: take_profit = set_auto_take_profit(stop_loss, price, abs(float(values['-profit_ratio-'])))
            else: take_profit = abs(float(values['-take_profit-'])) if (not simple_order and values['-take_profit-'] != '') else ''
        except Exception as e:
                print('Could not calculate take profit.\n Error: ', e)
                take_profit = ''

        # Calculate quantity
        try:
            if values['-auto_quantity-'] and stop_loss != '': quantity = set_max_quantity(stop_loss, price, values['-currency-'], direction, abs(int(values['-max_loss_percentage-'])))
            else: quantity = abs(float(values['-quantity-']))
        except Exception as e:
            print('Could not calculate quantity.\n Error: ', e)
            quantity = 0

        # Place order
        orders.autoOrder(ib, contract, direction, quantity, mktData, abs(float(values['-offset_targets-'])), take_profit, stop_loss, order_type=values['-order_type-'])#, TP_qty=delta_TP, SL_qty=delta_SL)#, ocaGroup=ocaGroup, ocaType=2)
        action = 'BUY' if direction == 1 else ('SELL' if direction == -1 else None)
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' - ' + action.lower() + ' order for ' + values['-symbol-'] + ': ' + str(direction * quantity) + ' shares @ ' + str(price) + '\n')
        ib.sleep(0.5)

        # Adjust bracket orders
        if values['-auto_adjust_bracket-']:
            ib.sleep(1)
            orders.adjust_bracket_orders(ib, values['-symbol-'], values['-currency-'])

    else: print('Could not execute order')


def create_bracket(ib, values):
    contract, mktData = helpers.get_symbol_mkt_data(ib, values['-symbol-'], currency=values['-currency-'])
    open_position = orders.get_positions_by_symbol(ib, contract.symbol)

    if mktData and open_position:

        direction = 1 if open_position[0].position > 0 else -1
        price = mktData.ask if direction == 1 else (mktData.bid if direction == -1 else None)
        price_bis = mktData.bid if direction == 1 else (mktData.ask if direction == -1 else None)
        take_profit, stop_loss, quantity = '', '', 0

        # Calculate Stop Loss
        try: stop_loss = set_auto_stop_loss(price_bis, direction, values) if values['-auto_stop_loss-'] else abs(float(values['-stop_loss-'])) if values['-stop_loss-'] != '' else ''
        except Exception as e: print('Could not calculate stop loss.\n Error: ', e)

        # Calculate Take Profit
        try: take_profit = set_auto_take_profit(stop_loss, price, abs(float(values['-profit_ratio-']))) if values['-auto_take_profit-'] else abs(float(values['-take_profit-'])) if values['-take_profit-'] != '' else ''
        except Exception as e:print('Could not calculate take profit.\n Error: ', e)

        trail = values['-sl_offset_value-'] if values['-sl_offset_type-'] == "Trail" and values['-sl_offset-'] and values['-auto_stop_loss-'] else None
        orders.create_bracket_orders(ib, contract, mktData, values['-quantity-'], take_profit, stop_loss, trail=trail, partial=values['-create_partial_brackets-'], partial_perc=values['-partial_percentage-'])


def adjust_bracket(ib, values):
    orders.adjust_bracket_orders(ib, values['-symbol-'], values['-currency-'])


def move_SL_to_BE(ib, values):
    trail = values['-sl_offset_value-'] if values['-sl_offset_type-'] == "Trail" and values['-sl_offset-'] and values['-auto_stop_loss-'] else None
    orders.move_SL_to_BE(ib, values['-symbol-'], offset=values['-offset_targets-'], trail=trail, currency=values['-currency-'])


def partial(ib, values, perc):
    contract, mktData = helpers.get_symbol_mkt_data(ib, values['-symbol-'], currency=values['-currency-'])
    position = orders.get_positions_by_symbol(ib, values['-symbol-'])

    if position:
        values_copy = values.copy()

        # Recording and closing all orders
        bo_list, TP_orders, SL_orders = orders.record_bracket(ib, contract)
        print('Closing all orders for symbol ', values_copy['-symbol-'], '\n')
        orders.cancel_orders_by_symbol(ib, values_copy['-symbol-'])
        ib.sleep(CONSTANTS.PROCESS_TIME['short'])

        print('\nPosition found: ', position, '\n')
        values_copy['-symbol-'] = position[0].contract.localSymbol if position[0].contract.localSymbol else position[0].contract.symbol
        values_copy['-quantity-'] = round(position[0].position * perc / 100)
        if values_copy['-quantity-'] > 0:
            direction = 1 if values_copy['-quantity-'] < 0 else -1
            print('Partiallinig position for symbol ', values_copy['-symbol-'])
            execute_order(ib, direction, values_copy, simple_order=True)
            ib.sleep(CONSTANTS.PROCESS_TIME['short'])

            print('\nRecreating brackets...')
            orders.recreate_bracket(ib, contract, bo_list, qty_factor=perc/100, TP=None, SL=None)
        else: print("Quantity for partial is 0 - abort partial")

        print()
    else:
        print('\nNo position found for this symbol')

    # Reconstituing bracket orders
    # Adjust bracket orders
    orders.adjust_bracket_orders(ib, values['-symbol-'], values['-currency-'])


def close_position(ib, values, partial=100):

    contract, mktData = helpers.get_symbol_mkt_data(ib, values['-symbol-'], currency=values['-currency-'])
    positions = orders.get_positions_by_symbol(ib, values['-symbol-'])
    if positions:
        values_copy = values.copy()
        BO_list, TP_orders, SL_orders = orders.record_bracket(ib, contract)
        for position in positions:
            print('\nPosition found: ', position)
            values_copy['-symbol-'] = position.contract.localSymbol if position.contract.localSymbol else position.contract.symbol
            values_copy['-quantity-'] = round(position.position * partial / 100)
            direction = 1 if values_copy['-quantity-'] < 0 else -1
            text_close = 'Closing' if partial == 100 else 'Partialling'
            print(text_close, ' position for symbol ', values_copy['-symbol-'])
            execute_order(ib, direction, values_copy, simple_order=True)
            ib.sleep(CONSTANTS.PROCESS_TIME['short'])

            print()
    else:
        print('\nNo position found for this symbol')

    # Adjust bracket orders
    orders.adjust_bracket_orders(ib, values['-symbol-'], values['-currency-'])


def set_max_quantity(ib, stop_loss, price, currency, direction, max_loss_percentage):
    equity = float([accountValue.value for accountValue in ib.accountValues() if accountValue.tag == 'CashBalance' and accountValue.currency == currency][0])
    # equity = 1500
    quantity = abs(math.floor((float(equity) * max_loss_percentage / 100.0) / (direction * (price - float(stop_loss)))))
    if quantity * price > equity: quantity = math.floor(equity / price)

    print('equity = ', equity)
    print('quantity = ', quantity)

    return quantity

def get_account_balance(ib, currency, max_loss_percentage):
    if currency == 'CAD': currency = 'BASE'
    try:
        values_list = ib.accountValues()
        cash_balance_BASE = [value.value for value in values_list if value.tag == 'CashBalance' and value.currency == 'BASE'][0]
        cash_balance_CURR = [value.value for value in values_list if value.tag == 'CashBalance' and value.currency == currency][0]
        # AccountValue(account='U14264367', tag='CashBalance', value='1297.3277', currency='BASE', modelCode='')
        max_loss = str(round(float(cash_balance_CURR) * float(max_loss_percentage) / 100, 2))
    except Exception as e:
        print(f"Couldn't fetch account balance and max loss. Error: {e}")
        cash_balance_CURR = '-1'
        max_loss = '-1'

    return cash_balance_CURR, max_loss


def get_symbol_data(self, symbol, currency):

    contract, mktData = helpers.get_symbol_mkt_data(ib, symbol, currency=currency)

    bid = mktData.bid if mktData else -1
    ask = mktData.ask if mktData else -1

    position_table = prettytable.PrettyTable()
    orders_table = prettytable.PrettyTable()

    open_position = orders.get_positions_by_symbol(ib, symbol)
    if open_position:
        position_table.title = symbol + ' Positions'
        position_table.field_names = ['Quantity', 'Avg Cost', 'Currency']
        position_table.add_row([open_position[0].position, open_position[0].avgCost, currency])
        print('\n', position_table, '\n')
        # print('Position:\n', open_position[0].position, ' shares @ ', open_position[0].avgCost, ' ', currency, '\n')
        # print('\n------------------------------------\n')
        # print('Portfolio = \n', ib.portfolio())
        # print('\n------------------------------------\n')

        # print('Limit Orders:')
        # for o in lmt_orders: print(o.action, ' order of qty ', o.totalQuantity, ' @ ', o.lmtPrice, ' - order ID ', o.orderId)
        # print('\nStop Orders:')
        # for o in stp_orders: print(o.action, ' order of qty ', o.totalQuantity, ' @ ', o.auxPrice, ' - order ID ', o.orderId)
        # print('\n------------------------------------\n')

    else: print('No open position')

    bracket_orders = orders.get_active_orders(ib, symbol=symbol)
    lmt_orders = [o for o in bracket_orders if bracket_orders and o.orderType == 'LMT']
    stp_orders = [o for o in bracket_orders if bracket_orders and o.orderType == 'STP']
    if lmt_orders or stp_orders:
        orders_table = prettytable.PrettyTable()
        orders_table.title = symbol + ' Orders'
        orders_table.field_names = ['Type', 'Action', 'Quantity', 'Price', 'ID']
        for o in sorted(lmt_orders + stp_orders, reverse=True, key=lambda x: x.lmtPrice if x.lmtPrice else x.auxPrice):
            orders_table.add_row([o.orderType, o.action, o.totalQuantity, o.lmtPrice if o.lmtPrice else o.auxPrice, o.orderId])
        print('\n', orders_table, '\n')

    # fundamentals = [
    #     ib.reqFundamentalData(contract, reportType='ReportsFinSummary', fundamentalDataOptions=[]),
    #     ib.reqFundamentalData(contract, reportType='ReportsOwnership', fundamentalDataOptions=[]),
    #     ib.reqFundamentalData(contract, reportType='ReportSnapshot', fundamentalDataOptions=[]),
    #     ib.reqFundamentalData(contract, reportType='ReportsFinStatements', fundamentalDataOptions=[]),
    #     ib.reqFundamentalData(contract, reportType='RESC', fundamentalDataOptions=[]),
    #     ib.reqFundamentalData(contract, reportType='CalendarReport', fundamentalDataOptions=[])]

    # for f in fundamentals:
    #     print(f)
    #     input()

    # fund = ib.reqFundamentalData(contract, reportType='ReportSnapshot', fundamentalDataOptions=[])
    # doc = ET.fromstring(fund)

    # indexes = doc.findall('.//Indexconstituet')
    # indexes = helpers.get_index_from_symbol(ib, symbol)
    # for index in indexes:
    #     print("index = ", index)
    #     print("Index ETF = ", helpers.get_index_etf(index))

    # print(contract)
    # portfolio_list = ib.portfolio()
    # values_list = ib.accountValues()
    # fills_list = ib.fills()
    # orders_list = ib.orders()
    # trades_list = ib.trades()
    # positions_list = ib.positions()
    # orderStatus_list = ib.orderStatusEvent()
    # ib.sleep(0.5)

    # # print('\n------------------------------------\n')
    # # print('Fills = ')
    # # for fill in ib.fills():
    # #     print(fill)
    # #     print()
    # print('\n------------------------------------\n')
    # print('Orders = ')
    # for order in orders_list:
    #     print(order)
    #     print()
    # print('\n------------------------------------\n')
    # print('Trades = ')
    # for trade in trades_list:
    #     if trade.isActive():
    #         print(trade)
    #         for order in ib.orders():
    #             if order.orderId == trade.order.orderId:
    #                 print('\n-----------------\n')
    #                 print(order)
    #                 ocaGroup = order.ocaGroup
    #         # print(trade.isActive())
    #         print()
    # print('\n------------------------------------\n')
    # print('Positions = ')
    # for position in positions_list:
    #     print(position)
    #     print()
    # print('\n------------------------------------\n')
    # print('\n------------------------------------\n')
    # # input()
    # print('Data for symbol ', symbol, '\n')

    # print(mktData)
    # print('\n------------------------------------\n')
    # for position in ib.positions():
    #     print(position)
    # print('\n++++++++++++++++++++++++++++++\n')
    # print('\n------------------------------------\n')
    # print('Portfolio = ', portfolio_list)
    # print('\n------------------------------------\n')

    # try:
    #     print('********  OPTION CHAIN *********')
    #     print(f'Requesting option chain for {contract.symbol}')

    #     # contract_details = self.ib.reqContractDetails(underlying)
    #     if contract:
    #         conId = contract.conId
    #         print(f'ConId for {contract.symbol}: {conId}')

    #         option_chain = ib.reqSecDefOptParams(contract.symbol, '', contract.secType, conId)
    #         for chain in option_chain:
    #             print(f'Exchange: {chain.exchange}')
    #             print(f'Underlying ConId: {chain.underlyingConId}')
    #             print(f'Trading Class: {chain.tradingClass}')
    #             print(f'Multiplier: {chain.multiplier}')
    #             print(f'Expirations: {chain.expirations}')
    #             print(f'Strikes: {chain.strikes}')
    #             print('-' * 50)

    #         expiry = chain.expirations[0]
    #         strike = 340#chain.strikes[0]
    #         call_contract = Option(symbol=contract.symbol, lastTradeDateOrContractMonth=expiry, strike=strike, right='C', exchange='SMART', currency=currency)
    #         ib.qualifyContracts(call_contract)
    #         # contract, mktData = helpers.get_symbol_mkt_data(ib, values['-symbol-'], currency=values['-currency-'])
    #         data_call = ib.reqMktData(contract=call_contract, genericTickList='',snapshot=False, regulatorySnapshot=False)
    #         # data_stock = ib.reqMktData(contract=contract, genericTickList='',snapshot=False, regulatorySnapshot=False)
    #         print()
    #         print('data_call = \n', data_call)
    #         # print('data_stock = \n', data_stock)
    #     print('********  END Here (Requesting option Chain) *********')
    # except Exception as e:
    #     print(f'Failed to get option chain response: {e}')

    # input()

    return bid, ask


def set_auto_stop_loss(price, direction, values):
    vwap = 1.35220
    next_level = 80

    tick_value = helpers.get_tick_value(vwap)
    offset_targets = abs(float(values['-offset_targets-'])) * tick_value

    stop_loss = ''

    if values['-sl_offset-']:

        offset_sl = abs(float(values['-sl_offset_value-']))
        stop_loss = price - direction * offset_sl

    elif values['-sl_level-']:

        if values['-sl_level_type-'] == 'VWAP':

            if direction == 1 and price < vwap: print('Price below VWAP, could not generate Stop Loss')
            elif direction == -1 and price > vwap: print('Price above VWAP, could not generate Stop Loss')
            else: stop_loss = vwap - direction * offset_targets

        elif values['-sl_level_type-'] == 'Auto': stop_loss = next_level - direction * offset_targets

    else: print('No auto stop loss mode defined')

    print('stop_loss = ', stop_loss)

    return stop_loss


def set_auto_take_profit(stop_loss, price, profit_ratio):
    return price + (price - stop_loss) * profit_ratio


def clear_TP_SL(values):

    clear_values = {'-take_profit-': '',
                    '-profit_ratio-': '',
                    '-stop_loss-': '',
                    '-auto_take_profit-': False,
                    '-auto_stop_loss-': False,
                    '-sl_offset-': True,
                    '-sl_offset_value-': values['-sl_offset_value-'],
                    '-sl_level-': False,
                    '-sl_level_type-': False}
    return clear_values

def close_order(self, values, all=False):

    if values['-close_order_all-'] or all:
        orders.cancel_orders_by_symbol(ib, values['-symbol-'])
    elif values['-close_order_id_value-']:
        orders.cancel_order_by_id(ib, values['-close_order_id_value-'])

        
def reset_values(values, keep_after_order=False, store_file=None):

    use_store_file = store_file != None

    if use_store_file:# and os.path.exists(os.path.join(os.getcwd(), store_file)):
        try:
            with open(store_file, 'rb') as fp:
                reset_values = pickle.load(fp)
        except Exception as e:
            print('Could not load from file ', store_file, '.\n Error: ', e)
            use_store_file = False

    if not use_store_file:
        reset_values = {'-symbol-': values['-symbol-'] if keep_after_order else '',
                      '-currency-': values['-currency-'] if keep_after_order else CONSTANTS.DEFAULT_CURRENCY,
                      '-take_profit-': '',
                      '-profit_ratio-': '',
                      '-stop_loss-': '',
                      '-quantity-': values['-quantity-'] if keep_after_order else '',
                    #   'bid': values['-bid-'] if keep_symbol else '-1',
                    #   'ask': values['-ask-'] if keep_symbol else '-1',
                      '-max_loss_percentage-': values['-max_loss_percentage-'],
                      '-offset_targets-': values['-offset_targets-'],
                      '-auto_take_profit-': False,
                      '-auto_stop_loss-': False,
                      '-sl_offset-': True,
                      '-sl_offset_value-': values['-sl_offset_value-'],
                    #   '-sl_level-': False,
                    #   '-sl_level_type-': False,
                      '-auto_quantity-': False,
                      '-close_order_all-': False,
                      '-close_order_id_value-': values['-close_order_id_value-'],
                    #   '-close_order_all-': False,
                      '-order_type-': values['-order_type-'] if keep_after_order else 'Fast',
                    #   '-limit_wait-': values['-limit_wait-'] if keep_after_order else False,
                    #   '-peg_order-': values['-peg_order-'],
                      '-auto_adjust_bracket-': values['-auto_adjust_bracket-'] if keep_after_order else False,
                      '-create_partial_brackets-': False,
                      '-partial_percentage-': ''}

    return reset_values


def save_default(values, store_file):
    with open(store_file, 'wb') as fp:
        pickle.dump(values, fp)
        print('Values saved successfully to ', store_file)


if __name__ == '__main__':

    args = sys.argv

    # TWS Connection
    paperTrading = False if len(args) > 1 and 'live' in args else True
    ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)

    # Setup:
    # ------
    store_file = 'trading_station_store.pkl'
    max_loss_percentage = '0.5'
    offset_targets = '10'#'0.1'

    default_values = reset_values({'-max_loss_percentage-': max_loss_percentage, '-offset_targets-': offset_targets}, store_file=store_file)

    executor = trade_executor.TradeExecutor(ib)

    # Graphic Interface
    # -----------------

    sg.theme('LightGreen')
    col_qty = sg.Column([[sg.Text('Quantity', font=('Helvetica', 12, 'bold'))],
                    [sg.Input(key='-quantity-', default_text=default_values['-quantity-'], size=(10,1), disabled=default_values['-auto_quantity-'])],
                    # [sg.Text('Profit Ratio', font=('Helvetica', 8, 'bold'))],
                    # [sg.Input(key='-profit_ratio-', default_text='', size=(10,1))],
                    [sg.CB('Auto Quantity', key='-auto_quantity-', enable_events=True, default=default_values['-auto_quantity-'])],
                    ], pad=0)

    col_TP = sg.Column([[sg.Text('Profit Target', font=('Helvetica', 12, 'bold'))],
                    [sg.Input(key='-take_profit-', default_text=default_values['-take_profit-'], size=(10,1), disabled=default_values['-auto_take_profit-'])],
                    [sg.Text('Profit Ratio', font=('Helvetica', 8, 'bold'))],
                    [sg.Input(key='-profit_ratio-', default_text=default_values['-profit_ratio-'], size=(10,1), disabled=not default_values['-auto_take_profit-'])],
                    [sg.CB('Auto Profit Target', key='-auto_take_profit-', enable_events=True, default=default_values['-auto_take_profit-'])]
                    ], pad=0)

    col_SL = sg.Column([[sg.Text('Stop Loss', font=('Helvetica', 12, 'bold'))],
                    [sg.Input(key='-stop_loss-', default_text=default_values['-stop_loss-'], size=(10,1), disabled=default_values['-auto_stop_loss-'])],
                    [sg.CB('Auto Stop Loss', key='-auto_stop_loss-', enable_events=True, default=default_values['-auto_stop_loss-'])],
                    [sg.Radio('Offset', 'radio_SL', key='-sl_offset-', enable_events=True, default=default_values['-sl_offset-'], disabled=not default_values['-auto_stop_loss-']),
                    sg.Combo(['Fixed', 'Trail'], default_value='Fixed', key='-sl_offset_type-', disabled=not default_values['-auto_stop_loss-']), sg.Input(key='-sl_offset_value-', default_text=default_values['-sl_offset_value-'], size=(5,1), disabled=not default_values['-auto_stop_loss-'])],
                    [sg.Radio('Level', 'radio_SL', key='-sl_level-', enable_events=True, default=not default_values['-sl_offset-'], disabled=not default_values['-auto_stop_loss-']),
                     sg.Combo(['Auto', 'VWAP'], default_value='Auto', key='-sl_level_type-', disabled=not default_values['-auto_stop_loss-'])],
                    # [sg.Radio('VWAP', 'radio_SL', key='-sl_vwap-', enable_events=True, default=not default_values['-sl_offset-'], disabled=not default_values['-auto_stop_loss-'])],
                    # [sg.Radio('Next Level', 'radio_SL', key='-sl_next_level-', enable_events=True, default=not default_values['-sl_offset-'], disabled=not default_values['-auto_stop_loss-'])]
                    ], pad=0)

    layout = [
        [sg.Text('Trading Station', font=('Helvetica', 16, 'bold'), size=(14,1))],
        [sg.Text()],
        [sg.Text('Symbol   '), sg.Input(key='-symbol-', default_text=default_values['-symbol-'], size=(11,1)), sg.Text(size=(8,1)), sg.Text('Offset Targets'), sg.Input(key='-offset_targets-', default_text=default_values['-offset_targets-'], size=(10,1))],
        [sg.Text('Currency'), sg.Input(key='-currency-', default_text=default_values['-currency-'], size=(10,1)), sg.Text(size=(10,1)), sg.Text('Max Loss %'), sg.Input(key='-max_loss_percentage-', default_text=default_values['-max_loss_percentage-'], size=(10,1))],
        [sg.Text(size=(30,1)), sg.Text('Account Balance: '), sg.Text(key='-account_balance-', text='-1')],
        [sg.Text(size=(36,1)), sg.Text('Max Loss: '), sg.Text(key='-max_loss-', text='-1')],
        [sg.Button('Get Data', key='-get_data-', size=(12,1)), sg.Text(size=(30 ,1)), sg.Button('Refresh', key='-refresh-', size=(7,1))],
        [sg.Text('Bid: '), sg.Text(key='-bid-', text='-1'), sg.Text('Ask: '), sg.Text(key='-ask-',  text='-1')],
        [sg.Text()],
        [sg.Button('Close Position', key='-close_position-', size=(12,1)), sg.Text(size=(22,1)), sg.Button('Close All Positions', key='-close_all_positions-', size=(15,1))],
        [sg.Text()],
        [sg.HorizontalSeparator(color='black')],
        [sg.vtop(col_qty), sg.VerticalSeparator(), sg.vtop(col_TP), sg.VerticalSeparator(), sg.vtop(col_SL)],
        [sg.Button('Clear TP/SL', key='-clear_TP_SL-', size=(10,1)), sg.Text(size=(1,1)), sg.Button('Close Order', key='-close_order-', size=(10,1)), sg.Text('ID: '),
                     sg.Input(key='-close_order_id_value-', default_text=default_values['-close_order_id_value-'], size=(5,1), disabled=not default_values['-close_order_all-']), sg.CB('All', key='-close_order_all-', enable_events=True, default=default_values['-close_order_all-'])],
        # [sg.Text(size=(31,1)), sg.Radio('All', 'radio_close_order', key='-close_order_all-', enable_events=True, disabled=default_values['-close_order_all-'])],
        [sg.Text()],
        [sg.Button('Buy', key='-buy-', size=(10,2), button_color = 'green', mouseover_colors = ('black', 'dark green')), sg.Button('Sell', key='-sell-', size=(10,2), button_color = 'red', mouseover_colors = ('black', 'dark red'))],
        [sg.Combo(['Fast', 'Pegged Best', 'Pegged Mid', 'Limit Wait', 'Adaptive Algo'], default_value='Fast', key='-order_type-')],
        # [sg.CB('Pegged Order', key='-peg_order-', enable_events=True, default=default_values['-peg_order-'])],
        # [sg.CB('Limit Wait Order', key='-limit_wait-', enable_events=True, default=default_values['-limit_wait-'])],
        [sg.Text()],
        [sg.Button('Partial 50%', key='-partial_50-', size=(10,1), button_color = 'grey', mouseover_colors = ('black', 'dark grey')),
         sg.Button('Partial 25%', key='-partial_25-', size=(10,1), button_color = 'grey', mouseover_colors = ('black', 'dark grey')),
         sg.Button('Partial 10%', key='-partial_10-', size=(10,1), button_color = 'grey', mouseover_colors = ('black', 'dark grey')),],
        [sg.Text()],
        [sg.Button('Adjust Bracket', key='-adjust_bracket-', size=(12,1)), sg.CB('Auto Adjust Bracket', key='-auto_adjust_bracket-', enable_events=True, default=default_values['-auto_adjust_bracket-'])],
        [sg.Button('Create Bracket', key='-create_bracket-', size=(12,1)), sg.CB('Create Partial Brackets', key='-create_partial_brackets-', enable_events=True, default=default_values['-create_partial_brackets-'])],
        [sg.Text(size=(14,1)), sg.Text('Partial %'), sg.Input(key='-partial_percentage-', default_text=default_values['-partial_percentage-'], size=(5,1), disabled=not default_values['-create_partial_brackets-'])],
        [sg.Button('Stop Loss --> BE', key='-move_SL_to_BE-', size=(14,1))],
        [sg.Text()],
        [sg.Button('Exit', size=(8,1)), sg.Text(size=(30,1)), sg.Button('Save As Default', key='-save_default-', size=(15,1))],
    ]

    window = sg.Window('Trading Station', layout, use_default_focus=False, finalize=True)

    window.bind('<Control-a>', '-hotkey_buy-')
    window.bind('<Control-d>', '-hotkey_sell-')
    window.bind('<Control-z>', '-hotkey_close_position-')
    window.bind('<Control-p>', '-hotkey_exit-')
    window.bind('<Control-c>', '-hotkey_move_SL-BE-')
    window.bind('<Control-1>', '-hotkey_partial_10-')
    window.bind('<Control-2>', '-hotkey_partial_25-')
    window.bind('<Control-5>', '-hotkey_partial_50-')

    while True:
        event, values = window.read()

        if event == '-auto_take_profit-':
            if values['-auto_take_profit-']:
                window['-take_profit-'].Widget.configure(state = 'disabled')
                window['-profit_ratio-'].Widget.configure(state = 'normal')
            else:
                window['-take_profit-'].Widget.configure(state = 'normal')
                window['-profit_ratio-'].Widget.configure(state = 'disabled')

        if event == '-auto_stop_loss-':
            if values['-auto_stop_loss-']:
                window['-stop_loss-'].Widget.configure(state = 'disabled')
                window['-sl_offset-'].Widget.configure(state = 'normal')
                window['-sl_offset_value-'].Widget.configure(state = 'normal')
                window['-sl_offset_type-'].Widget.configure(state = 'normal')
                window['-sl_level-'].Widget.configure(state = 'normal')
                window['-sl_level_type-'].Widget.configure(state = 'normal')
            else:
                window['-stop_loss-'].Widget.configure(state = 'normal')
                window['-sl_offset-'].Widget.configure(state = 'disabled')
                window['-sl_offset_value-'].Widget.configure(state = 'disabled')
                window['-sl_offset_type-'].Widget.configure(state = 'disabled')
                window['-sl_level-'].Widget.configure(state = 'disabled')
                window['-sl_level_type-'].Widget.configure(state = 'disabled')

        if event == '-auto_quantity-':
            if values['-auto_quantity-']:
                window['-quantity-'].Widget.configure(state = 'disabled')
            else:
                window['-quantity-'].Widget.configure(state = 'normal')

        if event == '-close_order_all-':
            if values['-close_order_all-']:
                window['-close_order_id_value-'].Widget.configure(state = 'disabled')
            else:
                window['-close_order_id_value-'].Widget.configure(state = 'normal')

        if event == '-create_partial_brackets-':
            if values['-create_partial_brackets-']:
                window['-partial_percentage-'].Widget.configure(state = 'normal')
            else:
                window['-partial_percentage-'].Widget.configure(state = 'disabled')

        if event in (sg.WIN_CLOSED, 'Exit', '-hotkey_exit-'):
            save_default(values, store_file)
            break

        if event == '-save_default-':
            save_default(values, store_file)

        # if event == 'Clear':
        #     window.FindElement('-output-').Update('')
        #     # print(f'CT path: {values['-CT-path-input-']}')

        if event in ('-buy-', '-sell-', '-hotkey_buy-', '-hotkey_sell-'):
            if event in ('-buy-', '-hotkey_buy-'):
                direction = 1
                print('\nBuying...\n')
            elif event in ('-sell-', '-hotkey_sell-'):
                direction = -1
                print('\nSelling...\n')
            contract = executor.execute_order(direction, values)

            # reset values
            values_resetted = reset_values(values, keep_after_order=True)
            for key in values_resetted.keys():
                window[key].update(value=values_resetted[key])
            window['-bid-'].update(value='-1')
            window['-ask-'].update(value='-1')

        if event == '-get_data-':
            bid, ask = executor.get_symbol_data(values['-symbol-'], values['-currency-'])
            window['-bid-'].update(value=bid)
            window['-ask-'].update(value=ask)

        if event == '-refresh-':
            account_balance, max_loss = executor.get_account_balance(values['-currency-'], values['-max_loss_percentage-'])
            window['-account_balance-'].update(value=account_balance)
            window['-max_loss-'].update(value=max_loss)

        if event == '-clear_TP_SL-':
            values_cleared = executor.clear_TP_SL(values)
            for key in values_cleared.keys():
                window[key].update(value=values_cleared[key])
            window['-stop_loss-'].Widget.configure(state = 'normal')
            window['-sl_offset-'].Widget.configure(state = 'disabled')
            window['-sl_offset_value-'].Widget.configure(state = 'disabled')
            window['-sl_offset_type-'].Widget.configure(state = 'disabled')
            window['-sl_level-'].Widget.configure(state = 'disabled')
            window['-sl_level_type-'].Widget.configure(state = 'disabled')
            window['-take_profit-'].Widget.configure(state = 'normal')
            window['-profit_ratio-'].Widget.configure(state = 'disabled')

        if event == '-close_order-':
            executor.close_order(values)

        if event in ('-close_position-', '-hotkey_close_position-'):
            executor.close_position(values)

            # reset values
            values_resetted = reset_values(values, keep_after_order=True)
            for key in values_resetted.keys():
                window[key].update(value=values_resetted[key])

        if event == '-close_all_positions-':
            if sg.popup_yes_no('Confirm close all positions?',  title='Close Confirmation') == 'Yes':
                values['-symbol-'] = 'all'
                executor.close_position(values)

                # reset values
                values_resetted = reset_values(values)
                for key in values_resetted.keys():
                    window[key].update(value=values_resetted[key])
            else:
                print('Abort cancel all positions')

        if event == '-adjust_bracket-':
            executor.adjust_bracket(values)

        if event == '-create_bracket-':
            executor.create_bracket(values)

        if event in ('-move_SL_to_BE-', '-hotkey_move_SL-BE-'):
            executor.move_SL_to_BE(values)

        if event in ('-partial_50-', '-hotkey_partial_50-'):
            executor.partial(values, perc=50)

        if event in ('-partial_25-', '-hotkey_partial_25-'):
            executor.partial(values, perc=25)

        if event in ('-partial_10-', '-hotkey_partial_10-'):
            executor.partial(values, perc=10)

    window.close()
    ib.disconnect()

    print('\n')
    # input('\nType anything to exit')

















































# # Case scenarios:
# # Enter with bracket same direction
#     # no exisiting bracket
#     # existing bracket
# # Enter with bracket opposite direction
#     # no exisiting bracket
#     # existing bracket
# # Enter without bracket same direction
#     # no exisiting bracket
#     # existing bracket
# # Enter without bracket opposite direction
#     # no exisiting bracket
#     # existing bracket


# def execute_order(ib, direction, values, simple_order=False):

#     # df = helpers.get_symbol_hist_data(ib, symbol=symbol, timeframe='1 min', query_time=helpers.date_local_to_EST(datetime.datetime.now()), duration='300 S', indicators_list = [])
#     # print(df.to_string())
#     # price = df.loc[len(df)-1, 'close']

#     contract, mktData = helpers.get_symbol_mkt_data(ib, values['-symbol-'], currency=values['-currency-'])

#     if mktData:

#         price = mktData.ask if direction == 1 else (mktData.bid if direction == -1 else None)
#         price_bis = mktData.bid if direction == 1 else (mktData.ask if direction == -1 else None)

#         # if position against:
#         #     simple_order = True
#         #     remove oldest bracket
#         # if no bracket and simple_order=False:
#         #     add bracket at same price as newest one

#         # active_orders = orders.get_active_orders(ib, values['-symbol-'])
#         # for order in active_orders:
#         #     print(order.ocaGroup)
#         # ocaGroup = active_orders[0].ocaGroup if active_orders else str(values['-symbol-']) + '_' + str(datetime.datetime.now())

#         # Check if trying to place stop loss and take profit when reducing existing position
#         open_position = orders.get_positions_by_symbol(ib, values['-symbol-'])
#         if open_position and (open_position[0].position >= 0 and direction == -1 or open_position[0].position < 0 and direction == 1):
#             # stop_loss = ''
#             # take_profit = ''
#             simple_order = True
#             # TP_orders, SL_orders = orders.find_bracket_orders(ib, contract)

#         # Calculate Stop Loss
#         try:
#             if values['-auto_stop_loss-'] and not simple_order: stop_loss = set_auto_stop_loss(price_bis, direction, values)
#             else: stop_loss = abs(float(values['-stop_loss-'])) if (not simple_order and values['-stop_loss-'] != '') else ''
#         except Exception as e:
#                 print('Could not calculate stop loss.\n Error: ', e)
#                 stop_loss = ''

#         # Calculate Take Profit
#         try:
#             if values['-auto_take_profit-'] and not simple_order: take_profit = set_auto_take_profit(stop_loss, price, abs(float(values['-profit_ratio-'])))
#             else: take_profit = abs(float(values['-take_profit-'])) if (not simple_order and values['-take_profit-'] != '') else ''
#         except Exception as e:
#                 print('Could not calculate take profit.\n Error: ', e)
#                 take_profit = ''

#         # Calculate quantity
#         try:
#             if values['-auto_quantity-'] and stop_loss != '': quantity = set_auto_quantity(stop_loss, price, values['-currency-'], direction, abs(int(values['-max_loss_percentage-'])))
#             else: quantity = abs(float(values['-quantity-']))
#         except Exception as e:
#             print('Could not calculate quantity.\n Error: ', e)
#             quantity = 0

#         # # Adjust bracket orders
#         # delta_TP, delta_SL, last_TP_order, last_SL_order = orders.adjust_bracket_orders(ib, contract, new_qty=quantity)
#         # print('delta_TP = ', delta_TP)
#         # print('delta_SL = ', delta_SL)

#         # if delta_SL > 0 and stop_loss == '' and last_SL_order:
#         #     stop_loss = last_SL_order.auxPrice

#         # if delta_TP > 0 and take_profit == '' and last_TP_order:
#         #     take_profit = last_TP_order.lmtPrice





#         # # Adjust take profit and stop loss
#         # TP_orders, SL_orders = orders.find_bracket_orders(ib, contract)
#         # if SL_orders:
#         #     last_SL_order = max(SL_orders, key=lambda x: x.orderId)
#         #     for order in SL_orders:
#         #         orders.cancel_order_by_id(ib, order.orderId)

#         #     if stop_loss == '': stop_loss = last_SL_order.auxPrice

#         # if (SL_orders and open_position) or (not SL_orders and open_position and stop_loss != ''):
#         #     SL_qty = abs(open_position[0].position + quantity * direction)
#         # else: SL_qty = None

#         # if TP_orders:
#         #     last_TP_order = max(TP_orders, key=lambda x: x.orderId)
#         #     for order in TP_orders:
#         #         orders.cancel_order_by_id(ib, order.orderId)

#         #     if take_profit == '': take_profit = last_TP_order.lmtPrice

#         # if (TP_orders and open_position) or (not TP_orders and open_position and take_profit != ''):
#         #     TP_qty = abs(open_position[0].position + quantity * direction)
#         # else: TP_qty = None

#         # Place order
#         # orders.autoOrder(ib, contract, direction, quantity, mktData, abs(float(values['-offset_targets-'])), take_profit, stop_loss, TP_qty=delta_TP, SL_qty=delta_SL)#, ocaGroup=ocaGroup, ocaType=2)
#         orders.autoOrder(ib, contract, direction, quantity, mktData, abs(float(values['-offset_targets-'])), take_profit, stop_loss, order_type=values['-order_type-'])#, TP_qty=delta_TP, SL_qty=delta_SL)#, ocaGroup=ocaGroup, ocaType=2)
#         action = 'BUY' if direction == 1 else ('SELL' if direction == -1 else None)
#         print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' - ' + action.lower() + ' order for ' + values['-symbol-'] + ': ' + str(direction * quantity) + ' shares @ ' + str(price) + '\n')
#         ib.sleep(0.5)

#         # Adjust bracket orders
#         if values['-auto_adjust_bracket-']:
#             ib.sleep(1)
#             orders.adjust_bracket_orders(ib, values['-symbol-'], values['-currency-'])

#     else: print('Could not execute order')


# def create_bracket(ib, values):

#     contract, mktData = helpers.get_symbol_mkt_data(ib, values['-symbol-'], currency=values['-currency-'])
#     open_position = orders.get_positions_by_symbol(ib, contract.symbol)

#     if mktData and open_position:

#         direction = 1 if open_position[0].position > 0 else -1
#         price = mktData.ask if direction == 1 else (mktData.bid if direction == -1 else None)
#         price_bis = mktData.bid if direction == 1 else (mktData.ask if direction == -1 else None)
#         take_profit, stop_loss, quantity = '', '', 0

#         # Calculate Stop Loss
#         try: stop_loss = set_auto_stop_loss(price_bis, direction, values) if values['-auto_stop_loss-'] else abs(float(values['-stop_loss-'])) if values['-stop_loss-'] != '' else ''
#         except Exception as e: print('Could not calculate stop loss.\n Error: ', e)

#         # Calculate Take Profit
#         try: take_profit = set_auto_take_profit(stop_loss, price, abs(float(values['-profit_ratio-']))) if values['-auto_take_profit-'] else abs(float(values['-take_profit-'])) if values['-take_profit-'] != '' else ''
#         except Exception as e:print('Could not calculate take profit.\n Error: ', e)

#         trail = values['-sl_offset_value-'] if values['-sl_offset_type-'] == "Trail" and values['-sl_offset-'] and values['-auto_stop_loss-'] else None
#         orders.create_bracket_orders(ib, contract, mktData, values['-quantity-'], take_profit, stop_loss, trail=trail, partial=values['-create_partial_brackets-'], partial_perc=values['-partial_percentage-'])

#     # open_position = orders.get_positions_by_symbol(ib, values['-symbol-'])

#     # # Determining Direction
#     # action = None
#     # if open_position:
#     #     if open_position[0].position < 0: action = 'BUY'
#     #     elif open_position[0].position > 0: action = 'SELL'
#     # elif values['-take_profit-'] and values['-stop_loss-']:
#     #         if values['-take_profit-'] < values['-stop_loss-']: action = 'BUY'
#     #         elif values['-take_profit-'] > values['-stop_loss-']: action = 'SELL'
#     # elif values['-take_profit-'] or values['-stop_loss-']:
#     #     action = sg.popup('Choose order direction', button_type=sg.POPUP_BUTTONS_YES_NO, custom_text=('BUY ','SELL'))

#     # if not action:
#     #     print('Could not determine a direction for the bracket order.')
#     # else:
#         # bracket_params =[{'quantity': values['-quantity-'], 'TP': values['-take_profit-'], 'SL': values['-stop_loss-']}]

#         # # Case Create Partial Brackets
#         # if values['-create_partial_brackets-']:
#         #     try:
#         #         partial_perc = abs(float(values['-partial_percentage-']))
#         #         bracket_params[0]['quantity'] = round(abs(float(values['-quantity-'])) * partial_perc / 100)

#         #     except Exception as e:
#         #         print(f'Failed to parse Partial % or Quantity: {e}')

#         # for params in bracket_params:
#         #     orders.placeTPSLOrders(ib, contract, action, params['quantity'], params['TP'], params['SL'])


# def adjust_bracket(ib, values):

#     orders.adjust_bracket_orders(ib, values['-symbol-'], values['-currency-'])


# def move_SL_to_BE(ib, values):

#     trail = values['-sl_offset_value-'] if values['-sl_offset_type-'] == "Trail" and values['-sl_offset-'] and values['-auto_stop_loss-'] else None
#     orders.move_SL_to_BE(ib, values['-symbol-'], offset=values['-offset_targets-'], trail=trail, currency=values['-currency-'])


# def partial(ib, values, perc):

#     contract, mktData = helpers.get_symbol_mkt_data(ib, values['-symbol-'], currency=values['-currency-'])
#     position = orders.get_positions_by_symbol(ib, values['-symbol-'])

#     if position:
#         values_copy = values.copy()

#         # Recording and closing all orders
#         bo_list, TP_orders, SL_orders = orders.record_bracket(ib, contract)
#         print('Closing all orders for symbol ', values_copy['-symbol-'], '\n')
#         orders.cancel_orders_by_symbol(ib, values_copy['-symbol-'])
#         ib.sleep(CONSTANTS.PROCESS_TIME['short'])

#         print('\nPosition found: ', position, '\n')
#         values_copy['-symbol-'] = position[0].contract.localSymbol if position[0].contract.localSymbol else position[0].contract.symbol
#         values_copy['-quantity-'] = round(position[0].position * perc / 100)
#         if values_copy['-quantity-'] > 0:
#             direction = 1 if values_copy['-quantity-'] < 0 else -1
#             print('Partiallinig position for symbol ', values_copy['-symbol-'])
#             execute_order(ib, direction, values_copy, simple_order=True)
#             ib.sleep(CONSTANTS.PROCESS_TIME['short'])

#             print('\nRecreating brackets...')
#             orders.recreate_bracket(ib, contract, bo_list, qty_factor=perc/100, TP=None, SL=None)
#         else: print("Quantity for partial is 0 - abort partial")

#         print()
#     else:
#         print('\nNo position found for this symbol')

#     # Reconstituing bracket orders
#     # Adjust bracket orders
#     orders.adjust_bracket_orders(ib, values['-symbol-'], values['-currency-'])
#     # if mktData:
#     #     # orders.recreate_bracket(ib, contract, qty_factor=perc/100)
#     #     close_position(ib, values, partial=perc)
#     # else:
#     #     print('\nCould not retrieve symbol data.\n')


# def close_position(ib, values, partial=100):

#     # if partial == 100:
#     #     # if values['-symbol'] == 'all':
#     #     print('Closing pending orders...')
#     #     orders.cancel_orders_by_symbol(ib, values['-symbol-'])
#     #     ib.sleep(0.5)

#     contract, mktData = helpers.get_symbol_mkt_data(ib, values['-symbol-'], currency=values['-currency-'])
#     positions = orders.get_positions_by_symbol(ib, values['-symbol-'])
#     if positions:
#         values_copy = values.copy()
#         BO_list, TP_orders, SL_orders = orders.record_bracket(ib, contract)
#         for position in positions:
#             print('\nPosition found: ', position)
#             values_copy['-symbol-'] = position.contract.localSymbol if position.contract.localSymbol else position.contract.symbol
#             values_copy['-quantity-'] = round(position.position * partial / 100)
#             direction = 1 if values_copy['-quantity-'] < 0 else -1
#             text_close = 'Closing' if partial == 100 else 'Partialling'
#             print(text_close, ' position for symbol ', values_copy['-symbol-'])
#             execute_order(ib, direction, values_copy, simple_order=True)
#             ib.sleep(CONSTANTS.PROCESS_TIME['short'])

#             # print('Closing all orders for symbol ', values_copy['-symbol-'], '\n')
#             # orders.cancel_orders_by_symbol(ib, values_copy['-symbol-'])
#             # ib.sleep(0.5)

#             print()
#     else:
#         print('\nNo position found for this symbol')

#     # Adjust bracket orders
#     orders.adjust_bracket_orders(ib, values['-symbol-'], values['-currency-'])


# def get_account_balance(ib, currency, max_loss_percentage):
#     if currency == 'CAD': currency = 'BASE'
#     try:
#         values_list = ib.accountValues()
#         cash_balance_BASE = [value.value for value in values_list if value.tag == 'CashBalance' and value.currency == 'BASE'][0]
#         cash_balance_CURR = [value.value for value in values_list if value.tag == 'CashBalance' and value.currency == currency][0]
#         # AccountValue(account='U14264367', tag='CashBalance', value='1297.3277', currency='BASE', modelCode='')
#         max_loss = str(round(float(cash_balance_CURR) * float(max_loss_percentage) / 100, 2))
#     except Exception as e:
#         print(f"Couldn't fetch account balance and max loss. Error: {e}")
#         cash_balance_CURR = '-1'
#         max_loss = '-1'


#     return cash_balance_CURR, max_loss


# def get_symbol_data(ib, symbol, currency):

#     contract, mktData = helpers.get_symbol_mkt_data(ib, symbol, currency=currency)

#     bid = mktData.bid if mktData else -1
#     ask = mktData.ask if mktData else -1

#     position_table = prettytable.PrettyTable()
#     orders_table = prettytable.PrettyTable()

#     open_position = orders.get_positions_by_symbol(ib, symbol)
#     if open_position:
#         position_table.title = symbol + ' Positions'
#         position_table.field_names = ['Quantity', 'Avg Cost', 'Currency']
#         position_table.add_row([open_position[0].position, open_position[0].avgCost, currency])
#         print('\n', position_table, '\n')
#         # print('Position:\n', open_position[0].position, ' shares @ ', open_position[0].avgCost, ' ', currency, '\n')
#         # print('\n------------------------------------\n')
#         # print('Portfolio = \n', ib.portfolio())
#         # print('\n------------------------------------\n')

#         # print('Limit Orders:')
#         # for o in lmt_orders: print(o.action, ' order of qty ', o.totalQuantity, ' @ ', o.lmtPrice, ' - order ID ', o.orderId)
#         # print('\nStop Orders:')
#         # for o in stp_orders: print(o.action, ' order of qty ', o.totalQuantity, ' @ ', o.auxPrice, ' - order ID ', o.orderId)
#         # print('\n------------------------------------\n')

#     else: print('No open position')

#     bracket_orders = orders.get_active_orders(ib, symbol=symbol)
#     lmt_orders = [o for o in bracket_orders if bracket_orders and o.orderType == 'LMT']
#     stp_orders = [o for o in bracket_orders if bracket_orders and o.orderType == 'STP']
#     if lmt_orders or stp_orders:
#         orders_table = prettytable.PrettyTable()
#         orders_table.title = symbol + ' Orders'
#         orders_table.field_names = ['Type', 'Action', 'Quantity', 'Price', 'ID']
#         for o in sorted(lmt_orders + stp_orders, reverse=True, key=lambda x: x.lmtPrice if x.lmtPrice else x.auxPrice):
#             orders_table.add_row([o.orderType, o.action, o.totalQuantity, o.lmtPrice if o.lmtPrice else o.auxPrice, o.orderId])
#         print('\n', orders_table, '\n')

#     # fundamentals = [
#     #     ib.reqFundamentalData(contract, reportType='ReportsFinSummary', fundamentalDataOptions=[]),
#     #     ib.reqFundamentalData(contract, reportType='ReportsOwnership', fundamentalDataOptions=[]),
#     #     ib.reqFundamentalData(contract, reportType='ReportSnapshot', fundamentalDataOptions=[]),
#     #     ib.reqFundamentalData(contract, reportType='ReportsFinStatements', fundamentalDataOptions=[]),
#     #     ib.reqFundamentalData(contract, reportType='RESC', fundamentalDataOptions=[]),
#     #     ib.reqFundamentalData(contract, reportType='CalendarReport', fundamentalDataOptions=[])]

#     # for f in fundamentals:
#     #     print(f)
#     #     input()

#     # fund = ib.reqFundamentalData(contract, reportType='ReportSnapshot', fundamentalDataOptions=[])
#     # doc = ET.fromstring(fund)

#     # indexes = doc.findall('.//Indexconstituet')
#     # indexes = helpers.get_index_from_symbol(ib, symbol)
#     # for index in indexes:
#     #     print("index = ", index)
#     #     print("Index ETF = ", helpers.get_index_etf(index))

#     # print(contract)
#     # portfolio_list = ib.portfolio()
#     # values_list = ib.accountValues()
#     # fills_list = ib.fills()
#     # orders_list = ib.orders()
#     # trades_list = ib.trades()
#     # positions_list = ib.positions()
#     # orderStatus_list = ib.orderStatusEvent()
#     # ib.sleep(0.5)

#     # # print('\n------------------------------------\n')
#     # # print('Fills = ')
#     # # for fill in ib.fills():
#     # #     print(fill)
#     # #     print()
#     # print('\n------------------------------------\n')
#     # print('Orders = ')
#     # for order in orders_list:
#     #     print(order)
#     #     print()
#     # print('\n------------------------------------\n')
#     # print('Trades = ')
#     # for trade in trades_list:
#     #     if trade.isActive():
#     #         print(trade)
#     #         for order in ib.orders():
#     #             if order.orderId == trade.order.orderId:
#     #                 print('\n-----------------\n')
#     #                 print(order)
#     #                 ocaGroup = order.ocaGroup
#     #         # print(trade.isActive())
#     #         print()
#     # print('\n------------------------------------\n')
#     # print('Positions = ')
#     # for position in positions_list:
#     #     print(position)
#     #     print()
#     # print('\n------------------------------------\n')
#     # print('\n------------------------------------\n')
#     # # input()
#     # print('Data for symbol ', symbol, '\n')

#     # print(mktData)
#     # print('\n------------------------------------\n')
#     # for position in ib.positions():
#     #     print(position)
#     # print('\n++++++++++++++++++++++++++++++\n')
#     # print('\n------------------------------------\n')
#     # print('Portfolio = ', portfolio_list)
#     # print('\n------------------------------------\n')

#     # try:
#     #     print('********  OPTION CHAIN *********')
#     #     print(f'Requesting option chain for {contract.symbol}')

#     #     # contract_details = self.ib.reqContractDetails(underlying)
#     #     if contract:
#     #         conId = contract.conId
#     #         print(f'ConId for {contract.symbol}: {conId}')

#     #         option_chain = ib.reqSecDefOptParams(contract.symbol, '', contract.secType, conId)
#     #         for chain in option_chain:
#     #             print(f'Exchange: {chain.exchange}')
#     #             print(f'Underlying ConId: {chain.underlyingConId}')
#     #             print(f'Trading Class: {chain.tradingClass}')
#     #             print(f'Multiplier: {chain.multiplier}')
#     #             print(f'Expirations: {chain.expirations}')
#     #             print(f'Strikes: {chain.strikes}')
#     #             print('-' * 50)

#     #         expiry = chain.expirations[0]
#     #         strike = 340#chain.strikes[0]
#     #         call_contract = Option(symbol=contract.symbol, lastTradeDateOrContractMonth=expiry, strike=strike, right='C', exchange='SMART', currency=currency)
#     #         ib.qualifyContracts(call_contract)
#     #         # contract, mktData = helpers.get_symbol_mkt_data(ib, values['-symbol-'], currency=values['-currency-'])
#     #         data_call = ib.reqMktData(contract=call_contract, genericTickList='',snapshot=False, regulatorySnapshot=False)
#     #         # data_stock = ib.reqMktData(contract=contract, genericTickList='',snapshot=False, regulatorySnapshot=False)
#     #         print()
#     #         print('data_call = \n', data_call)
#     #         # print('data_stock = \n', data_stock)
#     #     print('********  END Here (Requesting option Chain) *********')
#     # except Exception as e:
#     #     print(f'Failed to get option chain response: {e}')

#     # input()

#     return bid, ask


# def set_auto_stop_loss(price, direction, values):

#     vwap = 1.35220
#     next_level = 80

#     tick_value = helpers.get_tick_value(vwap)
#     offset_targets = abs(float(values['-offset_targets-'])) * tick_value

#     stop_loss = ''

#     if values['-sl_offset-']:

#         offset_sl = abs(float(values['-sl_offset_value-']))
#         stop_loss = price - direction * offset_sl

#     elif values['-sl_level-']:

#         if values['-sl_level_type-'] == 'VWAP':

#             if direction == 1 and price < vwap: print('Price below VWAP, could not generate Stop Loss')
#             elif direction == -1 and price > vwap: print('Price above VWAP, could not generate Stop Loss')
#             else: stop_loss = vwap - direction * offset_targets

#         elif values['-sl_level_type-'] == 'Auto': stop_loss = next_level - direction * offset_targets

#     else: print('No auto stop loss mode defined')

#     print('stop_loss = ', stop_loss)

#     return stop_loss


# def set_auto_take_profit(stop_loss, price, profit_ratio):

#     return price + (price - stop_loss) * profit_ratio


# def set_auto_quantity(stop_loss, price, currency, direction, max_loss_percentage):

#     equity = float([accountValue.value for accountValue in ib.accountValues() if accountValue.tag == 'CashBalance' and accountValue.currency == currency][0])
#     equity = 1500
#     quantity = abs(math.floor((float(equity) * max_loss_percentage / 100.0) / (direction * (price - float(stop_loss)))))
#     if quantity * price > equity: quantity = math.floor(equity / price)

#     print('equity = ', equity)
#     print('quantity = ', quantity)

#     return quantity


# def clear_TP_SL(values):

#     clear_values = {'-take_profit-': '',
#                       '-profit_ratio-': '',
#                       '-stop_loss-': '',
#                       '-auto_take_profit-': False,
#                       '-auto_stop_loss-': False,
#                       '-sl_offset-': True,
#                       '-sl_offset_value-': values['-sl_offset_value-'],
#                       '-sl_level-': False,
#                       '-sl_level_type-': False}

#     return clear_values


# def close_order(ib, values, all=False):

#     if values['-close_order_all-'] or all:
#         orders.cancel_orders_by_symbol(ib, values['-symbol-'])
#     elif values['-close_order_id_value-']:
#         orders.cancel_order_by_id(ib, values['-close_order_id_value-'])
