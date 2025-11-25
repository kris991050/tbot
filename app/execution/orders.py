import os, sys, json, datetime
from ib_insync import *
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers
from utils.constants import CONSTANTS

# Example for Forex: https://github.com/erdewit/ib_insync/blob/master/notebooks/ordering.ipynb
#                    https://github.com/erdewit/ib_insync/blob/master/notebooks/tick_data.ipynb


def placeOrder(ib, contract, action, qty, type='MKT', price=0.0, trail=0.2, isPT=False, ocaGroup=None, ocaType=2):

    # https://ib-insync.readthedocs.io/_modules/ib_insync/order.html
    if price: price = round(price, 2)
    if type == 'MKT':
        order = Order(action=action, orderType=type, totalQuantity=qty, tif='GTC', transmit=True, outsideRth=True)
    elif type == 'LMT':
        order = Order(action=action, orderType=type, lmtPrice=price, totalQuantity=qty, tif='GTC', transmit=not isPT, outsideRth=True)
    elif type == 'STP':
        order = Order(action=action, orderType=type, totalQuantity=qty, auxPrice=price, transmit=True, tif='GTC', outsideRth=True)
    elif type == 'TRAIL':
        order = Order(action=action, orderType=type, totalQuantity=qty, trailStopPrice=price, auxPrice=trail, transmit=True, tif='GTC', outsideRth=True)
    elif type == 'PEG BEST' or 'PEG MID':
        # https://interactivebrokers.github.io/tws-api/ibkrats.html#ibkrats_orders
        contract.exchange = "IBKRATS"
        order = Order(action=action, orderType=type, lmtPrice=price, totalQuantity=qty, tif='GTC', transmit=True, outsideRth=True, notHeld=True)
    elif type == 'ALGO':
        # https://interactivebrokers.github.io/tws-api/ibalgos.html
        priority = 'Normal' # 'Urgent' 'Patient'
        order = Order(action=action, orderType='LMT', lmtPrice=price, totalQuantity=qty, tif='GTC', transmit=True, outsideRth=True)
        order.algoStrategy = 'Adaptive'
        order.algoParams = [].append(TagValue("adaptivePriority", priority))

    if ocaGroup:
        order.ocaGroup = ocaGroup
        order.ocaType = ocaType

    if order.totalQuantity > 0:
        return ib.placeOrder(contract, order)
    else: return None


def placeBracketOrder(ib, contract, action, qty, TP, SL, TP_qty=None, SL_qty=None, type="MKT", lmtPrice=0.0, trail=None, ocaGroup=None, ocaType=2):

    # voir https://interactivebrokers.github.io/tws-api/bracket_order.html

    # bracket = ib.bracketOrder(action='BUY', quantity=qty, limitPrice=price, takeProfitPrice=price+TP, stopLossPrice=price-SL)
    # bracket = ib.bracketOrder(action=action, quantity=qty, limitPrice=lmtPrice, takeProfitPrice=TP, stopLossPrice=SL)

    if not TP_qty: TP_qty = qty
    if not SL_qty: SL_qty = qty
    if SL: SL = round(SL, 2)
    if TP: TP = round(TP, 2)

    parent_order_id = ib.client.getReqId()

    parent_order = Order(orderId=parent_order_id, action=action, totalQuantity=qty, orderType=type, transmit=False, tif='GTC', outsideRth=True)
    if parent_order.orderType == "LMT": parent_order.lmtPrice = lmtPrice

    if not trail: SL_order = Order(orderId=parent_order_id+2, action=actionInv[action], totalQuantity=SL_qty, orderType="STP", auxPrice=SL, parentId=parent_order_id, transmit=True, tif='GTC', outsideRth=True)
    else: SL_order = Order(orderId=parent_order_id+2, action=actionInv[action], totalQuantity=SL_qty, orderType="TRAIL", trailStopPrice=SL, auxPrice=trail, parentId=parent_order_id, transmit=True, tif='GTC', outsideRth=True)
    TP_order = Order(orderId=parent_order_id+1, action=actionInv[action], totalQuantity=TP_qty, orderType="LMT", lmtPrice=TP, parentId=parent_order_id, transmit=False, tif='GTC', outsideRth=True)

    ocaGroup = datetime.datetime.now().strftime('%Y%m%d%H:%M:%S')
    # Ajouter de verifier si parent order placed (isActive) before sending TP and SL? otherwise can have TP and SL without main order

    if ocaGroup:
        # parent_order.ocaGroup = ocaGroup
        # parent_order.ocaType = ocaType
        SL_order.ocaGroup = ocaGroup
        SL_order.ocaType = ocaType
        TP_order.ocaGroup = ocaGroup
        TP_order.ocaType = ocaType

    print("Parent Order = ", parent_order)
    print("TP Order = ", TP_order)
    print("SL Order = ", SL_order)

    bracket = [parent_order, TP_order, SL_order]

    for o in bracket:
        ib.placeOrder(contract, o)


def placeTPSLOrders(ib, contract, action, qty, TP, SL, trail=None, ocaGroup=None, ocaType=2):

    # voir https://interactivebrokers.github.io/tws-api/oca.html

    order_id_SL = ib.client.getReqId()
    order_id_TP = ib.client.getReqId()

    if not ocaGroup: ocaGroup = datetime.datetime.now().strftime('%Y%m%d%H:%M:%S')

    TP_order, SL_order = None, None
    bracket = []
    if SL:
        SL = round(SL, 2)
        if not trail: SL_order = Order(orderId=order_id_SL, action=action, totalQuantity=qty, orderType="STP", auxPrice=SL, ocaGroup=ocaGroup, ocaType=ocaType, transmit=True, tif='GTC', outsideRth=True)
        else: SL_order = Order(orderId=order_id_SL, action=action, totalQuantity=qty, orderType="TRAIL", trailStopPrice=SL, auxPrice=trail, ocaGroup=ocaGroup, ocaType=ocaType, transmit=True, tif='GTC', outsideRth=True)
        bracket.append(SL_order)
    if TP:
        TP = round(TP, 2)
        TP_order = Order(orderId=order_id_TP, action=action, totalQuantity=qty, orderType="LMT", lmtPrice=TP, ocaGroup=ocaGroup, ocaType=ocaType, transmit=True, tif='GTC', outsideRth=True)
        bracket.append(TP_order)

    print("Placing ", TP_order.action," TP order ID ", TP_order.orderId, ": Qty ", TP_order.totalQuantity, ", lmtPrice ", TP_order.lmtPrice) if TP_order else print("No TP order")
    print("Placing ", SL_order.action," SL order ID ", SL_order.orderId, ": Qty ", SL_order.totalQuantity, ", auxPrice ", SL_order.auxPrice) if SL_order else print("No SL order")

    for o in bracket:
        ib.placeOrder(contract, o)


def autoOrder(ib, contract, direction, qty, data, offset_targets, TP='', SL='', order_type='Fast', TP_qty=None, SL_qty=None, trail=None, ocaGroup=None, ocaType=2):

    time_now_EST = helpers.date_local_to_EST(datetime.datetime.now())
    trading_hours = [helpers.date_to_EST_aware(datetime.datetime(time_now_EST.year, time_now_EST.month, time_now_EST.day, 9, 30)),
                  helpers.date_to_EST_aware(datetime.datetime(time_now_EST.year, time_now_EST.month, time_now_EST.day, 16, 00))]
    outside_market_hours = time_now_EST < trading_hours[0] or time_now_EST > trading_hours[1]

    action = "BUY" if direction == 1 else ("SELL" if direction == -1 else None)
    if action == 'BUY': lmtPrice = data.ask if not order_type == 'Limit Wait' else data.bid
    elif action == 'SELL': lmtPrice = data.bid if not order_type == 'Limit Wait' else data.ask
    action_inv = {"BUY": "SELL", "SELL": "BUY"}

    if not order_type == 'Limit Wait': lmtPrice = lmtPrice + direction * offset_targets * helpers.get_tick_value(lmtPrice)

    if order_type == 'Pegged Best': orderType = 'PEG BEST'
    if order_type == 'Pegged Mid': orderType = 'PEG MID'
    elif order_type == 'Adaptive Algo': orderType = 'ALGO'
    elif order_type == 'Limit Wait' or (order_type == 'Fast' and outside_market_hours): orderType = "LMT"
    elif order_type == 'Fast' and not outside_market_hours: orderType = "MKT"


    # if TP == '' and SL == '':
    #     placeOrder(ib, contract, action, qty, type=orderType, price=lmtPrice, ocaGroup=ocaGroup, ocaType=ocaType)
    # else:
    #     placeBracketOrder(ib, contract, action, qty, TP, SL, TP_qty=TP_qty, SL_qty=SL_qty, type=orderType, lmtPrice=lmtPrice, ocaGroup=ocaGroup, ocaType=ocaType)

    order = placeOrder(ib, contract, action, qty, type=orderType, price=lmtPrice, trail=trail, ocaGroup=ocaGroup, ocaType=ocaType)
    TPSL_order = placeTPSLOrders(ib, contract, action_inv[action], qty, TP, SL, trail=trail, ocaGroup=ocaGroup, ocaType=ocaType)
    # if TP != '': placeOrder(ib, contract, action_inv[action], qty, type='LMT', price=TP, isPT=True, ocaGroup=ocaGroup, ocaType=ocaType)
    # if SL != '': placeOrder(ib, contract, action_inv[action], qty, type='STP', price=SL, ocaGroup=ocaGroup, ocaType=ocaType)
    return order, TPSL_order


def get_bracket_orders(ib, contract, action):

    open_orders = get_active_orders(ib, symbol=contract.symbol)
    # for order in open_orders:
    #     print('open_orders = ', order, "\n")
    # input()
    ib.sleep(CONSTANTS.PROCESS_TIME['short'])

    SL_orders = [order for order in open_orders if order.orderType == "STP" and order.action == action]# and order.parentId]
    TP_orders = [order for order in open_orders if order.orderType == "LMT" and order.action == action]# and order.parentId]

    # print("-------------------------------")
    # for o in open_orders:
    #     print(o)
    #     print()
    # print()
    # print(SL_orders)
    # print(TP_orders)
    # input()

    return TP_orders, SL_orders


def get_ext_bracket_order(orders, fct):

    if fct == "min": return min(orders, key=lambda x: x.orderId)
    elif fct == "max": return max(orders, key=lambda x: x.orderId)
    else:
        print("Function must be 'min' or 'max'")
        return None


def create_bracket_orders(ib, contract, data, quantity, TP, SL, trail=None, partial=False, partial_perc=0):

    open_position = get_positions_by_symbol(ib, contract.symbol)

    # Determining Direction
    action = None
    if open_position:
        if open_position[0].position < 0: action = "BUY"
        elif open_position[0].position > 0: action = "SELL"
    elif TP and SL:
            if TP < SL: action = "BUY"
            elif TP > SL: action = "SELL"
    # elif TP or SL:
        # action = sg.popup('Choose order direction', button_type=sg.POPUP_BUTTONS_YES_NO, custom_text=('BUY ','SELL'))
    else:
        action = None

    if not action:
        print("Could not determine a direction for the bracket order.")
    elif not partial:
        placeTPSLOrders(ib, contract, action, quantity, TP, SL, trail=trail)
    else:

        try:
            open_position = open_position[0]

            partial_perc = abs(float(partial_perc))
            # bracket_params[0]["quantity"] = round(abs(float(values["-quantity-"])) * partial_perc / 100)

            adjust_bracket_orders(ib, contract)
            ib.sleep(CONSTANTS.PROCESS_TIME['medium'])

            # Get last take profit and stop loss
            TP_orders, SL_orders = get_bracket_orders(ib, contract, action)
            last_SL_order = get_ext_bracket_order(SL_orders, "max") if SL_orders else None
            last_TP_order = get_ext_bracket_order(TP_orders, "max") if TP_orders else None

            print("----------------------------------------")
            print("last_TP_order = ", last_TP_order)
            print()
            print("last_SL_order = ", last_SL_order)
            print("----------------------------------------")

            partial_quantity = round(abs(float(last_TP_order.totalQuantity)) * partial_perc / 100)

            # if SL_orders:
            #     stop_loss = get_ext_bracket_order(SL_orders, "max").auxPrice
            # if TP_orders: take_profit = get_ext_bracket_order(TP_orders, "max").lmtPrice
            # Cancel current bracket and replace by partial one
            cancel_order_by_id(ib, last_SL_order.orderId)
            cancel_order_by_id(ib, last_TP_order.orderId)
            placeTPSLOrders(ib, contract, action, partial_quantity, last_TP_order.lmtPrice, last_SL_order.auxPrice, trail=trail)
            ib.sleep(CONSTANTS.PROCESS_TIME['medium'])

            # print("\n\ndata = ", data)
            # print("action = ", action)
            # print("last_TP_order.lmtPrice = ", last_TP_order.lmtPrice)
            # print("last_SL_order.lmtPrice = ", last_SL_order.auxPrice)
            # input()

            # Create new bracket with remaining quantity
            if action == "BUY": take_profit = last_TP_order.lmtPrice + (last_TP_order.lmtPrice - float(data.ask)) / 2
            elif action == "SELL": take_profit = last_TP_order.lmtPrice - (float(data.bid) - last_TP_order.lmtPrice) / 2
            placeTPSLOrders(ib, contract, action, last_TP_order.totalQuantity - partial_quantity, take_profit, last_SL_order.auxPrice, trail=trail)

            # adjust_bracket_orders(ib, contract)

        except Exception as e:
            print(f"Failed to create partial bracket orders: {e}")







        # bracket_params =[{"quantity": values["-quantity-"], "TP": values["-take_profit-"], "SL": values["-stop_loss-"]}]

        # # Case Create Partial Brackets
        # if partial:
        #     try:
        #         partial_perc = abs(float(values["-partial_percentage-"]))
        #         bracket_params[0]["quantity"] = round(abs(float(values["-quantity-"])) * partial_perc / 100)

        #     except Exception as e:
        #         print(f"Failed to parse Partial % or Quantity: {e}")

        # for params in bracket_params:
        #     orders.placeTPSLOrders(ib, contract, action, params["quantity"], params["TP"], params["SL"])


# def trim_bracket_orders(ib, contract, adjust_SL=True, adjust_TP=True):
def adjust_bracket_orders(ib, symbol, currency=CONSTANTS.DEFAULT_CURRENCY):#, new_qty):#, adjust_SL=True, adjust_TP=True):

    contract, mktData = helpers.get_symbol_mkt_data(ib, symbol, currency=currency)

    delta_TP, delta_SL = 0, 0
    last_TP_order, last_SL_order = None, None

    ib.sleep(CONSTANTS.PROCESS_TIME['short'])
    open_position = get_positions_by_symbol(ib, contract.symbol)

    if open_position:
        open_position = open_position[0]
        print("open_position = ", open_position.position)
        # projected_position = abs(open_position.position) + new_qty

        # Check exisiting stop losses and take profits
        # open_orders = get_orders_by_status(ib, status=["PreSubmitted", "Submitted"], symbol=contract.symbol)
        # open_orders = get_active_orders(ib, symbol=contract.symbol)
        # ib.sleep(CONSTANTS.PROCESS_TIME['short'])

        action = "BUY" if open_position.position < 0 else "SELL"
        action_inverse = "BUY" if open_position.position >= 0 else "SELL"
        stop_loss, take_profit = None, None

        # Cancel all stop losses and take profits opposite to current open position
        TP_orders_inverse, SL_orders_inverse = get_bracket_orders(ib, contract, action_inverse)
        for order in SL_orders_inverse + TP_orders_inverse:
            cancel_order_by_id(ib, order.orderId)


        # Equalize number of SL and TP orders
        # TP_orders, SL_orders = get_bracket_orders(ib, contract, action)
        # for SL_order in SL_orders:
        #     if SL_order.ocaGroup not in [TP_order.ocaGroup for TP_order in TP_orders]:
        #         cancel_order_by_id(ib, SL_order.orderId)

        # TP_orders, SL_orders = get_bracket_orders(ib, contract, action)
        # for TP_order in TP_orders:
        #     if TP_order.ocaGroup not in [SL_order.ocaGroup for SL_order in SL_orders]:
        #         cancel_order_by_id(ib, TP_order.orderId)


        # Verify if total stop losses match current position
        print("------------- Checking stop losses -------------")
        TP_orders, SL_orders = get_bracket_orders(ib, contract, action)
        total_SL_qty = sum(order.totalQuantity for order in SL_orders)
        total_TP_qty = sum(order.totalQuantity for order in TP_orders)
        if SL_orders: stop_loss = get_ext_bracket_order(SL_orders, "max").auxPrice
        if TP_orders: take_profit = get_ext_bracket_order(TP_orders, "max").lmtPrice
        print("Total SL = ", total_SL_qty)
        print("Total TP = ", total_TP_qty, '\n')

        while total_SL_qty > abs(open_position.position):
            TP_orders, SL_orders = get_bracket_orders(ib, contract, action)
            # total_SL_qty = sum(order.totalQuantity for order in SL_orders)
            if SL_orders: oldest_SL_order = get_ext_bracket_order(SL_orders, "min")
            if TP_orders: oldest_TP_order = get_ext_bracket_order(TP_orders, "min")
            else: break
            # print("-----------------------------------------------------")
            # print("total_SL_qty = ", total_SL_qty)
            # print("open.position = ", open_position.position)
            # print("cancelling order = ", oldest_SL_order)
            # # stop_loss = oldest_SL_order.auxPrice
            # # take_profit = oldest_TP_order.lmtPrice
            # cancel_order_by_id(ib, oldest_SL_order.orderId)
            # total_SL_qty -= oldest_SL_order.totalQuantity
            # print("New total_SL_qty = ", total_SL_qty)
            # print("-----------------------------------------------------")
            # input()
        print("Total SL after trimming = ", total_SL_qty)
        print("Total TP = ", total_TP_qty, '\n')
        print("Done with SL")
        # input()

        # delta_SL = projected_position - total_SL_qty


        # Verify if total take profits match current position and is not bigger than total SL orders
        print("------------- Checking take profits -------------")
        TP_orders, SL_orders = get_bracket_orders(ib, contract, action)
        total_TP_qty = sum(order.totalQuantity for order in TP_orders)
        print("Total TP = ", total_TP_qty, '\n')
        while total_TP_qty > abs(open_position.position) or total_TP_qty > total_SL_qty:
            TP_orders, SL_orders = get_bracket_orders(ib, contract, action)
            if TP_orders:
                oldest_TP_order = min(TP_orders, key=lambda x: x.orderId)
            else: break
            # print("-----------------------------------------------------")
            # print("total_TP_qty = ", total_TP_qty)
            # print("open.position = ", open_position.position)
            # print("cancelling order = ", oldest_TP_order)
            # # take_profit = oldest_TP_order.lmtPrice
            # cancel_order_by_id(ib, oldest_TP_order.orderId)
            # total_TP_qty -= oldest_TP_order.totalQuantity
            # print("New total_TP_qty = ", total_TP_qty)
            # print("-----------------------------------------------------")
            # total_TP_qty = sum(order.totalQuantity for order in TP_orders)

        print("Total TP after trimming = ", total_TP_qty, '\n')
        # delta_TP = projected_position - total_TP_qty
        # Final assessment
        print("------------- Final Assessment -------------")
        TP_orders, SL_orders = get_bracket_orders(ib, contract, action)
        total_SL_qty = sum(order.totalQuantity for order in SL_orders)
        total_TP_qty = sum(order.totalQuantity for order in TP_orders)
        delta_TP = abs(open_position.position) - total_TP_qty
        delta_SL = abs(open_position.position) - total_SL_qty
        print("Total SL reassessed = ", total_SL_qty)
        print("Total TP reassessed = ", total_TP_qty, '\n')
        print("Delta SL = ", delta_SL)
        print("Delta TP = ", delta_TP, '\n')

        # Calculate new last TP and SL orders
        # TP_orders, SL_orders = get_bracket_orders(ib, contract, action)
        last_SL_order = get_ext_bracket_order(SL_orders, "max") if SL_orders else None
        last_TP_order = get_ext_bracket_order(TP_orders, "max") if TP_orders else None
        trail = last_SL_order.auxPrice if last_SL_order and last_SL_order.orderType == 'TRAIL' else None

        # Create new stop loss if total stop losses smaller than current position
        if total_SL_qty < abs(open_position.position) and last_SL_order:
            stop_loss = last_SL_order.auxPrice

        # Create new take profit if total take profits smaller than current position
        if total_TP_qty < abs(open_position.position) and last_TP_order:
            take_profit = last_TP_order.lmtPrice

        print("delta_SL = ", delta_SL)
        print("delta_TP = ", delta_TP)
        print("stop_loss = ", stop_loss)
        print("take_profit = ", take_profit)
        if min(delta_SL, delta_TP) > 0: placeTPSLOrders(ib, contract, action, min(delta_SL, delta_TP), take_profit, stop_loss, trail=trail)

        # If still missing SL or TP
        delta_SL -= min(delta_SL, delta_TP)
        delta_TP -= min(delta_SL, delta_TP)
        if delta_SL > 0: placeOrder(ib, contract, action, delta_SL, type='STP', price=stop_loss)
        elif delta_TP > 0: placeOrder(ib, contract, action, delta_TP, type='LMT', price=take_profit)

    else:
        print("No open position for symbol ", contract.symbol, ". Closing all pending orders...")
        cancel_orders_by_symbol(ib, contract.symbol)
        ib.sleep(CONSTANTS.PROCESS_TIME['short'])

    # return delta_TP, delta_SL, last_TP_order, last_SL_order


def adjust_bracket_orders2(ib, contract):#, new_qty):#, adjust_SL=True, adjust_TP=True):

    contract, mktData = helpers.get_symbol_mkt_data(ib, contract.symbol, currency=contract.currency)

    delta_TP, delta_SL = 0, 0
    last_TP_order, last_SL_order = None, None

    ib.sleep(CONSTANTS.PROCESS_TIME['short'])
    open_position = get_positions_by_symbol(ib, contract.symbol)

    if open_position:
        open_position = open_position[0]
        print("open_position = ", open_position.position)
        # projected_position = abs(open_position.position) + new_qty

        # Check exisiting stop losses and take profits
        # open_orders = get_orders_by_status(ib, status=["PreSubmitted", "Submitted"], symbol=contract.symbol)
        # open_orders = get_active_orders(ib, symbol=contract.symbol)
        # ib.sleep(CONSTANTS.PROCESS_TIME['short'])

        action = "BUY" if open_position.position < 0 else "SELL"
        action_inverse = "BUY" if open_position.position >= 0 else "SELL"
        stop_loss, take_profit = None, None

        # Cancel all stop losses and take profits opposite to current open position
        TP_orders_inverse, SL_orders_inverse = get_bracket_orders(ib, contract, action_inverse)
        for order in SL_orders_inverse + TP_orders_inverse:
            cancel_order_by_id(ib, order.orderId)

        bo_list, TP_orders, SL_orders = record_bracket(ib, contract, action=action)
        for bo in bo_list:
            print(bo)
        sum_SL_orders = sum([bo['qty'] for bo in bo_list if bo['SL']])
        sum_TP_orders = sum([bo['qty'] for bo in bo_list if bo['TP']])

        diff_SL = sum_SL_orders - open_position.position

        # while sum_SL_orders > open_position.position:
        print("sum_SL_orders = ", sum_SL_orders)
        print("sum_TP_orders = ", sum_TP_orders)
        input()


    else:
        print("No open position for symbol ", contract.symbol, ". Closing all pending orders...")
        cancel_orders_by_symbol(ib, contract.symbol)
        ib.sleep(CONSTANTS.PROCESS_TIME['short'])


def record_bracket(ib, contract, action=None):

    if not action:
        open_position = get_positions_by_symbol(ib, contract.symbol)
        if open_position: action = "SELL" if open_position[0].position > 0 else "BUY"

    BO_list = []
    if action:
        TP_orders, SL_orders = get_bracket_orders(ib, contract, action)
        TP_orders_dict = [{'qty': TP_order.totalQuantity, 'TP': TP_order.lmtPrice, 'SL': '', 'trail': None, 'oca': TP_order.ocaGroup, 'id': TP_order.orderId} for TP_order in TP_orders]
        SL_orders_dict = []
        for SL_order in SL_orders:
            if SL_order.orderType == 'TRAIL':
                SL_orders_dict.append({'qty': SL_order.totalQuantity, 'TP': '', 'SL': SL_order.trailStopPrice, 'trail': SL_order.auxPrice, 'oca': SL_order.ocaGroup, 'id': SL_order.orderId})
            else: SL_orders_dict.append({'qty': SL_order.totalQuantity, 'TP': '', 'SL': SL_order.auxPrice, 'trail': None, 'oca': SL_order.ocaGroup, 'id': SL_order.orderId})
        BO_list = TP_orders_dict + SL_orders_dict

    return BO_list, TP_orders, SL_orders


def recreate_bracket(ib, contract, bo_list, action='auto', qty_factor=1, TP=None, SL=None):

    open_position = get_positions_by_symbol(ib, contract.symbol)
    print("\nopen_postion = ", open_position)

    if open_position:

        open_position = open_position[0]
        if action == 'auto': action = "SELL" if open_position.position > 0 else "BUY"

        for bo_order in bo_list:
            quantity = round(bo_order["qty"] * qty_factor, 0)
            take_profit = bo_order["TP"] if not TP else TP
            stop_loss = bo_order["SL"] if not SL else SL
            trail = bo_order["trail"]
            placeTPSLOrders(ib, contract, action, qty=quantity, TP=take_profit, SL=stop_loss, trail=trail)


def move_SL_to_BE(ib, symbol, offset, trail=None, currency=CONSTANTS.DEFAULT_CURRENCY):

    contract, mktData = helpers.get_symbol_mkt_data(ib, symbol, currency=currency)
    open_position = get_positions_by_symbol(ib, symbol)

    if mktData and open_position:
        open_position = open_position[0]
        action = "SELL" if open_position.position > 0 else "BUY"
        direction = 1 if open_position.position > 0 else -1
        price = mktData.bid if direction == 1 else (mktData.ask if direction == -1 else None)

        try:
            offset_targets = -float(offset) if action == "BUY" else float(offset)
            avgPosition = abs(round(float(open_position.avgCost), 2))
            avgPosition += offset_targets * helpers.get_tick_value(avgPosition)

            if direction == 1 and avgPosition < price or direction == -1 and avgPosition > price:

                # TP_orders, SL_orders = get_bracket_orders(ib, contract, action)
                bo_list, TP_orders, SL_orders = record_bracket(ib, contract, action=action)
                for bo in bo_list:
                    if bo['SL']: bo['SL'] = avgPosition

                cancel_orders_by_symbol(ib, symbol)
                ib.sleep(CONSTANTS.PROCESS_TIME['short'])

                if not SL_orders:
                    placeTPSLOrders(ib, contract, action, qty=abs(open_position.position), TP='', SL=avgPosition, trail=trail)
                
                    if TP_orders:
                        print("\nRecreating brackets...")
                        recreate_bracket(ib, contract, bo_list, TP=None, SL=None)

                else:
                    print("\nRecreating brackets...")
                    recreate_bracket(ib, contract, bo_list, TP=None, SL=None)

            else: print("Cannot move Stop Loss to break even as avgPrice ", "higher" if direction == 1 else "lower", " than current ", "ask" if direction == 1 else "bid")

        except Exception as e:
            print(f"Failed to move Stop Loss to Break Even: {e}")

    else:
        print("Could not fetch market data or no open position")

# Position(account='U14264367', contract=Stock(conId=324651164, symbol='QUBT', exchange='NASDAQ', currency=CONSTANTS.DEFAULT_CURRENCY, localSymbol='QUBT', tradingClass='SCM'), position=5.0, avgCost=18.31632345)


def get_orders_by_status(ib, status, symbol=None):

    printFunct = lambda trade: print("Order found with Id ", trade.order.orderId, " and status ", trade.orderStatus.status)

    if not symbol:
        orders = [trade.order for trade in ib.trades() if trade.orderStatus and trade.orderStatus.status in status]# and not printFunct(trade)]

    else:

        orders = []
        for trade in ib.trades():

            sym = trade.contract.localSymbol if trade.contract.localSymbol else ''
            if "." in sym: # Case Forex symbol
                sym = sym[0:3]

            if trade.order and (sym == symbol or symbol == "all") and trade.orderStatus and trade.orderStatus.status in status:
                orders.append(trade.order)
                printFunct(trade)

    return orders


def get_active_orders(ib, symbol):
    printFunct = lambda trade: print("Order found with Id ", trade.order.orderId, " and status ", trade.orderStatus.status)

    orders = []
    for trade in ib.trades():

        sym = trade.contract.localSymbol if trade.contract.localSymbol else ''
        if "." in sym: # Case Forex symbol
            sym = sym[0:3]

        if trade.order and (sym == symbol or symbol == "all" or symbol == None) and trade.isActive():
            orders.append(trade.order)

            # printFunct(trade)

    return orders


def get_order_by_id(ib, orderId):

    order = [trade.order for trade in ib.trades() if trade.order and str(trade.order.orderId) == str(orderId)] or [order for order in ib.orders() if order and str(order.orderId) == str(orderId)]

    if not order: print("Did not find open order ", orderId)

    return order


def cancel_orders_by_symbol(ib, symbol, status=None):

    # if not status: status=["PreSubmitted", "Submitted"]

    # for order in get_orders_by_status(ib, status=status, symbol=symbol):
    for order in get_active_orders(ib, symbol=symbol):
        cancel_order_by_id(ib, order.orderId)


def cancel_order_by_id(ib, orderId):
    if orderId == "all":
        ib.reqGlobalCancel()
        ib.sleep(CONSTANTS.PROCESS_TIME['short'])
    else:
        order = get_order_by_id(ib, orderId)

        if order:
            print("\nCancelling order ", orderId)
            ib.cancelOrder(order[-1])
            ib.sleep(CONSTANTS.PROCESS_TIME['short'])


def get_positions_by_symbol(ib, symbol):

    # positions = ib.positions()
    # ib.sleep(CONSTANTS.PROCESS_TIME['short'])

    # sym = trade.contract.localSymbol if trade.contract.localSymbol else ''
    # if "." in sym: # Case Forex symbol
    #     sym = sym[0:3]

    # print("/////////////////////////////////////////////////")
    # print(positions)
    # print("symbol = ", symbol)
    # print("/////////////////////////////////////////////////")

    # # return [position for position in positions if position.contract.symbol == symbol or symbol == "all"]
    positions = []
    for position in ib.positions():
        sym = position.contract.localSymbol if position.contract.localSymbol else ''
        if "." in sym: # Case Forex symbol
            sym = sym[0:3]

        # if position.contract.symbol == sym or symbol == "all":
        if symbol == sym or symbol == "all":
            positions.append(position)

    return positions


def trade_event_callback(trade):

    return trade.contract.symbol, trade.order.order_typee, trade.orderStatus.filled, trade.orderStatus.remaining



actionInv = {'BUY': 'SELL', 'SELL': 'BUY'}

if __name__ == '__main__':

    import helpers

    args = sys.argv

    data = json.loads(args[1].replace("'", "\""))
    print("data = ", data)

    ib, ibConnection = helpers.IBKRConnect(IB())

    ticker = data['ticker']
    exchange = data['ticker'][:data['ticker'].find(":")]
    symbol = data['ticker'][data['ticker'].find(":")+1:]
    profit_target = data['profit_target'][:7]
    stop_loss = data['stop_loss'][:7]

    contract, mktData = helpers.get_symbol_mkt_data(ib, symbol, exchange, data["currency"])

    autoOrder(ib, contract, data['action'], data['quantity'], mktData, profit_target, stop_loss)

    # if data['action'] == 'BUY':
    #     # placeBracketOrder(data['action'], data['quantity'], profit_target, stop_loss)
    #     placeOrder(data['action'], data['quantity'])
    # elif data['action'] == 'SELL':
    #     placeOrder(data['action'], data['quantity'])


    # placeOrder(data['action'], data['quantity'])
    # placeOrder('BUY', data['quantity'], type='MTL')

    # contract = Stock(symbol=data['ticker'], exchange='SMART', currency=data["currency"])


    # contract = Forex('USDCAD')
    # ib.qualifyContracts(contract)
    # print("contract = ", contract)
    # mktData = ib.reqMktData(contract = contract, genericTickList='',snapshot=False, regulatorySnapshot=False)
    # print("mktData = ", mktData)

    # contract = Contract(symbol='EUR', exchange='IDEALPRO', currency='CAD')
    # contract.SecType = "CASH"
    # print("contract = ", contract)
    # mktData = ib.reqHistoricalData(1, contract=contract, endDateTime='', durattionString='1 H', barSizeSetting='10 sec', whatToShow='MIDPOINT', useRTH=True, formatDate="yyyMMdd",keepUpToDate=True)

    print("\n\n")







# {
#     "ticker": "{{ticker}}",
#     "currency": "{{syminfo.currency}}",
#     "action": "{{strategy.order.action}}",
#     "direction": "{{strategy.market_position}}",
#     "quantity": "{{strategy.order.contracts}}",
#     "price": "{{close}}",
#     "timeframe": "{{interval}}",
#     "data": "{{strategy.order.alert_message}}"
# }
