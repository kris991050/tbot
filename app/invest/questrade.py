import os, sys, copy, prettytable, traceback#, time, math, pprint
from matplotlib import pyplot
from questrade_api import Questrade


# Doc Questrade API: https://pypi.org/project/questrade-api/
# Doc Security and Authorizations: https://www.questrade.com/api/documentation/authorization


def path_current_setup(path_current_file, ch_dir=True):

    # Current Directory Path Setup
    #name = os.path.basename(__file__)
    path_current = os.path.dirname(path_current_file)
    if ch_dir: os.chdir(path_current)
    print("\nCurrent Directory = ", os.getcwd(), "\n")

    return path_current

# https://login.questrade.com/oauth2/authorize?client_id=lRhSkqZB88pSLf04iDyUBxF4CQAqpQ&response_type=code&redirect_uri=https://www.example.com

# ACfJ8Y2v39kMobDIG5kYpAuAV8SKjPON0

# https://login.questrade.com/oauth2/authorize?client_id=lRhSkqZB88pSLf04iDyUBxF4CQAqpQ&response_type=token&redirect_uri=https://www.example.com

# https://www.example.com/#access_token=1Z1bLTXSvQRIyIYEQjYqZJgm8uV7R1840&refresh_token=cw-YBTOYqtNKClsiJtcm-tH_V_q2Q6630&token_type=Bearer&expires_in=1800&api_server=https://api03.iq.questrade.com/
# https://www.example.com/#access_token=UvgEh87Il91d2uRao5FVnaVSjoVBd9XB0&refresh_token=-xpMF5a8gTgFyENXJEYSuCJndbB3R8BM0&token_type=Bearer&expires_in=1800&api_server=https://api03.iq.questrade.com/
# https://www.example.com/#access_token=yVOPZLmf7VRLBJ8GXX_uNdPjWjzxPkTZ0&refresh_token=RCUFdhkc2sMwRBeYvc3SFSKTbYcwoUVG0&token_type=Bearer&expires_in=1800&api_server=https://api05.iq.questrade.com/


# access_token: yVOPZLmf7VRLBJ8GXX_uNdPjWjzxPkTZ0
# refresh_token: RCUFdhkc2sMwRBeYvc3SFSKTbYcwoUVG0
# token_type: Bearer
# expires_in: 1800
# api_server: https://api03.iq.questrade.com/

def calculate_positions(account_positions, totalMarketValue, allocations):

    positions_list = []
    if account_positions:
        for p in account_positions:
            percentage = round(100 * p['currentMarketValue'] / totalMarketValue, 2) if p['currentMarketValue'] else 0
            allocation = list(filter(lambda allocation: allocation['symbol'] == p['symbol'], allocations))
            if allocation:
                allocation = allocation[0]
                positions_list.append({
                    'symbol': p['symbol'],
                    'openQuantity': p['openQuantity'],
                    'currentMarketValue': p['currentMarketValue'],
                    'currentPrice': p['currentPrice'],
                    'percentage': percentage,
                    'precentage_from_allocation': round(100 * percentage / allocation['percentage'], 2)
            })
    else: print("\nNo positon found\n")

    return positions_list


# def balance_actions(q, account_title, account_number, allocations, currency, plot=True):
def balance_actions(account_positions, account_balances, account_title, account_number, allocations, currency, plot=True):

    # account_positions = q.account_positions(account_number)
    # balances_current = q.account_balances(account_number)['perCurrencyBalances']
    # balances = q.account_balances(account_number)['combinedBalances']

    balances = account_balances['combinedBalances']

    # balance_CAD = list(filter(lambda bal: bal['currency'] == 'CAD', balances_current))[0]['cash']
    # balance_USD = list(filter(lambda bal: bal['currency'] == 'USD', balances_current))[0]['cash']
    balance = list(filter(lambda bal: bal['currency'] == currency, balances))[0]['cash']

    # if account_title == 'FHSA': balance = round(balance -4000, 2)

    # print(sum([p['currentMarketValue'] for p in account_positions['positions']]))
    totalMarketValue = sum([p['currentMarketValue'] for p in account_positions['positions'] if p['currentMarketValue'] is not None]) + balance

    # if currency == "USD":
    #     print(account_positions)
    #     print()
    #     for p in account_positions['positions']:
    #         print("-----------------------------------")
    #         print(p)
    #     input()
    positions = calculate_positions(account_positions['positions'], totalMarketValue, allocations)

    # Calculate balancing actions
    # ---------------------------

    if positions:

        dynamic_balance = balance
        dynamic_totalMarketValue = round(totalMarketValue, 2)
        positions_to_balance = sorted(copy.deepcopy(positions), key=lambda p: p['precentage_from_allocation'])

        # for p in positions_to_balance:
        #     print(p, '\n')

        while dynamic_balance > 0:
            p = positions_to_balance[0]
            # value_to_buy = dynamic_totalMarketValue * p['percentage'] * ((100 / p['precentage_from_allocation']) - 1) / 100
            # shares_to_buy = math.floor(value_to_buy / p['currentPrice']) + 1
            shares_to_buy = 1
            cost = round(shares_to_buy * p['currentPrice'], 2)
            if dynamic_balance >= cost:
                dynamic_balance = round(dynamic_balance - cost, 2)
                # dynamic_totalMarketValue = round(dynamic_totalMarketValue - cost, 2)
                positions_to_balance[0]['openQuantity'] = round(positions_to_balance[0]['openQuantity'] + shares_to_buy, 2)
                positions_to_balance[0]['currentMarketValue'] = round(positions_to_balance[0]['currentMarketValue'] + cost, 2)
                positions_to_balance = calculate_positions(positions_to_balance, dynamic_totalMarketValue, allocations)
                positions_to_balance = sorted(positions_to_balance, key=lambda p: p['precentage_from_allocation'])

            else:
                break


        new_positions = sorted(positions_to_balance, key=lambda p: p['percentage'], reverse=True)
        # pprint.pprint(new_positions)
        # print()

        actions = []
        old_positions = sorted(positions, key=lambda p: p['symbol'])
        new_positions = sorted(new_positions, key=lambda p: p['symbol'])
        for index, p in enumerate(new_positions):
            actions.append({
                'symbol': p['symbol'],
                'shares_to_buy': p['openQuantity'] - old_positions[index]['openQuantity']
            })

    else:
        print("No position found for account number ", account_number)
        new_positions = []

    # Print and plot:
    # ---------------

    print("\nAccount Number: ", account_number, "\n")
    # Define Data
    if positions and plot:
        positions = sorted(positions, key=lambda p: p['percentage'], reverse = True)

        print("Old Balance = ", balance)
        print("Old Total Market Value = ", totalMarketValue, "\n\n")

        for p in positions:
            print(p['symbol'], " -  Qty: ", p['openQuantity'], " | Mkt Value: ", p["currentMarketValue"], " | ", p["percentage"], " %")
        cash_percentage = round(100 * balance / totalMarketValue, 2)
        print("Cash -  Balance: ", balance, " | ", cash_percentage, " %")

        print("\n\nActions:")
        # pprint.pprint(actions)
        for a in actions:
            print("Buy ", a['shares_to_buy'], " shares of ", a['symbol'])

        print("\n\nNew Balance = ", dynamic_balance)
        print("New Total Market Value = ", dynamic_totalMarketValue)

        # Plot current positions on pie chart and print current positions details

        # positions_symbol = [p['symbol'] + " - " + str(p['percentage']) + "%" for p in positions]
        positions_symbol = [p['symbol'] for p in positions]
        # positions_symbol.append('CASH' + " - " + str(cash_percentage) + "%")
        positions_symbol.append('CASH')
        positions_percentage = [p['percentage'] for p in positions]
        positions_percentage.append(cash_percentage)

        new_positions = sorted(new_positions, key=lambda p: p['percentage'], reverse = True)
        for p in new_positions:
            print(p['symbol'], " -  Qty: ", p['openQuantity'], " | Mkt Value: ", p["currentMarketValue"], " | ", p["percentage"], " %")
        print("Cash -  Balance: ", dynamic_balance, " | ", round(100 * dynamic_balance / dynamic_totalMarketValue, 2), " %")


        # Define Plots
        fig, (sub_fig1, sub_fig2) = pyplot.subplots(nrows=1, ncols=2, figsize=(9, 6), gridspec_kw={'width_ratios': [3, 1]})
        fig.tight_layout(pad=5.0)
        fig.suptitle(account_title)
        sub_fig1.pie(positions_percentage, labels=positions_symbol, counterclock=False, startangle=90, rotatelabels=True)

        # Define Data Table
        table_data = [[p['symbol'], p['openQuantity'], p['currentMarketValue'], p['percentage']] for p in positions]
        table_data.append(["Cash", "-", balance, cash_percentage])
        table_data.append(["Total", sum([p["openQuantity"] for p in positions]), dynamic_totalMarketValue, round(sum(positions_percentage), 2)])


        columns = ('Symbol', '#', currency, '%')
        col_widths = [0.5, 0.3, 0.5, 0.4]

        sub_fig2.table(cellText=table_data, colLabels=columns, colWidths=col_widths, loc="center")
        sub_fig2.axis("off")

        # pyplot.show(block=False)
        # pyplot.pause(0.001

    else:
        if plot: print("No position to plot")


def get_dummy_accounts():
    q_accounts = {'accounts': [{'type': 'TFSA', 'number': '53322283', 'status': 'Active', 'isPrimary': False, 'isBilling': False, 'clientAccountType': 'Individual'}, {'type': 'RRSP', 'number': '53311600', 'status': 'Active', 'isPrimary': False, 'isBilling': False, 'clientAccountType': 'Individual'}, {'type': 'FHSA', 'number': '53209246', 'status': 'Active', 'isPrimary': True, 'isBilling': True, 'clientAccountType': 'Individual'}], 'userId': 2055617}
    return q_accounts


def get_dummy_accounts_info(account_type):
    if account_type == 'FHSA':
        account_positions = {'positions': [{'symbol': 'KILO.B.TO', 'symbolId': 23563025, 'openQuantity': 18, 'closedQuantity': 0, 'currentMarketValue': 821.34, 'currentPrice': 45.63, 'averageEntryPrice': 39.175, 'dayPnl': -10.8, 'closedPnl': 0, 'openPnl': 116.19, 'totalCost': 705.15, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'VRE.TO', 'symbolId': 2874668, 'openQuantity': 10, 'closedQuantity': 0, 'currentMarketValue': 307.8, 'currentPrice': 30.78, 'averageEntryPrice': 29.711, 'dayPnl': -2, 'closedPnl': 0, 'openPnl': 10.69, 'totalCost': 297.11, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'ZAG.TO', 'symbolId': 9029, 'openQuantity': 148, 'closedQuantity': 0, 'currentMarketValue': 2023.16, 'currentPrice': 13.67, 'averageEntryPrice': 13.699798, 'dayPnl': -6.66, 'closedPnl': 0, 'openPnl': -4.410104, 'totalCost': 2027.570104, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'ZFL.TO', 'symbolId': 9037, 'openQuantity': 155, 'closedQuantity': 0, 'currentMarketValue': 1934.4, 'currentPrice': 12.48, 'averageEntryPrice': 13.077227, 'dayPnl': -11.625, 'closedPnl': 0, 'openPnl': -92.570185, 'totalCost': 2026.970185, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'XCSR.TO', 'symbolId': 30232426, 'openQuantity': 42, 'closedQuantity': 0, 'currentMarketValue': 3102.96, 'currentPrice': 73.88, 'averageEntryPrice': 65.212143, 'dayPnl': -24.78, 'closedPnl': 0, 'openPnl': 364.049994, 'totalCost': 2738.910006, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'HXS.TO', 'symbolId': 54301062, 'openQuantity': 54, 'closedQuantity': 0, 'currentMarketValue': 4652.64, 'currentPrice': 86.16, 'averageEntryPrice': 75.406112, 'dayPnl': -1.08, 'closedPnl': 0, 'openPnl': 580.709952, 'totalCost': 4071.930048, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'SVR.C.TO', 'symbolId': 2869421, 'openQuantity': 16, 'closedQuantity': 0, 'currentMarketValue': 260.32, 'currentPrice': 16.27, 'averageEntryPrice': 14.92625, 'dayPnl': -7.04, 'closedPnl': 0, 'openPnl': 21.5, 'totalCost': 238.82, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'XIC.TO', 'symbolId': 23995, 'openQuantity': 77, 'closedQuantity': 0, 'currentMarketValue': 3013.78, 'currentPrice': 39.14, 'averageEntryPrice': 35.761558, 'dayPnl': -28.875, 'closedPnl': 0, 'openPnl': 260.140034, 'totalCost': 2753.639966, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'XRE.TO', 'symbolId': 24016, 'openQuantity': 20, 'closedQuantity': 0, 'currentMarketValue': 292.2, 'currentPrice': 14.61, 'averageEntryPrice': 15.342, 'dayPnl': -1.8, 'closedPnl': 0, 'openPnl': -14.64, 'totalCost': 306.84, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'ETHX.B.TO', 'symbolId': 35693784, 'openQuantity': 23, 'closedQuantity': 0, 'currentMarketValue': 365.7, 'currentPrice': 15.9, 'averageEntryPrice': 14.196522, 'dayPnl': -21.62, 'closedPnl': 0, 'openPnl': 39.179994, 'totalCost': 326.520006, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'ZRE.TO', 'symbolId': 8938, 'openQuantity': 15, 'closedQuantity': 0, 'currentMarketValue': 301.35, 'currentPrice': 20.09, 'averageEntryPrice': 20.545333, 'dayPnl': -2.4, 'closedPnl': 0, 'openPnl': -6.829995, 'totalCost': 308.179995, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'BTCX.B.TO', 'symbolId': 34965750, 'openQuantity': 22, 'closedQuantity': 0, 'currentMarketValue': 436.26, 'currentPrice': 19.83, 'averageEntryPrice': 12.705455, 'dayPnl': -6.16, 'closedPnl': 0, 'openPnl': 156.73999, 'totalCost': 279.52001, 'isRealTime': True, 'isUnderReorg': False}]}
        account_balances = {'perCurrencyBalances': [{'currency': 'CAD', 'cash': 78.624, 'marketValue': 17511.91, 'totalEquity': 17590.534, 'buyingPower': 78.624, 'maintenanceExcess': 78.624, 'isRealTime': True}, {'currency': 'USD', 'cash': 0, 'marketValue': 0, 'totalEquity': 0, 'buyingPower': 0, 'maintenanceExcess': 0, 'isRealTime': True}], 'combinedBalances': [{'currency': 'CAD', 'cash': 78.624, 'marketValue': 17511.91, 'totalEquity': 17590.534, 'buyingPower': 78.624, 'maintenanceExcess': 78.624, 'isRealTime': True}, {'currency': 'USD', 'cash': 54.516711, 'marketValue': 12142.497573, 'totalEquity': 12197.014284, 'buyingPower': 53.317076, 'maintenanceExcess': 53.317076, 'isRealTime': True}], 'sodPerCurrencyBalances': [{'currency': 'CAD', 'cash': 78.62, 'marketValue': 17608.42, 'totalEquity': 17687.04, 'buyingPower': 78.624, 'maintenanceExcess': 78.624, 'isRealTime': True}, {'currency': 'USD', 'cash': 0, 'marketValue': 0, 'totalEquity': 0, 'buyingPower': 0, 'maintenanceExcess': 0, 'isRealTime': True}], 'sodCombinedBalances': [{'currency': 'CAD', 'cash': 78.62, 'marketValue': 17608.42, 'totalEquity': 17687.04, 'buyingPower': 78.624, 'maintenanceExcess': 78.624, 'isRealTime': True}, {'currency': 'USD', 'cash': 54.513937, 'marketValue': 12209.41617, 'totalEquity': 12263.930107, 'buyingPower': 53.317076, 'maintenanceExcess': 53.317076, 'isRealTime': True}]}
    elif account_type == 'RRSP':
        account_positions = {'positions': [{'symbol': 'REET', 'symbolId': 6849037, 'openQuantity': 25, 'closedQuantity': 0, 'currentMarketValue': 585, 'currentPrice': 23.4, 'averageEntryPrice': 25.558772, 'dayPnl': 4, 'closedPnl': 0, 'openPnl': -53.9693, 'totalCost': 638.9693, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'BETH', 'symbolId': 50657051, 'openQuantity': 3, 'closedQuantity': 0, 'currentMarketValue': 249.81, 'currentPrice': 83.27, 'averageEntryPrice': 65.966667, 'dayPnl': -9.2688, 'closedPnl': 0, 'openPnl': 51.909999, 'totalCost': 197.900001, 'isRealTime': False, 'isUnderReorg': False}, {'symbol': 'IGOV', 'symbolId': 23848, 'openQuantity': 6, 'closedQuantity': 0, 'currentMarketValue': 224.34, 'currentPrice': 37.39, 'averageEntryPrice': 40.10125, 'dayPnl': -0.54, 'closedPnl': 0, 'openPnl': -16.2675, 'totalCost': 240.6075, 'isRealTime': False, 'isUnderReorg': False}, {'symbol': 'VDY.TO', 'symbolId': 2874670, 'openQuantity': 1, 'closedQuantity': 0, 'currentMarketValue': 49.27, 'currentPrice': 49.27, 'averageEntryPrice': 49.77, 'dayPnl': -0.39, 'closedPnl': 0, 'openPnl': -0.5, 'totalCost': 49.77, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'SUSC', 'symbolId': 18154045, 'openQuantity': 10, 'closedQuantity': 0, 'currentMarketValue': 224.1, 'currentPrice': 22.41, 'averageEntryPrice': 23.11936, 'dayPnl': -0.6, 'closedPnl': 0, 'openPnl': -7.0936, 'totalCost': 231.1936, 'isRealTime': False, 'isUnderReorg': False}, {'symbol': 'VSGX', 'symbolId': 23014223, 'openQuantity': 8, 'closedQuantity': 0, 'currentMarketValue': 443.76, 'currentPrice': 55.47, 'averageEntryPrice': 59.3885, 'dayPnl': -1.92, 'closedPnl': 0, 'openPnl': -31.348, 'totalCost': 475.108, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'XVV', 'symbolId': 32392271, 'openQuantity': 40, 'closedQuantity': 0, 'currentMarketValue': 1796.4, 'currentPrice': 44.91, 'averageEntryPrice': 44.827255, 'dayPnl': 0.8, 'closedPnl': 0, 'openPnl': 3.3098, 'totalCost': 1793.0902, 'isRealTime': False, 'isUnderReorg': False}, {'symbol': 'NUDV', 'symbolId': 38292321, 'openQuantity': 44, 'closedQuantity': 0, 'currentMarketValue': 1210.88, 'currentPrice': 27.52, 'averageEntryPrice': 29.444318, 'dayPnl': 13.64, 'closedPnl': 0, 'openPnl': -84.669992, 'totalCost': 1295.549992, 'isRealTime': False, 'isUnderReorg': False}, {'symbol': 'GLTR', 'symbolId': 23196560, 'openQuantity': 6, 'closedQuantity': 0, 'currentMarketValue': 670.86, 'currentPrice': 111.81, 'averageEntryPrice': 117.199917, 'dayPnl': -10.32, 'closedPnl': 0, 'openPnl': -32.339502, 'totalCost': 703.199502, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'EGUS', 'symbolId': 46556819, 'openQuantity': 41, 'closedQuantity': 0, 'currentMarketValue': 1798.67, 'currentPrice': 43.87, 'averageEntryPrice': 42.642439, 'dayPnl': -14.8543, 'closedPnl': 0, 'openPnl': 50.330001, 'totalCost': 1748.339999, 'isRealTime': False, 'isUnderReorg': False}, {'symbol': 'GOVT', 'symbolId': 1651704, 'openQuantity': 28, 'closedQuantity': 0, 'currentMarketValue': 623.14, 'currentPrice': 22.255, 'averageEntryPrice': 22.745075, 'dayPnl': -0.42, 'closedPnl': 0, 'openPnl': -13.7221, 'totalCost': 636.8621, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'TLT', 'symbolId': 23803, 'openQuantity': 7, 'closedQuantity': 0, 'currentMarketValue': 597.8, 'currentPrice': 85.4, 'averageEntryPrice': 91.590428, 'dayPnl': -0.42, 'closedPnl': 0, 'openPnl': -43.332996, 'totalCost': 641.132996, 'isRealTime': True, 'isUnderReorg': False}]}
        account_balances = {'perCurrencyBalances': [{'currency': 'CAD', 'cash': 29.8175, 'marketValue': 49.27, 'totalEquity': 79.0875, 'buyingPower': 29.8175, 'maintenanceExcess': 29.8175, 'isRealTime': True}, {'currency': 'USD', 'cash': -0.0022, 'marketValue': 8427.21, 'totalEquity': 8427.2078, 'buyingPower': -0.0022, 'maintenanceExcess': -0.0022, 'isRealTime': True}], 'combinedBalances': [{'currency': 'CAD', 'cash': 29.814327, 'marketValue': 12202.992262, 'totalEquity': 12232.806589, 'buyingPower': 29.814256, 'maintenanceExcess': 29.814256, 'isRealTime': True}, {'currency': 'USD', 'cash': 20.67281, 'marketValue': 8461.373084, 'totalEquity': 8482.045895, 'buyingPower': 20.217859, 'maintenanceExcess': 20.217859, 'isRealTime': True}], 'sodPerCurrencyBalances': [{'currency': 'CAD', 'cash': 29.82, 'marketValue': 49.64, 'totalEquity': 79.46, 'buyingPower': 29.8175, 'maintenanceExcess': 29.8175, 'isRealTime': True}, {'currency': 'USD', 'cash': 0, 'marketValue': 8444.649, 'totalEquity': 8444.649, 'buyingPower': -0.0022, 'maintenanceExcess': -0.0022, 'isRealTime': True}], 'sodCombinedBalances': [{'currency': 'CAD', 'cash': 29.82, 'marketValue': 12228.512788, 'totalEquity': 12258.332788, 'buyingPower': 29.814256, 'maintenanceExcess': 29.814256, 'isRealTime': True}, {'currency': 'USD', 'cash': 20.676744, 'marketValue': 8479.068637, 'totalEquity': 8499.745381, 'buyingPower': 20.217859, 'maintenanceExcess': 20.217859, 'isRealTime': True}]}
    elif account_type == 'TFSA':
        account_positions = {'positions': [{'symbol': 'VDY.TO', 'symbolId': 2874670, 'openQuantity': 1, 'closedQuantity': 0, 'currentMarketValue': 49.27, 'currentPrice': 49.27, 'averageEntryPrice': 48.73, 'dayPnl': -0.39, 'closedPnl': 0, 'openPnl': 0.54, 'totalCost': 48.73, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'XDIV.TO', 'symbolId': 18070692, 'openQuantity': 2, 'closedQuantity': 0, 'currentMarketValue': 60, 'currentPrice': 30, 'averageEntryPrice': 29.63, 'dayPnl': -0.24, 'closedPnl': 0, 'openPnl': 0.74, 'totalCost': 59.26, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'XEI.TO', 'symbolId': 24023, 'openQuantity': 4, 'closedQuantity': 0, 'currentMarketValue': 108.36, 'currentPrice': 27.09, 'averageEntryPrice': 27.4475, 'dayPnl': -0.64, 'closedPnl': 0, 'openPnl': -1.43, 'totalCost': 109.79, 'isRealTime': True, 'isUnderReorg': False}]}
        account_balances = {'perCurrencyBalances': [{'currency': 'CAD', 'cash': 34.2555, 'marketValue': 217.63, 'totalEquity': 251.8855, 'buyingPower': 34.2555, 'maintenanceExcess': 34.2555, 'isRealTime': True}, {'currency': 'USD', 'cash': 0, 'marketValue': 0, 'totalEquity': 0, 'buyingPower': 0, 'maintenanceExcess': 0, 'isRealTime': True}], 'combinedBalances': [{'currency': 'CAD', 'cash': 34.2555, 'marketValue': 217.63, 'totalEquity': 251.8855, 'buyingPower': 34.2555, 'maintenanceExcess': 34.2555, 'isRealTime': True}, {'currency': 'USD', 'cash': 23.752254, 'marketValue': 150.901401, 'totalEquity': 174.653654, 'buyingPower': 23.229588, 'maintenanceExcess': 23.229588, 'isRealTime': True}], 'sodPerCurrencyBalances': [{'currency': 'CAD', 'cash': 34.26, 'marketValue': 218.54, 'totalEquity': 252.8, 'buyingPower': 34.2555, 'maintenanceExcess': 34.2555, 'isRealTime': True}, {'currency': 'USD', 'cash': 0, 'marketValue': 0, 'totalEquity': 0, 'buyingPower': 0, 'maintenanceExcess': 0, 'isRealTime': True}], 'sodCombinedBalances': [{'currency': 'CAD', 'cash': 34.26, 'marketValue': 218.54, 'totalEquity': 252.8, 'buyingPower': 34.2555, 'maintenanceExcess': 34.2555, 'isRealTime': True}, {'currency': 'USD', 'cash': 23.755374, 'marketValue': 151.532381, 'totalEquity': 175.287755, 'buyingPower': 23.229588, 'maintenanceExcess': 23.229588, 'isRealTime': True}]}

    return account_positions, account_balances


if __name__ == "__main__":

    # Path Setup
    current_path = os.path.realpath(__file__)
    path = path_current_setup(current_path)

    args = sys.argv

    # Define accounts
    accounts = [{'type': 'FHSA', 'currency': 'CAD', 'number': None, 'positions': [], 'balances': [], 'yields': [], 'allocations': [ # account number 53209246
        {'symbol': 'VDY.TO', 'percentage':  10},
        {'symbol': 'XEI.TO', 'percentage':  10},
        {'symbol': 'XDIV.TO', 'percentage':  10},
        {'symbol': 'ICAE.TO', 'percentage':  10},
        {'symbol': 'ZWB.TO', 'percentage':  4},
        {'symbol': 'TXF.B.TO', 'percentage':  4},
        {'symbol': 'HXS.TO', 'percentage':  8},
        {'symbol': 'XIC.TO', 'percentage':  6},
        {'symbol': 'XCSR.TO', 'percentage':  6},
        {'symbol': 'ZFL.TO', 'percentage':  10},
        {'symbol': 'ZAG.TO', 'percentage':  10},
        {'symbol': 'XRE.TO', 'percentage':  2.5},
        {'symbol': 'ZRE.TO', 'percentage':  2.5},
        {'symbol': 'VRE.TO', 'percentage':  2.5},
        {'symbol': 'KILO.B.TO', 'percentage':  2.625},
        {'symbol': 'SVR.C.TO', 'percentage':  0.875},
        {'symbol': 'BTCX.B.TO', 'percentage':  0.8},
        {'symbol': 'ETHX.B.TO', 'percentage':  0.2}]
    }, {'type': 'RRSP', 'currency': 'USD', 'number': None, 'positions': [], 'balances': [], 'yields': [],  'allocations': [ # account number 53311600
        {'symbol': 'XVV', 'percentage':  23},
        {'symbol': 'EGUS', 'percentage':  21},
        {'symbol': 'NUDV', 'percentage':  15},
        {'symbol': 'QYLD', 'percentage':  1},
        {'symbol': 'VSGX', 'percentage': 5},
        {'symbol': 'GOVT', 'percentage':  7.5},
        {'symbol': 'TLT', 'percentage':  7.5},
        {'symbol': 'SUSC', 'percentage':  2.5},
        {'symbol': 'IGOV', 'percentage':  2.5},
        {'symbol': 'REET', 'percentage':  3},
        {'symbol': 'ERET', 'percentage':  4},
        {'symbol': 'IYRI', 'percentage':  0.5},
        {'symbol': 'GLTR', 'percentage':  6},
        {'symbol': 'BETH', 'percentage':  1.5}]
    }, {'type': 'TFSA', 'currency': 'CAD', 'number': None, 'positions': [], 'balances': [], 'yields': [], 'allocations': [ # account number 53322283
        # {'symbol': 'XCSR.TO', 'percentage':  50},
        {'symbol': 'GGRO.TO', 'percentage':  65},
        {'symbol': 'VDY.TO', 'percentage':  4},
        {'symbol': 'XEI.TO', 'percentage':  4},
        {'symbol': 'XDIV.TO', 'percentage':  4},
        {'symbol': 'ICAE.TO', 'percentage':  4},
        {'symbol': 'ZWB.TO', 'percentage':  2},
        {'symbol': 'TXF.B.TO', 'percentage':  2},
        # {'symbol': 'HXS.TO', 'percentage':  0},
        # {'symbol': 'XIC.TO', 'percentage':  0},
        # {'symbol': 'ZFL.TO', 'percentage':  7.5},
        # {'symbol': 'ZAG.TO', 'percentage':  7.5},
        {'symbol': 'XRE.TO', 'percentage':  2.5},
        {'symbol': 'ZRE.TO', 'percentage':  2.5},
        {'symbol': 'VRE.TO', 'percentage':  2.5},
        {'symbol': 'KILO.B.TO', 'percentage':  4.5},
        {'symbol': 'SVR.C.TO', 'percentage':  1.5},
        {'symbol': 'BTCX.B.TO', 'percentage':  1.2},
        {'symbol': 'ETHX.B.TO', 'percentage':  0.3}]
    }]

    show_single_account = [arg for arg in args if len(args) > 1 and arg in [account['type'] for account in accounts]]
    if show_single_account: show_single_account = show_single_account[0]
    dummy_accounts = len(args) > 1 and "dummy" in args

    # Get Account Infos
    # -----------------
    # see https://www.reddit.com/r/Questrade/comments/1hf2zmn/api_down/
    #   - 2 changes in file venv3.12/lib/python3.12/site-packages/questrade_api/auth.py:
    #       - # r = request.urlopen(self.config['Auth']['RefreshURL'].format(token)) -> r = request.urlopen(request.Request(self.config['Auth']['RefreshURL'].format(token), headers={"User-Agent": "MyApp"}))
    #       - # if time.time() + 60 < int(self.token_data['expires_at']): -> if time.time() + 60 < int(self.token_data['expires_in']):
    # To get new token:
    #   - Follow steps to create new app, register it and create new refresh token: https://www.questrade.com/api/documentation/getting-started
    #   - At point 4, copy paste cmd in browser by adding new token value: https://login.questrade.com/oauth2/token?grant_type=refresh_token&refresh_token=
    #   - Copy paste the response to file ~/.questrade.json

    if not dummy_accounts:
        # PCITvwfFE7XaDCHqAykDoEdDg6AGUMaf0
        # q = Questrade(access_token='yVOPZLmf7VRLBJ8GXX_uNdPjWjzxPkTZ0')
        # q = Questrade(access_token='7vREbpOGIQMsq9UsL3cC5rwGsdA-OoN70')
        # q = Questrade(access_token='bbl32zzkfyVgsOQ0qSvwc_IA2DaYol0')
        # q = Questrade(refresh_token='LuBQ81IixWZ_wComiVcEc7qgBk3cV7TX0')
        q = Questrade()
        try:
            q_accounts = q.accounts
        except Exception as e:
            print("Could not load Questrade accounts. Error: ", e)

    else:
        q_accounts = get_dummy_accounts()

    # print(q_accounts)
    # input()
    # acc = {'positions': [{'symbol': 'KILO.B.TO', 'symbolId': 23563025, 'openQuantity': 19, 'closedQuantity': 0, 'currentMarketValue': 995.79, 'currentPrice': 52.41, 'averageEntryPrice': 39.808947, 'dayPnl': 0, 'closedPnl': 0, 'openPnl': 239.420007, 'totalCost': 756.369993, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'VRE.TO', 'symbolId': 2874668, 'openQuantity': 17, 'closedQuantity': 0, 'currentMarketValue': 523.43, 'currentPrice': 30.79, 'averageEntryPrice': 30.460588, 'dayPnl': 0, 'closedPnl': 0, 'openPnl': 5.600004, 'totalCost': 517.829996, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'ZAG.TO', 'symbolId': 9029, 'openQuantity': 158, 'closedQuantity': 0, 'currentMarketValue': 2216.74, 'currentPrice': 14.03, 'averageEntryPrice': 13.696013, 'dayPnl': 0, 'closedPnl': 0, 'openPnl': 52.769946, 'totalCost': 2163.970054, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'ZFL.TO', 'symbolId': 9037, 'openQuantity': 170, 'closedQuantity': 0, 'currentMarketValue': 2237.2, 'currentPrice': 13.16, 'averageEntryPrice': 13.033648, 'dayPnl': 0, 'closedPnl': 0, 'openPnl': 21.47984, 'totalCost': 2215.72016, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'XCSR.TO', 'symbolId': 30232426, 'openQuantity': 49, 'closedQuantity': 0, 'currentMarketValue': 3679.41, 'currentPrice': 75.09, 'averageEntryPrice': 66.840204, 'dayPnl': 0, 'closedPnl': 0, 'openPnl': 404.240004, 'totalCost': 3275.169996, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'HXS.TO', 'symbolId': 54301062, 'openQuantity': 70, 'closedQuantity': 0, 'currentMarketValue': 5741.4, 'currentPrice': 82.02, 'averageEntryPrice': 77.773001, 'dayPnl': 0, 'closedPnl': 0, 'openPnl': 297.28993, 'totalCost': 5444.11007, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'SVR.C.TO', 'symbolId': 2869421, 'openQuantity': 17, 'closedQuantity': 0, 'currentMarketValue': 317.56, 'currentPrice': 18.68, 'averageEntryPrice': 15.117647, 'dayPnl': 0, 'closedPnl': 0, 'openPnl': 60.560001, 'totalCost': 256.999999, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'XIC.TO', 'symbolId': 23995, 'openQuantity': 94, 'closedQuantity': 0, 'currentMarketValue': 3706.42, 'currentPrice': 39.43, 'averageEntryPrice': 36.510638, 'dayPnl': 0, 'closedPnl': 0, 'openPnl': 274.420028, 'totalCost': 3431.999972, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'XRE.TO', 'symbolId': 24016, 'openQuantity': 38, 'closedQuantity': 0, 'currentMarketValue': 565.44, 'currentPrice': 14.88, 'averageEntryPrice': 15.057895, 'dayPnl': 0, 'closedPnl': 0, 'openPnl': -6.76001, 'totalCost': 572.20001, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'ETHX.B.TO', 'symbolId': 35693784, 'openQuantity': 24, 'closedQuantity': 0, 'currentMarketValue': 229.44, 'currentPrice': 9.56, 'averageEntryPrice': 14.070417, 'dayPnl': 0, 'closedPnl': 0, 'openPnl': -108.250008, 'totalCost': 337.690008, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'ZRE.TO', 'symbolId': 8938, 'openQuantity': 25, 'closedQuantity': 0, 'currentMarketValue': 525.25, 'currentPrice': 21.01, 'averageEntryPrice': 20.5032, 'dayPnl': 0, 'closedPnl': 0, 'openPnl': 12.67, 'totalCost': 512.58, 'isRealTime': True, 'isUnderReorg': False}, {'symbol': 'BTCX.B.TO', 'symbolId': 34965750, 'openQuantity': 22, 'closedQuantity': 0, 'currentMarketValue': 387.75, 'currentPrice': 17.625, 'averageEntryPrice': 12.705455, 'dayPnl': 0, 'closedPnl': 0, 'openPnl': 108.22999, 'totalCost': 279.52001, 'isRealTime': True, 'isUnderReorg': False}]}
    # print(acc['positions'])
    # for position in acc['positions']:
    #     symbol = position['symbol']#[0]['symbol']
    #     print(symbol)
    #     try:
    #         q_symbol = q.symbols(names=symbol)['symbols'][0]
    #         print("Dividend for symbol ", q_symbol['symbol'], " = ", q_symbol['dividend'])
    #         print("Yield for symbol ", q_symbol['symbol'], " = ", q_symbol['yield'])
    #         print()
    #     except Exception as e:
    #         print("Could not load info for symbol ", symbol, ". Error: ", e)
    # input()

    # q_symbols = q.symbols(names=symbol)['symbols'][0]
    # yield_dict = [{'symbol': q.symbols(names=position['symbol'])['symbols'][0]['symbol'],
    #                'dividend': q.symbols(names=position['symbol'])['symbols'][0]['dividend'],
    #                'yield': q.symbols(names=position['symbol'])['symbols'][0]['yield']} for position in acc['positions']]


    if 'code' in q_accounts.keys():
        print("q_accounts = ", q_accounts, "/n")
        #q_accounts = {'code': 1017, 'message': 'Access token is invalid'}

    else:
        for account in accounts:

            try_again = True
            while try_again:
                try:
                    # Get Account Numbers
                    account['account_number'] = [q_account['number'] for q_account in q_accounts['accounts'] if q_account['type'] == account['type']][0]

                    # First populate all info from Questrade before starting script calculations
                    print("\nGetting data for ", account['type'], " Account # ", account['account_number'], "\n")
                    if not dummy_accounts:
                        try: account['positions'] = q.account_positions(account['account_number'])
                        except Exception as e: print("Could not fetch account positions for account # ", account['type'], ". Error: ", e, "  |  Full error: ", traceback.format_exc())

                        try: account['balances'] = q.account_balances(account['account_number'])
                        except Exception as e: print("Could not fetch account balances for account # ", account['type'], ". Error: ", e, "  |  Full error: ", traceback.format_exc())

                        # # Get Yields and Dividends
                        # table_yield = prettytable.PrettyTable()
                        # table_yield.clear_rows()
                        # if account['positions']:
                        #     print("Getting Yield and dividend infos")
                        #     for position in account['positions']['positions']:
                        #         try:
                        #             q_symbol = q.symbols(names=position['symbol'])['symbols'][0]
                        #             account['yields'].append({'symbol': q.symbols(names=position['symbol'])['symbols'][0]['symbol'],
                        #                                     'dividend': q.symbols(names=position['symbol'])['symbols'][0]['dividend'],
                        #                                     'yield': q.symbols(names=position['symbol'])['symbols'][0]['yield']})
                        #         except Exception as e:
                        #             print("Could not load yield and dividend info for symbol ", position['symbol'], ". Error: ", e, "  |  Full error: ", traceback.format_exc())

                        #     table_yield.field_names = ['Symbol', 'Yield', 'Dividend', '%']
                        #     for y in account['yields']:
                        #         allocation_perc = [allocation['percentage'] for allocation in account['allocations'] if allocation.get('symbol') == y['symbol']][0]
                        #         table_yield.add_row([y['symbol'], y['yield'], y['dividend'], allocation_perc])

                    else:
                        account['positions'], account['balances'] = get_dummy_accounts_info(account['type'])

                    if len(account['positions']) > 0 and len(account['balances']) > 0:
                        # Display Informations
                        if not show_single_account or show_single_account == account['type']:
                            print("\n=======================================================\nCalculating new balances for ", account['type'], " Account # ", account['account_number'], "\n=======================================================\n")

                            balance_actions(account['positions'], account['balances'], account['type'], account['account_number'], account['allocations'], account['currency'])
                            try_again = False

                            # print("\n", table_yield, "\n")

                except Exception as e:
                    print("Could not properly fetch data for ", account['type'], ". Error: ", e,  "  |  Full error: ", traceback.format_exc())
                    try_again_str = ' '
                    while str(try_again_str).lower() not in ["y", "n", "yes", 'no', '']:
                        try_again_str = input("Try again? (Y/N)")
                    try_again = str(try_again_str).lower() in ["y", "yes", ""]

        pyplot.show()

    print("\n\n")









#  # Define accounts
#     accounts = [{'type': 'TFSA', 'currency': 'CAD', 'number': None, 'positions': [], 'balances': [], 'yields': [], 'allocations': [ # account number 53209246
#         {'symbol': 'HXS.TO', 'percentage':  25},
#         {'symbol': 'XIC.TO', 'percentage':  20},
#         {'symbol': 'XCSR.TO', 'percentage':  20},
#         {'symbol': 'ZFL.TO', 'percentage':  10},
#         {'symbol': 'ZAG.TO', 'percentage':  10},
#         {'symbol': 'XRE.TO', 'percentage':  2.5},
#         {'symbol': 'ZRE.TO', 'percentage':  2.5},
#         {'symbol': 'VRE.TO', 'percentage':  2.5},
#         {'symbol': 'KILO.B.TO', 'percentage':  4.5},
#         {'symbol': 'SVR.C.TO', 'percentage':  1.5},
#         {'symbol': 'BTCX.B.TO', 'percentage':  1.2},
#         {'symbol': 'ETHX.B.TO', 'percentage':  0.3}]
#     }, {'type': 'RRSP', 'currency': 'USD', 'number': None, 'positions': [], 'balances': [], 'yields': [],  'allocations': [ # account number 53311600
#         {'symbol': 'XVV', 'percentage':  23},
#         {'symbol': 'EGUS', 'percentage':  21},
#         {'symbol': 'NUDV', 'percentage':  15},
#         {'symbol': 'QYLD', 'percentage':  1},
#         {'symbol': 'VSGX', 'percentage': 5},
#         {'symbol': 'GOVT', 'percentage':  7.5},
#         {'symbol': 'TLT', 'percentage':  7.5},
#         {'symbol': 'SUSC', 'percentage':  2.5},
#         {'symbol': 'IGOV', 'percentage':  2.5},
#         {'symbol': 'REET', 'percentage':  3},
#         {'symbol': 'ERET', 'percentage':  4},
#         {'symbol': 'IYRI', 'percentage':  0.5},
#         {'symbol': 'GLTR', 'percentage':  6},
#         {'symbol': 'BETH', 'percentage':  1.5}]
#     }, {'type': 'FHSA', 'currency': 'CAD', 'number': None, 'positions': [], 'balances': [], 'yields': [], 'allocations': [ # account number 53322283
#         {'symbol': 'VDY.TO', 'percentage':  24},
#         {'symbol': 'XEI.TO', 'percentage':  24},
#         {'symbol': 'XDIV.TO', 'percentage':  24},
#         {'symbol': 'ICAE.TO', 'percentage':  24},
#         {'symbol': 'ZWB.TO', 'percentage':  2},
#         {'symbol': 'TXF.B.TO', 'percentage':  2},]
#     }]
