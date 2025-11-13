import tradingview_ta as tv
from ib_insync import *
# sys.path.append("/Volumes/untitled/Trading/trading_app/scripts.path_setup")
# import path_setup

# Link: https://www.youtube.com/watch?v=lpaq_6kU5hg
#       https://www.youtube.com/watch?v=WhuB5cbr-kY




if __name__ == "__main__":

    # Path Setup
    # path = path_setup.path_current_setup(os.path.realpath(__file__))

    tsla = tv.TA_Handler(symbol='TSLA', screener='america', exchange='NASDAQ', interval=tv.Interval.INTERVAL_1_MINUTE)
    # for ind in tsla.get_analysis().indicators:
    #     print(ind)
    print("RSI = ", tsla.get_analysis().indicators["RSI"])
    print("close = ", tsla.get_analysis().indicators["close"])

    # TWS Connection
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)

    bars = ib.reqHistoricalData(
    	contract, endDateTime='', durationStr='30 D',
    	barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)

    # convert to pandas dataframe (pandas needs to be installed):
    df = util.df(bars)
    print(df)



print("\n\n")