import sys, requests, pandas as pd
from bs4 import BeautifulSoup
from tradingview_screener import Query, Column
from playwright.sync_api import sync_playwright
from utils import helpers, constants
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service as ChromeService
# from webdriver_manager.chrome import ChromeDriverManager
from ib_insync import *
# sys.path.append("/Volumes/untitled/Trading/trading_app/scripts.path_setup")
# import path_setup

# Doc ib_insync: https://ib-insync.readthedocs.io/api.html
# Doc IBKR API: https://interactivebrokers.github.io/tws-api/introduction.html
# util.startLoop()  # uncomment this line when in a notebook


#/cygdrive/c/Users/creis/Docs/Programs/python_kivy/python.exe "C:\Users\creis\Docs\Desktop\Medias\get_media.py"


def get_page(url, javascript=False, selector=''):
    # Doc requests_html: https://requests.readthedocs.io/projects/requests-html/en/latest/
    #                    https://www.youtube.com/watch?v=-PmNcIX9En4

    """Download a webpage and return a beautiful soup doc"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36'}

    if not javascript:
        response = requests.get(url, headers=headers)
        page_content = response.text
    else:
        # # driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
        # # driver.get(url)
        # session = requests_html.HTMLSession()
        # response = session.get(url, headers=headers)
        # response.html.render(sleep=5)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            # page.wait_for_selector("table[class=grid-table]")
            page.wait_for_selector(selector)
            # page.find('tbody', {'class': "ant-table-tbody"})
            # page_content = page.content()
            # page.wait_for_selector()
            # page_content.find('table', {'class': "grid-table "})
            page_content = page.content()

    try:
        # # if not response.ok:
        # #     print('Status code:', response.status_code)
        # #     raise Exception('Failed to load page {}'.format(url))
        # if not javascript:
        #     page_content = response.text
        # else:
        #     page_content = response.html.html
        #     # page_content = driver.page_source
        doc = BeautifulSoup(page_content, 'html.parser')
    except Exception as e:
        print("Page did not load. Error: ", e)
        doc = None
    return doc


def scannerIBKR(ib):
    # Links: https://github.com/erdewit/ib_insync/blob/master/notebooks/scanners.ipynb
    #        https://notebook.community/erdewit/ib_insync/notebooks/scanners
    #        https://interactivebrokers.github.io/tws-api/market_scanners.html

    sub = ScannerSubscription(instrument='STK', locationCode='STK.US.MAJOR', scanCode='TOP_PERC_GAIN')#scanCode='MOST_ACTIVE')
    ib.sleep(constants.CONSTANTS.PROCESS_TIME['long'])
    tagValues = [TagValue('priceAbove', 5), TagValue("volumeAbove", 500)]
    tagValues = [TagValue("changePercAbove", "5"),
                 TagValue('priceAbove', 5),#, TagValue('priceBelow', 50),
                 TagValue('sharesAvailableManyAbove', 1000000),# TagValue('sharesAvailableManyBelow', 50000000),
                 TagValue("volumeAbove", 500000)
                ]
    tagValues = [TagValue("changePercAbove", "5"),
                 TagValue('priceAbove', 2),#, TagValue('priceBelow', 50),
                 TagValue('sharesAvailableManyBelow', 20000000),# TagValue('sharesAvailableManyBelow', 50000000),
                 TagValue("volumeAbove", 100000)
                ]

    scanData = ib.reqScannerData(sub, [], tagValues) # the tagValues are given as 3rd argument; the 2nd argument must always be an empty list, (IB has not documented the 2nd argument and it's not clear what it does)
    symbols = [sd.contractDetails.contract.symbol for sd in scanData]

    return symbols


def scannerTradingView(mode, exclude_BB_15=False, exclude_RSI_1=False, side="both"):
    # Links: https://pypi.org/project/tradingview-screener/
    #        https://shner-elmo.github.io/TradingView-Screener/2.5.0/tradingview_screener.html
    # List of available columns: https://shner-elmo.github.io/TradingView-Screener/2.5.0/tradingview_screener/constants.html#COLUMNS

    if mode =="GG": # Gapper Go
        n_rows, df = (Query()
            .select('name', 'close', 'premarket_change', 'volume', 'premarket_volume', 'relative_volume_10d_calc', 'ATR', 'float_shares_outstanding')
            .set_markets('america')
            .where(
                # Column('market_cap_basic') <= 50000000,
                Column('float_shares_outstanding') <= 50000000,
                # Column('average_volume_10d_calc') >= 500000,
                Column('premarket_volume') >= 500000,
                # Column('volume') >= 500000,
                # Column('relative_volume_10d_calc') >= 1.2,
                Column('relative_volume_intraday|5') >= 2,
                # Column('close').between(2, 50),
                Column('close') >= 2,
                Column('ATR') >= 0.5,
                Column('premarket_change') >= 5
                # 'rate of change', 'premarket_gap', 'premarket_change_from_open', 'post_market_change'
            )
            .order_by('premarket_change', ascending=False)
            .get_scanner_data())

    if mode =="GapperUp": # GapperUp
        n_rows, df = (Query()
            .select('name', 'close', 'change', 'volume', 'premarket_change', 'premarket_volume', 'relative_volume', 'relative_volume_10d_calc', 'relative_volume_intraday|5', 'ATR', 'market_cap_basic', 'float_shares_outstanding', 'sector')#('name', 'close', 'change', 'volume', 'premarket_change', 'premarket_volume', 'relative_volume', 'relative_volume_10d_calc', 'relative_volume_intraday|5', 'ATR', 'market_cap', 'float_shares_outstanding', 'sector')
            .set_markets('america')
            .where(
                # Column('market_cap_basic') <= 50000000,
                # Column('float_shares_outstanding') >= 20000000,
                # Column('average_volume_10d_calc') >= 500000,
                Column('relative_volume_10d_calc') >= 2,
                # Column('relative_volume_intraday|5') >= 2,
                # Column('premarket_volume') >= 500000,
                Column('volume') >= 1000000,
                # Column('close').between(2, 50),
                Column('close') >= 2,
                # Column('ATR') >= 0.5,
                # Column('change') >= 5
                Column('premarket_change') >= 5
                # 'rate of change', 'premarket_gap', 'premarket_change_from_open', 'post_market_change'
            )
            .order_by('premarket_change', ascending=False)
            .get_scanner_data())

    if mode =="GapperDown": # GapperDown
        n_rows, df = (Query()
            .select('name', 'close', 'change', 'volume', 'premarket_change', 'premarket_volume', 'relative_volume', 'relative_volume_10d_calc', 'relative_volume_intraday|5', 'ATR', 'market_cap_basic', 'float_shares_outstanding', 'sector')#('name', 'close', 'change', 'volume', 'premarket_change', 'premarket_volume', 'relative_volume', 'relative_volume_10d_calc', 'relative_volume_intraday|5', 'ATR', 'market_cap', 'float_shares_outstanding', 'sector')
            .set_markets('america')
            .where(
                Column('relative_volume_10d_calc') >= 2,
                Column('volume') >= 1000000,
                Column('close') >= 2,
                Column('premarket_change') <= -5
            )
            .order_by('premarket_change', ascending=True)
            .get_scanner_data())

    if mode =="Earnings": # Recent Earnings

        # page_count = 5
        # all_dfs = []
        # for page in range(1, page_count + 1):
        #     _, df = (
        #         Query()
        #         .select('name', 'close', 'change', 'volume', 'premarket_change', 'premarket_volume', 'relative_volume', 'relative_volume_10d_calc', 'relative_volume_intraday|5', 'ATR', 'market_cap_basic', 'float_shares_outstanding', 'sector', 'earnings_release_date', 'earnings_release_time')
        #         .set_markets('america')
        #         .where(
        #             Column('volume') >= 1000000,
        #             Column('close') >= 2,
        #             Column('ATR') >= 0.5
        #         )
        #         .get_scanner_data(page=page)  # Use pagination here
        #     )

        #     # Stop if no data is returned (end of pages)
        #     if df.empty:
        #         break

        #     all_dfs.append(df)

        # # Combine all pages into one DataFrame
        # df = pd.concat(all_dfs, ignore_index=True)


        n_rows, df = (Query()
            .select('name', 'close', 'change', 'volume', 'premarket_change', 'premarket_volume', 'relative_volume', 'relative_volume_10d_calc', 'relative_volume_intraday|5', 'ATR', 'market_cap_basic', 'float_shares_outstanding', 'sector', 'earnings_release_date', 'earnings_release_time')
            .set_markets('america')
            .where(
                Column('volume') >= 1000000,
                Column('close') >= 2,
                Column('ATR') >= 0.5
            )
            .order_by('relative_volume_10d_calc', ascending=True)
            .get_scanner_data())

        df['earnings_release_date'] = pd.to_datetime(df['earnings_release_date'], unit='s', errors='coerce')
        df['earnings_release_time'] = df['earnings_release_time'].map({-1: 'amc', 1: 'bmo', 0: 'uncertain'}) # 'after market close', 'before market open'
        print(df)
        input()
        today = pd.Timestamp.today().normalize()
        yesterday = today - pd.Timedelta(days=1)

        # Earnings that happened yesterday amc or today bmo
        df = df[
            ((df['earnings_release_date'].dt.date == yesterday.date()) & (df['earnings_release_time'] == 'amc')) |
            ((df['earnings_release_date'].dt.date == today.date()) & (df['earnings_release_time'] == 'bmo'))]
        # df = df[df['earnings_release_date'].isin([today, yesterday])]

        print()
        print(df)
        input()

    if mode =="BB-RSI":
        n_rows_overbought, df_overbought = (Query()
            .select('name', 'close', 'premarket_change', 'volume', 'premarket_volume', 'relative_volume_10d_calc', 'RSI', 'BB.lower', 'BB.upper')
            .set_markets('america')
            .where(
                Column('float_shares_outstanding') >= 1000000,
                Column('average_volume_10d_calc') >= 500000,
                Column('ATR') >= 0.5,
                Column('close') >= 5,
                Column('close') > Column('BB.upper'),
                Column('RSI') >= 80
            )
            .order_by('RSI', ascending=False)
            .get_scanner_data())

        n_rows_oversold, df_oversold = (Query()
            .select('name', 'close', 'premarket_change', 'volume', 'premarket_volume', 'relative_volume_10d_calc', 'RSI', 'BB.lower', 'BB.upper')
            .set_markets('america')
            .where(
                Column('float_shares_outstanding') >= 1000000,
                Column('average_volume_10d_calc') >= 1000000,
                Column('ATR') >= 0.5,
                Column('close') >= 5,
                Column('close') < Column('BB.lower'),
                Column('RSI') <= 20
            )
            .order_by('RSI', ascending=True)
            .get_scanner_data())

        df = pd.concat([df_overbought, df_oversold])
        n_rows = n_rows_overbought + n_rows_oversold

    if mode =="RSI-Reversal":

        rsi_threshold = 25
        n_rows_overbought, df_overbought = (Query()
            .select('name', 'close', 'premarket_change', 'volume', 'premarket_volume', 'relative_volume_10d_calc', 'RSI', 'BB.lower', 'BB.upper')
            .set_markets('america')
            .where(
                Column('float_shares_outstanding') >= 1000000,
                Column('average_volume_10d_calc') >= 1000000,
                Column('ATR') >= 0.5,
                Column('close') >= 5,
                Column('RSI|1') > (100 - rsi_threshold),# or exclude_RSI_1,
                Column('RSI|5') > (100 - rsi_threshold),
                Column('RSI|15') > (100 - rsi_threshold),
                Column('RSI|60') > (100 - rsi_threshold),
                Column('close') > Column('BB.upper|5'),# or exclude_BB_15,
                Column('close') > Column('BB.upper|15'),# or exclude_BB_15,
                Column('close') > Column('BB.upper|60')
            )
            .order_by('RSI', ascending=False)
            .get_scanner_data())

        n_rows_oversold, df_oversold = (Query()
            .select('name', 'close', 'premarket_change', 'volume', 'premarket_volume', 'relative_volume_10d_calc', 'RSI', 'BB.lower', 'BB.upper')
            .set_markets('america')
            .where(
                Column('float_shares_outstanding') >= 1000000,
                Column('average_volume_10d_calc') >= 1000000,
                Column('ATR') >= 0.5,
                Column('close') >= 5,
                Column('RSI|1') < rsi_threshold,#  or exclude_RSI_1,
                Column('RSI|5') < rsi_threshold,
                Column('RSI|15') < rsi_threshold,
                Column('RSI|60') < rsi_threshold,
                Column('close') < Column('BB.lower|5'),# or exclude_BB_15,
                Column('close') < Column('BB.lower|15'),# or exclude_BB_15,
                Column('close') < Column('BB.lower|60')
            )
            .order_by('RSI', ascending=True)
            .get_scanner_data())

        if side == 'up': df = df_oversold
        elif side == 'down': df = df_overbought
        elif side == 'both': df = pd.concat([df_overbought, df_oversold])
        # n_rows = n_rows_overbought + n_rows_oversold

    # print(df.to_string())
    # print("\nn_rows = ", n_rows)
    # print(df.loc[0]["name"])

    symbols = [d for d in df.name]

    return symbols, df


def scannerFinviz(mode):

    symbols = []
    if mode == "E5D": # "Earnings 5 Days"

        url_finviz_next_earnings = "https://finviz.com/screener.ashx?v=161&p=d&f=earningsdate_nextdays5,sh_avgvol_o1000,sh_float_o1,sh_price_o5"
        page = get_page(url_finviz_next_earnings)

        screener_table = page.find('tr', {'id': "screener-table"})
        table_items = screener_table.find_all('a', {'class': "tab-link"})

        for item in table_items:
            symbol_info_page = get_page("https://finviz.com/" + item['href'])
            table_data_items = symbol_info_page.find('table', {'class': "screener_snapshot-table-body"})
            data_items = table_data_items.find_all('td', {'class': "snapshot-td2"})
            index_Earnings = [data_item.text for data_item in data_items].index("Earnings")
            date_next_earning = data_items[index_Earnings + 1].text
            symbols.append({'symbol': item.text, "next_earning": date_next_earning})

    if mode == "RE": # "Recent Earnings"

        url_finviz_earnings_yst_amc = "https://finviz.com/screener.ashx?v=161&p=d&f=earningsdate_yesterdayafter%2Csh_avgvol_o1000%2Csh_float_o1%2Csh_price_o2"
        url_finviz_earnings_bmo = "https://finviz.com/screener.ashx?v=161&p=d&f=earningsdate_todaybefore%2Csh_avgvol_o1000%2Csh_float_o1%2Csh_price_o2"
        for url in [url_finviz_earnings_yst_amc, url_finviz_earnings_bmo]:
            page = get_page(url)

            screener_table = page.find('tr', {'id': "screener-table"}) if page else None
            table_items = screener_table.find_all('a', {'class': "tab-link"}) if screener_table else []

            symbols += [item.text for item in table_items]

        symbols = list(set(symbols)) # Remove duplicates

    elif mode == "C5": # "Change 5%":

        url_finviz_5_change = "https://finviz.com/screener.ashx?v=111&p=d&f=sh_avgvol_o1000,sh_float_o10,sh_price_o5,ta_averagetruerange_o1,ta_gap_u5&ft=3"
        # url_finviz_5_change = "https://finviz.com/screener.ashx?v=111&p=d&f=cap_smallover,sh_avgvol_o500,sh_float_u50,sh_price_o5,ta_averagetruerange_o1,ta_gap_u5&ft=3"
        url_finviz_5_change = "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500,sh_float_o1,sh_price_o5,ta_change_u5&ft=3"
        url_finviz_10_change_small_cap = "https://finviz.com/screener.ashx?v=111&f=sh_float_u20,sh_price_o2,sh_relvol_o5,ta_change_u10"
        # page = get_page(url_finviz_5_change)
        page = get_page(url_finviz_10_change_small_cap)

        screener_table = page.find('tr', {'id': "screener-table"})
        table_items = screener_table.find_all('a', {'class': "tab-link"})

        symbols = [item.text for item in table_items]

    # symbols = [item.text for item in earning_items]
    return symbols


def scannerTradingTerminal(page):

    if page == 'scanners':
        # Deal with login: https://playwright.dev/python/docs/auth

        # Scrape from Trading Terminal scanners page
        url_tt = "https://scanner.tradingterminal.com/"
        # url_tt = "https://scanner.tradingterminal.com/dashboard"
        print("Getting page")
        # page = get_page(url_tt, True, "table[class=grid-table]")
        page = get_page(url_tt, True, "tile[tile-type=toplist]")
        # print(page)
        # # page = get_page(url_tt, False)

        # # screener_table = page.find('table', {'class': "grid-table "})
        print("Getting Table")
        screener_table = page.find('tile', {'tile-type': "toplist"})
        print(screener_table)
        # input()
        # screener_table = page.find_all('div', {'id': "outtlet"})
        # # screener_table = page.find_all('table', {'id': "6674bef8d118e81781744b7c"})
        # # print(screener_table)
        # # print(page)
        # table_items = screener_table.find_all('a', {'class': "tab-link"})


        # symbols = [item.text for item in table_items]

        symbols = []
        return symbols

    elif page == 'main':
        # Scrape from Trading Terminal main page
        url_tt = "https://tradingterminal.com/"

        # page = get_page(url_tt, True)
        page = get_page(url_tt, True, "tbody[class=ant-table-tbody]")
        # page = get_page(url_tt, True, "tr[class=ant-table-row-level-0]")
        # page = get_page(url_tt, True, "td[class=ant-table-cell]")
        # print(page)

        change_threshold = 5
        vol_PM_text = 'Biggest Volume Pre-Market'
        vol_trend_text = 'High Volume Trending Stocks'

        # print(page)
        # bttn = page.getByText(vol_PM_text)
        # # print(bttn)
        # input()

        sections_scanner = page.find_all('div', {'class': "ant-space-item"})

        section_scanner_vol_PM,  section_scanner_vol_trend = None, None
        for section_scanner in sections_scanner:
            # section_scanner_head_title = section_scanner.find('div', {'class': 'ant-card-head-title'})
            section_scanner_head_titles = section_scanner.find_all('div', {'class': 'ant-tabs-tab-btn'})
            for section_scanner_head_title in section_scanner_head_titles:

                # if section_scanner_head_title and section_scanner.find('div', {'class': 'ant-card-head-title'}).text == vol_PM_text:
                if section_scanner_head_title and section_scanner_head_title.text == vol_PM_text:
                    section_scanner_vol_PM = section_scanner

                elif section_scanner_head_title and section_scanner_head_title.text == vol_trend_text:
                    section_scanner_vol_trend = section_scanner
                    # print(section_scanner.find('div', {'class': 'ant-card-head-title'}).text)

        # print(section_scanner_vol_PM)
        # print()
        try:
            # table_rows_vol_PM = section_scanner_vol_PM.find_all('tr', {'class': "ant-table-row"})
            table_rows_vol_PM = section_scanner_vol_PM.find_all('tr', {'class': "ant-table-cell"})

            # print(table_rows_vol_PM)
            # input()

            dict_list_vol_PM = []
            for row in table_rows_vol_PM:
                row_text = [item.text for item in row]
                if abs(float(row_text[2][:-1])) >= change_threshold: dict_list_vol_PM.append({'time':row_text[0], 'symbol':row_text[1], 'change':float(row_text[2][:-1]) if row_text[2][:-1] else '', 'price':float(row_text[3][1:]) if row_text[3][:-1] else ''})

            table_rows_vol_trend = section_scanner_vol_trend.find_all('tr', {'class': "ant-table-row"})
            dict_list_vol_trend = []
            for row in table_rows_vol_trend:
                row_text = [item.text for item in row]
                if abs(float(row_text[2][:-1])) >= change_threshold: dict_list_vol_trend.append({'time':row_text[0], 'symbol':row_text[1], 'change':float(row_text[2][:-1]) if row_text[2][:-1] else '', 'price':float(row_text[3][1:]) if row_text[3][:-1] else ''})

            dict_list_vol_PM = sorted(dict_list_vol_PM, key=lambda i: i['change'], reverse=True)
            dict_list_vol_PM_pos = [d for d in dict_list_vol_PM if d['change'] >= 0]
            dict_list_vol_PM_neg = [d for d in dict_list_vol_PM if d['change'] < 0]
            dict_list_vol_trend = sorted(dict_list_vol_trend, key=lambda i: i['change'], reverse=True)
        except Exception as e:
            print("Could not fetch infos from Trading Terminal scanner. Error: ", e)
            dict_list_vol_PM, dict_list_vol_PM_pos, dict_list_vol_PM_neg, dict_list_vol_trend = [], [], [], []

        print("\n" + vol_PM_text + ":\n")
        for row in dict_list_vol_PM:
            print(row)
        print("\n" + vol_trend_text + ":\n")
        for row in dict_list_vol_trend:
            print(row)

        symbols_vol_PM = [row_dict['symbol'] for row_dict in dict_list_vol_PM_pos]
        symbols_vol_trend = [row_dict['symbol'] for row_dict in dict_list_vol_trend]

        return symbols_vol_PM


if __name__ == "__main__":

    # Path Setup
    # path = path_setup.path_current_setup(os.path.realpath(__file__))

    args = sys.argv

    continuous = False
    scanner_type = ''
    force = None
    time_wait = 10*60
    record = False
    for arg in args:
        if 'force' in arg: force = arg[5:]
        if 'ibkr' in arg: scanner_type = 'ibkr'
        if arg in ['tt', 'tv_rsi_reversal', 'tv_gapper_up', 'tv_gapper_down', 'earnings']: scanner_type = arg
        # if 'tt' == arg or 'tv_rsi_reversal' == arg or 'tv_gapper_up' == arg or 'tv_gapper_down' == arg or 'earnings' == arg: scanner_type = arg
        if 'record' in arg: record = True

    # TWS Connection
    paperTrading = False if len(args) > 1 and 'live' in args else True
    ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)
    if not ibConnection:
        paperTrading = not paperTrading
        ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)

    if scanner_type == 'ibkr':
        # TWS Connection
        try:
            symbolsIBKR = scannerIBKR(ib)
            if force: symbolsIBKR.append(force)
            print("Symbols from IBKR scanner =\n", symbolsIBKR, "\n\n")

        except Exception as e:
            print("\n\nCould not load IBKR scanner. Error: ", e, "\n\n")

    elif scanner_type == 'tt':

        symbolsTT = scannerTradingTerminal("main")
        # symbolsTT = scannerTradingTerminal("scanners")
        if force: symbolsTT.append(force)
        print("\nSymbols from Trading Terminal scanner =\n")
        print(symbolsTT)
        if record: helpers.save_to_daily_csv(ib, symbolsTT, constants.PATHS.daily_csv_files['gapper_up'])

    elif scanner_type == 'tv_rsi_reversal':

        symbolsTV, dfTV = scannerTradingView("RSI-Reversal")
        symbolsTV_no15, dfTV = scannerTradingView("RSI-Reversal", exclude_BB_15=True, exclude_RSI_1=True)
        if force: symbolsTV_no15.append(force)
        print("\nSymbols from TradingView scanner RSI_Reversal=\n")
        print(symbolsTV)
        print(symbolsTV_no15)
        if record: helpers.save_to_daily_csv(ib, symbolsTV_no15, constants.PATHS.daily_csv_files['rsi-reversal'])

    elif scanner_type == 'tv_gapper_up':

        # # https://pypi.org/project/tradingview-screener/2.0.0/
        # df = tradingview_screener.Scanner.premarket_gainers.get_data()

        symbolsTV, dfTV = scannerTradingView("GapperUp")
        # symbolsTV = scannerTradingView("GG")
        if force: symbolsTV.append(force)

        table_dfTV = helpers.df_to_table(dfTV.round(2))
        print("\n", table_dfTV, "\n")

        print("\nSymbols from TradingView scanner GapperUp=\n")
        print(symbolsTV)
        if record: helpers.save_to_daily_csv(ib, symbolsTV, constants.PATHS.daily_csv_files['gapper_up'])

    elif scanner_type == 'tv_gapper_down':

        symbolsTV, dfTV = scannerTradingView("GapperDown")
        if force: symbolsTV.append(force)

        table_dfTV = helpers.df_to_table(dfTV.round(2))
        print("\n", table_dfTV, "\n")

        print("\nSymbols from TradingView scanner GapperDown=\n")
        print(symbolsTV)
        if record: helpers.save_to_daily_csv(ib, symbolsTV, constants.PATHS.daily_csv_files['gapper_down'])

    elif scanner_type == 'earnings':
        # symbolsTV, dfTV = scannerTradingView("Earnings")
        # if force: symbolsTV.append(force)

        # table_dfTV = helpers.df_to_table(dfTV.round(2))
        # print("\n", table_dfTV, "\n")

        symbolsFinviz = scannerFinviz("RE")
        if force: symbolsFinviz.append(force)
        print("\nSymbols from Finviz scanner =\n")
        print(symbolsFinviz)
        if record: helpers.save_to_daily_csv(ib, symbolsFinviz, constants.PATHS.daily_csv_files['earnings'])

    else:
        # symbolsFinviz = scannerFinviz("C5")
        symbolsFinviz = scannerFinviz("RE")
        if force: symbolsFinviz.append(force)
        print("\nSymbols from Finviz scanner =\n")
        print(symbolsFinviz)



    print("\n\n")


# if __name__ == "__main__":

#     # Path Setup
#     # path = path_setup.path_current_setup(os.path.realpath(__file__))

#     args = sys.argv

#     continuous = False
#     scanner_type = ''
#     force = None
#     time_wait = 10*60
#     record = False
#     for arg in args:
#         if 'force' in arg: force = arg[5:]
#         if 'cont' in arg: continuous = True
#         if 'ibkr' in arg: scanner_type = 'ibkr'
#         if 'tt' == arg or 'tv_rsi_reversal' == arg or 'tv_gapper_up' == arg: scanner_type = arg
#         # if 'tv' in arg: scanner_type = 'tv'
#         if 'wait' in arg:
#             try: time_wait = int(float(arg[4:]) * 60)
#             except Exception as e: print("Could not get time_wait parameter. Error: ", e)
#         if 'record' in arg: record = True

#     # TWS Connection
#     paperTrading = False if len(args) > 1 and 'live' in args else True
#     ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)
#     if not ibConnection:
#         paperTrading = not paperTrading
#         ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)

#     counter = 100
#     while counter > 0:
#         if scanner_type == 'ibkr':
#             # TWS Connection
#             try:
#                 ib = IB()
#                 ib.connect('127.0.0.1', 7497, clientId=1)

#                 symbolsIBKR = scannerIBKR(ib)
#                 if force: symbolsIBKR.append(force)
#                 print("Symbols from IBKR scanner =\n", symbolsIBKR, "\n\n")

#             except Exception as e:
#                 print("\n\nCould not load IBKR scanner. Error: ", e, "\n\n")

#         elif scanner_type == 'tt':

#             symbolsTT = scannerTradingTerminal("main")
#             # symbolsTT = scannerTradingTerminal("scanners")
#             if force: symbolsTT.append(force)
#             print("\nSymbols from Trading Terminal scanner =\n")
#             print(symbolsTT)
#             if record: helpers.save_to_daily_csv(ib, symbolsTT, constants.PATHS.daily_csv_files['gapper_up'])

#         elif scanner_type == 'tv_rsi_reversal':

#             symbolsTV, dfTV = scannerTradingView("RSI-Reversal")
#             symbolsTV_no15, dfTV = scannerTradingView("RSI-Reversal", exclude_15=True)
#             if force: symbolsTV_no15.append(force)
#             print("\nSymbols from TradingView scanner RSI_Reversal=\n")
#             print(symbolsTV)
#             print(symbolsTV_no15)
#             if record: helpers.save_to_daily_csv(ib, symbolsTV_no15, constants.PATHS.daily_csv_files['rsi-reversal'])

#         elif scanner_type == 'tv_gapper_up':

#             # # https://pypi.org/project/tradingview-screener/2.0.0/
#             # df = tradingview_screener.Scanner.premarket_gainers.get_data()

#             symbolsTV, dfTV = scannerTradingView("GapperUp")
#             # symbolsTV = scannerTradingView("GG")
#             if force: symbolsTV.append(force)

#             table_dfTV = helpers.df_to_table(dfTV.round(2))
#             print("\n", table_dfTV, "\n")

#             print("\nSymbols from TradingView scanner GapperUp=\n")
#             print(symbolsTV)
#             if record: helpers.save_to_daily_csv(ib, symbolsTV, constants.PATHS.daily_csv_files['gapper_up'])

#         else:
#             symbolsFinviz = scannerFinviz("C5")
#             if force: symbolsFinviz.append(force)
#             print("\nSymbols from Finviz scanner =\n")
#             print(symbolsFinviz)

#         if continuous:
#             counter = counter - 1
#             helpers.sleep_display(time_wait)

#         else: counter = 0



#     print("\n\n")





    # # Scanner Test:
    # # -------------

    # print('\n-------------------------------------------------\n')

    # # Link: https://www.youtube.com/watch?v=E2JE8fPaC5s
    # #       https://github.com/adidror005/youtube-videos/blob/main/Scanners-Video.ipynb


    # # def display_with_stock_symbol_and_market_data(scanData,tickers_dict):
    # #     df=display_with_stock_symbol(scanData)
    # #     market_data_df = util.df(tickers_dict.values())
    # #     market_data_df['symbol']=market_data_df.apply(lambda l:l['contract'].symbol,axis=1)
    # #     df_merged=df.merge(market_data_df[['symbol','bid','ask','last','close','open']],on='symbol')
    # #     df_merged['% Change']=(df_merged['last']-df_merged['close'])/df_merged['close']
    # #     df_merged['GAP %']=(df_merged['open']-df_merged['close'])/df_merged['close']
    # #     return df_merged[['rank','symbol','bid','ask','last','close','open','% Change','GAP %']]

    # # def display_with_stock_symbol(scanData):
    # #     df=util.df(scanData)
    # #     df['contract']=df.apply(lambda l:l['contractDetails'].contract,axis=1)
    # #     df['symbol']=df.apply(lambda l:l['contract'].symbol,axis=1)
    # #     return df[['rank','contractDetails','contract','symbol']]


    # # sub = ScannerSubscription(numberOfRows=50, instrument='STK', locationCode='STK.US.MAJOR', scanCode='TOP_PERC_LOSE', marketCapAbove=300_000, abovePrice=100, aboveVolume=100000

    # #                         )
    # # scanData = ib.reqScannerData(sub)

    # # util.df(scanData)


    # ticker_dict = {}
    # for contract in display_with_stock_symbol(scanData).contract.tolist():
    #     ticker_dict[contract]=ib.reqMktData(contract=contract,genericTickList="",snapshot=True,regulatorySnapshot=False)
    # ib.sleep(2)