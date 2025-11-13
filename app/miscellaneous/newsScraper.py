import os, requests, pytz, numpy, yfinance
from bs4 import BeautifulSoup
from datetime import datetime, timedelta#, timezone
from ib_insync import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from finvizfinance.quote import finvizfinance
from finvizfinance.news import News
from utils import helpers, constants

#/cygdrive/c/Users/creis/Docs/Programs/python_kivy/python.exe "C:\Users\creis\Docs\Desktop\Medias\get_media.py"


def get_page(url):
    """Download a webpage and return a beautiful soup doc"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36'}
    response = requests.get(url, headers=headers)

    try:
        if not response.ok:
            print('Status code:', response.status_code)
            raise Exception('Failed to load page {}'.format(url))
        page_content = response.text
        doc = BeautifulSoup(page_content, 'html.parser')
    except Exception as e:
        print("Page did not load. Error: ", e)
        doc = None
    return doc


def time_condition(time_span, max_days):

    if "hour" in time_span: elapsed_days = 0
    elif "yesterday" in time_span: elapsed_days = 1
    elif "days" in time_span:
        indexDays = time_span.split(" ").index('days')
        elapsed_days = int(time_span.split(" ")[indexDays - 1])
    else: elapsed_days = 32

    return elapsed_days <= max_days


def keywords_sentiment(text, keywords):

    score = 0
    for level, keyword_list in enumerate(keywords):
        for keyword in keyword_list:
            score += text.lower().count(keyword) * (level + 1)

    return score


def getNewsYfinance(symbols, max_days, date_range=None, print_live=False):

    # News scraper from Yahoo finance
    # -------------------------------
    # Link News Scraper: https://medium.com/@vinodvidhole/web-scraping-yahoo-finance-using-python-7c4612fab70c#4366
    # see also finviz: https://sahajgodhani777.medium.com/analyzing-stock-price-news-sentiment-with-machine-learning-models-in-python-1d94fb680b3d
    # link for dates: https://stackoverflow.com/questions/59744589/how-can-i-convert-the-string-2020-01-06t000000-000z-into-a-datetime-object

    newsList = []
    for symbol in symbols:

        print()
        print("\nGetting news for ", symbol, " from Yahoo finance")
        url_yf = "https://finance.yahoo.com/quote/" + symbol + "/news"
        # url_yf = "https://finance.yahoo.com/quote/MRNA/news"
        # url_yf = "https://ca.finance.yahoo.com/quote/MRNA/"
        # url_yf = "https://finance.yahoo.com/topic/stock-market-news/"


        page = get_page(url_yf)

        news_items = page.find_all('li', {'class': "stream-item"})

        newsList_item = {'symbol':symbol, 'news':[]}

        # tickers = div_tags[0].find_all('a', {'data-testid': "ticker-container"})
        for index_news, news_item in enumerate(news_items):

            tickers = [ticker.text for ticker in news_item.find_all('span', {'class': "symbol"})]
            print("Analyzing news # ", index_news + 1, " out of ", len (news_items), end="\r")

            if len(tickers) > 0:
                hyperlink = news_item.find_all('a', {'class': 'subtle-link'})[0]['href']
                date = get_page(hyperlink).find('time')['datetime']
                date_EST = datetime.fromisoformat(date[:-1] + '+00:00').astimezone(tz=pytz.timezone(constants.CONSTANTS.TZ_WORK))

                time_from_news = news_item.find_all('div', {'class': 'publishing'})[0].text

                if date_range:
                    date_condition = date > date_range[0] and date < date_range[1]
                else:
                    date_condition = time_condition(time_from_news, max_days) # search using number of days displayed on first page (faster)

                if date_condition:
                    title = news_item.find_all('h3', {'class': 'clamp'})[0].text
                    abstract = news_item.find_all('p', {'class': 'clamp'})[0].text
                    sentiment_score = analyzer.polarity_scores(abstract)['compound']
                    keywords_score = keywords_sentiment(abstract, keywords)

                    newsList_item['news'].append({'title': title,
                                                  'tickers': tickers,
                                                  'sentiment_score': sentiment_score,
                                                  'keywords_score': keywords_score,
                                                  'date': str(date_EST),
                                                  'hyperlink': hyperlink})

        newsList.append(newsList_item)

        if print_live:
            if len(newsList_item['news']) > 0:
                print("News found for ", newsList_item['symbol'], ":\n")
                for news in newsList_item['news']:
                    print(news['date'], ":   ", news['title'])
                print("\nAverage Sentiment: ", numpy.average([n['sentiment_score'] for n in news['news']]))
                print("Average Keywords Sentiment: ", numpy.average([n['keywords_score'] for n in news['news']]))
                print("\n----------------------------------\n")
            else: print("No news found for symbol ", newsList_item['symbol'], '\n')

    if print_live: print("\n\n++++++++++++++++++++++++++++++++++++++++++++\n\n")

    return newsList


def getNewsFinviz(symbols, max_days, date_range=None):
    # https://pypi.org/project/finvizfinance/
    # https://finvizfinance.readthedocs.io/en/latest/

    newsList = []
    for symbol in symbols:

        stock = finvizfinance(symbol)
        # url_chart = stock.ticker_charts()
        news_items = stock.ticker_news()

        newsList_item = {'symbol':symbol, 'news':[]}
        for index_news, date in enumerate(news_items.iloc[:, 0]):

            date_EST = helpers.date_to_EST_aware(date)
            time_now = datetime.now(tz=pytz.timezone(constants.CONSTANTS.TZ_WORK))
            if date_range:
                date_condition = date_EST > date_range[0] and date_EST < date_range[1]
            else:
                date_condition = time_now - date_EST <= timedelta(days=max_days)

            if date_condition:
                title = news_items.iloc[index_news, 1]
                hyperlink = news_items.iloc[index_news, 2]
                sentiment_score = analyzer.polarity_scores(title)['compound']
                keywords_score = keywords_sentiment(title, keywords)

                newsList_item['news'].append({'title': title,
                                              'sentiment_score': sentiment_score,
                                              'keywords_score': keywords_score,
                                              'date': str(date_EST),
                                              'hyperlink': hyperlink})

        newsList.append(newsList_item)

    return newsList










# def getNewsIBKR(symbols, ib, max_days):
#     # # Test get News from IBKR and sentiment analysis
#     # # ----------------------------------------------

#     # ib.reqNewsBulletins(True)
#     # ib.sleep(5)
#     # print(ib.newsBulletins())

#     analyzer = SentimentIntensityAnalyzer()

#     for symbol in symbols:
#         stock = Stock(symbol, 'SMART', 'USD')
#         ib.qualifyContracts(stock)

#         try:
#             headlines = ib.reqHistoricalNews(stock.conId, codes, '', '', 10)
#             latest = headlines[0]
#             time_since_last_news = datetime.now() - latest.time
#             if time_since_last_news < max_days:
#                 article = ib.reqNewsArticle(latest.providerCode, latest.articleId).articleText
#                 sentiment_score = analyzer.polarity_scores(article)['compound']

#                 print("\nSymbol = ", symbol, "       |    Date = ", latest.time)
#                 print("\nlen(headlines) = ", len(headlines))
#                 print("\nArticle = ", article)
#                 print("\nSentiment Score = ", sentiment_score)
#                 input()
#             else:
#                 print("Last news for ", symbol, " was ", time_since_last_news)
#         except Exception as e:
#             print(e, "\nNo news found for symbol ", symbol)
#         print("\n++++++++++++++++++++++++++++++++++++++++\n")


def printNews(newsList, display_hyperlink=False):

    # Print News Results:
    for news in newsList:
        if len(news['news']) > 0:
            print('News found for ', news['symbol'], ':\n')
            for news_item in news['news']:
                hyperlink_text = ''
                if display_hyperlink:
                    hyperlink_text = "\n                                   " + news_item['hyperlink']
                print('\n', news_item['date'], ":   ", news_item['title'], hyperlink_text)
            print("\nAverage Sentiment: ", numpy.average([n['sentiment_score'] for n in news['news']]))
            print("Average Keywords Sentiment: ", numpy.average([n['keywords_score'] for n in news['news']]))
            print("\n----------------------------------\n")
        else: print("No news found for symbol ", news['symbol'], '\n')


# Links Sentiment Analyzer: https://medium.com/@deepml1818/financial-sentiment-analysis-and-stock-information-retrieval-with-python-a1d6e821deb6
#                           https://medium.com/@prasannaraut/extract-market-sentiment-with-google-news-and-python-nltk-eb02b8c4a04e


# # Need to download the "vader_lexicon" once
# import nltk
# nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

keywords = [["phase I", "grant", "investor", "accept", "new", "sign", "merger", "gain"],
                ["phase II", "receives", "FDA", "approval", "benefit", "beneficial", "fast track", "breakout", "acquire", "acquisition", "expand", "expansion", "contract", "complete", "promising", "achieve", "achievement", "launch"],
                ["phase III", "positive", "top-line", "significant", "demonstrates", "treatment", "trial", "agreement", "cancer", "partnership", "collaboration", "improvement", "successful", "billionaire", "carl itchhan", "increase", "awarded", "primary"],
                ["positive endpoint", "positive ceo statement"]]


if __name__ == "__main__":

    # Path Setup
    # path = path_setup.path_current_setup(os.path.realpath(__file__))

    # # symbols = ['GME','SQSP','RXRX','FULC','INCY','NVAX','ZK','ARM','BILI','RIOT','DJT','BABA','SMR','MARA','CLSK','STLA','JD','RDDT','SOUN','SG','HOOD','VOD','COIN','NIO','PHG']
    # symbols = ['GME', 'AMC', 'SRCL', 'SMMT', 'PARA', 'HOOD', 'CLSK', 'MARA', 'NVDA', 'RIOT', 'GPCR', 'NIO', 'QCOM']
    # symbols = ['NVDA', 'RIOT', 'GPCR', 'NIO', 'QCOM']
    # symbols = ['CORZ', 'ANNX']
    symbols = ['NRIX', 'ASTS', 'BILI', 'KR', 'DELL', 'NVDA', 'RIOT', 'MU', 'CLSK', 'TSM', 'PLTR']
    symbols = ['FLNC', 'RVMD', 'ZAPP', 'INTC', 'BP', 'SMR']
    symbols = ['MARA', 'TSM', 'NVDA', 'AMD', 'AGEN']
    symbols = ['ZENA', 'NRIX', 'ASTS', 'BILI', 'KR']

    symbols = ['ZENA', 'NRIX', 'KR']

    max_days = 1
    date_range = [datetime.fromisoformat("2024-07-04 00:00:00-04:00"),
                  datetime.fromisoformat("2024-07-06 09:00:00-04:00")]

    # newsList = getNewsYfinance(symbols, max_days)
    # newsList = getNewsYfinance(symbols, max_days, date_range=date_range, print_live=False)
    newsList = getNewsFinviz(symbols, max_days, date_range=None)
    print()
    printNews(newsList, display_hyperlink=True)

    print("\n\n")
    # input("\nEnter anything to exit")









    # def getNewsYfinance(symbols, max_days, date_range=None, print_live=False):

    # # News scraper from Yahoo finance
    # # -------------------------------
    # # Link News Scraper: https://medium.com/@vinodvidhole/web-scraping-yahoo-finance-using-python-7c4612fab70c#4366
    # # see also finviz: https://sahajgodhani777.medium.com/analyzing-stock-price-news-sentiment-with-machine-learning-models-in-python-1d94fb680b3d
    # # link for dates: https://stackoverflow.com/questions/59744589/how-can-i-convert-the-string-2020-01-06t000000-000z-into-a-datetime-object

    # newsList = []
    # for symbol in symbols:

    #     print()
    #     print("\nGetting news for ", symbol, " from Yahoo finance")
    #     url_yf = "https://finance.yahoo.com/quote/" + symbol + "/news"
    #     # url_yf = "https://finance.yahoo.com/quote/MRNA/news"
    #     # url_yf = "https://ca.finance.yahoo.com/quote/MRNA/"
    #     # url_yf = "https://finance.yahoo.com/topic/stock-market-news/"


    #     page = get_page(url_yf)

    #     news_items = page.find_all('li', {'class': "stream-item"})

    #     newsList_item = {"symbol":symbol, "tickers":[], "titles":[], "sentiment_scores":[], "keywords_scores":[], "dates":[], "hyperlink":[]}

    #     # tickers = div_tags[0].find_all('a', {'data-testid': "ticker-container"})
    #     for index_news, news_item in enumerate(news_items):
    #         tickers = [ticker.text for ticker in news_item.find_all('span', {'class': "symbol"})]
    #         print("Analyzing news # ", index_news + 1, " out of ", len (news_items), end="\r")

    #         if len(tickers) > 0:
    #             hyperlink = news_item.find_all('a', {'class': 'subtle-link'})[0]['href']
    #             date = get_page(hyperlink).find('time')['datetime']
    #             date = datetime.fromisoformat(date[:-1] + '+00:00').astimezone(tz=pytz.timezone('US/Eastern'))
    #             time_now = datetime.now(tz=pytz.timezone('US/Eastern'))
    #             time_from_news = news_item.find_all('div', {'class': 'publishing'})[0].text

    #             if date_range:
    #                 date_condition = date > date_range[0] and date < date_range[1]
    #             else:
    #                 date_condition = time_condition(time_from_news, max_days) # search using number of days displayed on first page (faster)

    #             if date_condition:
    #                 title = news_item.find_all('h3', {'class': 'clamp'})[0].text
    #                 abstract = news_item.find_all('p', {'class': 'clamp'})[0].text
    #                 sentiment_score = analyzer.polarity_scores(abstract)['compound']
    #                 keywords_score = keywords_sentiment(abstract, keywords)

    #                 newsList_item["tickers"].append(tickers)
    #                 newsList_item["titles"].append(title)
    #                 newsList_item["sentiment_scores"].append(sentiment_score)
    #                 newsList_item["keywords_scores"].append(keywords_score)
    #                 newsList_item["dates"].append(str(date))
    #                 newsList_item["hyperlink"].append(str(hyperlink))

    #     if len(newsList_item["titles"]) > 0:
    #         newsList.append(newsList_item)
    #         if print_live:
    #             print("News found for ", newsList_item["symbol"], ":\n")
    #             for index in range(0, len(newsList_item["titles"])):
    #                 # print(datetime.strptime(sym["dates"][index], "%Y-%m-%d %H:%M:%S+HH:MM"), ":   ", sym["titles"][index])
    #                 print(newsList_item["dates"][index], ":   ", newsList_item["titles"][index])
    #             print("\nAverage Sentiment: ", numpy.average(newsList_item["sentiment_scores"]))
    #             print("Average Keywords Sentiment: ", numpy.average(newsList_item["keywords_scores"]))
    #             print("\n----------------------------------\n")

    # if print_live: print("\n\n++++++++++++++++++++++++++++++++++++++++++++\n\n")

    # return newsList




# def getNewsFinviz(symbols, max_days, date_range=None):
#     # https://pypi.org/project/finvizfinance/
#     # https://finvizfinance.readthedocs.io/en/latest/

#     newsList = []
#     for symbol in symbols:

#         stock = finvizfinance(symbol)
#         # url_chart = stock.ticker_charts()
#         news_items = stock.ticker_news()

#         newsList_item = {"symbol":symbol, "tickers":[], "titles":[], "sentiment_scores":[], "keywords_scores":[], "dates":[], "hyperlink":[]}
#         for index_news, date in enumerate(news_items.iloc[:, 0]):
#             date_EST = datetime.fromisoformat(str(date) + '-04:00').astimezone(tz=pytz.timezone('US/Eastern'))
#             time_now = datetime.now(tz=pytz.timezone('US/Eastern'))
#             if date_range:
#                 date_condition = date_EST > date_range[0] and date_EST < date_range[1]
#             else:
#                 date_condition = time_now - date_EST <= timedelta(days=max_days)

#             if date_condition:
#                 title = news_items.iloc[index_news, 1]
#                 hyperlink = news_items.iloc[index_news, 2]
#                 sentiment_score = analyzer.polarity_scores(title)['compound']
#                 keywords_score = keywords_sentiment(title, keywords)

#                 # newsList_item["tickers"].append(symbol)
#                 newsList_item["titles"].append(title)
#                 newsList_item["sentiment_scores"].append(sentiment_score)
#                 newsList_item["keywords_scores"].append(keywords_score)
#                 newsList_item["dates"].append(str(date_EST))
#                 newsList_item["hyperlink"].append(str(hyperlink))

#         if len(newsList_item["titles"]) > 0:
#             newsList.append(newsList_item)

#     return newsList



    # def printNews(newsList, display_hyperlink=False):

    # # Print News Results:
    # for sym in newsList:
    #     print("News found for ", sym["symbol"], ":\n")
    #     for index in range(0, len(sym["titles"])):
    #         hyperlink_text = ""
    #         if display_hyperlink:
    #             hyperlink_text = "\n                                   " + sym["hyperlink"][index]
    #         # print(datetime.strptime(sym["dates"][index], "%Y-%m-%d %H:%M:%S+HH:MM"), ":   ", sym["titles"][index])
    #         print('\n', sym["dates"][index], ":   ", sym["titles"][index], hyperlink_text)
    #     print("\nAverage Sentiment: ", numpy.average(sym["sentiment_scores"]))
    #     print("Average Keywords Sentiment: ", numpy.average(sym["keywords_scores"]))
    #     print("\n----------------------------------\n")
