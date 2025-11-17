import os, sys, numpy, math, csv, prettytable, traceback
# import backtrader as bt
from ib_insync import *
from matplotlib import pyplot
from collections import defaultdict
from utils import helpers
from utils.constants import CONSTANTS
from datetime import datetime
# from scripts import newsScraper


# def get_symbols():

#     daily_data_folder = helpers.get_path_daily_data_folder()

#     symbols_gapper_up_csv_path = os.path.join(daily_data_folder, constants.PATHS.local_csv_files['gapper_up'])
#     symbols_rsi_reversal_csv_path = os.path.join(daily_data_folder, constants.PATHS.local_csv_files['rsi-reversal'])
#     symbols_main_csv_path = constants.PATHS.csv_files['main']
#     symbols_index_csv_path = constants.PATHS.csv_files['index']

#     symbols_gapper_up = [row[1] for i, row in enumerate(helpers.read_csv_file(symbols_gapper_up_csv_path)) if i > 0 and row != [] and row[0] != '']
#     symbols_rsi_reversal = [row[1] for i, row in enumerate(helpers.read_csv_file(symbols_rsi_reversal_csv_path)) if i > 0 and row != [] and row[0] != '']
#     symbols_main = [row[0] for row in helpers.read_csv_file(symbols_main_csv_path) if row != [] and row[0] != '']
#     symbols_index = [row[0] for row in helpers.read_csv_file(symbols_index_csv_path) if row != [] and row[0] != '']

#     return symbols_rsi_reversal + symbols_gapper_up + symbols_main + symbols_index


# def transform_symbols(symbols):
#     # Modify symbols shape from 1 x n to 2 x n/2
#     symbol_odd = len(symbols) % 2 != 0
#     if symbol_odd:
#         last_el = symbols[-1]
#         symbols = symbols[:-1]
#     symbols = [list(x) for x in zip(*[iter(symbols)]*2)]
#     if symbol_odd: symbols.append([last_el])

#     return symbols


class DOMLevel:

    def __init__(self, price, size, marketMaker):
        self.price = price
        self.size = size
        self.marketMaker = marketMaker


class DummyMktDepth:

    def __init__(self):
        self.domBids = [DOMLevel(price=122.9, size=100.0, marketMaker='NSDQ'), DOMLevel(price=122.86, size=200.0, marketMaker='ARCA'), DOMLevel(price=122.86, size=200.0, marketMaker='DRCTEDGE'), DOMLevel(price=122.86, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.84, size=100.0, marketMaker='NSDQ'), DOMLevel(price=122.83, size=200.0, marketMaker='ARCA'), DOMLevel(price=122.82, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.81, size=100.0, marketMaker='NSDQ'), DOMLevel(price=122.8, size=200.0, marketMaker='MEMX'), DOMLevel(price=122.8, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.79, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.78, size=500.0, marketMaker='NSDQ'), DOMLevel(price=122.77, size=100.0, marketMaker='NSDQ'), DOMLevel(price=122.75, size=100.0, marketMaker='NSDQ'), DOMLevel(price=122.7, size=100.0, marketMaker='NSDQ'), DOMLevel(price=122.65, size=100.0, marketMaker='ARCA'), DOMLevel(price=122.61, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.6, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.6, size=200.0, marketMaker='CHX'), DOMLevel(price=122.45, size=2800.0, marketMaker='NSDQ'), DOMLevel(price=122.4, size=10200.0, marketMaker='NSDQ'), DOMLevel(price=122.35, size=100.0, marketMaker='ARCA'), DOMLevel(price=122.31, size=200.0, marketMaker='ARCA'), DOMLevel(price=122.2, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.17, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.1, size=200.0, marketMaker='ARCA'), DOMLevel(price=122.01, size=1000.0, marketMaker='ARCA'), DOMLevel(price=122.01, size=400.0, marketMaker='NSDQ'), DOMLevel(price=122.0, size=900.0, marketMaker='ARCA'), DOMLevel(price=122.0, size=1900.0, marketMaker='NSDQ'), DOMLevel(price=121.96, size=800.0, marketMaker='ARCA'), DOMLevel(price=121.92, size=100.0, marketMaker='ARCA'), DOMLevel(price=121.88, size=500.0, marketMaker='NSDQ'), DOMLevel(price=121.6, size=100.0, marketMaker='ARCA'), DOMLevel(price=121.6, size=500.0, marketMaker='NSDQ'), DOMLevel(price=121.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=121.5, size=100.0, marketMaker='NSDQ'), DOMLevel(price=121.3, size=300.0, marketMaker='NSDQ'), DOMLevel(price=121.23, size=100.0, marketMaker='NSDQ'), DOMLevel(price=121.2, size=4000.0, marketMaker='ARCA'), DOMLevel(price=121.01, size=900.0, marketMaker='BYX'), DOMLevel(price=121.01, size=200.0, marketMaker='NSDQ'), DOMLevel(price=121.0, size=1300.0, marketMaker='ARCA'), DOMLevel(price=121.0, size=1500.0, marketMaker='NSDQ'), DOMLevel(price=120.58, size=100.0, marketMaker='ARCA'), DOMLevel(price=120.54, size=100.0, marketMaker='ARCA'), DOMLevel(price=120.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=120.38, size=200.0, marketMaker='NSDQ'), DOMLevel(price=120.18, size=1000.0, marketMaker='NSDQ'), DOMLevel(price=120.1, size=100.0, marketMaker='NSDQ'), DOMLevel(price=120.04, size=100.0, marketMaker='NSDQ'), DOMLevel(price=120.0, size=2500.0, marketMaker='ARCA'), DOMLevel(price=120.0, size=2900.0, marketMaker='NSDQ'), DOMLevel(price=119.91, size=100.0, marketMaker='ARCA'), DOMLevel(price=119.9, size=100.0, marketMaker='ARCA'), DOMLevel(price=119.09, size=100.0, marketMaker='ARCA'), DOMLevel(price=119.0, size=600.0, marketMaker='ARCA'), DOMLevel(price=119.0, size=800.0, marketMaker='NSDQ'), DOMLevel(price=118.8, size=200.0, marketMaker='ARCA'), DOMLevel(price=118.54, size=100.0, marketMaker='NSDQ'), DOMLevel(price=118.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=118.25, size=100.0, marketMaker='ARCA'), DOMLevel(price=118.2, size=100.0, marketMaker='ARCA'), DOMLevel(price=118.18, size=100.0, marketMaker='ARCA'), DOMLevel(price=118.18, size=100.0, marketMaker='NSDQ'), DOMLevel(price=118.11, size=100.0, marketMaker='NSDQ'), DOMLevel(price=118.02, size=200.0, marketMaker='ARCA'), DOMLevel(price=118.0, size=1400.0, marketMaker='ARCA'), DOMLevel(price=118.0, size=700.0, marketMaker='NSDQ'), DOMLevel(price=117.93, size=100.0, marketMaker='ARCA'), DOMLevel(price=117.87, size=100.0, marketMaker='ARCA'), DOMLevel(price=117.87, size=100.0, marketMaker='NSDQ'), DOMLevel(price=117.8, size=100.0, marketMaker='ARCA'), DOMLevel(price=117.67, size=100.0, marketMaker='NSDQ'), DOMLevel(price=117.12, size=100.0, marketMaker='ARCA'), DOMLevel(price=117.0, size=400.0, marketMaker='ARCA'), DOMLevel(price=116.86, size=100.0, marketMaker='ARCA'), DOMLevel(price=116.81, size=100.0, marketMaker='NSDQ'), DOMLevel(price=116.59, size=100.0, marketMaker='ARCA'), DOMLevel(price=116.55, size=200.0, marketMaker='ARCA'), DOMLevel(price=116.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=116.46, size=300.0, marketMaker='ARCA'), DOMLevel(price=116.3, size=100.0, marketMaker='ARCA'), DOMLevel(price=116.16, size=100.0, marketMaker='ARCA'), DOMLevel(price=116.15, size=800.0, marketMaker='ARCA'), DOMLevel(price=116.0, size=1800.0, marketMaker='ARCA'), DOMLevel(price=116.0, size=100.0, marketMaker='NSDQ'), DOMLevel(price=115.88, size=100.0, marketMaker='NSDQ'), DOMLevel(price=115.69, size=100.0, marketMaker='ARCA'), DOMLevel(price=115.57, size=100.0, marketMaker='NSDQ'), DOMLevel(price=115.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=115.26, size=100.0, marketMaker='ARCA'), DOMLevel(price=115.12, size=100.0, marketMaker='ARCA'), DOMLevel(price=115.1, size=400.0, marketMaker='ARCA'), DOMLevel(price=115.07, size=200.0, marketMaker='ARCA'), DOMLevel(price=115.0, size=5600.0, marketMaker='ARCA'), DOMLevel(price=115.0, size=300.0, marketMaker='NSDQ'), DOMLevel(price=114.97, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.88, size=200.0, marketMaker='NSDQ'), DOMLevel(price=114.8, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.66, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.6, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.57, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.55, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.51, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.5, size=600.0, marketMaker='ARCA'), DOMLevel(price=114.41, size=200.0, marketMaker='ARCA'), DOMLevel(price=114.4, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.2, size=600.0, marketMaker='ARCA'), DOMLevel(price=114.14, size=400.0, marketMaker='ARCA'), DOMLevel(price=114.1, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.0, size=2600.0, marketMaker='ARCA'), DOMLevel(price=114.0, size=100.0, marketMaker='NSDQ'), DOMLevel(price=113.75, size=1000.0, marketMaker='ARCA'), DOMLevel(price=113.73, size=100.0, marketMaker='ARCA'), DOMLevel(price=113.7, size=100.0, marketMaker='ARCA'), DOMLevel(price=113.69, size=100.0, marketMaker='ARCA'), DOMLevel(price=113.68, size=100.0, marketMaker='ARCA'), DOMLevel(price=113.6, size=100.0, marketMaker='ARCA'), DOMLevel(price=113.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=113.5, size=100.0, marketMaker='NSDQ'), DOMLevel(price=113.44, size=500.0, marketMaker='ARCA'), DOMLevel(price=113.33, size=300.0, marketMaker='ARCA'), DOMLevel(price=113.3, size=300.0, marketMaker='NSDQ'), DOMLevel(price=113.01, size=100.0, marketMaker='ARCA'), DOMLevel(price=113.0, size=1900.0, marketMaker='ARCA'), DOMLevel(price=112.88, size=100.0, marketMaker='ARCA'), DOMLevel(price=112.83, size=200.0, marketMaker='ARCA'), DOMLevel(price=112.53, size=100.0, marketMaker='ARCA'), DOMLevel(price=112.51, size=100.0, marketMaker='ARCA'), DOMLevel(price=112.36, size=100.0, marketMaker='NSDQ'), DOMLevel(price=112.2, size=100.0, marketMaker='ARCA'), DOMLevel(price=112.12, size=400.0, marketMaker='ARCA'), DOMLevel(price=112.1, size=100.0, marketMaker='ARCA'), DOMLevel(price=112.0, size=1700.0, marketMaker='ARCA'), DOMLevel(price=111.88, size=100.0, marketMaker='ARCA'), DOMLevel(price=111.81, size=100.0, marketMaker='NSDQ'), DOMLevel(price=111.7, size=100.0, marketMaker='ARCA'), DOMLevel(price=111.46, size=100.0, marketMaker='ARCA'), DOMLevel(price=111.11, size=400.0, marketMaker='ARCA'), DOMLevel(price=111.06, size=100.0, marketMaker='ARCA'), DOMLevel(price=111.01, size=100.0, marketMaker='ARCA'), DOMLevel(price=111.0, size=900.0, marketMaker='ARCA'), DOMLevel(price=110.67, size=100.0, marketMaker='ARCA'), DOMLevel(price=110.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=110.2, size=200.0, marketMaker='ARCA'), DOMLevel(price=110.11, size=200.0, marketMaker='ARCA'), DOMLevel(price=110.08, size=100.0, marketMaker='ARCA'), DOMLevel(price=110.05, size=200.0, marketMaker='ARCA'), DOMLevel(price=110.01, size=100.0, marketMaker='ARCA'), DOMLevel(price=110.0, size=8500.0, marketMaker='ARCA'), DOMLevel(price=110.0, size=700.0, marketMaker='NSDQ'), DOMLevel(price=109.78, size=400.0, marketMaker='ARCA'), DOMLevel(price=109.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=108.0, size=700.0, marketMaker='NSDQ'), DOMLevel(price=107.5, size=100.0, marketMaker='NSDQ'), DOMLevel(price=106.01, size=1200.0, marketMaker='NSDQ'), DOMLevel(price=105.0, size=800.0, marketMaker='NSDQ'), DOMLevel(price=101.0, size=100.0, marketMaker='NSDQ'), DOMLevel(price=100.0, size=200.0, marketMaker='NSDQ'), DOMLevel(price=93.23, size=500.0, marketMaker='NSDQ'), DOMLevel(price=80.0, size=200.0, marketMaker='NSDQ'), DOMLevel(price=0.06, size=1300.0, marketMaker='NSDQ'), DOMLevel(price=0.01, size=500.0, marketMaker='NSDQ')]
        self.domAsks = [DOMLevel(price=122.91, size=100.0, marketMaker='DRCTEDGE'), DOMLevel(price=122.93, size=200.0, marketMaker='ARCA'), DOMLevel(price=122.93, size=300.0, marketMaker='NSDQ'), DOMLevel(price=122.99, size=100.0, marketMaker='MEMX'), DOMLevel(price=122.99, size=300.0, marketMaker='NSDQ'), DOMLevel(price=123.0, size=1100.0, marketMaker='ARCA'), DOMLevel(price=123.0, size=300.0, marketMaker='NSDQ'), DOMLevel(price=123.02, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.08, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.1, size=1100.0, marketMaker='ARCA'), DOMLevel(price=123.15, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.16, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.2, size=200.0, marketMaker='CHX'), DOMLevel(price=123.2, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.23, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.26, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.27, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.28, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.3, size=1000.0, marketMaker='NSDQ'), DOMLevel(price=123.35, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.4, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.41, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.48, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=123.5, size=300.0, marketMaker='NSDQ'), DOMLevel(price=123.59, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.6, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.63, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.69, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.7, size=1500.0, marketMaker='ARCA'), DOMLevel(price=123.76, size=100.0, marketMaker='ARCA'), DOMLevel(price=123.77, size=1000.0, marketMaker='NSDQ'), DOMLevel(price=123.8, size=400.0, marketMaker='ARCA'), DOMLevel(price=123.83, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.87, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.88, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.89, size=1000.0, marketMaker='ARCA'), DOMLevel(price=123.89, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.91, size=100.0, marketMaker='ARCA'), DOMLevel(price=123.95, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.99, size=800.0, marketMaker='NSDQ'), DOMLevel(price=124.0, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.0, size=2200.0, marketMaker='NSDQ'), DOMLevel(price=124.05, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.15, size=600.0, marketMaker='NSDQ'), DOMLevel(price=124.22, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.25, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.25, size=1000.0, marketMaker='NSDQ'), DOMLevel(price=124.3, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.34, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.47, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.5, size=700.0, marketMaker='NSDQ'), DOMLevel(price=124.57, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.6, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.62, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.73, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.75, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.8, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.81, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.95, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.95, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.98, size=700.0, marketMaker='NSDQ'), DOMLevel(price=125.0, size=1100.0, marketMaker='ARCA'), DOMLevel(price=125.0, size=10600.0, marketMaker='NSDQ'), DOMLevel(price=125.02, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.05, size=200.0, marketMaker='ARCA'), DOMLevel(price=125.05, size=200.0, marketMaker='NSDQ'), DOMLevel(price=125.09, size=3000.0, marketMaker='NSDQ'), DOMLevel(price=125.11, size=600.0, marketMaker='NSDQ'), DOMLevel(price=125.18, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.19, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.2, size=200.0, marketMaker='NSDQ'), DOMLevel(price=125.21, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.25, size=300.0, marketMaker='ARCA'), DOMLevel(price=125.25, size=700.0, marketMaker='NSDQ'), DOMLevel(price=125.28, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.3, size=100.0, marketMaker='ARCA'), DOMLevel(price=125.3, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.45, size=200.0, marketMaker='ARCA'), DOMLevel(price=125.45, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.46, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.5, size=1300.0, marketMaker='ARCA'), DOMLevel(price=125.5, size=1100.0, marketMaker='NSDQ'), DOMLevel(price=125.56, size=600.0, marketMaker='ARCA'), DOMLevel(price=125.6, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.68, size=600.0, marketMaker='NSDQ'), DOMLevel(price=125.75, size=100.0, marketMaker='ARCA'), DOMLevel(price=125.8, size=600.0, marketMaker='ARCA'), DOMLevel(price=125.8, size=400.0, marketMaker='NSDQ'), DOMLevel(price=125.88, size=300.0, marketMaker='ARCA'), DOMLevel(price=125.9, size=200.0, marketMaker='NSDQ'), DOMLevel(price=126.0, size=3600.0, marketMaker='ARCA'), DOMLevel(price=126.0, size=2300.0, marketMaker='NSDQ'), DOMLevel(price=126.2, size=100.0, marketMaker='NSDQ'), DOMLevel(price=126.25, size=200.0, marketMaker='NSDQ'), DOMLevel(price=126.28, size=1000.0, marketMaker='ARCA'), DOMLevel(price=126.5, size=700.0, marketMaker='ARCA'), DOMLevel(price=126.55, size=100.0, marketMaker='NSDQ'), DOMLevel(price=126.6, size=100.0, marketMaker='ARCA'), DOMLevel(price=126.6, size=200.0, marketMaker='NSDQ'), DOMLevel(price=126.7, size=400.0, marketMaker='NSDQ'), DOMLevel(price=126.75, size=500.0, marketMaker='NSDQ'), DOMLevel(price=126.87, size=100.0, marketMaker='ARCA'), DOMLevel(price=126.9, size=100.0, marketMaker='NSDQ'), DOMLevel(price=126.94, size=200.0, marketMaker='ARCA'), DOMLevel(price=126.98, size=500.0, marketMaker='NSDQ'), DOMLevel(price=127.0, size=1300.0, marketMaker='ARCA'), DOMLevel(price=127.0, size=1600.0, marketMaker='NSDQ'), DOMLevel(price=127.03, size=100.0, marketMaker='ARCA'), DOMLevel(price=127.47, size=200.0, marketMaker='ARCA'), DOMLevel(price=127.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=127.5, size=200.0, marketMaker='NSDQ'), DOMLevel(price=127.58, size=400.0, marketMaker='ARCA'), DOMLevel(price=127.6, size=100.0, marketMaker='NSDQ'), DOMLevel(price=127.75, size=200.0, marketMaker='NSDQ'), DOMLevel(price=127.86, size=100.0, marketMaker='ARCA'), DOMLevel(price=128.0, size=500.0, marketMaker='ARCA'), DOMLevel(price=128.0, size=1000.0, marketMaker='NSDQ'), DOMLevel(price=128.02, size=200.0, marketMaker='NSDQ'), DOMLevel(price=128.08, size=400.0, marketMaker='NSDQ'), DOMLevel(price=128.2, size=200.0, marketMaker='NSDQ'), DOMLevel(price=128.28, size=400.0, marketMaker='NSDQ'), DOMLevel(price=128.42, size=100.0, marketMaker='ARCA'), DOMLevel(price=128.48, size=700.0, marketMaker='NSDQ'), DOMLevel(price=128.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=128.5, size=100.0, marketMaker='NSDQ'), DOMLevel(price=128.66, size=100.0, marketMaker='NSDQ'), DOMLevel(price=128.73, size=100.0, marketMaker='ARCA'), DOMLevel(price=128.75, size=800.0, marketMaker='ARCA'), DOMLevel(price=128.75, size=100.0, marketMaker='NSDQ'), DOMLevel(price=128.8, size=100.0, marketMaker='ARCA'), DOMLevel(price=128.99, size=600.0, marketMaker='ARCA'), DOMLevel(price=129.0, size=1500.0, marketMaker='ARCA'), DOMLevel(price=129.0, size=1100.0, marketMaker='NSDQ'), DOMLevel(price=129.2, size=500.0, marketMaker='NSDQ'), DOMLevel(price=129.31, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.4, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.4, size=100.0, marketMaker='NSDQ'), DOMLevel(price=129.45, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.46, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.5, size=800.0, marketMaker='ARCA'), DOMLevel(price=129.53, size=1000.0, marketMaker='ARCA'), DOMLevel(price=129.68, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.77, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.8, size=200.0, marketMaker='NSDQ'), DOMLevel(price=129.88, size=800.0, marketMaker='NSDQ'), DOMLevel(price=129.9, size=200.0, marketMaker='ARCA'), DOMLevel(price=129.9, size=100.0, marketMaker='NSDQ'), DOMLevel(price=129.91, size=200.0, marketMaker='ARCA'), DOMLevel(price=129.91, size=100.0, marketMaker='NSDQ'), DOMLevel(price=129.97, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.99, size=300.0, marketMaker='ARCA'), DOMLevel(price=129.99, size=1000.0, marketMaker='NSDQ'), DOMLevel(price=130.0, size=9400.0, marketMaker='ARCA'), DOMLevel(price=130.0, size=2100.0, marketMaker='NSDQ'), DOMLevel(price=130.2, size=500.0, marketMaker='ARCA'), DOMLevel(price=130.24, size=300.0, marketMaker='ARCA'), DOMLevel(price=130.24, size=100.0, marketMaker='NSDQ'), DOMLevel(price=130.25, size=100.0, marketMaker='NSDQ'), DOMLevel(price=130.32, size=100.0, marketMaker='ARCA'), DOMLevel(price=130.48, size=300.0, marketMaker='ARCA'), DOMLevel(price=130.5, size=1700.0, marketMaker='ARCA'), DOMLevel(price=130.68, size=100.0, marketMaker='ARCA'), DOMLevel(price=130.69, size=200.0, marketMaker='ARCA'), DOMLevel(price=130.8, size=100.0, marketMaker='ARCA'), DOMLevel(price=130.81, size=100.0, marketMaker='ARCA'), DOMLevel(price=130.98, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.0, size=3900.0, marketMaker='ARCA'), DOMLevel(price=131.0, size=400.0, marketMaker='NSDQ'), DOMLevel(price=131.05, size=200.0, marketMaker='ARCA'), DOMLevel(price=131.3, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.31, size=300.0, marketMaker='ARCA'), DOMLevel(price=131.45, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.47, size=600.0, marketMaker='ARCA'), DOMLevel(price=131.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.53, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.6, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.66, size=200.0, marketMaker='ARCA'), DOMLevel(price=131.7, size=200.0, marketMaker='ARCA'), DOMLevel(price=131.8, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.88, size=500.0, marketMaker='NSDQ'), DOMLevel(price=131.99, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.99, size=200.0, marketMaker='NSDQ'), DOMLevel(price=132.0, size=13500.0, marketMaker='ARCA'), DOMLevel(price=132.1, size=100.0, marketMaker='ARCA'), DOMLevel(price=132.1, size=100.0, marketMaker='NSDQ'), DOMLevel(price=132.34, size=100.0, marketMaker='ARCA'), DOMLevel(price=132.37, size=100.0, marketMaker='ARCA'), DOMLevel(price=132.44, size=100.0, marketMaker='ARCA'), DOMLevel(price=132.46, size=100.0, marketMaker='ARCA'), DOMLevel(price=132.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=132.8, size=300.0, marketMaker='NSDQ'), DOMLevel(price=132.9, size=300.0, marketMaker='ARCA'), DOMLevel(price=132.96, size=200.0, marketMaker='ARCA'), DOMLevel(price=133.0, size=2900.0, marketMaker='ARCA'), DOMLevel(price=133.09, size=1100.0, marketMaker='ARCA'), DOMLevel(price=133.22, size=200.0, marketMaker='ARCA'), DOMLevel(price=133.34, size=300.0, marketMaker='ARCA'), DOMLevel(price=133.4, size=100.0, marketMaker='NSDQ')]

        # self.domBids = [DOMLevel(price=122.9, size=100.0, marketMaker='NSDQ')]#, DOMLevel(price=122.70, size=15000.0, marketMaker='ARCA'), DOMLevel(price=122.86, size=200.0, marketMaker='ARCA'), DOMLevel(price=122.86, size=200.0, marketMaker='DRCTEDGE'), DOMLevel(price=122.86, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.84, size=100.0, marketMaker='NSDQ'), DOMLevel(price=122.83, size=200.0, marketMaker='ARCA'), DOMLevel(price=122.82, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.81, size=100.0, marketMaker='NSDQ'), DOMLevel(price=122.8, size=200.0, marketMaker='MEMX'), DOMLevel(price=122.8, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.79, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.78, size=500.0, marketMaker='NSDQ'), DOMLevel(price=122.77, size=100.0, marketMaker='NSDQ'), DOMLevel(price=122.75, size=100.0, marketMaker='NSDQ'), DOMLevel(price=122.7, size=100.0, marketMaker='NSDQ'), DOMLevel(price=122.65, size=100.0, marketMaker='ARCA'), DOMLevel(price=122.61, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.6, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.6, size=200.0, marketMaker='CHX'), DOMLevel(price=122.45, size=2800.0, marketMaker='NSDQ'), DOMLevel(price=122.4, size=10200.0, marketMaker='NSDQ'), DOMLevel(price=122.35, size=100.0, marketMaker='ARCA'), DOMLevel(price=122.31, size=200.0, marketMaker='ARCA'), DOMLevel(price=122.2, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.17, size=200.0, marketMaker='NSDQ'), DOMLevel(price=122.1, size=200.0, marketMaker='ARCA'), DOMLevel(price=122.01, size=1000.0, marketMaker='ARCA'), DOMLevel(price=122.01, size=400.0, marketMaker='NSDQ'), DOMLevel(price=122.0, size=900.0, marketMaker='ARCA'), DOMLevel(price=122.0, size=1900.0, marketMaker='NSDQ'), DOMLevel(price=121.96, size=800.0, marketMaker='ARCA'), DOMLevel(price=121.92, size=100.0, marketMaker='ARCA'), DOMLevel(price=121.88, size=500.0, marketMaker='NSDQ'), DOMLevel(price=121.6, size=100.0, marketMaker='ARCA'), DOMLevel(price=121.6, size=500.0, marketMaker='NSDQ'), DOMLevel(price=121.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=121.5, size=100.0, marketMaker='NSDQ'), DOMLevel(price=121.3, size=300.0, marketMaker='NSDQ'), DOMLevel(price=121.23, size=100.0, marketMaker='NSDQ'), DOMLevel(price=121.2, size=4000.0, marketMaker='ARCA'), DOMLevel(price=121.01, size=900.0, marketMaker='BYX'), DOMLevel(price=121.01, size=200.0, marketMaker='NSDQ'), DOMLevel(price=121.0, size=1300.0, marketMaker='ARCA'), DOMLevel(price=121.0, size=1500.0, marketMaker='NSDQ'), DOMLevel(price=120.58, size=100.0, marketMaker='ARCA'), DOMLevel(price=120.54, size=100.0, marketMaker='ARCA'), DOMLevel(price=120.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=120.38, size=200.0, marketMaker='NSDQ'), DOMLevel(price=120.18, size=1000.0, marketMaker='NSDQ'), DOMLevel(price=120.1, size=100.0, marketMaker='NSDQ'), DOMLevel(price=120.04, size=100.0, marketMaker='NSDQ'), DOMLevel(price=120.0, size=2500.0, marketMaker='ARCA'), DOMLevel(price=120.0, size=2900.0, marketMaker='NSDQ'), DOMLevel(price=119.91, size=100.0, marketMaker='ARCA'), DOMLevel(price=119.9, size=100.0, marketMaker='ARCA'), DOMLevel(price=119.09, size=100.0, marketMaker='ARCA'), DOMLevel(price=119.0, size=600.0, marketMaker='ARCA'), DOMLevel(price=119.0, size=800.0, marketMaker='NSDQ'), DOMLevel(price=118.8, size=200.0, marketMaker='ARCA'), DOMLevel(price=118.54, size=100.0, marketMaker='NSDQ'), DOMLevel(price=118.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=118.25, size=100.0, marketMaker='ARCA'), DOMLevel(price=118.2, size=100.0, marketMaker='ARCA'), DOMLevel(price=118.18, size=100.0, marketMaker='ARCA'), DOMLevel(price=118.18, size=100.0, marketMaker='NSDQ'), DOMLevel(price=118.11, size=100.0, marketMaker='NSDQ'), DOMLevel(price=118.02, size=200.0, marketMaker='ARCA'), DOMLevel(price=118.0, size=1400.0, marketMaker='ARCA'), DOMLevel(price=118.0, size=700.0, marketMaker='NSDQ'), DOMLevel(price=117.93, size=100.0, marketMaker='ARCA'), DOMLevel(price=117.87, size=100.0, marketMaker='ARCA'), DOMLevel(price=117.87, size=100.0, marketMaker='NSDQ'), DOMLevel(price=117.8, size=100.0, marketMaker='ARCA'), DOMLevel(price=117.67, size=100.0, marketMaker='NSDQ'), DOMLevel(price=117.12, size=100.0, marketMaker='ARCA'), DOMLevel(price=117.0, size=400.0, marketMaker='ARCA'), DOMLevel(price=116.86, size=100.0, marketMaker='ARCA'), DOMLevel(price=116.81, size=100.0, marketMaker='NSDQ'), DOMLevel(price=116.59, size=100.0, marketMaker='ARCA'), DOMLevel(price=116.55, size=200.0, marketMaker='ARCA'), DOMLevel(price=116.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=116.46, size=300.0, marketMaker='ARCA'), DOMLevel(price=116.3, size=100.0, marketMaker='ARCA'), DOMLevel(price=116.16, size=100.0, marketMaker='ARCA'), DOMLevel(price=116.15, size=800.0, marketMaker='ARCA'), DOMLevel(price=116.0, size=1800.0, marketMaker='ARCA'), DOMLevel(price=116.0, size=100.0, marketMaker='NSDQ'), DOMLevel(price=115.88, size=100.0, marketMaker='NSDQ'), DOMLevel(price=115.69, size=100.0, marketMaker='ARCA'), DOMLevel(price=115.57, size=100.0, marketMaker='NSDQ'), DOMLevel(price=115.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=115.26, size=100.0, marketMaker='ARCA'), DOMLevel(price=115.12, size=100.0, marketMaker='ARCA'), DOMLevel(price=115.1, size=400.0, marketMaker='ARCA'), DOMLevel(price=115.07, size=200.0, marketMaker='ARCA'), DOMLevel(price=115.0, size=5600.0, marketMaker='ARCA'), DOMLevel(price=115.0, size=300.0, marketMaker='NSDQ'), DOMLevel(price=114.97, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.88, size=200.0, marketMaker='NSDQ'), DOMLevel(price=114.8, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.66, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.6, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.57, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.55, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.51, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.5, size=600.0, marketMaker='ARCA'), DOMLevel(price=114.41, size=200.0, marketMaker='ARCA'), DOMLevel(price=114.4, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.2, size=600.0, marketMaker='ARCA'), DOMLevel(price=114.14, size=400.0, marketMaker='ARCA'), DOMLevel(price=114.1, size=100.0, marketMaker='ARCA'), DOMLevel(price=114.0, size=2600.0, marketMaker='ARCA'), DOMLevel(price=114.0, size=100.0, marketMaker='NSDQ'), DOMLevel(price=113.75, size=1000.0, marketMaker='ARCA'), DOMLevel(price=113.73, size=100.0, marketMaker='ARCA'), DOMLevel(price=113.7, size=100.0, marketMaker='ARCA'), DOMLevel(price=113.69, size=100.0, marketMaker='ARCA'), DOMLevel(price=113.68, size=100.0, marketMaker='ARCA'), DOMLevel(price=113.6, size=100.0, marketMaker='ARCA'), DOMLevel(price=113.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=113.5, size=100.0, marketMaker='NSDQ'), DOMLevel(price=113.44, size=500.0, marketMaker='ARCA'), DOMLevel(price=113.33, size=300.0, marketMaker='ARCA'), DOMLevel(price=113.3, size=300.0, marketMaker='NSDQ'), DOMLevel(price=113.01, size=100.0, marketMaker='ARCA'), DOMLevel(price=113.0, size=1900.0, marketMaker='ARCA'), DOMLevel(price=112.88, size=100.0, marketMaker='ARCA'), DOMLevel(price=112.83, size=200.0, marketMaker='ARCA'), DOMLevel(price=112.53, size=100.0, marketMaker='ARCA'), DOMLevel(price=112.51, size=100.0, marketMaker='ARCA'), DOMLevel(price=112.36, size=100.0, marketMaker='NSDQ'), DOMLevel(price=112.2, size=100.0, marketMaker='ARCA'), DOMLevel(price=112.12, size=400.0, marketMaker='ARCA'), DOMLevel(price=112.1, size=100.0, marketMaker='ARCA'), DOMLevel(price=112.0, size=1700.0, marketMaker='ARCA'), DOMLevel(price=111.88, size=100.0, marketMaker='ARCA'), DOMLevel(price=111.81, size=100.0, marketMaker='NSDQ'), DOMLevel(price=111.7, size=100.0, marketMaker='ARCA'), DOMLevel(price=111.46, size=100.0, marketMaker='ARCA'), DOMLevel(price=111.11, size=400.0, marketMaker='ARCA'), DOMLevel(price=111.06, size=100.0, marketMaker='ARCA'), DOMLevel(price=111.01, size=100.0, marketMaker='ARCA'), DOMLevel(price=111.0, size=900.0, marketMaker='ARCA'), DOMLevel(price=110.67, size=100.0, marketMaker='ARCA'), DOMLevel(price=110.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=110.2, size=200.0, marketMaker='ARCA'), DOMLevel(price=110.11, size=200.0, marketMaker='ARCA'), DOMLevel(price=110.08, size=100.0, marketMaker='ARCA'), DOMLevel(price=110.05, size=200.0, marketMaker='ARCA'), DOMLevel(price=110.01, size=100.0, marketMaker='ARCA'), DOMLevel(price=110.0, size=8500.0, marketMaker='ARCA'), DOMLevel(price=110.0, size=700.0, marketMaker='NSDQ'), DOMLevel(price=109.78, size=400.0, marketMaker='ARCA'), DOMLevel(price=109.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=108.0, size=700.0, marketMaker='NSDQ'), DOMLevel(price=107.5, size=100.0, marketMaker='NSDQ'), DOMLevel(price=106.01, size=1200.0, marketMaker='NSDQ'), DOMLevel(price=105.0, size=800.0, marketMaker='NSDQ'), DOMLevel(price=101.0, size=100.0, marketMaker='NSDQ'), DOMLevel(price=100.0, size=200.0, marketMaker='NSDQ'), DOMLevel(price=93.23, size=500.0, marketMaker='NSDQ'), DOMLevel(price=80.0, size=200.0, marketMaker='NSDQ'), DOMLevel(price=0.06, size=1300.0, marketMaker='NSDQ'), DOMLevel(price=0.01, size=500.0, marketMaker='NSDQ')]
        # self.domAsks = []#DOMLevel(price=122.91, size=100.0, marketMaker='DRCTEDGE'), DOMLevel(price=123.11, size=16000.0, marketMaker='ARCA'), DOMLevel(price=122.93, size=200.0, marketMaker='ARCA'), DOMLevel(price=122.93, size=300.0, marketMaker='NSDQ'), DOMLevel(price=122.99, size=100.0, marketMaker='MEMX'), DOMLevel(price=122.99, size=300.0, marketMaker='NSDQ'), DOMLevel(price=123.0, size=1100.0, marketMaker='ARCA'), DOMLevel(price=123.0, size=300.0, marketMaker='NSDQ'), DOMLevel(price=123.02, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.08, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.1, size=1100.0, marketMaker='ARCA'), DOMLevel(price=123.15, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.16, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.2, size=200.0, marketMaker='CHX'), DOMLevel(price=123.2, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.23, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.26, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.27, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.28, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.3, size=1000.0, marketMaker='NSDQ'), DOMLevel(price=123.35, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.4, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.41, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.48, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=123.5, size=300.0, marketMaker='NSDQ'), DOMLevel(price=123.59, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.6, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.63, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.69, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.7, size=1500.0, marketMaker='ARCA'), DOMLevel(price=123.76, size=100.0, marketMaker='ARCA'), DOMLevel(price=123.77, size=1000.0, marketMaker='NSDQ'), DOMLevel(price=123.8, size=400.0, marketMaker='ARCA'), DOMLevel(price=123.83, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.87, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.88, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.89, size=1000.0, marketMaker='ARCA'), DOMLevel(price=123.89, size=100.0, marketMaker='NSDQ'), DOMLevel(price=123.91, size=100.0, marketMaker='ARCA'), DOMLevel(price=123.95, size=200.0, marketMaker='NSDQ'), DOMLevel(price=123.99, size=800.0, marketMaker='NSDQ'), DOMLevel(price=124.0, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.0, size=2200.0, marketMaker='NSDQ'), DOMLevel(price=124.05, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.15, size=600.0, marketMaker='NSDQ'), DOMLevel(price=124.22, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.25, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.25, size=1000.0, marketMaker='NSDQ'), DOMLevel(price=124.3, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.34, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.47, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.5, size=700.0, marketMaker='NSDQ'), DOMLevel(price=124.57, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.6, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.62, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.73, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.75, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.8, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.81, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.95, size=100.0, marketMaker='ARCA'), DOMLevel(price=124.95, size=100.0, marketMaker='NSDQ'), DOMLevel(price=124.98, size=700.0, marketMaker='NSDQ'), DOMLevel(price=125.0, size=1100.0, marketMaker='ARCA'), DOMLevel(price=125.0, size=10600.0, marketMaker='NSDQ'), DOMLevel(price=125.02, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.05, size=200.0, marketMaker='ARCA'), DOMLevel(price=125.05, size=200.0, marketMaker='NSDQ'), DOMLevel(price=125.09, size=3000.0, marketMaker='NSDQ'), DOMLevel(price=125.11, size=600.0, marketMaker='NSDQ'), DOMLevel(price=125.18, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.19, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.2, size=200.0, marketMaker='NSDQ'), DOMLevel(price=125.21, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.25, size=300.0, marketMaker='ARCA'), DOMLevel(price=125.25, size=700.0, marketMaker='NSDQ'), DOMLevel(price=125.28, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.3, size=100.0, marketMaker='ARCA'), DOMLevel(price=125.3, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.45, size=200.0, marketMaker='ARCA'), DOMLevel(price=125.45, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.46, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.5, size=1300.0, marketMaker='ARCA'), DOMLevel(price=125.5, size=1100.0, marketMaker='NSDQ'), DOMLevel(price=125.56, size=600.0, marketMaker='ARCA'), DOMLevel(price=125.6, size=100.0, marketMaker='NSDQ'), DOMLevel(price=125.68, size=600.0, marketMaker='NSDQ'), DOMLevel(price=125.75, size=100.0, marketMaker='ARCA'), DOMLevel(price=125.8, size=600.0, marketMaker='ARCA'), DOMLevel(price=125.8, size=400.0, marketMaker='NSDQ'), DOMLevel(price=125.88, size=300.0, marketMaker='ARCA'), DOMLevel(price=125.9, size=200.0, marketMaker='NSDQ'), DOMLevel(price=126.0, size=3600.0, marketMaker='ARCA'), DOMLevel(price=126.0, size=2300.0, marketMaker='NSDQ'), DOMLevel(price=126.2, size=100.0, marketMaker='NSDQ'), DOMLevel(price=126.25, size=200.0, marketMaker='NSDQ'), DOMLevel(price=126.28, size=1000.0, marketMaker='ARCA'), DOMLevel(price=126.5, size=700.0, marketMaker='ARCA'), DOMLevel(price=126.55, size=100.0, marketMaker='NSDQ'), DOMLevel(price=126.6, size=100.0, marketMaker='ARCA'), DOMLevel(price=126.6, size=200.0, marketMaker='NSDQ'), DOMLevel(price=126.7, size=400.0, marketMaker='NSDQ'), DOMLevel(price=126.75, size=500.0, marketMaker='NSDQ'), DOMLevel(price=126.87, size=100.0, marketMaker='ARCA'), DOMLevel(price=126.9, size=100.0, marketMaker='NSDQ'), DOMLevel(price=126.94, size=200.0, marketMaker='ARCA'), DOMLevel(price=126.98, size=500.0, marketMaker='NSDQ'), DOMLevel(price=127.0, size=1300.0, marketMaker='ARCA'), DOMLevel(price=127.0, size=1600.0, marketMaker='NSDQ'), DOMLevel(price=127.03, size=100.0, marketMaker='ARCA'), DOMLevel(price=127.47, size=200.0, marketMaker='ARCA'), DOMLevel(price=127.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=127.5, size=200.0, marketMaker='NSDQ'), DOMLevel(price=127.58, size=400.0, marketMaker='ARCA'), DOMLevel(price=127.6, size=100.0, marketMaker='NSDQ'), DOMLevel(price=127.75, size=200.0, marketMaker='NSDQ'), DOMLevel(price=127.86, size=100.0, marketMaker='ARCA'), DOMLevel(price=128.0, size=500.0, marketMaker='ARCA'), DOMLevel(price=128.0, size=1000.0, marketMaker='NSDQ'), DOMLevel(price=128.02, size=200.0, marketMaker='NSDQ'), DOMLevel(price=128.08, size=400.0, marketMaker='NSDQ'), DOMLevel(price=128.2, size=200.0, marketMaker='NSDQ'), DOMLevel(price=128.28, size=400.0, marketMaker='NSDQ'), DOMLevel(price=128.42, size=100.0, marketMaker='ARCA'), DOMLevel(price=128.48, size=700.0, marketMaker='NSDQ'), DOMLevel(price=128.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=128.5, size=100.0, marketMaker='NSDQ'), DOMLevel(price=128.66, size=100.0, marketMaker='NSDQ'), DOMLevel(price=128.73, size=100.0, marketMaker='ARCA'), DOMLevel(price=128.75, size=800.0, marketMaker='ARCA'), DOMLevel(price=128.75, size=100.0, marketMaker='NSDQ'), DOMLevel(price=128.8, size=100.0, marketMaker='ARCA'), DOMLevel(price=128.99, size=600.0, marketMaker='ARCA'), DOMLevel(price=129.0, size=1500.0, marketMaker='ARCA'), DOMLevel(price=129.0, size=1100.0, marketMaker='NSDQ'), DOMLevel(price=129.2, size=500.0, marketMaker='NSDQ'), DOMLevel(price=129.31, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.4, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.4, size=100.0, marketMaker='NSDQ'), DOMLevel(price=129.45, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.46, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.5, size=800.0, marketMaker='ARCA'), DOMLevel(price=129.53, size=1000.0, marketMaker='ARCA'), DOMLevel(price=129.68, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.77, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.8, size=200.0, marketMaker='NSDQ'), DOMLevel(price=129.88, size=800.0, marketMaker='NSDQ'), DOMLevel(price=129.9, size=200.0, marketMaker='ARCA'), DOMLevel(price=129.9, size=100.0, marketMaker='NSDQ'), DOMLevel(price=129.91, size=200.0, marketMaker='ARCA'), DOMLevel(price=129.91, size=100.0, marketMaker='NSDQ'), DOMLevel(price=129.97, size=100.0, marketMaker='ARCA'), DOMLevel(price=129.99, size=300.0, marketMaker='ARCA'), DOMLevel(price=129.99, size=1000.0, marketMaker='NSDQ'), DOMLevel(price=130.0, size=9400.0, marketMaker='ARCA'), DOMLevel(price=130.0, size=2100.0, marketMaker='NSDQ'), DOMLevel(price=130.2, size=500.0, marketMaker='ARCA'), DOMLevel(price=130.24, size=300.0, marketMaker='ARCA'), DOMLevel(price=130.24, size=100.0, marketMaker='NSDQ'), DOMLevel(price=130.25, size=100.0, marketMaker='NSDQ'), DOMLevel(price=130.32, size=100.0, marketMaker='ARCA'), DOMLevel(price=130.48, size=300.0, marketMaker='ARCA'), DOMLevel(price=130.5, size=1700.0, marketMaker='ARCA'), DOMLevel(price=130.68, size=100.0, marketMaker='ARCA'), DOMLevel(price=130.69, size=200.0, marketMaker='ARCA'), DOMLevel(price=130.8, size=100.0, marketMaker='ARCA'), DOMLevel(price=130.81, size=100.0, marketMaker='ARCA'), DOMLevel(price=130.98, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.0, size=3900.0, marketMaker='ARCA'), DOMLevel(price=131.0, size=400.0, marketMaker='NSDQ'), DOMLevel(price=131.05, size=200.0, marketMaker='ARCA'), DOMLevel(price=131.3, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.31, size=300.0, marketMaker='ARCA'), DOMLevel(price=131.45, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.47, size=600.0, marketMaker='ARCA'), DOMLevel(price=131.5, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.53, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.6, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.66, size=200.0, marketMaker='ARCA'), DOMLevel(price=131.7, size=200.0, marketMaker='ARCA'), DOMLevel(price=131.8, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.88, size=500.0, marketMaker='NSDQ'), DOMLevel(price=131.99, size=100.0, marketMaker='ARCA'), DOMLevel(price=131.99, size=200.0, marketMaker='NSDQ'), DOMLevel(price=132.0, size=13500.0, marketMaker='ARCA'), DOMLevel(price=132.1, size=100.0, marketMaker='ARCA'), DOMLevel(price=132.1, size=100.0, marketMaker='NSDQ'), DOMLevel(price=132.34, size=100.0, marketMaker='ARCA'), DOMLevel(price=132.37, size=100.0, marketMaker='ARCA'), DOMLevel(price=132.44, size=100.0, marketMaker='ARCA'), DOMLevel(price=132.46, size=100.0, marketMaker='ARCA'), DOMLevel(price=132.5, size=200.0, marketMaker='ARCA'), DOMLevel(price=132.8, size=300.0, marketMaker='NSDQ'), DOMLevel(price=132.9, size=300.0, marketMaker='ARCA'), DOMLevel(price=132.96, size=200.0, marketMaker='ARCA'), DOMLevel(price=133.0, size=2900.0, marketMaker='ARCA'), DOMLevel(price=133.09, size=1100.0, marketMaker='ARCA'), DOMLevel(price=133.22, size=200.0, marketMaker='ARCA'), DOMLevel(price=133.34, size=300.0, marketMaker='ARCA'), DOMLevel(price=133.4, size=100.0, marketMaker='NSDQ')]


class DomL2:

    __slots__ = ('ib', 'symbol', 'date', 'data_folder', 'bids', 'asks', 'timezone', 'forexFactor', 'bidsAggregated', 'asksAggregated', '_bidsSizes', '_bidsPrices', '_asksSizes', '_asksPrices', 'bidsSumSizes', 'bidsSumPrices', 'asksSumSizes', 'asksSumPrices', 'bidsAvgSizes', 'bidsAvgPrices', 'asksAvgSizes', 'asksAvgPrices', 'bidsStdSizes', 'bidsStdPrices', 'asksStdSizes', 'asksStdPrices', 'sumSizes', 'avgSizes', 'stdSizes', 'dom_table', 'full_dom_table', 'sum_table')

    def __init__(self, ib:IB, symbol:str, data_folder:str, bids:list=[], asks:list=[], timezone=CONSTANTS.TZ_WORK):
        self.ib = ib
        self.symbol = symbol.upper()
        self.forexFactor = 1000000 if symbol[:3] in helpers.get_forex_symbols_list() else 1
        self.bids = bids
        self.asks = asks
        self.data_folder = data_folder
        self.dom_table = prettytable.PrettyTable()
        self.full_dom_table = prettytable.PrettyTable()
        self.sum_table = prettytable.PrettyTable()
        self.timezone = timezone
        self.__initiate(bids, asks)

    def __initiate(self, bids, asks):
        self.bids = bids
        self.asks = asks
        self.date = datetime.now(tz=self.timezone)
        self.bidsAggregated = self.__aggregate_levels(bids, reverse=True)
        self.asksAggregated = self.__aggregate_levels(asks)
        self._bidsSizes = [level['size']/self.forexFactor for level in self.bidsAggregated]
        self._asksSizes = [level['size']/self.forexFactor for level in self.asksAggregated]
        self._bidsPrices = [level['price'] for level in self.bidsAggregated]
        self._asksPrices = [level['price'] for level in self.asksAggregated]
        self.bidsSumSizes, self.bidsAvgSizes, self.bidsStdSizes = self.__stats_levels(self._bidsSizes)
        self.asksSumSizes, self.asksAvgSizes, self.asksStdSizes = self.__stats_levels(self._asksSizes)
        self.bidsSumPrices, self.bidsAvgPrices, self.bidsStdPrices = self.__stats_levels(self._bidsPrices)
        self.asksSumPrices, self.asksAvgPrices, self.asksStdPrices = self.__stats_levels(self._asksPrices)
        self.sumSizes, self.avgSizes, self.stdSizes = self.__stats_levels(self._bidsSizes + self._asksSizes)
        self.dom_table.title = "DOM " + self.symbol + '     ' + str(self.date)
        self.sum_table.title = "SUMs " + self.symbol + '     ' + str(self.date)
        self.full_dom_table.title = "FULL DOM " + self.symbol + '     ' + str(self.date)

    def get_dom(self, currency='USD'):
        contract, mktData = helpers.get_symbol_mkt_data(self.ib, self.symbol, currency=currency)#, exchange="NASDAQ")
        MD = self.ib.reqMktDepth(contract, numRows=2000, isSmartDepth=True, mktDepthOptions=None)
        self.ib.sleep(CONSTANTS.PROCESS_TIME['long'])

        self.__initiate(MD.domBids, MD.domAsks)

        return MD

    def __aggregate_levels(self, levels, reverse=False):

        if levels:
            # Sort levels by 'price' and group by price and summing sizes
            levels_grouped = defaultdict(list)
            for obj in sorted(levels, reverse=reverse, key=lambda x: x.price): levels_grouped[obj.price].append(obj)

            levels_aggregated = [{'price': price,
                                'size': sum(obj.size for obj in group),
                                'num_levels': len(group),
                                'distance': round(price - [level for _, level in levels_grouped.items()][0][0].price, 2)}
                                for price, group in levels_grouped.items()]

        else: levels_aggregated = []

        return levels_aggregated


    def __stats_levels(self, levels):

        if levels:
            levelsSum = sum(levels)
            levelsAvg = numpy.average(levels)
            levelsStd = numpy.std(levels)

        else: levelsSum, levelsAvg, levelsStd = [], [], []

        return levelsSum, levelsAvg, levelsStd


    def get_sign_bids(self, stdOrder=None):
        return [l for l in self.bidsAggregated if l['size'] > int(stdOrder) * self.stdSizes]


    def get_sign_asks(self, stdOrder=None):
        return [l for l in self.asksAggregated if l['size'] > int(stdOrder) * self.stdSizes]


    def __trim_levels(self, levels, col, trim_order=3):
        _, levels_avg, levels_std = self.__stats_levels([l[col] for l in levels])

        return list(filter(lambda x: x[col] > levels_avg - trim_order*levels_std and x[col] < levels_avg + trim_order*levels_std, levels))


    def plot_dom(self):

        bids_aggregated_trimmed = self.__trim_levels(self.bidsAggregated, 'price', 2)
        asks_aggregated_trimmed = self.__trim_levels(self.asksAggregated, 'price', 2)
        print('self._asksSizes[0] = ', self._asksSizes[0], "   ", type(self._asksSizes[0]))
        print("bids_aggregated_trimmed[-1]['price'] = ", bids_aggregated_trimmed[-1]['price'], "   ", type(bids_aggregated_trimmed[-1]['price']))
        range_bids = [round(r, 2) for r in numpy.arange(self._bidsPrices[0], bids_aggregated_trimmed[-1]['price'], -0.01)]
        range_asks = [round(r, 2) for r in numpy.arange(self._asksPrices[0], asks_aggregated_trimmed[-1]['price'], 0.01)]

        bids_plot_array, asks_plot_array = [], []

        for price in range_bids: bids_plot_array.append([bid['size'] for bid in bids_aggregated_trimmed if bid['price'] == price][0]) if price in [bid['price'] for bid in bids_aggregated_trimmed] else bids_plot_array.append(0)
        for price in range_asks: asks_plot_array.append([ask['size'] for ask in asks_aggregated_trimmed if ask['price'] == price][0]) if price in [ask['price'] for ask in asks_aggregated_trimmed] else asks_plot_array.append(0)

        # Define Plots
        fig, (sub_fig_bids, sub_fig_asks) = pyplot.subplots(nrows=2, ncols=1, figsize=(10, 8))
        fig.tight_layout(pad=5.0)
        fig.suptitle(self.symbol + " DOM Level 2", fontsize='x-large')

        # range_bids = [*range(1, len(self._bidsSizes)+1)]
        # range_asks = [*range(1, len(self._asksSizes)+1)]
        sub_fig_bids.bar(range_bids, bids_plot_array, tick_label=str(self._bidsPrices), width = 0.8, color = 'green')
        sub_fig_bids.set_title('Bids', fontsize='large')
        sub_fig_bids.set_xlabel('Prices')
        sub_fig_bids.set_ylabel('Sizes')
        sub_fig_asks.bar(range_asks, asks_plot_array, tick_label=str(self._asksPrices), width = 0.8, color = 'green')
        sub_fig_asks.set_title('Asks', fontsize='large')
        sub_fig_asks.set_xlabel('Prices')
        sub_fig_asks.set_ylabel('Sizes')

        pyplot.show()


    def __get_sentiment(self, criteriaAsk, criteriaBid, inverse=False):

        perc_diff = round(abs(100 - (min(criteriaAsk, criteriaBid) * 100 / max(criteriaAsk, criteriaBid))), 2) if (criteriaAsk != criteriaBid and max(criteriaAsk, criteriaBid)) else 0
        if criteriaBid > criteriaAsk: perc_diff = -perc_diff
        if inverse: perc_diff = -perc_diff

        return perc_diff


    def __populate_dom_table(self, levels, full_dom=False):

        self.dom_table.clear_rows()
        self.dom_table.field_names = ['Bids P', 'Bids S', 'Dist', 'Asks S', 'Asks L']
        dom_table_list = []
        for i, level in enumerate(levels):
            el_price = str(level['price']) + ' (' + str(level['num_levels']) + ')'
            if i > 0 and level['distance'] == -levels[i-1]['distance']:
                if level['distance'] < 0:
                    dom_table_list[-1][0] = el_price
                    dom_table_list[-1][1] = float(dom_table_list[-1][1]) + level['size'] if dom_table_list[-1][1] != '' else 0
                else:
                    dom_table_list[-1][4] = el_price
                    dom_table_list[-1][3] = float(dom_table_list[-1][3]) + level['size'] if dom_table_list[-1][3] != '' else 0

            else:
                if level['distance'] < 0: dom_table_list.append([el_price, level['size'], -level['distance'], '', ''])
                else: dom_table_list.append(['', '', level['distance'], level['size'], el_price])

        if not full_dom: self.dom_table.add_rows(dom_table_list)
        else: self.full_dom_table.add_rows(dom_table_list)


    def __populate_sum_table(self, sign_levels, sum_ponderated, sum_sign_ponderated, sum_direct):
        self.sum_table.clear_rows()
        self.sum_table.add_column('Type', ['Levels Pond', 'Ponderated', 'Direct'])
        self.sum_table.add_column('Bids', [len(sign_levels['bids']), round(sum_ponderated['bids'], 2), sum_direct['bids']])
        self.sum_table.add_column('Asks', [len(sign_levels['asks']), round(sum_ponderated['asks'], 2), sum_direct['asks']])
        self.sum_table.add_column('%', [sum_sign_ponderated['perc_diff'], sum_ponderated['perc_diff'], sum_direct['perc_diff']])
        self.sum_table.add_column('Scores', [sum_sign_ponderated['score_diff'], sum_ponderated['score_diff'], sum_direct['score_diff']])


    def __save_to_csv(self, sign_levels, sum_ponderated, sum_sign_ponderated, sum_direct):

        dataL2_csv_file = os.path.join(self.data_folder, 'dataL2_' + self.date.strftime('%Y-%m-%d') + '.csv')
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)
            print(f"Directory '{self.data_folder}' created successfully.")

        helpers.initialize_symbols_csv_file(dataL2_csv_file, ['Time', 'Symbol', 'Score levels', '% Levels', 'N Levels Bids', 'N Levels Asks', 'Levels Bids', 'Levels Asks', 'Score Sum', '% Sum', 'Sum Bids', 'Sum Asks', 'Score Direct', '% Direct', 'Direct Bids', 'Direct Asks'])

        with open(dataL2_csv_file, mode='a') as file:
            journal_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # Write to csv file
            text_line_infos = [self.date.strftime('%Y-%m-%dT%H:%M:%S'), self.symbol]
            text_line_sum = [sum_ponderated['score_diff'], sum_ponderated['perc_diff'], round(sum_ponderated['bids'], 2), round(sum_ponderated['asks'], 2)]
            text_line_levels = [sign_levels['score_diff'], sign_levels['perc_diff'], len(sign_levels['bids']), len(sign_levels['asks']), [l['price'] for l in sign_levels['bids']], [l['price'] for l in sign_levels['asks']]]
            text_line_direct = [sum_direct['score_diff'], sum_direct['perc_diff'], round(sum_direct['bids'], 2), round(sum_direct['asks'], 2)]
            # journal_writer.writerow([''])
            journal_writer.writerow(text_line_infos + text_line_levels + text_line_sum + text_line_direct)
            print("\nRecorded data for ", self.symbol, " at ", dataL2_csv_file, "\n")


    def print_dom(self, full_dom=False):
        try:
            if full_dom: print(self.full_dom_table, '\n')
            print(self.dom_table, '\n\n', self.sum_table)
        except Exception as e: print("Could not print tables. Error: ", e, "  |  Full error: ", traceback.format_exc())


    def assess_dom(self, full_dom=False, save_to_file=False):

        try:
            # Calculate weighted sum of sizes for each prices
            bids_aggregated_trimmed = self.__trim_levels(self.bidsAggregated, 'price')
            asks_aggregated_trimmed = self.__trim_levels(self.asksAggregated, 'price')

            _, _, bids_aggregated_trimmed_prices_std = self.__stats_levels([level['price'] for level in bids_aggregated_trimmed])
            _, _, asks_aggregated_trimmed_prices_std = self.__stats_levels([level['price'] for level in asks_aggregated_trimmed])

            weights_bids = [numpy.exp(-((price - self._bidsPrices[0])**2)/(2 * bids_aggregated_trimmed_prices_std**2)) for price in self._bidsPrices[2:]]
            weights_asks = [numpy.exp(-((price - self._asksPrices[0])**2)/(2 * asks_aggregated_trimmed_prices_std**2)) for price in self._asksPrices[2:]]

            sum_ponderated = {'bids': sum(x * y for x, y in zip(self._bidsSizes, weights_bids)),# / sum(weights_bids)
                              'asks': sum(x * y for x, y in zip(self._asksSizes, weights_asks)),# / sum(weights_asks)
                              'perc_diff': '', 'score_diff': ''}
            sum_ponderated['perc_diff'] = self.__get_sentiment(sum_ponderated['asks'], sum_ponderated['bids'])
            sum_ponderated['score_diff'] = math.floor(sum_ponderated['perc_diff'] / 10)

            # Calculate significant levels's number and ponderated sum
            sign_levels = {'bids': self.get_sign_bids(3), 'asks': self.get_sign_asks(3), 'perc_diff': '', 'score_diff': ''}
            sign_bids_asks_levels = sorted(sign_levels['bids'] + sign_levels['asks'], key=lambda x: abs(x['distance']))
            sign_levels['perc_diff'] = self.__get_sentiment(len(sign_levels['asks']), len(sign_levels['bids']))
            sign_levels['score_diff'] = math.floor(sign_levels['perc_diff'] / 10)

            _, _, sign_bids_asks_levels_std = self.__stats_levels([level['price'] for level in sign_bids_asks_levels])
            if sign_bids_asks_levels_std == 0: sign_bids_asks_levels_std = 0.1 # Case of only one level total amongst bids and asks -> std = 0

            weights_sign_bids = [numpy.exp(-(level['distance']**2)/(2 * sign_bids_asks_levels_std**2)) for level in sign_levels['bids']]
            weights_sign_asks = [numpy.exp(-(level['distance']**2)/(2 * sign_bids_asks_levels_std**2)) for level in sign_levels['asks']]

            sum_sign_ponderated = {'bids': sum(x * y for x, y in zip([level['size'] for level in sign_levels['bids']], weights_sign_bids)),# / sum(weights_sign_bids) if sign_bids_levels else 0
                                   'asks': sum(x * y for x, y in zip([level['size'] for level in sign_levels['asks']], weights_sign_asks)),# / sum(weights_sign_asks) if sign_asks_levels else 0
                                   'perc_diff': '', 'score_diff': ''}
            sum_sign_ponderated['perc_diff'] = self.__get_sentiment(sum_sign_ponderated['asks'], sum_sign_ponderated['bids'])
            sum_sign_ponderated['score_diff'] = math.floor(sum_sign_ponderated['perc_diff'] / 10)

            # Calculate sum of prices for first two levels
            sum_direct = {'bids': self.bidsAggregated[0]['size'] + self.bidsAggregated[1]['size'],
                          'asks': self.asksAggregated[0]['size'] + self.asksAggregated[1]['size'],
                          'perc_diff': '', 'score_diff': ''}
            sum_direct['perc_diff'] = self.__get_sentiment(sum_direct['asks'], sum_direct['bids'], inverse=True)
            sum_direct['score_diff'] = math.floor(sum_direct['perc_diff'] / 10)

            # Populate tables and save to CSV
            if full_dom:
                bids_asks_aggregated = sorted(self.bidsAggregated + self.asksAggregated, key=lambda x: abs(x['distance']))
                self.__populate_dom_table(bids_asks_aggregated, full_dom=full_dom)

            self.__populate_dom_table(sign_bids_asks_levels)
            self.__populate_sum_table(sign_levels, sum_ponderated, sum_sign_ponderated, sum_direct)
            if save_to_file: self.__save_to_csv(sign_levels, sum_ponderated, sum_sign_ponderated, sum_direct)

            return sum_ponderated

        except Exception as e:
            print("Could not assess DOM. Error: ", e, "  |  Full error: ", traceback.format_exc())
            return [], []






if __name__ == "__main__":

    import helpers

    args = sys.argv

    symbol = 'TSLA'
    currency = 'USD'
    dummy = False
    full_dom = False
    time_wait = 5*60
    for arg in args:
        if 'sym' in arg: symbol = arg[3:]
        if 'cont' in arg: continuous = True
        if 'dummy' in arg: dummy = True
        if 'full' in arg: full_dom = True
        if 'wait' in arg:
            try: time_wait = int(float(arg[4:]) * 60)
            except Exception as e: print("Could not get time_wait parameter. Error: ", e, "  |  Full error: ", traceback.format_exc())

    paperTrading = False if len(args) > 1 and 'live' in args else True

    # TWS Connection
    ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)
    if not ibConnection:
        paperTrading = not paperTrading
        ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)

    daily_data_folder = helpers.get_path_daily_data_folder()

    if dummy:
        MD = DummyMktDepth()
        domL2 = DomL2(ib, symbol, daily_data_folder, MD.domBids, MD.domAsks)
    else:
        print("\nFetching Market Depth for symbol ", symbol)
        domL2 = DomL2(ib, symbol, daily_data_folder)
        MD = domL2.get_dom()
        ib.sleep(CONSTANTS.PROCESS_TIME['medium'])

    if MD.domBids and MD.domAsks:
        # FD = ib.reqFundamentalData(contract, reportType='ReportsFinSummary', fundamentalDataOptions=[]) #https://ib-insync.readthedocs.io/api.html#ib_insync.ib.IB.reqMktDepth
        # ib.sleep(1)

        # for i, md in enumerate(domL2.bids):
        #     print(md.price, "    ", md.size, "   |   ", domL2.asks[i].price, "    ", domL2.asks[i].size)
        # print()

        domL2.assess_dom(full_dom=full_dom, save_to_file=True)
        domL2.print_dom(full_dom=full_dom)

    else:
        print("Could not fetch Market Depth data.")

print("\n\n")





# if __name__ == "__main__":

#     import helpers

#     args = sys.argv

#     currency = 'USD'
#     dummy = False
#     continuous = False
#     symbol_single = None
#     full_dom = False
#     time_wait = 5*60
#     for arg in args:
#         if 'sym' in arg: symbol_single = [arg[3:]]
#         if 'cont' in arg: continuous = True
#         if 'dummy' in arg: dummy = True
#         if 'full' in arg: full_dom = True
#         if 'wait' in arg:
#             try: time_wait = int(float(arg[4:]) * 60)
#             except Exception as e: print("Could not get time_wait parameter. Error: ", e, "  |  Full error: ", traceback.format_exc())

#     paperTrading = False if len(args) > 1 and 'live' in args else True

#     # TWS Connection
#     ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)
#     if not ibConnection:
#         paperTrading = not paperTrading
#         ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)

#     daily_data_folder = helpers.get_path_daily_data_folder()

#     counter = 60
#     while counter > 0:

#         symbols = transform_symbols(get_symbols()) if not symbol_single else [symbol_single]

#         for symbols_subset in symbols:
#             print("Subset Symbols: ", symbols_subset)

#             for symbol in symbols_subset:
#                 if dummy:
#                     MD = DummyMktDepth()
#                     domL2 = DomL2(ib, symbol, daily_data_folder, MD.domBids, MD.domAsks)
#                 else:
#                     print("\nFetching Market Depth for symbol ", symbol)
#                     domL2 = DomL2(ib, symbol, daily_data_folder)
#                     MD = domL2.get_dom()
#                     ib.sleep(CONSTANTS.PROCESS_TIME['medium'])

#                 if MD.domBids and MD.domAsks:
#                     # FD = ib.reqFundamentalData(contract, reportType='ReportsFinSummary', fundamentalDataOptions=[]) #https://ib-insync.readthedocs.io/api.html#ib_insync.ib.IB.reqMktDepth
#                     # ib.sleep(1)

#                     # for i, md in enumerate(domL2.bids):
#                     #     print(md.price, "    ", md.size, "   |   ", domL2.asks[i].price, "    ", domL2.asks[i].size)
#                     # print()

#                     domL2.assess_dom(full_dom=full_dom)
#                     domL2.print_dom(full_dom=full_dom)

#                 else:
#                     print("Could not fetch Market Depth data.")

#             if continuous:
#                 print("\nDisconnecting IB")
#                 ib.disconnect()
#                 ib.sleep(2)
#                 print("\nReconnecting IB")
#                 ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)

#         if continuous:

#             print("\nDisconnecting IB")
#             ib.disconnect()

#             counter = counter - 1
#             helpers.sleep_display(time_wait)
#             print("\nReconnecting IB")
#             ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)

#         else: counter = 0

# print("\n\n")
