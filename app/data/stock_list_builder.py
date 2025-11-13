import sys, os, yfinance as yf, pandas as pd, numpy as np
from ib_insync import *
from typing import List, Optional

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import constants, helpers

class StockUniverseBuilder:
    def __init__(self, ib: IB = IB(), use_sp500: bool = True, use_russell2000: bool = False, russell_csv_path: Optional[str] = None,
                 use_custom: bool = False, custom_csv_path: Optional[str] = None,
                 use_existing: bool = False, existing_csv_path: Optional[str] = None):
        self.ib = ib
        self.use_sp500 = use_sp500
        self.use_russell2000 = use_russell2000
        self.russell_csv_path = russell_csv_path
        self.use_custom = use_custom
        self.custom_csv_path = custom_csv_path
        self.use_existing = use_existing
        self.existing_csv_path = existing_csv_path
        self.symbols = []

    def get_sp500_symbols(self) -> List[str]:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return table[0]['Symbol'].tolist()

    def get_russell2000_symbols(self) -> List[str]:
        if self.russell_csv_path:
            df = pd.read_csv(self.russell_csv_path)
            return df['symbol'].dropna().tolist()
        else:
            raise ValueError("Russell 2000 CSV path not provided.")

    def get_custom_symbols(self) -> List[str]:
        if self.custom_csv_path:
            return pd.read_csv(self.russell_csv_path)['symbol'].dropna().tolist()
        else:
            raise ValueError("Custom path not provided.")

    def get_existing_symbols(self) -> pd.DataFrame:
        if self.existing_csv_path:
            return pd.read_csv(self.existing_csv_path)
        else:
            raise ValueError("Existiing CSV path not provided.")

    def fetch_symbols(self):
        symbols = []
        if self.use_sp500:
            symbols += self.get_sp500_symbols()
        if self.use_russell2000:
            symbols += self.get_russell2000_symbols()
        if self.use_custom:
            symbols += self.get_custom_symbols()
        self.symbols = list(set(symbols))

    def classify_stock(self, symbol: str) -> Optional[dict]:
        try:
            # stock = yf.Ticker(symbol)
            # market_cap = stock.info.get("marketCap")
            # avg_volume = stock.info.get("averageVolume", 0)
            # price = stock.info.get("previousClose", 0)
            # sector = stock.info.get("sector", "Unknown")
            market_cap = helpers.convert_large_numbers(helpers.get_stock_info_from_Finviz(symbol, "Market Cap"))
            avg_volume = helpers.convert_large_numbers(helpers.get_stock_info_from_Finviz(symbol, "Avg Volume"))
            price = float(helpers.get_stock_info_from_Finviz(symbol, "Price"))
            sector = helpers.get_index_from_symbol(self.ib, symbol)

            if not market_cap or avg_volume < 500_000:
                return None

            # cap_cat = 'large' if market_cap >= 200e9 else 'mid' if market_cap >= 10e9 else 'small'
            cap_cat = helpers.categorize_market_cap(market_cap)
            price_cat = 'low' if price < 20 else 'mid' if price < 100 else 'high'

            return {
                "symbol": symbol,
                "market_cap": market_cap,
                "avg_volume": avg_volume,
                "price": price,
                "sector": sector,
                "cap_cat": cap_cat,
                "price_cat": price_cat
            }

        except Exception:
            return None

    def build_stock_list(self, n_per_category: int = 15, seed: Optional[int] = None) -> pd.DataFrame:

        if self.use_existing:
            df = self.get_existing_symbols()

        else:
            self.fetch_symbols()
            classified = []

            for index, symbol in enumerate(self.symbols):
                print(f"Fetching info for symbol {symbol} - Remaining: {len(self.symbols) - index}")
                data = self.classify_stock(symbol)
                if data:
                    classified.append(data)

            df = pd.DataFrame(classified)

        if df.empty:
            return pd.DataFrame(), df
        else:
            # Ensure we have enough for each category
            df_select = pd.concat([
                df[df["cap_cat"] == "large"].sample(n=min(n_per_category, len(df[df["cap_cat"] == "large"])), random_state=seed),
                df[df["cap_cat"] == "mid"].sample(n=min(n_per_category, len(df[df["cap_cat"] == "mid"])), random_state=seed),
                df[df["cap_cat"] == "small"].sample(n=min(n_per_category, len(df[df["cap_cat"] == "small"])), random_state=seed),
            ])

            return df_select.reset_index(drop=True), df

    # def save_universe_to_csv(self, df: pd.DataFrame, file_path: str = "stock_list.csv"):
    #     df.to_csv(file_path, index=False)
    #     print(f"Saved stock list to {file_path}")



if __name__ == "__main__":

    args = sys.argv

    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    # Arguments management
    args = sys.argv
    paperTrading = not 'live' in args
    use_sp500 = 'sp500' in args
    use_russell2000 = 'russell2000' in args
    use_custom = 'custom' in args
    use_existing = 'existing' in args
    seed = next((arg[5:] for arg in args if arg.startswith('seed=')), None)

    # TWS Connection
    ib, ibConnection = helpers.IBKRConnect_any(IB(), paper=paperTrading)

    # # TWS Connection
    # ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)
    # if not ibConnection:
    #     paperTrading = not paperTrading
    #     ib, ibConnection = helpers.IBKRConnect(IB(), paper=paperTrading)

    # Setup
    if seed is not None: seed = int(seed)
    ext_seed = f'_seed{seed}.csv' if seed is not None else '.csv'
    hist_folder = constants.PATHS.folders_path['market_data']
    market_data_folder = constants.PATHS.folders_path['market_data']

    russell_csv_path = constants.PATHS.csv_files['russell2000']
    custom_csv_path = None
    stock_list_csv = os.path.splitext(constants.PATHS.csv_files['stock_list'])[0] + ext_seed
    full_stock_list_csv = os.path.join(constants.PATHS.folders_path['market_data'], 'stock_list_full.csv')
    existing_csv_path = full_stock_list_csv if os.path.exists(full_stock_list_csv) else None


    builder = StockUniverseBuilder(ib, use_sp500=use_sp500, use_russell2000=use_russell2000, russell_csv_path=russell_csv_path,
                                   use_custom=use_custom, custom_csv_path=custom_csv_path,
                                   use_existing=use_existing, existing_csv_path=existing_csv_path)
    stock_list_df, full_stock_list_df = builder.build_stock_list(n_per_category=15, seed=seed)
    print()

    helpers.save_df_to_file(stock_list_df, stock_list_csv, file_format='csv')
    helpers.save_df_to_file(full_stock_list_df, full_stock_list_csv, file_format='csv')
