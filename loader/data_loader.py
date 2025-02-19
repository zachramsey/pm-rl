import os

from util.util import get_index_tickers
from loader.load_yf import LoadYF
from loader.asset_pool import AssetPool

from config.base import DEBUG, PICKLE_DIR, TICKERS, MIN_VOLUME, SCALER, TRAIN_RATIO, WINDOW_SIZE, NUM_ASSETS, INDICATORS

class StockDataLoader:
    def __init__(self):
        self.pool = None                # Stock data class
        self.train_data = None          # Training data
        self.test_data = None           # Testing data

        # Load the tickers and stock data
        print("-"*50)
        if DEBUG and os.path.exists(PICKLE_DIR):
            print("Loading Asset Pool...", end='\r')
            self.pool = AssetPool.load(PICKLE_DIR)
            print("Loading Asset Pool... Done!")
        else:
            print("Fetching Tickers...", end='\r')
            tickers = TICKERS if isinstance(TICKERS, list) else get_index_tickers(TICKERS)
            print("Fetching Tickers... Done!")

            print("Fetching Stock Data...", end='\r')
            asset_data = LoadYF(tickers)
            prop = asset_data.get_prop()
            data = asset_data.get_data()
            print("Fetching Stock Data... Done!")

            print("Building Asset Pool...", end='\r')
            self.pool = AssetPool(prop, data)
            print("Building Asset Pool... Done!")

            if DEBUG and not os.path.exists(PICKLE_DIR):
                print("Saving Asset Pool...", end='\r')
                self.pool.save(PICKLE_DIR)
                print("Saving Asset Pool... Done!")
        print("-"*50)

        print("Picking Assets...", end='\r')
        self._condition_data()
        print("Picking Assets... Done!")

        print("Adding Indicators...", end='\r')
        self.pool.add_indicators(INDICATORS)
        print("Adding Indicators... Done!")

        print("Making Data Stationary...", end='\r')
        self.pool.stationary()
        print("Making Data Stationary... Done!")

        print("Scaling Data...", end='\r')
        self.pool.scale(SCALER)
        print("Scaling Data... Done!")

        print("Splitting Data...", end='\r')
        self.train_data, self.test_data = self.pool.split(TRAIN_RATIO)
        print("Splitting Data... Done!")

        print("-"*50)

    def _condition_data(self):
        # Filter tickers with insufficient average volume
        for ticker, volume in self.pool.volume().items():
            if volume.mean() < MIN_VOLUME:
                self.pool.del_asset(ticker)

        # Choose tickers with the most data
        lengths = self.pool.num_dates()
        lowest_tickers = sorted(lengths.keys(), key=lambda x: lengths[x])[:len(lengths)-NUM_ASSETS]
        for ticker in lowest_tickers:
            self.pool.del_asset(ticker)
