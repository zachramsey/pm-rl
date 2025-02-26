import os
from util.util import get_index_tickers
from data.load_yf import LoadYF
from data.asset_pool import AssetPool
from config.base import LOAD_LOCAL, PICKLE_DIR, TICKERS, MIN_VOLUME, SCALER, TRAIN_RATIO, WINDOW_SIZE, NUM_ASSETS, INDICATORS

class StockDataLoader:
    def __init__(self):
        self.pool = None                # Stock data class
        self.train_dl = None            # Training data loader
        self.test_dl = None             # Testing data loader
        self.info = None                # Information about the data

        # Load the tickers and stock data
        print("-"*50)
        if LOAD_LOCAL and os.path.exists(PICKLE_DIR + "asset_pool.pkl"):
            print("Loading Asset Pool...", end='\r')
            self.pool = AssetPool.load(PICKLE_DIR, "asset_pool")
            print("Loading Asset Pool... Done!")
        else:
            print("Fetching Tickers...", end='\r')
            tickers = get_index_tickers(TICKERS, ['HUBB'])
            print("Fetching Tickers... Done!")

            print("Fetching Stock Data...", end='\r')
            asset_data = LoadYF(tickers.keys())
            data = asset_data.get_data()
            print("Fetching Stock Data... Done!")

            self.pool = AssetPool.from_data(tickers, data)
 
            if LOAD_LOCAL and not os.path.exists(PICKLE_DIR + "asset_pool.pkl"):
                print("Saving Asset Pool...", end='\r')
                self.pool.save(PICKLE_DIR, "asset_pool")
                print("Saving Asset Pool... Done!")
        print("-"*50)

        print("Adding Indicators...", end='\r')
        self.pool.add_indicators(INDICATORS)
        print("Adding Indicators... Done!")

        print("Making Data Stationary...", end='\r')
        self.pool.stationary(pickle_dir=PICKLE_DIR, printout=True)
        print("Making Data Stationary... Done!")

        print("Picking Assets...", end='\r')
        # Filter tickers with insufficient average volume
        for ticker, volume in self.pool.volume().items():
            if volume.mean() < MIN_VOLUME:
                del self.pool[ticker]

        # Choose tickers with the most data
        lengths = {t: len(self.pool[t]) for t in self.pool.tickers}
        lowest_tickers = sorted(lengths.keys(), key=lambda x: lengths[x])[:len(lengths)-NUM_ASSETS]
        for ticker in lowest_tickers:
            tickers = self.pool.tickers
            del self.pool[ticker]
        print("Picking Assets... Done!")

        print("Splitting Data...", end='\r')
        train_pool, test_pool = self.pool.split(TRAIN_RATIO)
        print("Splitting Data... Done!")

        print("Scaling Data...", end='\r')
        train_pool.scale(SCALER)
        test_pool.scale(SCALER)
        print("Scaling Data... Done!")

        print("Windowing Data...", end='\r')
        train_pool.window(WINDOW_SIZE)
        test_pool.window(WINDOW_SIZE)
        print("Windowing Data... Done!")

        print("Creating Dataloaders...", end='\r')
        self.train_dl = train_pool.get_loader()
        self.test_dl = test_pool.get_loader()
        print("Creating Dataloaders... Done!")

        self.info = {
            "num_assets": len(self.pool),
            "tickers": self.pool.tickers,
            "num_features": len(self.pool.features),
            "features": self.pool.features,
            "train_len": len(train_pool),
            "train_dates": train_pool.datetimes,
            "train_prices": train_pool.prices,
            "test_len": len(test_pool),
            "test_dates": test_pool.datetimes,
            "test_prices": test_pool.prices
        }
        print("-"*50)

    @property
    def num_assets(self): return self.info["num_assets"]
    @property
    def tickers(self): return self.info["tickers"]
    @property
    def num_features(self): return self.info["num_features"]
    @property
    def features(self): return self.info["features"]
    @property
    def train_len(self): return self.info["train_len"]
    @property
    def train_dates(self): return self.info["train_dates"]
    @property
    def train_prices(self): return self.info["train_prices"]
    @property
    def test_len(self): return self.info["test_len"]
    @property
    def test_dates(self): return self.info["test_dates"]
    @property
    def test_prices(self): return self.info["test_prices"]    
