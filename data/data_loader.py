import os
from util.util import get_index_tickers
from data.load_yf import LoadYF
from data.instrument_pool import InstrumentPool
from torch.utils.data import DataLoader
from util.logger import TrainingLogger
from config.base import LOAD_LOCAL, PICKLE_DIR, TARGETS, CONTEXT, MIN_VOLUME, SCALER, TRAIN_RATIO, WINDOW_SIZE, NUM_ASSETS, INDICATORS

class StockDataLoader:
    pool: InstrumentPool    # Stock data class
    train_dl: DataLoader    # Training data loader
    test_dl: DataLoader     # Testing data loader
    info: dict              # Information about the data

    def __init__(self, logger: TrainingLogger):
        self.logger = logger
        self.logger.format_banner("PREPARING DATA")

        # Load the tickers and stock data
        if LOAD_LOCAL and os.path.exists(PICKLE_DIR + "asset_pool.pkl"):
            self.pool = self.logger.is_done_wrapper("Loading Asset Pool", InstrumentPool.load, PICKLE_DIR, "asset_pool")
        else:
            tickers = self.logger.is_done_wrapper("Fetching Tickers", get_index_tickers, TARGETS, ['HUBB'])

            targets_data = LoadYF(tickers.keys()).get_data()
            context_data = LoadYF(CONTEXT).get_data()
            self.pool = InstrumentPool.from_data(targets_data, context_data)
 
            if LOAD_LOCAL and not os.path.exists(PICKLE_DIR + "asset_pool.pkl"):
                self.logger.is_done_wrapper("Saving Asset Pool", self.pool.save, PICKLE_DIR, "asset_pool")

        # Remove tickers with insufficient data
        for ticker in self.pool.target_tickers:
            if len(self.pool.targets[ticker]) < 12000:
                del self.pool.targets[ticker]

        self.logger.is_done_wrapper("Adding Indicators", self.pool.add_indicators, INDICATORS)
        self.pool.stationary(pickle_dir=PICKLE_DIR, log=True)
        self.logger.is_done_wrapper("Filtering Assets", self._filter_assets, self.pool, MIN_VOLUME, NUM_ASSETS)
        train_pool, test_pool = self.logger.is_done_wrapper("Splitting Data", self.pool.split, TRAIN_RATIO)
        self.logger.is_done_wrapper("Scaling Data", self._scale, train_pool, test_pool, SCALER)
        self.logger.is_done_wrapper("Windowing Data", self._window, train_pool, test_pool, WINDOW_SIZE)
        self.train_dl, self.test_dl = self.logger.is_done_wrapper("Building DataLoaders", self._build_dataloaders, train_pool, test_pool)

        self.info = {
            "n_targets": len(self.pool.target_tickers),
            "n_context": len(self.pool.context_tickers),
            "n_features": len(self.pool.features),
            "len_train": len(train_pool.datetimes),
            "len_test": len(test_pool),
            "targets": self.pool.target_tickers,
            "context": self.pool.context_tickers,
            "features": self.pool.features,
            "train_dates": train_pool.datetimes,
            "test_dates": test_pool.datetimes,
        }

    @staticmethod
    def _filter_assets(pool, min_volume, num_assets):
        # Filter tickers with insufficient average volume
        for ticker, volume in pool.volume()['targets'].items():
            if volume.mean() < min_volume:
                del pool.targets[ticker]

        # Choose tickers with the most data
        lengths = {t: len(pool.targets[t]) for t in pool.target_tickers}
        lowest_tickers = sorted(lengths.keys(), key=lambda x: lengths[x])[:len(lengths)-num_assets]
        for ticker in lowest_tickers:
            del pool.targets[ticker]

    @staticmethod
    def _scale(train_pool, test_pool, scaler):
        train_pool.scale(scaler)
        test_pool.scale(scaler)

    @staticmethod
    def _window(train_pool, test_pool, window_size):
        train_pool.window(window_size)
        test_pool.window(window_size)

    @staticmethod
    def _build_dataloaders(train_pool, test_pool):
        return train_pool.get_loader(), test_pool.get_loader()

    @property
    def n_targets(self): return self.info["n_targets"]
    @property
    def n_context(self): return self.info["n_context"]
    @property
    def n_features(self): return self.info["n_features"]
    @property
    def len_train(self): return self.info["len_train"]
    @property
    def len_test(self): return self.info["len_test"]
    @property
    def targets(self): return self.info["targets"]
    @property
    def context(self): return self.info["context"]
    @property
    def features(self): return self.info["features"]
    @property
    def train_dates(self): return self.info["train_dates"]
    @property
    def test_dates(self): return self.info["test_dates"]
