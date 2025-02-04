from config.base import DATA_DIR

import os
import sys

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

import platformdirs as ad
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

from loader.tickers import Tickers
from config.base import PICKLE_DIR, DEBUG

class StockData:
    def __init__(self, tickers: Tickers):
        self.tickers = tickers                          # Tickers class
        self.data = {}                                  # Initialize dictionary to store data

        if DEBUG and os.path.exists(PICKLE_DIR):
            self.load()                                 # Load data from a pickle file
        else:
            self.session = self._configure_session()    # Configure the requests session
            self.nyse = mcal.get_calendar('NYSE')       # NYSE calendar for valid trading days
            if not os.path.exists(DATA_DIR):            # Create data directory if it does not exist
                os.makedirs(DATA_DIR)
            self._download_data()                       # Download the historical data

    def __call__(self) -> dict[str, pd.DataFrame]:
        '''Return the historical data for all stock tickers'''
        return self.data

    def __len__(self) -> int:
        '''Return the number of stock tickers'''
        return len(self.data)

    def __getitem__(self, ticker: str) -> pd.DataFrame:
        '''Return the historical data for the given ticker'''
        return self.data[ticker]
    
    def __setitem__(self, ticker: str, data: pd.DataFrame) -> None:
        '''Set the historical data for the given ticker'''
        self.data[ticker] = data
    
    def __iter__(self) -> iter:
        '''Return an iterator for the historical data'''
        return iter(self.data.items())
    
    def __repr__(self) -> str:
        '''Return a string representation of the historical data'''
        num_tickers, num_dates, num_features = self.shape()
        return "|" + "="*48 + "|\n \
                |                STOCK DATA INFO                 |\n \
                |" + "="*48 + "|\n \
                |" + f"{'Number of Tickers':<20}| {num_tickers:<25}|\n \
                |" + f"{'Number of Features':<20}| {num_features:<25}|\n \
                |" + f"{'Number of Dates':<20}| {num_dates:<25}|\n \
                |" + f"{'Start Date':<20}| {self.get_dates().min().strftime('%b %d, %Y'):<25}|\n \
                |" + f"{'End Date':<20}| {self.get_dates().max().strftime('%b %d, %Y'):<25}|\n \
                |" + "="*48 + "|\n"

    def _download_data(self) -> None:
        '''Separate tickers into existing and new based on data directory'''
        i = 0
        for ticker in self.tickers:
            file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col='Date')   # Load data from csv
                df.index = pd.to_datetime(df.index)             # Ensure index is in datetime format

                # Check if the data is up-to-date
                last_local_date = df.index[-1].date()               # Get the last local date
                last_trading_date = self._get_last_trading_date()   # Get the last trading date
                if last_local_date < last_trading_date:
                    new_data = yf.download(ticker, start=df.index[-1] + pd.Timedelta(days=1),
                                        end=last_trading_date + pd.Timedelta(days=1),
                                        repair=True, progress=False, rounding=True, session=self.session)
                    if not new_data.empty:
                        new_data = new_data[['Open', 'High', 'Low', 'Close', 'Volume']]
                        new_data.to_csv(file_path, mode='a', header=False)  # Append to csv
                        df = pd.concat([df, new_data])                      # Add new data to the dataframe
                self.data[ticker] = df                                      # Add to the dictionary
                i += 1
            else:
                last_trading_date = self._get_last_trading_date()
                df = yf.download(ticker, end=last_trading_date + pd.Timedelta(days=1),
                                repair=True, progress=False, rounding=True, session=self.session)
                if not df.empty:
                    df.index = pd.to_datetime(df.index)
                    df = df.droplevel("Ticker", axis=1) if 'Ticker' in df.columns.names else df
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df.to_csv(file_path)        # Save to csv
                    self.data[ticker] = df      # Add to the dictionary
                    i += 1
                else:
                    self.tickers.remove(ticker) # Remove the ticker if no data is available
            print(f'{"Fetching Data: ":<25}{i:>5} / {len(self.tickers):<5} | {i/len(self.tickers)*100:.2f}%', end='\r')
        print()

    def _get_last_trading_date(self) -> pd.Timestamp:
        '''Get the last trading date from the NYSE calendar'''
        curr_dt = pd.Timestamp.now(tz='America/New_York')
        today = curr_dt.date()
        schedule = self.nyse.schedule(start_date=today - pd.DateOffset(days=7), end_date=today)
        last_trading_date = schedule['market_close'].iloc[-2 if curr_dt.hour < 16 else -1].date()
        return last_trading_date

    def _configure_session(self) -> Session:
        '''Configure the requests session for rate-limiting and caching'''
        # Add the parent directory to the system path
        _parent_dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        _src_dp = _parent_dp
        sys.path.insert(0, _src_dp)

        # Use adjacent cache folder for testing, delete if already exists and older than today
        testing_cache_dirpath = os.path.join(ad.user_cache_dir(), "py-yfinance-testing")
        yf.set_tz_cache_location(testing_cache_dirpath)
        if os.path.isdir(testing_cache_dirpath):
            mtime = pd.Timestamp(os.path.getmtime(testing_cache_dirpath), unit='s', tz='UTC')
            if mtime.date() < pd.Timestamp.now().date():
                import shutil
                shutil.rmtree(testing_cache_dirpath)

        # Setup a session to rate-limit and cache persistently:
        class CachedLimiterSession(CacheMixin, LimiterMixin, Session): pass
        history_rate = RequestRate(1, Duration.SECOND*0.1)
        limiter = Limiter(history_rate)
        cache_fp = os.path.join(testing_cache_dirpath, "unittests-cache")
        self.session = CachedLimiterSession(
            limiter=limiter,
            bucket_class=MemoryQueueBucket,
            backend=SQLiteCache(cache_fp, expire_after=pd.Timedelta(hours=1)),
        )

    def store(self) -> None:
        '''Store the historical data in a pickle file'''
        print("Saving Pickled Data...", end='\r')
        pd.to_pickle(self.data, PICKLE_DIR + "data.pkl")
        print(f"Saving Pickled Data... Done!")

    def load(self) -> None:
        '''Load the historical data from a pickle file''' 
        print("Loading Pickled Data...", end='\r')
        self.data = pd.read_pickle(PICKLE_DIR + "data.pkl")
        print(f"Loading Pickled Data... Done!")

    def filter(self, tickers: str | list[str]) -> None:
        '''Update the data to only include the given tickers'''
        if isinstance(tickers, str):
            tickers = [tickers]
        self.data = {ticker: self.data[ticker] for ticker in tickers}

    def get_dates(self, ticker: str) -> pd.DatetimeIndex:
        '''Return the dates included in a ticker's data or all common dates'''
        return self.data[ticker].index
    
    def get_features(self, ticker: str) -> list[str]:
        '''Return the features included in a ticker's data'''
        return self.data[ticker].columns
    
    def shape(self) -> tuple[int, int, int]:
        '''Return the shape of the data (number of tickers, number of dates, number of features)'''
        num_tickers = len(self.data)
        num_dates = len(self.data[self.tickers[0]].index)
        num_features = len(self.data[self.tickers[0]].columns)
        return num_tickers, num_dates, num_features