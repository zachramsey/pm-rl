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

from config.base import DATA_DIR
from util.logger import TrainingLogger

class LoadYF:
    '''Class to fetch asset data from Yahoo Finance'''
    
    def __init__(self, tickers: list[str]):
        self.tickers = tickers                      # List of tickers
        self.data = {}                              # Initialize dictionary to store data

        self.session = self._configure_session()    # Configure the requests session
        self.nyse = mcal.get_calendar('NYSE')       # NYSE calendar for valid trading days
        if not os.path.exists(DATA_DIR):            # Create data directory if it does not exist
            os.makedirs(DATA_DIR)
        self._download_data()                       # Download the historical data
    
    def get_data(self, ticker: str = None) -> dict:
        '''Return the data for the given ticker or all tickers'''
        if ticker is None:
            return self.data
        return self.data[ticker]

    def _download_data(self) -> None:
        '''Separate tickers into existing and new based on data directory'''
        for i, ticker in enumerate(self.tickers):
            TrainingLogger.progress_logger("Fetching Data", i, len(self.tickers))
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
