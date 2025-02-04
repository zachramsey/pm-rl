import os
import talib as ta
import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas_market_calendars as mcal

from config.base import DEBUG, PICKLE_DIR, TICKERS, MIN_VOLUME, NORM, TRAIN_RATIO, WINDOW_SIZE, NUM_ASSETS, INDICATORS
from loader.tickers import Tickers
from loader.stock_data import StockData

if NORM == "diff":
    from loader.data_set import TrajDataset as Dataset
elif NORM == "traj":
    from loader.data_set import TrajNormDataset as Dataset
    
class StockDataLoader:
    def __init__(self):
        self.tickers = None             # Ticker data class
        self.data = None                # Stock data class

        self.dates = None               # Common dates among all stock tickers
        self.features = None            # Features for each stock ticker
        self.targets = None             # Targets for each stock ticker

        self.train_dataloader = None    # Training data loader
        self.eval_dataloader = None     # Evaluation data loader

        # Load the tickers and stock data
        print("-"*50)
        if DEBUG and os.path.exists(PICKLE_DIR):
            self.tickers = Tickers()                # Load locally stored tickers
            self.data = StockData(self.tickers)     # Load locally stored stock data
            print("-"*50)
        else:
            self.tickers = Tickers(TICKERS)         # Fetch the tickers
            self.data = StockData(self.tickers)     # Fetch the stock data
            print("-"*50)

            self._preprocess_data()                 # Preprocess the raw data
            print("-"*50)

            if DEBUG and not os.path.exists(PICKLE_DIR):
                os.makedirs(PICKLE_DIR)
                self.tickers.store()
                self.data.store()
                print("-"*50)
            
        self._prepare_data()            # Prepare the data for training
        print("-"*50)

        self._create_dataloaders()      # Create the data loaders
        print("-"*50)

    def _preprocess_data(self):
        '''Clean the stock data'''
        # Ensure tickers have data for all valid trading days within their date range
        nyse = mcal.get_calendar('NYSE')
        for i, ticker in enumerate(self.tickers):
            print(f'{"Pre-Processing Data: ":<25}{(i+1):>5} / {len(self.tickers):<5} | {(i+1)/len(self.tickers)*100:.2f}%', end='\r')
            dates = self.data.get_dates(ticker)
            valid_days = nyse.valid_days(dates[0], dates[-1]).date

            df = self.data[ticker].copy()
            df = df.reindex(valid_days)             # Reindex to include all valid trading days
            df['Volume'] = df['Volume'].fillna(0)   # Volume: Fill missing with 0
            df = df.ffill()                         # Prices: Fill missing with the last valid
            self.data[ticker] = df
        print()

    def _prepare_data(self):
        print("Preparing Data...", end='\r')
        self._condition_data()          # Condition the data
        self._add_indicators()          # Add indicators to the data
        self._cleanup_data()            # Clean up the modified data
        self._initialize_tensors()      # Convert the data to tensors
        print("Preparing Data... Done!")

    def _condition_data(self):
        # Filter tickers with insufficient average volume
        tickers = []
        for ticker in self.tickers:
            if self.data[ticker]['Volume'].mean() > MIN_VOLUME:
                tickers.append(ticker)

        # Filter tickers to include those with the most data
        tickers = [(ticker, len(self.data[ticker])) for ticker in tickers]
        tickers = sorted(tickers, key=lambda x: x[1])
        tickers = [ticker for ticker, _ in tickers[-(NUM_ASSETS-1):]]

        # Filter the data to include only the selected tickers
        self.tickers.filter(tickers)
        self.data.filter(tickers)

    def _add_indicators(self):
        '''Extract features from the stock data'''
        if INDICATORS:
            for i, ticker in enumerate(self.tickers):
                print(f'{"Adding Indicators: ":<25}{(i+1):>5} / {len(self.tickers):<5} | {(i+1)/len(self.tickers)*100:.2f}%', end='\r')
                df = self.data[ticker].copy()
                df['sma_30'] = ta.SMA(df['Close'], timeperiod=30)
                df['sma_60'] = ta.SMA(df['Close'], timeperiod=60)
                df['macd'], _, _ = ta.MACD(df['Close'])
                df['bbu'], _, df['bbl'] = ta.BBANDS(df['Close'])
                df['rsi_30'] = ta.RSI(df['Close'], timeperiod=30)
                df['dx_30'] = ta.DX(df['High'], df['Low'], df['Close'], timeperiod=30)
                self.data[ticker] = df
            print()

    def _cleanup_data(self):
        '''Clean up the data'''
        # Get common dates among all tickers
        self.dates = self.data.get_dates(self.tickers[0])
        for ticker in self.tickers[1:]:
            self.dates = self.dates.intersection(self.data.get_dates(ticker))
        self.dates = sorted(list(self.dates))
        
        for ticker in self.tickers:
            df = self.data[ticker].copy()
            df = df.reindex(self.dates)     # Filter data to include only common dates
            df = df.drop(columns=['Open'])  # Remove 'Open' column (assumes curr open == last Close)

            # Normalize the data
            if NORM == "diff":
                df['High'] = (df['High']/df['Close'].shift(1))-1
                df['Low'] = (df['Low']/df['Close'].shift(1))-1
                df['Close'] = (df['Close']/df['Close'].shift(1))-1
                df['Volume'] = (df['Volume']/df['Volume'].shift(1))-1
                df = df.drop(index=self.dates[0])           # Drop the first row
                df = df.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
                df = df.fillna(0)                           # Fill NaN with 0

            self.data[ticker] = df          # Update the data

    def _initialize_tensors(self):
        '''Initialize the tensors for the features and targets'''
        num_tickers, num_dates, num_features = self.data.shape()
        self.targets = torch.zeros(num_tickers, num_dates)
        self.features = torch.zeros(num_tickers, num_dates, num_features)

        # Fill prototype tensors with data
        for i, (ticker, df) in enumerate(self.data):
            if NORM == "diff":
                price_relative = df['Close']+1
            else:
                price_relative = df['Close']/df['Close'].shift(1)
                price_relative = price_relative.drop(index=self.dates[0])
                price_relative = price_relative.replace([np.inf, -np.inf], np.nan)
                price_relative = price_relative.fillna(1)
            self.targets[i] = torch.from_numpy(price_relative.to_numpy())
            self.features[i] = torch.from_numpy(df.to_numpy())

        # Prepend additional asset representing cash with everything set to 1
        self.targets = torch.cat([torch.ones((1, num_dates)), self.targets], dim=0)
        self.features = torch.cat([torch.zeros((1, num_dates, num_features)), self.features], dim=0)

        # Append additional feature representing asset weights with everything set to 0
        self.features = torch.cat([self.features, torch.zeros((num_tickers+1, num_dates, 1))], dim=2)

    def _create_dataloaders(self):
        '''Create the data loaders for training and testing'''
        print(f'{"Creating Data Loaders..."}', end='\r')
        start_eval = int(len(self.dates) * TRAIN_RATIO)

        # Slice the data into training and testing sets
        train_dates = self.dates[:start_eval]
        train_features = self.features[:, :start_eval]
        train_targets = self.targets[:, :start_eval]

        eval_dates = self.dates[start_eval:]
        eval_features = self.features[:, start_eval:]
        eval_targets = self.targets[:, start_eval:]

        # Create the data loaders
        train_dataset = Dataset(train_dates, train_features, train_targets, WINDOW_SIZE)
        self.train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        eval_dataset = Dataset(eval_dates, eval_features, eval_targets, WINDOW_SIZE)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
        print("Creating Data Loaders... Done!")

    def get_train_data(self):
        return self.train_dataloader
    
    def get_eval_data(self):
        return self.eval_dataloader
    
    def get_train_len(self):
        return len(self.train_dataloader.dataset)
    
    def get_eval_len(self):
        return len(self.eval_dataloader.dataset)
    
    def get_train_dates(self):
        return self.train_dataloader.dataset.dates
    
    def get_eval_dates(self):
        return self.eval_dataloader.dataset.dates
    
    def get_assets(self):
        return ['$'] + list(self.tickers.data.keys())
    
    def get_num_assets(self):
        return self.features.size(0)
    
    def get_num_features(self):
        return self.features.size(2)