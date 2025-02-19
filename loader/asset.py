from datetime import datetime
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from talib import abstract
import tensordict as td
from tensordict.tensorclass import tensorclass
import torch
from typing import Any

@tensorclass
class Asset:
    ''' ### TensorClass for handling asset data

        Asset data is structured as follows:  
        ```python
        Asset: {  
            'ticker': str,  
            'info': {  
                  'sector': str,  
                'industry': str  
                     '...': ...  
            },  
            'datetimes': torch.Tensor,  
            'prices': torch.Tensor,  
            'features': {  
                  'open': torch.Tensor,  
                  'high': torch.Tensor,  
                   'low': torch.Tensor,  
                 'close': torch.Tensor,  
                'volume': torch.Tensor,  
                   '...': ...  
            }  
        }  
        ```
    '''
    datetimes: np.ndarray      # Timestamps for the prices and feature data
    ticker: str                 # The stock ticker
    info: td.TensorDict         # Asset information
    prices: torch.Tensor        # Relative closing prices
    features: td.TensorDict     # Historical (time-series) feature data
    

    ''' ##################################################################
        ###                       Magic Methods                        ###
        ################################################################## '''
    
    def __init__(self, 
                 ticker: str,
                 info: td.TensorDict,
                 datetimes: np.ndarray,
                 prices: torch.Tensor,
                 features: td.TensorDict
                ) -> None:
        ''' ### Initialize the asset data '''
        self.ticker = ticker
        self.info = info
        self.datetimes = datetimes
        self.prices = prices
        self.features = features
    

    def __len__(self) -> int:
        ''' Return the length of the asset data '''
        return len(self.datetimes)
    

    def __iter__(self):
        ''' ### Iterate over the asset data

            Yields:
                Asset: The asset data
        '''
        assert len(self.datetimes) == len(self.prices) == len(self.features), "Data must have the same number of datetimes, prices, and features"
        for idx in range(len(self)):
            yield self.at(idx)


    def __str__(self) -> str:
        ''' Return a string representation of the asset data'''
        return f'Ticker: {self.ticker}\nSector: {self.sector}\nIndustry: {self.industry}'


    def __repr__(self) -> str:
        ''' Return a string representation of the asset data'''
        return f'ticker: {self.ticker}, sector: {self.sector}, industry: {self.industry}'


    ''' ##################################################################
        ###                    Asset Data Accessors                    ###
        ################################################################## '''
    
    @property
    def ticker(self) -> str:
        ''' The stock ticker symbol '''
        return self['ticker']


    @property
    def datetimes(self) -> list[datetime]:
        ''' The datetimes of the asset data '''
        return self['datetimes']
    

    @property
    def prices(self) -> torch.Tensor:
        ''' The historical relative closing prices for the asset '''
        return self['prices']
    

    def at(self, idx: int) -> 'Asset':
        ''' ### Return the asset data at the given index

            Args:
                idx (int): The index of the asset data

            Returns:
                Asset: The asset data
        '''
        return Asset(
            ticker=self.ticker,
            info=self.info,
            datetimes=self.datetimes[idx],
            prices=self.prices[idx],
            features=self.features[idx]
        )
    

    ''' ##################################################################
        ###                  Asset Property Accessors                  ###
        ################################################################## '''
    
    @property
    def info(self) -> td.TensorDict:
        ''' The asset info '''
        return self['info']

    
    @property
    def sector(self) -> str:
        ''' The sector of the asset '''
        return self.info['sector']
    

    @property
    def industry(self) -> str:
        ''' The industry of the asset '''
        return self.info['industry']
    
    
    def get_info(self, key: str) -> Any:
        ''' ### Return the info with the given key

            Args:
                key (str): The key of the info

            Returns:
                Any: The value of the info
        '''
        return self.info[key]
    

    def set_info(self, key: str, value: str) -> None:
        ''' ### Set the info with the given key
        
            Args:
                key (str): The key of the info
                value (str): The value of the info
        '''
        self.info[key] = value


    def del_info(self, key: str) -> None:
        ''' ### Remove the info with the given key
        
            Args:
                key (str): The key of the info
        '''
        self.info.pop(key)
    

    ''' ##################################################################
        ###                  Asset Feature Accessors                   ###
        ################################################################## '''
    
    @property
    def features(self) -> td.TensorDict:
        ''' The historical feature data for the asset '''
        return self['features']
    
    
    @property
    def open(self) -> torch.Tensor:
        ''' The historical opening prices for the asset '''
        return self.features['open']
    

    @property
    def high(self) -> torch.Tensor:
        ''' The historical high prices for the asset '''
        return self.features['high']
    

    @property
    def low(self) -> torch.Tensor:
        ''' The historical low prices for the asset '''
        return self.features['low']
    
    
    @property
    def close(self) -> torch.Tensor:
        ''' The historical closing prices for the asset '''
        return self.features['close']
    

    @property
    def volume(self) -> torch.Tensor:
        ''' The historical trading volumes for the asset '''
        return self.features['volume']
    

    def get_ohlcv(self) -> td.TensorDict:
        ''' The historical OHLCV data for the asset '''
        return td.TensorDict({key: self.features[key] for key in ['open', 'high', 'low', 'close', 'volume']})
    

    def get_ohlcv_dict(self) -> dict[str, np.ndarray]:
        ''' The historical OHLCV data for the asset as a dictionary '''
        return {key: self.features[key].numpy() for key in ['open', 'high', 'low', 'close', 'volume']}

    
    def get_feature(self, feature: str) -> torch.Tensor:
        ''' ### Return the feature data
        
            Args:
                feature (str): The name of the feature
        
            Returns:
                torch.Tensor: The feature data
        '''
        return self.features[feature]
    

    def set_feature(self, feature: str, value: torch.Tensor) -> None:
        ''' ### Set the feature data
        
            Args:
                feature (str): The name of the feature
                value (torch.Tensor): The feature data
        '''
        value = value.flatten()
        assert len(value) == len(self), f'Feature data must have the same length as the asset data, found {len(value)} != {len(self)}'
        self.features[feature] = value


    def del_feature(self, feature: str) -> None:
        ''' ### Remove the feature data
        
            Args:
                feature (str): The name of the feature
        '''
        self.features.pop(feature)


    def add_indicators(self, indicators: dict) -> int:
        ''' ### Add indicators to the feature data
        
            Args:
                indicators (dict): The indicators and their arguments {name: {arg: value, ...}, ...}

            Returns:
                int: The lookback period of the indicator
        '''
        lookback = 0
        for name, params in indicators.items():
            func = abstract.Function(name)
            func.set_parameters(params)
            lookback = max(lookback, func.lookback)
            output_names = func.output_names
            ins = {key: value.astype(np.float64, copy=True) for key, value in self.get_ohlcv_dict().items()}
            outs = func.run(ins)
            if not isinstance(outs, list): outs = [outs]
            for name_, output in zip(output_names, outs):
                if name_ == 'real': name_ = name
                name_ = name_ + '_' + '_'.join([f'{value}' for value in params.values()])
                self.features[name_] = torch.from_numpy(output).to(torch.float32)
        return lookback
            

    ''' ##################################################################
        ###                     Data Augmentation                      ###
        ################################################################## '''
    
    def clip(self, start: int=0, end: int=None) -> None:
        ''' ### Clip the data to the given range
        
            *Note: This method is applied in-place; cloning the asset is advised.*
        
            Args:
                start (int): The starting index
                end (int): The ending index
        '''
        if end is None: end = len(self)
        self.features = self.features[start:end]
        self.prices = self.prices[start:end]
        self.datetimes = self.datetimes[start:end]

    
    def stationary(self, thres: float = 1e-5) -> int:
        ''' ### Make the data stationary

            Uses the fixed-width window fractional difference method described in Chapter 5
            of "Advances in Financial Machine Learning" (Marcos Lopez de Prado, 2018)

            *Note: This method is applied in-place; cloning the asset is advised.*
            
            Args:
                thres (float): The threshold for the differentiation

            Returns:
                int: The maximum differentiation width
        '''
        max_width = 0
        for feature, data in self.features.items():
            data = torch.log(data)
            for d in np.linspace(0, 1, 21):
                # Compute weights for the data
                w = [1.]
                for k in range(1, 100):
                    assert k < len(data), f'Insufficient data for feature {feature}'
                    w_ = -w[-1]/k * (d-k+1)
                    if abs(w_) < thres: break
                    w.append(w_)
                w = torch.tensor(w[::-1], dtype=torch.float32)
                width = len(w)-1
                # Apply the weights to the data
                data_ = torch.zeros((len(data)-width,), dtype=torch.float32)
                for i in range(width, len(data)):
                    data_[i-width] = torch.dot(w, data[i-width:i+1])
                # Check for stationarity
                if adfuller(data_)[1] < 0.05:
                    self.features[feature] = data_
                    max_width = max(max_width, width)
                    break
        return max_width
    

    def split(self, purge_length: int, train_ratio: float = 0.8) -> tuple['Asset', 'Asset']:
        ''' ### Split the data into training and testing sets

            ## Notes:
                - Train data contains the first `train_ratio` of the data
                - Test data contains the remaining data after clipping the first `purge_length` datetimes

            Args:
                purge_length (int): The number of datetimes to purge from the beginning of the testing data
                train_ratio (float): The ratio of training data to total data

            Returns:
                tuple[Asset, Asset]: The training and testing data
        '''
        split = int(train_ratio * len(self))
        train_data = self.clone()
        test_data = self.clone()

        train_data.features = train_data.features[:split]
        train_data.prices = train_data.prices[:split]
        train_data.datetimes = train_data.datetimes[:split]

        test_data.features = test_data.features[split+purge_length:]
        test_data.prices = test_data.prices[split+purge_length:]
        test_data.datetimes = test_data.datetimes[split+purge_length:]

        return train_data, test_data
    

    def scale(self, method: str = 'minmax') -> 'Asset':
        ''' ### Scale the data feature(s) using the given method

            ## Notes:
                - This method is applied in-place; clone the asset first if the original data is needed.
                - Scaling considers the whole time-series; split the data first to prevent data leakage.
            
            Args:
                method (str): The scaling method to use
        '''
        for f in self.features.keys():
            assert len(self.features[f].size()) == 1, f'Cannot scale feature {f} with multi-dimensional shape {self.features[f].size()}'
            if method == 'standard': scaler = StandardScaler()
            elif method == 'minmax': scaler = MinMaxScaler()
            else: raise ValueError(f'Invalid scaling method: {method}')
            self.features[f] = torch.tensor(scaler.fit_transform(self.features[f].reshape(-1, 1)).flatten())
    
    
    def window(self, window_size: int) -> None:
        ''' ### Return the windowed data

            ## Notes:
                - This method is applied in-place; clone the asset first if the original data is needed.
                
            Args:
                window_size (int): The window size
        '''
        features_ = td.TensorDict({feature: torch.zeros((len(self)-window_size+1, window_size), dtype=torch.float32) for feature in self.features.keys()})
        for i in range(len(self)-window_size+1):
            features_[i] = self.features[i:i+window_size]
        self.features = features_
        self.prices = self.prices[window_size-1:]
        self.datetimes = self.datetimes[window_size-1:]



    ''' ##################################################################
        ###                       Helper Methods                       ###
        ################################################################## '''
    
    @classmethod      
    def from_data(cls, ticker: str, info: dict[str,], data: pd.DataFrame) -> 'Asset':
        ''' ### Create an AssetData object from the given info and data

            ## Notes:
            - The info must contain the sector and industry of the asset
            - The data must contain OHLCV data

            Args:
                ticker (str): The stock ticker
                info (dict[str,]): Asset information of the form {info_key: info_value}
                data (DataFrame): Time-series data with feature columns and a datetime index

            Returns:
                Asset: The asset data
        '''
        # Check that data is correctly formatted and contains the minimum required elements
        ohlcv = ['open', 'high', 'low', 'close', 'volume']
        data = data.rename(columns=lambda x: x.lower())
        assert all([feature in data.columns for feature in ohlcv]), "Data must include OHLCV data, missing {}".format([feature for feature in ohlcv if feature not in data])
        assert all([len(data[feature].dropna()) == len(data['close'].dropna()) for feature in data]), "All features must be the same length, found {}".format({feature: len(data[feature].dropna()) for feature in data})
        assert 'sector' in info and 'industry' in info, "Info must include sector and industry, missing {}".format([key for key in info if key not in ['sector', 'industry']])

        df = data.copy()
        df = df.rename(columns=lambda x: x.lower())                 # Lowercase column names
        nyse = mcal.get_calendar('NYSE')                            # Get NYSE calendar
        valid_days = nyse.valid_days(df.index[0], df.index[-1])     # Get valid trading days
        df = df.reindex(valid_days.tz_localize(None))               # Reindex to include all valid trading days
        df['volume'] = df['volume'].fillna(0)                       # Volume: Fill missing with 0
        df = df.ffill()                                             # Prices: Fill missing with the last valid

        # Create the asset
        asset = cls(
            ticker=ticker,
            info=td.TensorDict(info),
            datetimes=df.index[1:].to_pydatetime(),
            prices=torch.tensor((df['close']/df['close'].shift(1)).to_numpy()[1:], dtype=torch.float32),
            features=td.TensorDict({key: torch.tensor(value[1:], dtype=torch.float32) for key, value in df.to_dict(orient='list').items()}, batch_size=len(df)-1)
        )
        return asset