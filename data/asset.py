from datetime import datetime
import numpy as np
import os
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from talib import abstract
import tensordict as td
from tensordict.tensorclass import tensorclass
import torch
from typing import Any

from data.ffd import FixedFracDiff

@tensorclass
class Asset:
    ''' ### TensorClass for handling asset data

        Asset data is structured as follows:  
        ```python
        Asset: {  
            'ticker': str,  
            'meta': {  
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
    meta: td.TensorDict         # Asset metadata
    prices: torch.Tensor        # Relative closing prices
    features: td.TensorDict     # Historical (time-series) feature data
    

    ''' ##################################################################
        ###                       Magic Methods                        ###
        ################################################################## '''
    
    def __init__(self, 
                 ticker: str,
                 meta: td.TensorDict,
                 datetimes: np.ndarray,
                 prices: torch.Tensor,
                 features: td.TensorDict
                ) -> None:
        ''' ### Initialize the asset data '''
        self.ticker = ticker
        self.meta = meta
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
            meta=self.meta,
            datetimes=self.datetimes[idx],
            prices=self.prices[idx],
            features=self.features[idx]
        )
    

    ''' ##################################################################
        ###                  Asset Property Accessors                  ###
        ################################################################## '''
    
    @property
    def meta(self) -> td.TensorDict:
        ''' The asset metadata '''
        return self['meta']

    
    @property
    def sector(self) -> str:
        ''' The sector of the asset '''
        return self.meta['sector']
    

    @property
    def industry(self) -> str:
        ''' The industry of the asset '''
        return self.meta['industry']
    
    
    def get_meta(self, key: str) -> Any:
        ''' ### Return the metadata with the given key

            Args:
                key (str): The key of the metadata

            Returns:
                Any: The value of the metadata
        '''
        return self.meta[key]
    

    def set_meta(self, key: str, value: str) -> None:
        ''' ### Set the metadata with the given key
        
            Args:
                key (str): The key of the metadata
                value (str): The value of the metadata
        '''
        self.meta[key] = value


    def del_meta(self, key: str) -> None:
        ''' ### Remove the metadata with the given key
        
            Args:
                key (str): The key of the metadata
        '''
        self.meta.pop(key)
    

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
        self.clip(start=lookback)
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


    def stationary(self, thres: float = 1e-5, pickle_dir: str = None, printout: str = None) -> int:
        ''' ### Make the data stationary

            Uses the fixed-width window fractional difference method described in Chapter 5
            of "Advances in Financial Machine Learning" (Marcos Lopez de Prado, 2018)

            *Note: This method is applied in-place; cloning the asset is advised.*
            
            Args:
                thres (float): The threshold for the differentiation

            Returns:
                int: The maximum differentiation width
        '''
        if pickle_dir is not None:
            file = pickle_dir + f'd_opt_{self.ticker}.pkl'
            exists = os.path.exists(file)
            d_opt = pd.read_pickle(file) if exists else None
        ffd = FixedFracDiff(self.features, thres, d_opt)
        self.features = ffd.fit_transform(printout)
        if pickle_dir is not None and not exists: 
            pd.to_pickle(ffd.d_opt, file)
        return ffd.get_max_width()
    

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
    def from_data(cls, ticker: str, meta: dict[str,], data: pd.DataFrame) -> 'Asset':
        ''' ### Create an AssetData object from the given metadata and data

            ## Notes:
            - The metadata must contain the sector and industry of the asset
            - The data must contain OHLCV data

            Args:
                ticker (str): The stock ticker
                meta (dict[str,]): Asset metadata
                data (DataFrame): Time-series data with feature columns and a datetime index

            Returns:
                Asset: The asset data
        '''
        # Check that data is correctly formatted and contains the minimum required elements
        ohlcv = ['open', 'high', 'low', 'close', 'volume']
        data = data.rename(columns=lambda x: x.lower())
        assert all([feature in data.columns for feature in ohlcv]), "Data must include OHLCV data, missing {}".format([feature for feature in ohlcv if feature not in data])
        assert all([len(data[feature].dropna()) == len(data['close'].dropna()) for feature in data]), "All features must be the same length, found {}".format({feature: len(data[feature].dropna()) for feature in data})
        assert 'sector' in meta and 'industry' in meta, "Metadata must include sector and industry, missing {}".format([key for key in meta if key not in ['sector', 'industry']])

        data = data.rename(columns=lambda x: x.lower())                 # Lowercase column names
        nyse = mcal.get_calendar('NYSE')                                # Get NYSE calendar
        valid_days = nyse.valid_days(data.index[0], data.index[-1])     # Get valid trading days
        data = data.reindex(valid_days.tz_localize(None))               # Reindex to all valid trading days
        data['volume'] = data['volume'].fillna(0)                       # Missing Volume: Fill w/ 0
        data = data.ffill()                                             # Missing Prices: Fill w/ the prior valid

        f_labels = data.columns
        f_datas = torch.from_numpy(data.to_numpy(dtype=np.float32).T)
        feature_data = td.TensorDict({
            f_label: f_data for f_label, f_data in zip(f_labels, f_datas)
        }, batch_size=len(data))

        # Create the asset
        asset = cls(
            ticker=ticker,
            meta=td.TensorDict(meta),
            datetimes=data.index[1:].to_pydatetime(),
            prices=feature_data['close'][1:]/feature_data['close'][:-1],
            features=feature_data[1:]
        )
        return asset
    

    def feature_tensor(self) -> torch.Tensor:
        ''' ### Return the asset meta (expanded) and feature data as a tensor

            Returns:
                torch.Tensor: The asset data of shape (data_length, num_meta + num_features)
        '''
        meta = torch.tensor([self.meta[key] for key in self.meta.keys()], dtype=torch.float32)
        meta = meta.unsqueeze(0).expand(len(self), -1)
        data = torch.stack([self.features[key] for key in self.features.keys()], dim=1)
        return torch.cat((meta, data), dim=1)