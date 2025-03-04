
import os
import torch

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import tensordict as td

from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from talib import abstract
from tensordict.tensorclass import tensorclass

from data.ffd import FixedFracDiff

@tensorclass
class Instrument:
    ''' TensorClass for handling financial instrument time-series data

        Instrument data is structured as follows:  
        ```python
        Instrument: {  
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
    datetimes: np.ndarray       # Timestamps for the prices and feature data
    prices: torch.Tensor        # Relative closing prices
    features: td.TensorDict     # Historical (time-series) feature data

    @classmethod      
    def from_data(cls, data: pd.DataFrame) -> 'Instrument':
        ''' Create an Instrument object from the given data

            Parameters
            ----------
                data : DataFrame
                    Time-series data with feature columns and a datetime index

            Returns
            -------
                Instrument: The instrument data

            Notes
            -----
            - The data must contain OHLCV data
        '''
        # Check that data is correctly formatted and contains the minimum required elements
        ohlcv = ['open', 'high', 'low', 'close', 'volume']
        data = data.rename(columns=lambda x: x.lower())
        assert all([feature in data.columns for feature in ohlcv]), "Data must include OHLCV data, missing {}".format([feature for feature in ohlcv if feature not in data])
        assert all([len(data[feature].dropna()) == len(data['close'].dropna()) for feature in data]), "All features must be the same length, found {}".format({feature: len(data[feature].dropna()) for feature in data})

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

        # Create the instrument
        instrument = cls(
            datetimes=data.index[1:].to_pydatetime(),
            prices=feature_data['close'][1:]/feature_data['close'][:-1],
            features=feature_data[1:]
        )
        return instrument
    

    ''' ##################################################################
        ###                       Magic Methods                        ###
        ################################################################## '''
    
    def __init__(self, datetimes: np.ndarray, prices: torch.Tensor, features: td.TensorDict) -> None:
        self.datetimes = datetimes
        self.prices = prices
        self.features = features

    def __len__(self) -> int:
        return len(self.datetimes)
    

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


    def __getitem__(self, key: str) -> any:
        return getattr(self, key)


    ''' ##################################################################
        ###                    Instrument Data Accessors               ###
        ################################################################## '''
    
    @property
    def datetimes(self) -> list[datetime]:
        ''' The datetimes of the instrument data '''
        return self['datetimes']

    @property
    def prices(self) -> torch.Tensor:
        ''' The historical relative closing prices for the instrument '''
        return self['prices']
    
    @property
    def features(self) -> td.TensorDict:
        ''' The historical feature data for the instrument '''
        return self['features']
    
    @property
    def open(self) -> torch.Tensor:
        ''' The historical opening prices for the instrument '''
        return self.features['open']

    @property
    def high(self) -> torch.Tensor:
        ''' The historical high prices for the instrument '''
        return self.features['high']

    @property
    def low(self) -> torch.Tensor:
        ''' The historical low prices for the instrument '''
        return self.features['low']
    
    @property
    def close(self) -> torch.Tensor:
        ''' The historical closing prices for the instrument '''
        return self.features['close']

    @property
    def volume(self) -> torch.Tensor:
        ''' The historical trading volumes for the instrument '''
        return self.features['volume']

    def get_ohlcv(self) -> td.TensorDict:
        ''' The historical OHLCV data for the instrument '''
        return self.features.select('open', 'high', 'low', 'close', 'volume')


    def get_feature_tensor(self) -> torch.Tensor:
        ''' Return the instrument feature data as a tensor

            Returns
            -------
                Tensor : The instrument data of shape (data_length, num_features)
        '''
        return torch.stack([self.features[key] for key in self.features.keys()], dim=1)

    
    def get_feature(self, feature: str) -> torch.Tensor:
        ''' Return the feature data
        
            Parameters
            ----------
                feature : str
                    The name of the feature
        
            Returns
            -------
                Tensor : The feature data
        '''
        return self.features[feature]
    

    def set_feature(self, feature: str, value: torch.Tensor) -> None:
        ''' Set the feature data
        
            Parameters
            ----------
                feature : str
                    The name of the feature
                value : Tensor
                    The feature data
        '''
        value = value.flatten()
        assert len(value) == len(self), f'Feature data must have the same length as the instrument data, found {len(value)} != {len(self)}'
        self.features[feature] = value


    def del_feature(self, feature: str) -> None:
        ''' Remove the feature data
        
            Parameters
            ----------
                feature : str
                    The name of the feature
        '''
        del self.features[feature]


    def add_indicators(self, indicators: dict) -> int:
        ''' Add indicators to the feature data
        
            Parameters
            ----------
                indicators : dict
                    The indicators and their arguments {name: {arg: value, ...}, ...}

            Returns
            -------
                int: The lookback period of the indicator
        '''
        lookback = 0
        for indicator, params in indicators.items():
            func = abstract.Function(indicator)
            func.set_parameters(params)
            lookback = max(lookback, func.lookback)
            out_names = func.output_names
            res = func.run(self.get_ohlcv().to(torch.float64).numpy())
            if not isinstance(res, list): res = [res]
            for name, output in zip(out_names, res):
                if name == 'real': name = indicator
                name = name + '_' + '_'.join([f'{value}' for value in params.values()])
                self.features[name] = torch.from_numpy(output).to(torch.float32)
        self.clip(start=lookback)
        return lookback
            

    ''' ##################################################################
        ###                     Data Augmentation                      ###
        ################################################################## '''
    
    def clip(self, start: int=0, end: int=None) -> None:
        ''' Clip the data to the given range
        
            *Note: This method is applied in-place; cloning the instrument is advised.*
        
            Parameters
            ----------
                start : int
                    The starting index
                end : int
                    The ending index
        '''
        if end is None: end = len(self)
        self.features = self.features[start:end]
        self.prices = self.prices[start:end]
        self.datetimes = self.datetimes[start:end]


    def stationary(self, thres: float = 1e-5, pickle_path: str = None) -> int:
        ''' Make the data stationary

            Uses the fixed-width window fractional difference method described in Chapter 5
            of "Advances in Financial Machine Learning" (Marcos Lopez de Prado, 2018)

            *Note: This method is applied in-place; cloning the instrument is advised.*
            
            Parameters
            ----------
                thres : float
                    The threshold for the differentiation

            Returns
            -------
                int: The maximum differentiation width
        '''
        if pickle_path is not None:
            exists = os.path.exists(pickle_path)
            d_opt = pd.read_pickle(pickle_path) if exists else None
        ffd = FixedFracDiff(self.features, thres, d_opt)
        self.features = ffd.fit_transform()
        if pickle_path is not None and not exists: 
            pd.to_pickle(ffd.d_opt, pickle_path)
        return ffd.get_max_width()
    

    def split(self, purge_length: int, train_ratio: float = 0.8) -> tuple['Instrument', 'Instrument']:
        ''' Split the data into training and testing sets

            Parameters
            ----------
                purge_length : int
                    The number of datetimes to purge from the beginning of the testing data
                train_ratio : float
                    The ratio of training data to total data

            Returns
            -------
                tuple[Instrument, Instrument] : The training and testing data
                
            Notes
            -----
                - Train data contains the first `train_ratio` of the data
                - Test data contains the remaining data after clipping the first `purge_length` datetimes
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
    

    def scale(self, method: str = 'minmax') -> 'Instrument':
        ''' Scale the data feature(s) using the given method
          
            Parameters
            ----------
                method : str
                    The scaling method to use

            Notes
            -----
            - This method is applied in-place; clone the instrument first if the original data is needed.
            - Scaling considers the whole time-series; split the data first to prevent data leakage.
        '''
        for f in self.features.keys():
            assert len(self.features[f].size()) == 1, f'Cannot scale feature {f} with multi-dimensional shape {self.features[f].size()}'
            if method == 'standard': scaler = StandardScaler()
            elif method == 'minmax': scaler = MinMaxScaler()
            else: raise ValueError(f'Invalid scaling method: {method}')
            self.features[f] = torch.tensor(scaler.fit_transform(self.features[f].reshape(-1, 1)).flatten())
    
    
    def window(self, window_size: int) -> None:
        ''' Return the windowed data
    
            Parameters
            ----------
                window_size : int
                    The window size

            Notes
            -----
            - This method is applied in-place; clone the instrument first if the original data is needed.
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
    
    
