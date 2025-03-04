
import os
import pickle
import torch

import numpy as np
import pandas as pd
import tensordict as td

from tensordict.tensorclass import tensorclass
from torch.utils.data import DataLoader

from data.instrument import Instrument
from util.logger import TrainingLogger

@tensorclass
class InstrumentPool:
    ''' Class for handling a pool of financial instrument data '''
    targets: td.TensorDict
    context: td.TensorDict
    lookback_length: int

    @classmethod
    def from_data(cls, targets: dict[str, pd.DataFrame], context: dict[str, pd.DataFrame]) -> 'InstrumentPool':
        ''' Create an instrument pool from the given data
        
            Parameters
            ----------
                targets : dict[str, DataFrame]
                    The instrument data of the form {ticker: DataFrame}
                context : dict[str, DataFrame]
                    The context data of the form {ticker: DataFrame}

            returns:
                InstrumentPool: The instrument pool
        '''
        # Validate that input is homogeneous and correctly formatted
        assert isinstance(targets, dict), "Target data must be dictionary"
        assert isinstance(context, dict), "Context data must be dictionary"
        assert all([isinstance(targets[t], pd.DataFrame) for t in list(targets.keys())]), "All targets data must be of type DataFrame"
        assert all([isinstance(context[t], pd.DataFrame) for t in list(context.keys())]), "All context data must be of type DataFrame"
        assert all([targets[list(targets.keys())[0]].columns.equals(targets[t].columns) for t in list(targets.keys())[1:]]), "Columns do not match for all target tickers"
        assert all([context[list(context.keys())[0]].columns.equals(context[t].columns) for t in list(context.keys())[1:]]), "Columns do not match for all context tickers"

        target_pool = td.TensorDict({})
        n_targets = len(targets)
        for i, (ticker, asset_data) in enumerate(targets.items()):
            TrainingLogger.progress_logger("Building Target Instrument Pool", i, n_targets)
            target_pool[ticker] = Instrument.from_data(asset_data)

        context_pool = td.TensorDict({})
        n_context = len(context)
        for i, (ticker, context_data) in enumerate(context.items()):
            TrainingLogger.progress_logger("Building Context Instrument Pool", i, n_context)
            context_pool[ticker] = Instrument.from_data(context_data)

        return cls(target_pool, context_pool)
    

    ''' ##################################################################
        ###                       Magic Methods                        ###
        ################################################################## '''

    def __init__(self, targets: td.TensorDict, context: td.TensorDict, lookback_length: int = 0) -> None:
        ''' Initialize the instrument pool

            Parameters
            ----------
                targets : TensorDict
                    Primary data for the target instruments (those to be acted upon)
                context : TensorDict
                    Auxiliary data for contextualizing the targets  
                    *e.g., market data, sector data, benchmark indices, etc.*
                lookback_length : int
                    The lookback length for the data

            Notes
            -----
            Targets and Context must both be of the form:
            ```python
            TensorDict({
                'ticker': Instrument
            })
        '''

        # Validate that constituent instrument have homogeneous data
        assert all([isinstance(val, Instrument) for val in list(targets.values())+list(context.values())]), "All data must be of type Instrument"
        assert all([targets[list(targets.keys())[0]]['features'].keys() == targets[t]['features'].keys() for t in list(targets.keys())[1:]]), "Features do not match for all target tickers"
        assert all([context[list(context.keys())[0]]['features'].keys() == context[t]['features'].keys() for t in list(context.keys())[1:]]), "Features do not match for all context tickers"
        
        # Initialize the instrument pool
        self.targets = targets
        self.context = context
        self.lookback_length = lookback_length


    def __len__(self) -> int:
        return len(self.datetimes)
    

    def __getitem__(self, key: str) -> Instrument:
        if key in self.target_tickers:
            return self.targets[key]
        return self.context[key]
    

    def __delitem__(self, key: str) -> None:
        if key in self.target_tickers:
            del self.targets[key]
        else:
            del self.context[key]


    def __str__(self) -> str:
        return f"InstrumentPool:\nDates: {self.datetimes[0]} to {self.datetimes[-1]}\nFeatures:\n{", ".join(self.features)}\nTarget Instruments:\n{', '.join(self.target_tickers)}\nContext Instruments\n{', '.join(self.context_tickers)}"

    
    ''' ##################################################################
        ###                  Instrument Data Accessors                 ###
        ################################################################## '''
    
    @property
    def target_tickers(self) -> list[str]:
        ''' The list of target tickers '''
        return list(self.targets.keys())

    @property
    def context_tickers(self) -> list[str]:
        ''' The list of context tickers '''
        return list(self.context.keys())

    @property
    def datetimes(self) -> list[pd.Timestamp]:
        ''' The array of datetimes '''
        datetimes = self.targets[self.target_tickers[0]]['datetimes']
        assert all([np.array_equal(datetimes, self.targets[t]['datetimes']) for t in self.target_tickers]), "Dates do not match for all tickers"
        return datetimes

    @property
    def features(self) -> list[str]:
        ''' The list of feature names '''
        names = self.targets[self.target_tickers[0]]['features'].keys()
        assert all([names == self.targets[t]['features'].keys() for t in self.target_tickers]), "Features do not match for all tickers"
        return list(names)

    def target_shape(self) -> tuple[int, int, int]:
        ''' The shape of the target data  
            *(n_instruments, n_steps, n_features)* '''
        return (len(self.target_tickers), len(self.datetimes), len(self.features))
    
    def context_shape(self) -> tuple[int, int, int]:
        ''' The shape of the context data  
            *(n_instruments, n_steps, n_features)* '''
        return (len(self.context_tickers), len(self.datetimes), len(self.features))
    

    ''' #################################################################
        ###                     Feature Accessors                     ###
        ################################################################# '''    

    def open(self, ticker: str = None) -> td.TensorDict | torch.Tensor:
        ''' Shortcut to get open prices for an instrument (default: all instruments)  
            *See InstrumentPool.get_feature()*
        '''
        return self.get_feature('open', ticker)

    def high(self, ticker: str = None) -> td.TensorDict | torch.Tensor:
        ''' Shortcut to get high prices for an instrument (default: all instruments)  
            *See InstrumentPool.get_feature()*
        '''
        return self.get_feature('high', ticker)

    def low(self, ticker: str = None) -> td.TensorDict | torch.Tensor:
        ''' Shortcut to get low prices for an instrument (default: all instruments)  
            *See InstrumentPool.get_feature()*
        '''
        return self.get_feature('low', ticker)

    def close(self, ticker: str = None) -> td.TensorDict | torch.Tensor:
        ''' Shortcut to get close prices for an instrument (default: all instruments)  
            *See InstrumentPool.get_feature()*
        '''
        return self.get_feature('close', ticker)

    def volume(self, ticker: str = None) -> td.TensorDict | torch.Tensor:
        ''' Shortcut to get volume for an instrument (default: all instruments)  
            *See InstrumentPool.get_feature()* '''
        return self.get_feature('volume', ticker)


    def get_ohlcv(self, ticker: str = None) -> td.TensorDict:
        ''' Return the open, high, low, close, and volume data for the instrument
        
            Parameters
            ----------
                ticker : str
                    The ticker of the instrument
                
            Returns
            -------
                TensorDict : The open, high, low, close, and volume data

            Notes
            -----
                If ticker is 'None', the feature data for all instruments is returned as:
                ```python
                TensorDict({
                    'targets': TensorDict({ticker: feature_data}),
                    'context': TensorDict({ticker: feature_data})
                })
                ```
        '''
        if ticker is None:
            return td.TensorDict({
                'targets': td.TensorDict({t: self.targets[t].get_ohlcv() for t in self.target_tickers}),
                'context': td.TensorDict({t: self.context[t].get_ohlcv() for t in self.context_tickers})
            })
        elif ticker in self.target_tickers:
            return self.targets[ticker].get_ohlcv()
        return self.context[ticker].get_ohlcv()
    

    def get_feature(self, feature: str, ticker: str = None) -> td.TensorDict | torch.Tensor:
        ''' Return the feature data for the instrument
        
            Parameters
            ----------
                feature : str
                    The feature to return
                ticker : str
                    The ticker of the instrument
                
            Returns
            -------
                TensorDict | Tensor : The feature data

            Notes
            -----
                If ticker is 'None', the feature data for all instruments is returned as:
                ```python
                TensorDict({
                    'targets': TensorDict({ticker: feature_data}),
                    'context': TensorDict({ticker: feature_data})
                })
                ```
        '''
        if ticker is None:
            return td.TensorDict({
                'targets': td.TensorDict({t: self.targets[t].get_feature(feature) for t in self.target_tickers}),
                'context': td.TensorDict({t: self.context[t].get_feature(feature) for t in self.context_tickers})
            })
        elif ticker in self.target_tickers:
            return self.targets[ticker].get_feature(feature)
        return self.context[ticker].get_feature(feature)
    

    def set_feature(self, ticker: str, feature: str, value: pd.DataFrame) -> None:
        ''' Set the feature data for the instrument
        
            Parameters
            ----------
                ticker : str
                    The ticker of the instrument
                feature : str
                    The feature to set
                value : DataFrame
                    The feature data
        '''
        drop_idcs = value.index.difference(self.targets[ticker]['datetimes'])
        if len(drop_idcs) > 0: value = value.drop(drop_idcs)
        assert len(value) == len(self.datetimes), f"Expected data length ({len(self.datetimes)}), got {len(value)} when setting feature {feature} for ticker {ticker}"
        data = torch.tensor(value.to_numpy(), dtype=torch.float32)
        if ticker in self.target_tickers:
            self.targets[ticker].set_feature(feature, data)
        else:
            self.context[ticker].set_feature(feature, data)


    def del_feature(self, feature: str) -> None:
        ''' Delete the feature for **all** instruments
        
            Parameters
            ----------
                feature : str
                    The feature to delete
        '''
        for ticker in self.target_tickers + self.context_tickers:
            if feature in self.targets[ticker]['features'].keys():
                self.targets[ticker].del_feature(feature)


    def add_indicators(self, indicators: dict) -> None:
        ''' Add indicators to the instrument data
        
            Parameters
            ----------
                indicators (dict): The indicators and their arguments {name: {arg: value, ...}, ...}
        '''
        if indicators is None or len(indicators) == 0: return
        max_lookback = 0
        for ticker in self.target_tickers:
            max_lookback = max(max_lookback, self.targets[ticker].add_indicators(indicators))
        for ticker in self.context_tickers:
            max_lookback = max(max_lookback, self.context[ticker].add_indicators(indicators))
        self.lookback_length = max(max_lookback, self.lookback_length)
    

    ''' #################################################################
        ###                     Data Augmentation                     ###
        ################################################################# '''
    
    def clip(self, start: int = 0, end: int = None) -> None:
        ''' Clip the data to the given indices
        
            Parameters
            ----------
                start (int): The start index to clip the data
                end (int): The end index to clip the data

            Notes
            -----
                This method is applied in-place; clone the instrument pool first if the original data is needed.
        '''
        for ticker in self.target_tickers:
            self.targets[ticker].clip(start, end)
        for ticker in self.context_tickers:
            self.context[ticker].clip(start, end)


    def stationary(self, thresh: float = 1e-5, pickle_dir: str = None, log: bool = False) -> None:
        ''' Make the data stationary

            Uses the fixed-width window fractional difference method described in Chapter 5
            of "Advances in Financial Machine Learning" (Marcos Lopez de Prado, 2018)
            
            Parameters
            ----------
                thresh : float
                    The threshold for the differentiation

            Returns
            -------
                InstrumentPool:
                    The stationary data

            Notes
            -----
                This method is applied in-place; clone the instrument pool first if the original data is needed.
        '''
        max_width = 0
        for i, ticker in enumerate(self.target_tickers):
            if log: TrainingLogger.progress_logger(f"Making Targets Data Stationary | {ticker}", i, len(self.target_tickers) - 1)
            pickle_path = pickle_dir + f'd_opt_{ticker}.pkl'
            max_width = max(max_width, self.targets[ticker].stationary(thresh, pickle_path))
        for i, ticker in enumerate(self.context_tickers):
            if log: TrainingLogger.progress_logger(f"Making Context Data Stationary | {ticker}", i, len(self.context_tickers) - 1)
            pickle_path = pickle_dir + f'd_opt_{ticker}.pkl'
            max_width = max(max_width, self.context[ticker].stationary(thresh, pickle_path))
        self.lookback_length = max(max_width, self.lookback_length)
    
    
    def window(self, window_size: int = 32) -> None:
        ''' Window the data into sequences of the given size
        
            Parameters
            ----------
                window_size : int
                    The window size for the data

            Notes
            -----
            - This method is applied in-place; clone the instrument pool first if the original data is needed.
            - It is recommended to split the data into training and testing sets, then window each.
        '''
        for ticker in self.tickers:
            self.targets[ticker].window(window_size)
        self.lookback_length = max(window_size, self.lookback_length)


    def split(self, train_ratio: float = 0.8) -> tuple['InstrumentPool', 'InstrumentPool']:
        ''' Split the data into training and testing sets
        
            Parameters
            ----------
                train_ratio : float
                    The ratio of data to use for training

            Returns
            -------
                tuple[InstrumentPool, InstrumentPool]: The training and testing data
        '''
        train_targets, train_context = td.TensorDict({}), td.TensorDict({})
        test_targets, test_context = td.TensorDict({}), td.TensorDict({})

        for ticker in self.target_tickers:
            train_targets[ticker], test_targets[ticker] = self.targets[ticker].split(self.lookback_length, train_ratio)
        for ticker in self.context_tickers:
            train_context[ticker], test_context[ticker] = self.context[ticker].split(self.lookback_length, train_ratio)
        
        train_pool = InstrumentPool(train_targets, train_context, self.lookback_length)
        test_pool = InstrumentPool(test_targets, test_context, self.lookback_length)

        return train_pool, test_pool
    

    def scale(self, method: str = 'minmax') -> None:
        ''' Scale the data using the given method

            Notes
            ----- 
                - This method is applied in-place; clone the instrument pool first if the original data is needed.
                - Scaling considers the whole time-series; split the data first to prevent data leakage.
                
            Parameters
            ----------
                method : str
                    The scaling method to use ('minmax'/'standard')
        '''
        for ticker in self.target_tickers:
            self.targets[ticker].scale(method)
        for ticker in self.context_tickers:
            self.context[ticker].scale(method)
    

    ''' ##################################################################
        ###                       Helper Methods                       ###
        ################################################################## '''

    def save(self, path: str, name: str) -> None:
        ''' Save the instrument pool to a pickle file

            Parameters
            ----------
                path : str
                    The directory to save the pickle files
                name : str
                    The name of the pickle file
        '''
        if not os.path.exists(path): os.makedirs(path)
        pickle.dump((self.targets, self.context, self.lookback_length), open(path + name + ".pkl", "wb"))


    @classmethod
    def load(cls, path: str, name: str) -> 'InstrumentPool':
        
        ''' Load the instrument pool from a pickle file

            Parameters
            ----------
                path : str
                    The directory to load the pickle files
                name : str
                    The name of the pickle file

            Returns
            -------
                InstrumentPool: The instrument pool
        '''
        assert os.path.exists(path), f"Pickle directory {path} not found"
        pool, context, lookback_length = pickle.load(open(path + name + ".pkl", "rb"))
        return cls(pool, context, lookback_length)
    

    def get_loader(self) -> DataLoader:
        ''' Get a dataloader for the instrument pool

            Returns
            -------
                DataLoader: The dataloader
        '''
        data = td.TensorDict({
            'datetimes': np.array(self.datetimes),
            'prices': torch.stack([self.targets[t]['prices'] for t in self.target_tickers], dim=0),
            'targets': torch.stack([self.targets[t].feature_tensor() for t in self.target_tickers], dim=0),
            'context': torch.stack([self.context[t].feature_tensor() for t in self.context_tickers], dim=0)
        })
        return DataLoader(data, batch_size=1, shuffle=False, collate_fn=lambda x: x)


    def wf_loaders(self, 
                   train_ratio: float = 0.8,
                   thres: float | None = None,
                   window_size: int = 1,
                   scale: str | None = None
                   ) -> DataLoader:
        ''' Get dataloaders for training and backtesting with the walk-forward method

            Notes
            -----
                Dataloaders yield the tuple (datetime, prices, data):
                    **datetime** (ndarray[1,]): The datetime of the data  
                    **prices** (Tensor[n_instrument, window_size]): Relative price data  
                    **data** (Tensor[n_instrument, window_size, n_features]): Feature data
        
            Parameters
            ----------
                train_ratio : float
                    The ratio of data to use for training
                thres : float
                    The threshold for differentiation    *(e.g. 1e-5)*
                window_size (int): The window size for the data     *(e.g. 32)*
                scale : str
                    The scaling method to use              *(None/'minmax'/'standard')*

            Returns
            -------
                (tuple[DataLoader, DataLoader]): The training and testing dataloaders
        '''
        pool = self.clone()

        if thres is not None: pool.stationary(thres)
        train_pool, test_pool = pool.split(train_ratio)
        if scale is not None:
            train_pool.scale(scale)
            test_pool.scale(scale)
        if window_size > 1:
            train_pool.window(window_size)
            test_pool.window(window_size)

        train_data = td.TensorDict({
            'datetimes': np.array(train_pool.datetimes),
            'prices': torch.stack([train_pool.pool[t]['prices'] for t in train_pool.tickers], dim=0),
            'data': torch.stack([train_pool.pool[t].get_feature_tensor() for t in train_pool.tickers], dim=0)
        })
        
        test_data = td.TensorDict({
            'datetimes': np.array(test_pool.datetimes),
            'prices': torch.stack([test_pool.pool[t]['prices'] for t in test_pool.tickers], dim=0),
            'data': torch.stack([test_pool.pool[t].get_feature_tensor() for t in test_pool.tickers], dim=0)
        })

        train_loader = DataLoader(train_data, batch_size=1, shuffle=False, collate_fn=lambda x: x)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=lambda x: x)

        return train_loader, test_loader
    