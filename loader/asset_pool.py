
import numpy as np
import os
import pandas as pd
import pickle
import tensordict as td
from tensordict.tensorclass import tensorclass
import torch
from typing import Any

from asset import Asset

@tensorclass
class AssetPool:
    ''' ### Class for handling asset data '''
    pool: td.TensorDict
    lookback_length: int
    

    ''' ##################################################################
        ###                       Magic Methods                        ###
        ################################################################## '''

    def __init__(self, pool: dict[str, Asset] | td.TensorDict, lookback_length: int = 0) -> None:
        ''' ### Initialize the asset pool '''
        # Validate that constituent assets have homogeneous data
        assert isinstance(pool, dict) or isinstance(pool, td.TensorDict), "Asset pool must be created from dict or TensorDict"
        assert all([isinstance(pool[t], Asset) for t in pool.keys()]), "All assets must be of type Asset"
        assert all([pool[list(pool.keys())[0]].info.keys() == pool[t].info.keys() for t in list(pool.keys())[1:]]), "Properties must match for all assets"
        assert all([pool[list(pool.keys())[0]].features.keys() == pool[t].features.keys() for t in list(pool.keys())[1:]]), "Features must match for all assets"
        assert all([np.array_equal(pool[list(pool.keys())[0]].datetimes, pool[t].datetimes) for t in list(pool.keys())[1:]]), "Dates must match for all assets"
        
        # Initialize the asset pool
        self.pool = pool if isinstance(pool, td.TensorDict) else td.TensorDict(pool)
        self.lookback_length = lookback_length


    def __len__(self) -> int:
        ''' ### Return the length of the time series data '''
        return len(self.datetimes)
    

    def __iter__(self):
        ''' ### Return the iterator for the asset pool '''
        assert all([len(self.pool[t]) == len(self) for t in self.tickers]), "Data lengths do not match for all tickers"
        for idx in range(len(self)):
            yield self.at(idx)
    

    def __setitem__(self, key: str, value: Asset | tuple[dict[str,], pd.DataFrame]) -> None:
        ''' ### Add an asset to the pool
        
            Args:
                key (str): The asset ticker symbol
                value (tuple[dict[str,], pd.DataFrame]): The asset info and data
        '''
        if isinstance(value, Asset):
            self.pool[key] = value
        else:
            assert isinstance(value, tuple) and len(value) == 2, "Expected tuple of length 2"
            assert isinstance(value[0], dict) and isinstance(value[1], pd.DataFrame), "Expected tuple of (dict, pd.DataFrame)"
            self.pool[key] = Asset.from_data(key, value[0], value[1])


    def __getitem__(self, key: str) -> Asset:
        ''' ### Return the asset from the pool
        
            Args:
                key (str): The asset ticker symbol
        '''
        return self.pool[key]
    

    def __delitem__(self, key: str) -> None:
        ''' ### Remove an asset from the pool
        
            Args:
                key (str): The asset ticker symbol
        '''
        del self.pool[key]


    def __str__(self) -> str:
        ''' ### Return a string representation of the asset pool '''
        return f"AssetPool:\nDates: {self.datetimes[0]} to {self.datetimes[-1]}\nFeatures: {",".join(self.features)}\nTickers\n{'\n'.join([str(a) for a in self.pool.values()])}"

    
    ''' ##################################################################
        ###                     Asset Data Accessors                    ###
        ################################################################## '''
    
    @property
    def tickers(self) -> list[str]:
        ''' The list of tickers '''
        return list(self.pool.keys())
    

    @property
    def info(self) -> list[str]:
        ''' ### The list of info names '''
        names = self.pool[self.tickers[0]].info.keys()
        assert all([names == self.pool[t].info.keys() for t in self.tickers]), "Properties do not match for all tickers"
        return list(names)
    

    @property
    def datetimes(self) -> list[pd.Timestamp]:
        ''' The array of datetimes '''
        datetimes = self.pool[self.tickers[0]].datetimes
        assert all([np.array_equal(datetimes, self.pool[t].datetimes) for t in self.tickers]), "Dates do not match for all tickers"
        return datetimes
    

    @property
    def prices(self) -> td.TensorDict:
        ''' ### The price data '''
        return td.TensorDict({t: self.pool[t].prices for t in self.tickers})
    

    @property
    def features(self) -> list[str]:
        ''' The list of feature names '''
        names = self.pool[self.tickers[0]].features.keys()
        assert all([names == self.pool[t].features.keys() for t in self.tickers]), "Features do not match for all tickers"
        return list(names)
    

    def shape(self) -> tuple[int, int, int]:
        ''' ### The shape of the data '''
        return (len(self.tickers), len(self.datetimes), len(self.features))
    

    def at(self, idx: int) -> 'AssetPool':
        ''' ### Return the asset pool at the given index
        
            Args:
                idx (int): The index to return
        
            Returns:
                AssetPool: The asset pool at the given index
        '''
        return AssetPool({t: self.pool[t].at(idx) for t in self.tickers})
    

    ''' ##################################################################
        ###                     Property Accessors                     ###
        ################################################################## '''    

    def get_industry(self, ticker: str = None) -> dict[str, str] | str:
        ''' ### Return the industry of the asset '''
        return self.pool[ticker].industry
    
    
    def get_info(self, info: str, ticker: str = None) -> dict[str, Any] | Any:
        ''' ### Return the info of the asset
        
            Args:
                info (str): The info to return
                ticker (str): The ticker of the asset (default: all tickers)
                
            Returns:
                (dict[str, Any] | Any): The info data
        '''
        if ticker is None:
            return {t: self.pool[t].info[info] for t in self.tickers}
        return self.pool[ticker].get_info(info)
    

    def set_info(self, ticker: str, info: str, value) -> None:
        ''' ### Set the info of the asset

            Args:
                info (str): The info to set
                value: The info data
        '''
        for ticker in self.tickers:
            if info in self.pool[ticker].info.keys():
                assert isinstance(value, type(self.pool[ticker].info[info])), f"Expected type {type(self.pool[ticker].info[info])}, got {type(value)} for info {info}"
        self.pool[ticker].set_info(info, value)


    def del_info(self, info: str) -> None:
        ''' ### Delete the info for all assets

            Args:
                info (str): The info to delete
        '''
        for ticker in self.tickers:
            if info in self.pool[ticker].info.keys():
                self.pool[ticker].del_info(info)


    def get_sector(self, ticker: str = None) -> dict[str, str] | str:
        ''' ### Return the sector of the asset
            
            Args:
                ticker (str): The ticker of the asset (default: all tickers)
            
            Returns:
                (dict[str, str] | str): The sector data
        '''
        return self.pool[ticker].sector
    

    ''' #################################################################
        ###                     Feature Accessors                     ###
        ################################################################# '''
    
    def get_feature(self, feature: str, ticker: str = None) -> td.TensorDict | torch.Tensor:
        ''' ### Return the feature of the asset
        
            Args:
                feature (str): The feature to return
                ticker (str): The ticker of the asset (default: all tickers)
                
            Returns:
                td.TensorDict | torch.Tensor: The feature data
        '''
        if ticker is None:
            return td.TensorDict({t: self.pool[t].get_feature(feature) for t in self.tickers})
        return self.pool[ticker].get_feature(feature)
    

    def set_feature(self, ticker: str, feature: str, value: pd.DataFrame) -> None:
        ''' ### Set the feature of the asset 
        
            Args:
                feature (str): The feature to set
                value (pd.DataFrame): The feature data
        '''
        drop_idcs = value.index.difference(self.pool[ticker].datetimes)
        if len(drop_idcs) > 0: value = value.drop(drop_idcs)
        assert len(value) == len(self.datetimes), f"Expected data length ({len(self.datetimes)}), got {len(value)} when setting feature {feature} for ticker {ticker}"
        self.pool[ticker].set_feature(feature, torch.tensor(value.to_numpy(), dtype=torch.float32))


    def del_feature(self, feature: str) -> None:
        ''' ### Delete the feature for all assets
        
            Args:
                feature (str): The feature to delete
        '''
        for ticker in self.tickers:
            if feature in self.pool[ticker].features.keys():
                self.pool[ticker].del_feature(feature)


    def add_indicators(self, indicators: dict) -> None:
        ''' ### Add an indicator to the asset pool
        
            Args:
                indicators (dict): The indicators and their arguments {name: {arg: value, ...}, ...}
        '''
        if indicators is None or len(indicators) == 0: return
        max_lookback = 0
        for ticker in self.tickers:
            max_lookback = max(max_lookback, self.pool[ticker].add_indicators(indicators))
        self.lookback_length = max(max_lookback, self.lookback_length)
    

    def open(self, ticker: str = None) -> td.TensorDict | torch.Tensor:
        ''' ### Return the opening prices

            Args:
                ticker (str): The ticker of the asset (default: all tickers)

            Returns:
                td.TensorDict | torch.Tensor: The opening prices
        '''
        if ticker is None:
            return td.TensorDict({t: self.pool[t].open for t in self.tickers})
        return self.pool[ticker].open
    

    def high(self, ticker: str = None) -> td.TensorDict | torch.Tensor:
        ''' ### Return the high prices

            Args:
                ticker (str): The ticker of the asset (default: all tickers)

            Returns:
                td.TensorDict | torch.Tensor: The high prices
        '''
        if ticker is None:
            return td.TensorDict({t: self.pool[t].high for t in self.tickers})
        return self.pool[ticker].high
    

    def low(self, ticker: str = None) -> td.TensorDict | torch.Tensor:
        ''' ### Return the low prices

            Args:
                ticker (str): The ticker of the asset (default: all tickers)

            Returns:
                td.TensorDict | torch.Tensor: The low prices
        '''
        if ticker is None:
            return td.TensorDict({t: self.pool[t].low for t in self.tickers})
        return self.pool[ticker].low
    

    def close(self, ticker: str = None) -> td.TensorDict | torch.Tensor:
        ''' ### Return the closing prices

            Args:
                ticker (str): The ticker of the asset (default: all tickers)

            Returns:
                td.TensorDict | torch.Tensor: The closing prices
        '''
        if ticker is None:
            return td.TensorDict({t: self.pool[t].close for t in self.tickers})
        return self.pool[ticker].close
    

    def volume(self, ticker: str = None) -> td.TensorDict | torch.Tensor:
        ''' ### Return the trading volume

            Args:
                ticker (str): The ticker of the asset (default: all tickers)

            Returns:
                td.TensorDict | torch.Tensor: The trading volume
        '''
        if ticker is None:
            return td.TensorDict({t: self.pool[t].volume for t in self.tickers})
        return self.pool[ticker].volume
    

    def get_ohlcv(self, ticker: str = None) -> td.TensorDict:
        ''' ### Return the OHLCV data

            Args:
                ticker (str): The ticker of the asset (default: all tickers)

            Returns:
                Tensor: The OHLCV data
        '''
        if ticker is None:
            return td.TensorDict({t: self.pool[t].get_ohlcv() for t in self.tickers})
        return self.pool[ticker].get_ohlcv()
    

    def get_ohlcv_dict(self) -> dict[str, torch.Tensor]:
        ''' ### Return the OHLCV data

            Returns:
                dict[str, Tensor]: The OHLCV data for all assets
        '''
        return {t: self.pool[t].get_ohlcv_dict() for t in self.tickers}
    

    ''' #################################################################
        ###                     Data Augmentation                     ###
        ################################################################# '''
    
    def clip(self, start: int = 0, end: int = None) -> None:
        ''' ### Clip the data to the given start and end indices

            ## Notes:
                - This method is applied in-place; clone the asset pool first if the original data is needed.
        
            Args:
                start (int): The start index to clip the data
                end (int): The end index to clip the data
        '''
        for ticker in self.tickers:
            self.pool[ticker].clip(start, end)


    def stationary(self, thresh: float = 1e-5) -> None:
        ''' ### Recturn the asset pool with the data differentiated to be stationary

            Uses the fixed-width window fractional difference method described in Chapter 5
            of "Advances in Financial Machine Learning" (Marcos Lopez de Prado, 2018)

            ## Notes:
                - This method is applied in-place; clone the asset pool first if the original data is needed.
            
            Args:
                thresh (float): The threshold for the differentiation

            Returns:
                AssetPool: The stationary data
        '''
        max_width = 0
        for ticker in self.tickers:
            max_width = max(max_width, self.pool[ticker].stationary(thresh))
        self.lookback_length = max(max_width, self.lookback_length)
    
    
    def window(self, window_size: int) -> None:
        ''' ### Return the asset pool with the data windowed

            ## Notes:
                - This method is applied in-place; clone the asset pool first if the original data is needed.
                - Splitting the data into training and testing sets, then windowing each is recommended.
        
            Args:
                window_size (int): The window size for the data
        '''
        for ticker in self.tickers:
            self.pool[ticker].window(window_size)
        self.lookback_length = max(window_size, self.lookback_length)


    def split(self, train_ratio: float = 0.8) -> tuple['AssetPool', 'AssetPool']:
        ''' ### Return copies of the asset pool split into training and testing data
        
            Args:
                train_ratio (float): The ratio of data to use for training

            Returns:
                tuple[AssetPool, AssetPool]: The training and testing data
        '''
        train_pool, test_pool = {}, {}
        for ticker in self.tickers:
            train_pool[ticker], test_pool[ticker] = self.pool[ticker].split(self.lookback_length, train_ratio)
        return AssetPool(train_pool), AssetPool(test_pool)
    

    def scale(self, method: str = 'minmax') -> None:
        ''' ### Return the asset pool with the data scaled

            ## Notes: 
                - This method is applied in-place; clone the asset pool first if the original data is needed.
                - Scaling considers the whole time-series; split the data first to prevent data leakage.
                
            Args:
                method (str): The scaling method to use
        '''
        for ticker in self.tickers:
            self.pool[ticker].scale(method)
    

    ''' ##################################################################
        ###                       Helper Methods                       ###
        ################################################################## '''
    
    @classmethod
    def from_data(cls, info: dict[str, dict[str,]], data: dict[str, pd.DataFrame]) -> 'AssetPool':
        ''' ### Create an asset pool from the given data
        
            Args:
                info (dict[str, dict[str,]]): The asset info of the form {ticker: {info_key: info_val}}
                data (dict[str, DataFrame]): The asset data of the form {ticker: DataFrame}

            returns:
                AssetPool: The asset pool
        '''
        # Validate that input is homogeneous and correctly formatted
        assert isinstance(info, dict) and isinstance(data, dict), "Properties and data must be dictionaries"
        assert info.keys() == data.keys(), "Properties and data must have the same keys"
        assert all([isinstance(info[t], dict) for t in info.keys()]), "All info must be of type dict"
        assert all([isinstance(data[t], pd.DataFrame) for t in data.keys()]), "All data must be of type pd.DataFrame"
        assert all([info[list(info.keys())[0]] == info[t] for t in list(info.keys())[1:]]), "Properties must match for all tickers"
        assert all([data[list(data.keys())[0]].columns.equals(data[t].columns) for t in list(data.keys())[1:]]), "Columns must match for all data"

        return cls({ticker: Asset.from_data(ticker, info[ticker], data[ticker]) for ticker in info.keys()})


    def save(self, pickle_dir: str) -> None:
        ''' ### Save the asset pool to the pickle directory

            Args:
                pickle_dir (str): The directory to save the pickle files
        '''
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        pickle.dump((self.pool, self.lookback_length), open(pickle_dir, 'wb'))


    @classmethod
    def load(cls, pickle_dir: str) -> 'AssetPool':
        
        ''' ### Load the asset pool from the pickle directory

            Args:
                pickle_dir (str): The directory to load the pickle files

            Returns:
                AssetPool: The asset pool
        '''
        assert os.path.exists(pickle_dir), f"Pickle directory {pickle_dir} not found"
        pool, lookback_length = pickle.load(open(pickle_dir, 'rb'))
        return cls(pool, lookback_length)
