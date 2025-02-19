import unittest
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import tensordict as td
from asset_pool import AssetPool  # Assuming asset_pool.py is the file containing AssetPool
from asset import Asset

class TestAssetPoolTensorclass(unittest.TestCase):

    def setUp(self):
        # Sample Data and Assets for testing
        self.tickers = ["TEST1", "TEST2"]
        self.info = {"sector": "Technology", "industry": "Software"}
        self.datetimes = pd.date_range(start='2023-03-15', periods=100, freq='D')
        self.prices = torch.tensor([100.0 + i for i in range(100)], dtype=torch.float32)
        self.features_data = {
            "open": torch.tensor([99.0 + i for i in range(100)], dtype=torch.float32),
            "high": torch.tensor([102.0 + i for i in range(100)], dtype=torch.float32),
            "low": torch.tensor([98.0 + i for i in range(100)], dtype=torch.float32),
            "close": torch.tensor([100.0 + i for i in range(100)], dtype=torch.float32),
            "volume": torch.tensor([1000 + i*100 for i in range(100)], dtype=torch.float32),
        }
        self.features = td.TensorDict(self.features_data, batch_size=[100])
        self.asset1 = Asset(
            ticker=self.tickers[0],
            info=self.info,
            datetimes=self.datetimes,
            prices=self.prices,
            features=self.features.clone()
        )
        self.asset2 = Asset(
            ticker=self.tickers[1],
            info=self.info,
            datetimes=self.datetimes,
            prices=self.prices,
            features=self.features.clone()
        )
        self.asset_dict = {self.tickers[0]: self.asset1, self.tickers[1]: self.asset2}
        self.asset_pool_dict = AssetPool(self.asset_dict)
        self.asset_pool_tensordict = AssetPool(td.TensorDict(self.asset_dict)) # Test TensorDict input

        self.sample_df = pd.DataFrame({
            "open": [99.0 + i for i in range(100)],
            "high": [102.0 + i for i in range(100)],
            "low": [98.0 + i for i in range(100)],
            "close": [100.0 + i for i in range(100)],
            "volume": [1000 + i*100 for i in range(100)]
        }, index=pd.date_range(start='2023-03-15', periods=100, freq='D'))
        self.sample_info_df = {"sector": "Finance", "industry": "Investments"}
        self.sample_data_dict = {
            self.tickers[0]: self.sample_df,
            self.tickers[1]: self.sample_df
        }
        self.sample_info_dict = {
            self.tickers[0]: self.sample_info_df,
            self.tickers[1]: self.sample_info_df
        }
        self.ohlcv_keys = ['open', 'high', 'low', 'close', 'volume']


    def test_initialization_dict_input(self):
        self.assertIsInstance(self.asset_pool_dict, AssetPool)
        self.assertIsInstance(self.asset_pool_dict.pool, td.TensorDict)
        self.assertEqual(self.asset_pool_dict.tickers, self.tickers)
        self.assertEqual(self.asset_pool_dict.lookback_length, 0)

    def test_initialization_tensordict_input(self):
        self.assertIsInstance(self.asset_pool_tensordict, AssetPool)
        self.assertIsInstance(self.asset_pool_tensordict.pool, td.TensorDict)
        self.assertEqual(self.asset_pool_tensordict.tickers, self.tickers)
        self.assertEqual(self.asset_pool_tensordict.lookback_length, 0)

    def test_initialization_invalid_input_type_error(self):
        with self.assertRaises(AssertionError):
            AssetPool([self.asset1, self.asset2]) # List input

    def test_initialization_inhomogeneous_info_error(self):
        inhomogeneous_asset = Asset(
            ticker="BAD",
            info={"sector": "Different", "industry": "Different"},
            datetimes=self.datetimes,
            prices=self.prices,
            features=self.features.clone()
        )
        bad_asset_dict = AssetPool({self.tickers[0]: self.asset1, self.tickers[1]: self.asset2})
        bad_asset_dict["BAD"] = inhomogeneous_asset
        with self.assertRaises(AssertionError):
            AssetPool(bad_asset_dict)

    def test_initialization_inhomogeneous_features_error(self):
        inhomogeneous_features = td.TensorDict({
            "bad_feature": torch.tensor([99.0 + i for i in range(100)], dtype=torch.float32),
        }, batch_size=[100])
        inhomogeneous_asset = Asset(
            ticker="BAD",
            info=self.info,
            datetimes=self.datetimes,
            prices=self.prices,
            features=inhomogeneous_features
        )
        bad_asset_dict = {self.tickers[0]: self.asset1, self.tickers[1]: self.asset2}
        bad_asset_dict["BAD"] = inhomogeneous_asset
        with self.assertRaises(AssertionError):
            AssetPool(bad_asset_dict)

    def test_initialization_inhomogeneous_dates_error(self):
        inhomogeneous_datetimes = pd.date_range(start='2023-03-16', periods=100, freq='D').to_pydatetime()
        inhomogeneous_asset = Asset(
            ticker="BAD",
            info=self.info,
            datetimes=inhomogeneous_datetimes,
            prices=self.prices,
            features=self.features.clone()
        )
        bad_asset_dict = {self.tickers[0]: self.asset1, self.tickers[1]: self.asset2}
        bad_asset_dict["BAD"] = inhomogeneous_asset
        with self.assertRaises(AssertionError):
            AssetPool(bad_asset_dict)

    def test_length(self):
        self.assertEqual(len(self.asset_pool_dict), len(self.datetimes))

    def test_iteration(self):
        for idx, asset_pool_point in enumerate(self.asset_pool_dict):
            self.assertIsInstance(asset_pool_point, AssetPool)
            self.assertEqual(asset_pool_point.tickers, self.tickers)
            for ticker in self.tickers:
                self.assertTrue(np.array_equal(asset_pool_point[ticker].datetimes, self.datetimes[idx]))
                self.assertTrue(torch.equal(asset_pool_point[ticker].prices, self.prices[idx]))
                self.assertTrue(torch.equal(asset_pool_point[ticker].features['open'], self.features['open'][idx]))

    def test_iteration_mismatched_lengths_error(self):
        short_datetimes = pd.date_range(start='2023-03-15', periods=5, freq='D').to_pydatetime()
        short_asset = Asset(
            ticker="SHORT",
            info=self.info,
            datetimes=short_datetimes,
            prices=self.prices[:5],
            features=td.TensorDict({k: v[:5] for k, v in self.features.items()}, batch_size=[5])
        )
        bad_asset_pool_dict = AssetPool({self.tickers[0]: self.asset1, self.tickers[1]: self.asset2})
        bad_asset_pool_dict["SHORT"] = short_asset
        with self.assertRaises(AssertionError):
            for _ in bad_asset_pool_dict: # Should raise error during iteration
                pass

    def test_setitem_getitem_delitem(self):
        new_ticker = "NEW"
        self.asset_pool_dict[new_ticker] = (self.sample_info_df, self.sample_df)
        self.assertIn(new_ticker, self.asset_pool_dict.tickers)
        self.assertIsInstance(self.asset_pool_dict[new_ticker], Asset)
        del self.asset_pool_dict[new_ticker]
        self.assertNotIn(new_ticker, self.asset_pool_dict.tickers)
        with self.assertRaises(KeyError):
            _ = self.asset_pool_dict[new_ticker] # Should raise KeyError after deletion

    def test_tickers_property(self):
        self.assertEqual(self.asset_pool_dict.tickers, self.tickers)

    def test_info_property(self):
        info_names = self.asset_pool_dict.info
        self.assertIsInstance(info_names, list)
        self.assertEqual(info_names, list(self.info.keys()))

    def test_info_property_inhomogeneous_error(self):
        inhomogeneous_asset = Asset(
            ticker="BAD",
            info={"bad_sector": "Different", "bad_industry": "Different"},
            datetimes=self.datetimes,
            prices=self.prices,
            features=self.features.clone()
        )
        bad_asset_pool_dict = AssetPool({self.tickers[0]: self.asset1, self.tickers[1]: self.asset2})
        bad_asset_pool_dict["BAD"] = inhomogeneous_asset
        with self.assertRaises(AssertionError):
            _ = bad_asset_pool_dict.info

    def test_dates_property(self):
        datetimes = self.asset_pool_dict.datetimes
        self.assertIsInstance(datetimes, np.ndarray)
        self.assertTrue(np.array_equal(datetimes, self.datetimes))

    def test_dates_property_inhomogeneous_error(self):
        inhomogeneous_datetimes = pd.date_range(start='2023-03-16', periods=100, freq='D').to_pydatetime()
        inhomogeneous_asset = Asset(
            ticker="BAD",
            info=self.info,
            datetimes=inhomogeneous_datetimes,
            prices=self.prices,
            features=self.features.clone()
        )
        bad_asset_pool_dict = AssetPool({self.tickers[0]: self.asset1, self.tickers[1]: self.asset2})
        bad_asset_pool_dict["BAD"] = inhomogeneous_asset
        with self.assertRaises(AssertionError):
            _ = bad_asset_pool_dict.datetimes

    def test_features_property(self):
        feature_names = self.asset_pool_dict.features
        self.assertIsInstance(feature_names, list)
        self.assertEqual(feature_names, list(self.features_data.keys()))

    def test_features_property_inhomogeneous_error(self):
        inhomogeneous_features = td.TensorDict({
            "bad_feature": torch.tensor([99.0 + i for i in range(100)], dtype=torch.float32),
        }, batch_size=[100])
        inhomogeneous_asset = Asset(
            ticker="BAD",
            info=self.info,
            datetimes=self.datetimes,
            prices=self.prices,
            features=inhomogeneous_features
        )
        bad_asset_pool_dict = AssetPool({self.tickers[0]: self.asset1, self.tickers[1]: self.asset2})
        bad_asset_pool_dict["BAD"] = inhomogeneous_asset
        with self.assertRaises(AssertionError):
            _ = bad_asset_pool_dict.features

    def test_shape_method(self):
        shape = self.asset_pool_dict.shape()
        self.assertEqual(shape, (len(self.tickers), len(self.datetimes), len(self.features_data)))

    def test_at_method(self):
        idx = 3
        asset_pool_at_idx = self.asset_pool_dict.at(idx)
        self.assertIsInstance(asset_pool_at_idx, AssetPool)
        self.assertEqual(asset_pool_at_idx.tickers, self.tickers)
        for ticker in self.tickers:
            self.assertIsInstance(asset_pool_at_idx[ticker], Asset)
            self.assertTrue(np.array_equal(asset_pool_at_idx[ticker].datetimes, self.datetimes[idx]))
            self.assertTrue(torch.equal(asset_pool_at_idx[ticker].prices, self.prices[idx]))
            self.assertTrue(torch.equal(asset_pool_at_idx[ticker].features['open'], self.features['open'][idx]))

    def test_get_industry_method(self):
        industry = self.asset_pool_dict.get_industry(self.tickers[0])
        self.assertEqual(industry, self.info['industry'])

    def test_get_info_method_single_ticker(self):
        sector = self.asset_pool_dict.get_info('sector', self.tickers[0])
        self.assertEqual(sector, self.info['sector'])
        with self.assertRaises(KeyError):
            self.asset_pool_dict.get_info('invalid_info', self.tickers[0])

    def test_get_info_method_all_tickers(self):
        sectors = self.asset_pool_dict.get_info('sector')
        self.assertIsInstance(sectors, dict) # Expecting dict now
        self.assertEqual(sectors[self.tickers[0]], self.info['sector'])
        self.assertEqual(sectors[self.tickers[1]], self.info['sector'])

    def test_set_info_method(self):
        self.asset_pool_dict.set_info(self.tickers[0], 'test_key', 'test_value')
        self.assertEqual(self.asset_pool_dict.get_info('test_key', self.tickers[0]), 'test_value')

    def test_set_info_type_error(self):
        self.asset_pool_dict.set_info(self.tickers[0], 'sector', 'NewSector') # Valid str type
        with self.assertRaises(AssertionError):
            self.asset_pool_dict.set_info(self.tickers[0], 'sector', 123) # Invalid int type

    def test_del_info_method(self):
        self.asset_pool_dict.del_info('sector')
        with self.assertRaises(KeyError):
            self.asset_pool_dict.get_info('sector', self.tickers[0])
        with self.assertRaises(KeyError):
            self.asset_pool_dict.get_info('sector', self.tickers[1])

    def test_get_sector_method(self):
        sector = self.asset_pool_dict.get_sector(self.tickers[0])
        self.assertEqual(sector, self.info['sector'])

    def test_get_feature_method_single_ticker(self):
        open_feature = self.asset_pool_dict.get_feature('open', self.tickers[0])
        self.assertTrue(torch.equal(open_feature, self.features['open']))
        with self.assertRaises(KeyError):
            self.asset_pool_dict.get_feature('invalid_feature', self.tickers[0])

    def test_get_feature_method_all_tickers(self):
        open_features = self.asset_pool_dict.get_feature('open')
        self.assertIsInstance(open_features, td.TensorDict) # Expecting TensorDict now
        self.assertTrue(torch.equal(open_features[self.tickers[0]], self.features['open']))
        self.assertTrue(torch.equal(open_features[self.tickers[1]], self.features['open']))

    def test_set_feature_method(self):
        new_feature_df = pd.DataFrame({'test_feature': [float(i) for i in range(100)]}, index=pd.to_datetime(self.datetimes))
        self.asset_pool_dict.set_feature(self.tickers[0], 'test_feature', new_feature_df)
        self.assertTrue('test_feature' in self.asset_pool_dict[self.tickers[0]].features.keys())
        self.assertTrue(torch.equal(self.asset_pool_dict.get_feature('test_feature', self.tickers[0]), torch.tensor(new_feature_df['test_feature'].values, dtype=torch.float32)))

    def test_set_feature_length_mismatch_error(self):
        short_df = self.sample_df[:-2] # shorter by 2 rows
        with self.assertRaises(AssertionError):
            self.asset_pool_dict.set_feature(self.tickers[0], 'volume', short_df)

    def test_del_feature_method(self):
        self.asset_pool_dict.del_feature('open')
        for ticker in self.tickers:
            with self.assertRaises(KeyError):
                self.asset_pool_dict.get_feature('open', ticker)

    def test_add_indicators_method(self):
        indicators = {"RSI": {"timeperiod": 3}}
        self.asset_pool_dict.add_indicators(indicators)
        for ticker in self.tickers:
            self.assertTrue('RSI_3' in self.asset_pool_dict[ticker].features.keys())
            self.assertIsInstance(self.asset_pool_dict[ticker].features['RSI_3'], torch.Tensor)
            self.assertEqual(len(self.asset_pool_dict[ticker].features['RSI_3']), len(self.datetimes))

    def test_ohlcv_methods_single_ticker(self):
        for ticker in self.tickers:
            self.assertTrue(torch.equal(self.asset_pool_dict.open(ticker), self.features['open']))
            self.assertTrue(torch.equal(self.asset_pool_dict.high(ticker), self.features['high']))
            self.assertTrue(torch.equal(self.asset_pool_dict.low(ticker), self.features['low']))
            self.assertTrue(torch.equal(self.asset_pool_dict.close(ticker), self.features['close']))
            self.assertTrue(torch.equal(self.asset_pool_dict.volume(ticker), self.features['volume']))

    def test_ohlcv_methods_all_tickers(self):
        open_prices = self.asset_pool_dict.open()
        self.assertIsInstance(open_prices, td.TensorDict) # Expecting TensorDict
        self.assertTrue(torch.equal(open_prices[self.tickers[0]], self.features['open']))
        self.assertTrue(torch.equal(open_prices[self.tickers[1]], self.features['open']))

        high_prices = self.asset_pool_dict.high()
        self.assertIsInstance(high_prices, td.TensorDict) # Expecting TensorDict
        self.assertTrue(torch.equal(high_prices[self.tickers[0]], self.features['high']))
        self.assertTrue(torch.equal(high_prices[self.tickers[1]], self.features['high']))

        low_prices = self.asset_pool_dict.low()
        self.assertIsInstance(low_prices, td.TensorDict) # Expecting TensorDict
        self.assertTrue(torch.equal(low_prices[self.tickers[0]], self.features['low']))
        self.assertTrue(torch.equal(low_prices[self.tickers[1]], self.features['low']))

        close_prices = self.asset_pool_dict.close()
        self.assertIsInstance(close_prices, td.TensorDict) # Expecting TensorDict
        self.assertTrue(torch.equal(close_prices[self.tickers[0]], self.features['close']))
        self.assertTrue(torch.equal(close_prices[self.tickers[1]], self.features['close']))

        volumes = self.asset_pool_dict.volume()
        self.assertIsInstance(volumes, td.TensorDict) # Expecting TensorDict
        self.assertTrue(torch.equal(volumes[self.tickers[0]], self.features['volume']))
        self.assertTrue(torch.equal(volumes[self.tickers[1]], self.features['volume']))


    def test_get_ohlcv_method_single_ticker(self):
        for ticker in self.tickers:
            ohlcv_tensordict = self.asset_pool_dict.get_ohlcv(ticker)
            self.assertIsInstance(ohlcv_tensordict, td.TensorDict) # Expecting TensorDict
            for key in self.ohlcv_keys:
                self.assertTrue(torch.equal(ohlcv_tensordict[key], self.features[key]))

    def test_get_ohlcv_method_all_tickers(self):
        ohlcv_pool_tensordict = self.asset_pool_dict.get_ohlcv()
        self.assertIsInstance(ohlcv_pool_tensordict, td.TensorDict) # Expecting TensorDict
        for ticker in self.tickers:
            ohlcv_tensordict = ohlcv_pool_tensordict[ticker]
            self.assertIsInstance(ohlcv_tensordict, td.TensorDict) # Expecting TensorDict for each ticker
            for key in self.ohlcv_keys:
                self.assertTrue(torch.equal(ohlcv_tensordict[key], self.features[key]))

    def test_get_ohlcv_dict_method(self):
        ohlcv_pool_dict = self.asset_pool_dict.get_ohlcv_dict()
        self.assertIsInstance(ohlcv_pool_dict, dict) # Expecting dict
        for ticker in self.tickers:
            ohlcv_dict = ohlcv_pool_dict[ticker]
            self.assertIsInstance(ohlcv_dict, dict) # Expecting dict for each ticker
            for key in self.ohlcv_keys:
                self.assertTrue(np.array_equal(ohlcv_dict[key], self.features_data[key].numpy()))

    def test_clip_method(self):
        start_clip = 2
        end_clip = 8
        self.asset_pool_dict.clip(start=start_clip, end=end_clip)
        for ticker in self.tickers:
            self.assertEqual(len(self.asset_pool_dict[ticker]), end_clip - start_clip)
            self.assertTrue(np.array_equal(self.asset_pool_dict[ticker].datetimes, self.datetimes[start_clip:end_clip]))
            self.assertTrue(torch.equal(self.asset_pool_dict[ticker].prices, self.prices[start_clip:end_clip]))
            self.assertTrue(torch.equal(self.asset_pool_dict[ticker].features['open'], self.features['open'][start_clip:end_clip]))

    def test_stationary_method(self):
        original_features = {ticker: self.asset_pool_dict[ticker].features.clone() for ticker in self.tickers}
        self.asset_pool_dict.stationary()
        for ticker in self.tickers:
            self.assertGreaterEqual(self.asset_pool_dict.lookback_length, 0)
            self.assertEqual(self.asset_pool_dict[ticker].features.keys(), original_features[ticker].keys()) # Basic check if features are still there

    def test_window_method(self):
        window_size = 3
        self.asset_pool_dict.window(window_size=window_size)
        for ticker in self.tickers:
            self.assertEqual(len(self.asset_pool_dict[ticker]), len(self.datetimes) - window_size + 1)
            self.assertEqual(self.asset_pool_dict[ticker].features['open'].shape, torch.Size([len(self.datetimes) - window_size + 1, window_size]))
        self.assertEqual(self.asset_pool_dict.lookback_length, window_size)

    def test_split_method(self):
        train_pool, test_pool = self.asset_pool_dict.split(train_ratio=0.7)
        self.assertIsInstance(train_pool, AssetPool)
        self.assertIsInstance(test_pool, AssetPool)
        expected_train_len = int(0.7 * len(self.asset_pool_dict))
        expected_test_len = len(self.asset_pool_dict) - expected_train_len
        for ticker in self.tickers:
            self.assertIsInstance(train_pool[ticker], Asset)
            self.assertIsInstance(test_pool[ticker], Asset)
            self.assertEqual(len(train_pool[ticker]), expected_train_len)
            self.assertEqual(len(test_pool[ticker]), expected_test_len)

    def test_scale_method_minmax(self):
        original_open = {ticker: self.asset_pool_dict[ticker].features['open'].clone() for ticker in self.tickers}
        self.asset_pool_dict.scale(method='minmax')
        for ticker in self.tickers:
            scaled_open = self.asset_pool_dict.open(ticker)
            self.assertTrue(torch.all(scaled_open >= 0.0))
            self.assertTrue(torch.all(scaled_open <= 1.0))
            self.assertFalse(torch.equal(scaled_open, original_open[ticker]))

    def test_scale_method_standard(self):
        original_open = {ticker: self.asset_pool_dict[ticker].features['open'].clone() for ticker in self.tickers}
        self.asset_pool_dict.scale(method='standard')
        for ticker in self.tickers:
            scaled_open = self.asset_pool_dict.open(ticker)
            self.assertFalse(torch.equal(scaled_open, original_open[ticker]))

    def test_scale_method_invalid(self):
        with self.assertRaises(ValueError):
            self.asset_pool_dict.scale(method='invalid_method')

    def test_from_data_classmethod(self):
        asset_pool_from_data = AssetPool.from_data(info=self.sample_info_dict, data=self.sample_data_dict)
        self.assertIsInstance(asset_pool_from_data, AssetPool)
        self.assertEqual(asset_pool_from_data.tickers, self.tickers)
        for ticker in self.tickers:
            self.assertIsInstance(asset_pool_from_data[ticker], Asset)
            self.assertEqual(asset_pool_from_data[ticker].ticker, ticker)
            self.assertEqual(asset_pool_from_data[ticker].info['sector'], self.sample_info_dict[ticker]['sector'])
            self.assertEqual(asset_pool_from_data[ticker].info['industry'], self.sample_info_dict[ticker]['industry'])
        self.assertEqual(len(asset_pool_from_data), len(self.sample_df)-1) # Dates are aligned and prices shifted by 1 in Asset.from_data

    def test_from_data_inhomogeneous_keys_error(self):
        bad_data_dict = {self.tickers[0]: self.sample_df} # Missing data for TEST2
        with self.assertRaises(AssertionError):
            AssetPool.from_data(info=self.sample_info_dict, data=bad_data_dict)

    def test_from_data_inhomogeneous_info_type_error(self):
        bad_info_dict = {self.tickers[0]: self.sample_info_df, self.tickers[1]: "bad_info"} # bad info type for TEST2
        with self.assertRaises(AssertionError):
            AssetPool.from_data(info=bad_info_dict, data=self.sample_data_dict)

    def test_from_data_inhomogeneous_data_type_error(self):
        bad_data_dict = {self.tickers[0]: self.sample_df, self.tickers[1]: [1,2,3]} # bad data type for TEST2
        with self.assertRaises(AssertionError):
            AssetPool.from_data(info=self.sample_info_dict, data=bad_data_dict)

    def test_from_data_inhomogeneous_properties_error(self):
        bad_info_dict = {self.tickers[0]: self.sample_info_df, self.tickers[1]: {"sector": "Different", "industry": "Different"}}
        with self.assertRaises(AssertionError):
            AssetPool.from_data(info=bad_info_dict, data=self.sample_data_dict)

    def test_from_data_inhomogeneous_columns_error(self):
        bad_df = self.sample_df.drop(columns=['close']) # Missing close column
        bad_data_dict = {self.tickers[0]: self.sample_df, self.tickers[1]: bad_df}
        with self.assertRaises(AssertionError):
            AssetPool.from_data(info=self.sample_info_dict, data=bad_data_dict)


if __name__ == '__main__':
    unittest.main()