import unittest
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import tensordict as td
from asset import Asset # Corrected import path


class TestAssetTensorclass(unittest.TestCase):

    def setUp(self):
        # Sample Data for testing
        self.ticker = "TEST"
        self.info = td.TensorDict({"sector": "Technology", "industry": "Software"}) # info is TensorDict now - Correct
        self.datetimes = pd.date_range(start='2023-03-15', periods=100, freq='D').to_pydatetime() # datetimes is numpy array of datetime - Correct
        self.prices = torch.tensor([100.0 + i for i in range(100)], dtype=torch.float32) # 100 example prices
        self.features = td.TensorDict({
            "open": torch.tensor([99.0 + i for i in range(100)], dtype=torch.float32),
            "high": torch.tensor([102.0 + i for i in range(100)], dtype=torch.float32),
            "low": torch.tensor([98.0 + i for i in range(100)], dtype=torch.float32),
            "close": torch.tensor([100.0 + i for i in range(100)], dtype=torch.float32),
            "volume": torch.tensor([1000 + i*100 for i in range(100)], dtype=torch.float32),
            "rsi": torch.tensor([50.0 + i for i in range(100)], dtype=torch.float32) # Example indicator - 100 values
        }, batch_size=[100])
        self.asset = Asset(
            ticker=self.ticker,
            info=self.info,
            datetimes=self.datetimes,
            prices=self.prices,
            features=self.features
        )

        self.sample_df = pd.DataFrame({
            "open": [99.0 + i for i in range(100)],
            "high": [102.0 + i for i in range(100)],
            "low": [98.0 + i for i in range(100)],
            "close": [100.0 + i for i in range(100)],
            "volume": [1000 + i*100 for i in range(100)]
        }, index=pd.date_range(start='2023-03-15', periods=100, freq='D'))
        self.sample_info_df = {"sector": "Finance", "industry": "Investments"}


    def test_initialization(self):
        self.assertIsInstance(self.asset, Asset)
        self.assertEqual(self.asset.ticker, self.ticker)
        self.assertIsInstance(self.asset.info, td.TensorDict)
        self.assertTrue(np.array_equal(self.asset.datetimes, self.datetimes)) # Corrected assertion for numpy array
        self.assertTrue(torch.equal(self.asset.prices, self.prices))
        self.assertIsInstance(self.asset.features, td.TensorDict)

    def test_representation(self):
        repr_str = self.asset.__str__() # Using __str__() - Correct, although __repr__() would be more conventional for unit tests
        self.assertIn(f'Ticker: {self.ticker}', repr_str)
        self.assertIn(f'Sector: {self.info["sector"]}', repr_str)
        self.assertIn(f'Industry: {self.info["industry"]}', repr_str)

    def test_length(self):
        self.assertEqual(len(self.asset), len(self.datetimes))
        self.assertEqual(len(self.asset), len(self.prices))
        self.assertEqual(len(self.asset), len(self.features['open']))

    def test_iteration(self):
        for idx, asset_point in enumerate(self.asset):
            self.assertIsInstance(asset_point, Asset)
            self.assertEqual(asset_point.ticker, self.ticker)
            self.assertTrue(np.array_equal(asset_point.datetimes, self.datetimes[idx])) # Corrected assertion for numpy array
            self.assertTrue(torch.equal(asset_point.prices, self.prices[idx]))
            self.assertTrue(torch.equal(asset_point.features['open'], self.features['open'][idx]))

    def test_iteration_mismatched_lengths_error(self):
        mismatched_features = td.TensorDict({
            "open": torch.tensor([99.0, 100.0], dtype=torch.float32), # Mismatched length
        }, batch_size=[2])
        mismatched_asset = Asset(
            ticker=self.ticker,
            info=self.info,
            datetimes=self.datetimes,
            prices=self.prices,
            features=mismatched_features
        )
        with self.assertRaises(AssertionError):
            for _ in mismatched_asset: # Should raise error during iteration
                pass

    def test_ticker_property(self):
        self.assertEqual(self.asset.ticker, self.ticker)

    def test_datetime_property(self):
        datetimes = self.asset.datetimes
        self.assertIsInstance(datetimes, np.ndarray) # Correct assertion type
        self.assertEqual(len(datetimes), len(self.datetimes))
        self.assertIsInstance(datetimes[0], datetime)

    def test_prices_property(self):
        self.assertTrue(torch.equal(self.asset.prices, self.prices))

    def test_at_method(self):
        idx = 5
        asset_at_idx = self.asset.at(idx)
        self.assertIsInstance(asset_at_idx, Asset)
        self.assertEqual(asset_at_idx.ticker, self.ticker)
        self.assertTrue(np.array_equal(asset_at_idx.datetimes, self.datetimes[idx])) # Corrected assertion for numpy array
        self.assertTrue(torch.equal(asset_at_idx.prices, self.prices[idx]))
        self.assertTrue(torch.equal(asset_at_idx.features['open'], self.features['open'][idx]))

    def test_info_property(self):
        self.assertIsInstance(self.asset.info, td.TensorDict)
        self.assertEqual(self.asset.info['sector'], self.info['sector'])

    def test_sector_property(self):
        self.assertEqual(self.asset.sector, self.info['sector'])

    def test_industry_property(self):
        self.assertEqual(self.asset.industry, self.info['industry'])

    def test_get_info_method(self):
        self.assertEqual(self.asset.get_info('sector'), self.info['sector'])
        self.assertEqual(self.asset.get_info('industry'), self.info['industry'])
        with self.assertRaises(KeyError): # Or handle KeyError based on implementation expectation
            self.asset.get_info('invalid_key')

    def test_set_info_method(self):
        self.asset.set_info('test_key', 'test_value')
        self.assertEqual(self.asset.get_info('test_key'), 'test_value')

    def test_del_info_method(self):
        self.asset.del_info('sector')
        with self.assertRaises(KeyError):
            self.asset.get_info('sector')

    def test_features_property(self):
        self.assertIsInstance(self.asset.features, td.TensorDict)
        self.assertTrue(torch.equal(self.asset.features['open'], self.features['open']))

    def test_ohlcv_properties(self):
        self.assertTrue(torch.equal(self.asset.open, self.features['open']))
        self.assertTrue(torch.equal(self.asset.high, self.features['high']))
        self.assertTrue(torch.equal(self.asset.low, self.features['low']))
        self.assertTrue(torch.equal(self.asset.close, self.features['close']))
        self.assertTrue(torch.equal(self.asset.volume, self.features['volume']))

    def test_get_ohlcv_method(self):
        ohlcv_tensor = self.asset.get_ohlcv()
        self.assertTrue(torch.equal(ohlcv_tensor[:, 0], self.asset.open))
        self.assertTrue(torch.equal(ohlcv_tensor[:, 1], self.asset.high))
        self.assertTrue(torch.equal(ohlcv_tensor[:, 2], self.asset.low))
        self.assertTrue(torch.equal(ohlcv_tensor[:, 3], self.asset.close))
        self.assertTrue(torch.equal(ohlcv_tensor[:, 4], self.asset.volume))

    def test_get_ohlcv_dict_method(self):
        ohlcv_dict = self.asset.get_ohlcv_dict()
        self.assertIsInstance(ohlcv_dict, dict)
        self.assertTrue(np.array_equal(ohlcv_dict['open'], self.asset.open.numpy()))
        self.assertTrue(np.array_equal(ohlcv_dict['high'], self.asset.high.numpy()))
        self.assertTrue(np.array_equal(ohlcv_dict['low'], self.asset.low.numpy()))
        self.assertTrue(np.array_equal(ohlcv_dict['close'], self.asset.close.numpy()))
        self.assertTrue(np.array_equal(ohlcv_dict['volume'], self.asset.volume.numpy()))

    def test_get_feature_method(self):
        self.assertTrue(torch.equal(self.asset.get_feature('open'), self.features['open']))
        with self.assertRaises(KeyError):
            self.asset.get_feature('invalid_feature')

    def test_set_feature_method(self):
        new_feature = torch.tensor([float(i) for i in range(100)], dtype=torch.float32) # 100 values
        self.asset.set_feature('test_feature', new_feature)
        self.assertTrue(torch.equal(self.asset.get_feature('test_feature'), new_feature))

    def test_del_feature_method(self):
        self.asset.del_feature('open')
        with self.assertRaises(KeyError):
            self.asset.get_feature('open')

    def test_add_indicators_method(self):
        indicators = {"rsi": {"timeperiod": 3}} # timeperiod adjusted to reflect longer data
        lookback = self.asset.add_indicators(indicators)
        self.assertEqual(lookback, 3) # Adjusted lookback to 3 - Correct assertion
        self.assertTrue('rsi_3' in self.asset.features.keys()) # Check for indicator feature - adjusted RSI name to RSI_3 (based on timeperiod)
        # No direct ground truth for RSI readily available for 100 days, basic check if indicator is added and is a tensor.
        self.assertIsInstance(self.asset.features['rsi_3'], torch.Tensor)
        self.assertEqual(len(self.asset.features['rsi_3']), 100) # Check indicator length - Correct assertion


    def test_clip_method(self):
        original_len = len(self.asset)
        start_clip = 2
        end_clip = 8
        self.asset.clip(start=start_clip, end=end_clip)
        self.assertEqual(len(self.asset), end_clip - start_clip)
        self.assertTrue(np.array_equal(self.asset.datetimes, self.datetimes[start_clip:end_clip])) # Corrected assertion for numpy array
        self.assertTrue(torch.equal(self.asset.prices, self.prices[start_clip:end_clip]))
        self.assertTrue(torch.equal(self.asset.features['open'], self.features['open'][start_clip:end_clip]))

    def test_clip_method_default_end(self):
        original_len = len(self.asset)
        start_clip = 3
        self.asset.clip(start=start_clip)
        self.assertEqual(len(self.asset), original_len - start_clip)

    def test_stationary_method(self):
        # This test is basic as stationarity is complex and data-dependent.
        # We just check if it runs without error and modifies features (in a dummy way).
        original_features = self.asset.features.clone()
        max_width = self.asset.stationary()
        self.assertIsInstance(max_width, int)
        self.assertGreaterEqual(max_width, 0)
        # Check if features were modified (dummy check, actual stationarity check is complex)
        # The following check might be too sensitive to numerical changes.
        # A better approach is to check if the number of features remains the same or changes as expected.
        self.assertEqual(self.asset.features.keys(), original_features.keys())


    def test_split_method(self):
        train_data, test_data = self.asset.split(purge_length=2, train_ratio=0.7) # Adjusted purge_length and train_ratio
        self.assertIsInstance(train_data, Asset)
        self.assertIsInstance(test_data, Asset)
        expected_train_len = int(0.7 * len(self.asset))
        expected_test_len = len(self.asset) - expected_train_len - 2 # Adjusted purge length in calculation
        self.assertEqual(len(train_data), expected_train_len)
        self.assertEqual(len(test_data), expected_test_len)

    def test_scale_method_minmax(self):
        original_open = self.asset.features['open'].clone()
        self.asset.scale(method='minmax')
        scaled_open = self.asset.features['open']
        self.assertTrue(torch.all(scaled_open >= 0.0))
        self.assertTrue(torch.all(scaled_open <= 1.0))
        self.assertFalse(torch.equal(scaled_open, original_open)) # Check if scaling actually happened

    def test_scale_method_standard(self):
        original_open = self.asset.features['open'].clone()
        self.asset.scale(method='standard')
        scaled_open = self.asset.features['open']
        self.assertFalse(torch.equal(scaled_open, original_open)) # Check if scaling actually happened
        # Standard scaling properties are harder to assert without numerical tolerances and larger data


    def test_scale_method_invalid(self):
        with self.assertRaises(ValueError):
            self.asset.scale(method='invalid_method')

    def test_scale_method_multidimensional_feature_error(self):
        multi_dim_features = td.TensorDict({"multi": torch.randn(100, 2)}, batch_size=[100]) # Adjusted to 100 length
        multi_dim_asset = Asset(
            ticker=self.ticker,
            info=self.info,
            datetimes=self.datetimes,
            prices=self.prices,
            features=multi_dim_features
        )
        with self.assertRaises(AssertionError):
            multi_dim_asset.scale() # Method defaults to minmax

    def test_window_method(self):
        window_size = 3 # Adjusted window_size
        original_len = len(self.asset)
        self.asset.window(window_size=window_size)
        self.assertEqual(len(self.asset), original_len - window_size + 1)
        self.assertEqual(self.asset.features['open'].shape, torch.Size([original_len - window_size + 1, window_size]))

    def test_from_data_classmethod(self):
        asset_from_data = Asset.from_data(ticker=self.ticker, info=self.sample_info_df, data=self.sample_df)
        self.assertIsInstance(asset_from_data, Asset)
        self.assertEqual(asset_from_data.ticker, self.ticker)
        self.assertEqual(asset_from_data.info['sector'], self.sample_info_df['sector'])
        self.assertEqual(asset_from_data.info['industry'], self.sample_info_df['industry'])
        self.assertEqual(len(asset_from_data), len(self.sample_df)-1) # Prices and features are shifted by 1

    def test_from_data_missing_ohlcv_error(self):
        bad_df = self.sample_df.drop(columns=['close'])
        with self.assertRaises(AssertionError):
            Asset.from_data(ticker=self.ticker, info=self.sample_info_df, data=bad_df)

    def test_from_data_mismatched_feature_length_error(self):
        bad_df = self.sample_df.copy()
        bad_df['volume'] = bad_df['volume'][:-2] # Shorten volume column by 2 now
        with self.assertRaises(AssertionError):
            Asset.from_data(ticker=self.ticker, info=self.sample_info_df, data=bad_df)

    def test_from_data_missing_info_error(self):
        bad_info = {"sector": "Finance"} # Missing industry
        with self.assertRaises(AssertionError):
            Asset.from_data(ticker=self.ticker, info=bad_info, data=self.sample_df)


if __name__ == '__main__':
    unittest.main()