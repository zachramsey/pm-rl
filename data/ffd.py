
import torch
import tensordict as td
from statsmodels.tsa.stattools import adfuller

class FixedFracDiff:    
    def __init__(self, data: td.TensorDict, thres: float = 1e-5, d_opt: dict[str, float] = None) -> None:
        """ ### Fixed Fractional Differencing
        This class implements the Fixed-Width Window Fractional Differencing method proposed 
        by Marcos Lopez de Prado in Chapter 5 of *Advances in Financial Machine Learning* (2018).

        Args:
            data (td.TensorDict): A dictionary of time series data.
            thres (float, optional): The threshold for the differencing. Defaults to 1e-3.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = data.to(self.device)
        self.thres = thres

        feat_ignore = ['volume', 'adx', 'adxr', 'apo', 'aroon', 'aroonosc', 'bop', 'cci', 'cmo', 'dx', 'macd', 'macdext', 'macdfix', 'mfi', 'minus_di', 'minus_dm', 'mom', 'plus_di', 'plus_dm', 'ppo', 'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'stoch', 'stochf', 'stochrsi', 'trix', 'ultosc', 'willr']
        
        self.len_data = len(data)
        self.feats = [f for f in data.keys() if f not in feat_ignore]
        self.data_tensor = torch.zeros((len(self.feats), self.len_data), device=self.device)
        for i, f in enumerate(self.feats):
            self.data_tensor[i] = data[f]

        self.widths = [0] * len(self.feats)
        self.weights = [None] * len(self.feats)

        self.d_opt = {f: 0.0 for f in self.feats} if d_opt is None else d_opt
        self.k = torch.arange(1, self.len_data, dtype=torch.float32, device=self.device)
        self.i = 0

    def _objective(self, d: float) -> float:
        # Calculate the weights
        factor = torch.zeros(self.len_data-1, device=self.device) - (d+1)
        proto = torch.cat([torch.tensor([1.0], device=self.device), torch.div(factor, self.k)+1])
        weights = torch.cumprod(proto, dim=0)

        # Calculate the width
        width = torch.where(weights.abs() > self.thres)[0].max()
        self.widths[self.i] = width

        # Truncate the weights
        weights = weights[:width].flip(0)
        self.weights[self.i] = weights

        # Calculate the differenced data
        data_seq = self.data_tensor[self.i].unsqueeze(0).unsqueeze(0)
        weights = weights.unsqueeze(0).unsqueeze(0)
        diff = torch.nn.functional.conv1d(data_seq, weights, padding=0).squeeze()
    
        # Calculate the p-value
        pval = adfuller(diff)[1]
        return pval

    def fit(self) -> None:
        for i, f in enumerate(self.feats):
            self.i = i
            low, high, pval = 0.0, 1.0, 1.0
            if self.d_opt[f] > 0.0:
                pval = self._objective(self.d_opt[f])
            else:
                while pval > 0.05 or pval < 0.049:
                    d_mid = (low + high) / 2
                    if d_mid < 1e-5:
                        self.d_opt[f] = 0.0
                        break
                    if abs(self.d_opt[f] - d_mid) < 1e-5:
                        break
                    self.d_opt[f] = d_mid
                    pval = self._objective(d_mid)
                    if pval > 0.05:
                        low = d_mid
                    else:
                        high = d_mid

    def transform(self) -> None:
        max_width = self.get_max_width()
        data = self.data[max_width:]
        for i, f in enumerate(self.feats):
            if self.d_opt[f] > 0:
                data_seq = self.data_tensor[i].unsqueeze(0).unsqueeze(0)
                w = self.weights[i].unsqueeze(0).unsqueeze(0)
                diff = torch.nn.functional.conv1d(data_seq, w, padding=0).squeeze()
                data[f] = diff[-(self.len_data-max_width):]
        self.data = data.to('cpu')
    
    def fit_transform(self) -> td.TensorDict:
        self.fit()
        self.transform()
        return self.data
    
    def get_max_width(self) -> int:
        return int(max(self.widths))
    
    @property
    def get_d_opt(self) -> dict:
        return self.d_opt