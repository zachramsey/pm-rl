import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
from config.base import PICKLE_DIR, DEBUG

class Tickers:
    def __init__(self, tickers: str | list[str] = None):
        self.data = {}

        if DEBUG and os.path.exists(PICKLE_DIR + "tickers.pkl"):
            self.load()
        else:
            print("Fetching Tickers...", end='\r')
            if isinstance(tickers, list):
                raise NotImplementedError('List of tickers not yet implemented')
            elif isinstance(tickers, str):
                if tickers == "sp500":
                    self._fetch_sp500()
                    self.remove(['HUBB'])
                else:
                    raise NotImplementedError(f'Tickers for {tickers} not yet implemented')
            else:
                raise TypeError(f'Tickers must be a string or list of strings, got {type(tickers)}')
            print("Fetching Tickers... Done!")

    def __call__(self) -> list[str]:
        '''Return the list of stock tickers'''
        return list(self.data.keys())

    def __len__(self) -> int:
        '''Return the number of stock tickers'''
        return len(self.data)
    
    def __getitem__(self, idx: int) -> str:
        '''Return the ticker at the given index'''
        return list(self.data.keys())[idx]
    
    def __repr__(self) -> str:
        '''Return a string representation of the stock tickers'''
        return f'Tickers:\n' + '\n'.join([f'{ticker}: {self.sectors[sector]}, {self.industries[industry]}' for ticker, (sector, industry) in self.data.items()])
    
    def __iter__(self) -> iter:
        '''Return an iterator for the stock tickers'''
        return iter(self.data.keys())

    def _fetch_sp500(self) -> None:
        '''Fetch HTML content from the given URL'''
        response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        data = pd.read_html(StringIO(str(table)))[0].to_dict('records')
        for company in data:
            ticker = company['Symbol'].replace('.', '-')
            sector = company['GICS Sector']
            industry = company['GICS Sub-Industry']
            self.data[ticker] = (sector, industry)

    def store(self) -> None:
        '''Store the tickers in a pickle file'''
        print("Saving Pickled Tickers...", end='\r')
        pd.to_pickle(self.data, PICKLE_DIR + "tickers.pkl")
        print(f"Saving Pickled Tickers... Done!")

    def load(self) -> None:
        '''Load the tickers from a pickle file'''
        print("Loading Pickled Tickers...", end='\r')
        self.data = pd.read_pickle(PICKLE_DIR + "tickers.pkl")
        print(f"Loading Pickled Tickers... Done!")

    def filter(self, tickers: str | list[str]) -> None:
        '''Update the ticker data to only include the given tickers'''
        if isinstance(tickers, str):
            tickers = [tickers]
        self.data = {ticker: self.data[ticker] for ticker in tickers}

    def remove(self, tickers: str | list[str]) -> None:
        '''Remove the tickers that are in the given list'''
        if isinstance(tickers, str):
            tickers = [tickers]
        self.data = {ticker: self.data[ticker] for ticker in self.data if ticker not in tickers}

