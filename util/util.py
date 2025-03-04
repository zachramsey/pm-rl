import sys
import torch
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

def get_index_tickers(index: str, ignore: list[str] = []) -> dict[str, dict[str,]]:
    '''Return the tickers for the given index'''
    if index == "sp500": 
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    elif index == "sp100": 
        url = 'https://en.wikipedia.org/wiki/S%26P_100'
    elif index == "djia": 
        url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    else: 
        raise NotImplementedError(f'Tickers for {index} not yet implemented')
    
    resp = requests.get(url)
    resp.raise_for_status()
    table = BeautifulSoup(resp.text, 'html.parser').find('table', {'id': 'constituents'})
    data = pd.read_html(StringIO(str(table)))[0].to_dict('records')
    
    tickers = {}
    for company in data:
        if company['Symbol'] not in ignore:
            sector = company['GICS Sector']
            industry = company['GICS Sub-Industry']
            tickers[company['Symbol'].replace('.', '-')] = {'sector': sector, 'industry': industry}
    return tickers


def actual_portfolio_selection(final_scores, num_assets, prop_winners=1):
    if prop_winners == 1:
        portfolio = torch.nn.functional.softmax(final_scores, dim=-1)
    else:
        num_selected = int(num_assets * prop_winners)
        assert num_selected > 0
        assert num_selected <= num_assets

        values, indices = torch.topk(final_scores, k=num_selected)
        winners_mask = torch.ones_like(final_scores, device=final_scores.device)
        winners_mask.scatter_(1, indices, 0).detach()
        portfolio = torch.nn.functional.softmax(final_scores - 1e9 * winners_mask, dim=-1)

    assert not torch.isnan(portfolio).any(), f"Portfolio contains NaN values: {portfolio}"
    assert torch.all(portfolio >= 0), f"Portfolio contains non-negative values: {portfolio}"
    return portfolio
