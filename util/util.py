import sys
import torch
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

def get_index_tickers(index: str) -> list[str]:
    '''Return the tickers for the given index'''
    if tickers == "sp500": 
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    elif tickers == "sp100": 
        url = 'https://en.wikipedia.org/wiki/S%26P_100'
    elif tickers == "djia": 
        url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    else: 
        raise NotImplementedError(f'Tickers for {tickers} not yet implemented')
    
    resp = requests.get(url)
    resp.raise_for_status()
    table = BeautifulSoup(resp.text, 'html.parser').find('table', {'id': 'constituents'})
    data = pd.read_html(StringIO(str(table)))[0].to_dict('records')
    
    tickers = [company['Symbol'].replace('.', '-') for company in data]
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

def print_inline_every(iter, freq, term, msg):
    if iter % freq == 0 or iter == term - 1:
        if iter > 0: sys.stdout.write("\033[F\033[K")
        print(msg)

from config.base import LOG_DIR, ALGORITHM
from config.base import BATCH_SIZE, NUM_ASSETS, WINDOW_SIZE
from config.base import NORM, MIN_VOLUME, INDICATORS
from config.base import SEED_EPOCHS, UPDATE_STEPS
from config.base import INITIAL_CASH, COMISSION
from config.base import REWARD, REWARD_SCALE, RISK_FREE_RATE
from datetime import datetime as dt

def write_config():
    with open(LOG_DIR + "latest.log", "w") as f:
        f.write("|" + "="*48 + "|\n")
        f.write("|                 CONFIGURATION                  |\n")
        f.write("|" + "="*48 + "|\n")
        f.write(f"| {'Model':<20}| {ALGORITHM:<25}|\n")
        f.write(f"| {'Date':<20}| {dt.now().strftime("%b %d, %Y %I:%M %p"):<25}|\n")
        f.write("|" + "-"*48 + "|\n")
        f.write(f"| {'Batch Size':<20}| {BATCH_SIZE:<25}|\n")
        f.write(f"| {'Window Size':<20}| {WINDOW_SIZE:<25}|\n")
        f.write(f"| {'Number of Assets':<20}| {NUM_ASSETS:<25}|\n")
        f.write("|" + "-"*48 + "|\n")
        f.write(f"| {'Normalization':<20}| {NORM[0] + ' / ' + NORM[1]:<25}|\n")
        f.write(f"| {'Minimum Volume':<20}| {MIN_VOLUME:<25}|\n")
        # f.write(f"| {'Indicators':<20}| {INDICATORS:<25}|\n")
        f.write("|" + "-"*48 + "|\n")
        f.write(f"| {'Seed Epochs':<20}| {SEED_EPOCHS:<25}|\n")
        f.write(f"| {'Update Steps':<20}| {UPDATE_STEPS:<25}|\n")
        f.write("|" + "-"*48 + "|\n")
        f.write(f"| {'Initial Cash':<20}| {INITIAL_CASH:<25}|\n")
        f.write(f"| {'Comission':<20}| {COMISSION:<25}|\n")
        f.write("|" + "-"*48 + "|\n")
        f.write(f"| {'Reward':<20}| {REWARD:<25}|\n")
        f.write(f"| {'Reward Scale':<20}| {REWARD_SCALE:<25}|\n")
        f.write(f"| {'Risk-Free Rate':<20}| {RISK_FREE_RATE:<25}|\n")
        f.write("|" + "="*48 + "|\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("|" + "="*48 + "|\n")
        f.write("|                 TRAINING INFO                  |\n")
        f.write("|" + "="*48 + "|\n")
        f.write("\n")