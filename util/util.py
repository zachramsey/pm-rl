import sys

def print_inline_every(iter, freq, term, msg):
    if iter % freq == 0 or iter == term - 1:
        if iter > 0: sys.stdout.write("\033[F\033[K")
        print(msg)

from config.base import LOG_DIR, MODEL
from config.base import BATCH_SIZE, NUM_ASSETS, WINDOW_SIZE
from config.base import NORM, MIN_VOLUME, INDICATORS
from config.base import SEED_EPOCHS, UPDATE_STEPS
from config.base import BUFFER_SIZE, PERCENT_LATEST
from config.base import INITIAL_CASH, COMISSION
from config.base import REWARD, REWARD_SCALE, RISK_FREE_RATE
from datetime import datetime as dt

def write_config():
    with open(LOG_DIR + "latest.log", "w") as f:
        f.write("|" + "="*48 + "|\n")
        f.write("|                 CONFIGURATION                  |\n")
        f.write("|" + "="*48 + "|\n")
        f.write(f"| {'Model':<20}| {MODEL:<25}|\n")
        f.write(f"| {'Date':<20}| {dt.now().strftime("%b %d, %Y %I:%M %p"):<25}|\n")
        f.write("|" + "-"*48 + "|\n")
        f.write(f"| {'Batch Size':<20}| {BATCH_SIZE:<25}|\n")
        f.write(f"| {'Window Size':<20}| {WINDOW_SIZE:<25}|\n")
        f.write(f"| {'Number of Assets':<20}| {NUM_ASSETS:<25}|\n")
        f.write("|" + "-"*48 + "|\n")
        f.write(f"| {'Normalization':<20}| {NORM:<25}|\n")
        f.write(f"| {'Minimum Volume':<20}| {MIN_VOLUME:<25}|\n")
        # f.write(f"| {'Indicators':<20}| {INDICATORS:<25}|\n")
        f.write("|" + "-"*48 + "|\n")
        f.write(f"| {'Seed Epochs':<20}| {SEED_EPOCHS:<25}|\n")
        f.write(f"| {'Update Steps':<20}| {UPDATE_STEPS:<25}|\n")
        f.write("|" + "-"*48 + "|\n")
        f.write(f"| {'Buffer Size':<20}| {BUFFER_SIZE:<25}|\n")
        f.write(f"| {'Percent Latest':<20}| {PERCENT_LATEST:<25}|\n")
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