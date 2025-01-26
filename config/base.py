
DEBUG = False

# {model} = model name | {type} = model/log/plot | {date} = current date
SAVE_PATTERN = "{model}_{type}_{date:%Y-%m-%d_%H-%M-%S}"

# Directories
DATA_DIR = "data/stock_data/"
PICKLE_DIR = "data/pickles/"
MODEL_DIR = "data/models/"
LOG_DIR = "data/logs/"
PLOT_DIR = "data/plots/"

# Data
MIN_VOLUME = 100000
TRAIN_RATIO = 0.8
NUM_ASSETS = 128
NUM_INVESTED = 128
WINDOW_SIZE = 50

# Environment
INITIAL_CASH = 25000
SELL_PENALTY = 0.0025
BUY_PENALTY = 0.0025

# Experience Replay Buffer
BUFFER_SIZE = 100000
BUFFER_MIN_SIZE = 1000
PERCENT_LATEST = 0.5