
DEBUG = True
MODEL = "DSAC"  # "DSAC"/"LSRE-CANN"/"TD3"/"DreamerV3"

# {model} = model name | {type} = log/plot | {date} = current date
SAVE_PATTERN = "{model}_{type}_{date:%Y-%m-%d_%H-%M-%S}"

# Directories
DATA_DIR = "data/stock_data/"
PICKLE_DIR = "data/pickles/"
MODEL_DIR = "data/models/"
LOG_DIR = "data/logs/"
PLOT_DIR = "data/plots/"

# Logging
EVAL_FREQ = 1
PLOT_FREQ = 1
SAVE_FREQ = 25

# Console
COLLECT_PRINT_FREQ = 500
TRAIN_PRINT_FREQ = 1
EVAL_PRINT_FREQ = 200

# Data
MIN_VOLUME = 100000
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
WINDOW_SIZE = 50

# Dimensions
NUM_ASSETS = 128
FEAT_DIM = None
TRAIN_LEN = None
EVAL_LEN = None

# Training
NUM_EPOCHS = 500
UPDATE_STEPS = 50
NUM_INVESTED = 128

# Environment
INITIAL_CASH = 25000
SELL_PENALTY = 0.0025
BUY_PENALTY = 0.0025

# Experience Replay Buffer
BUFFER_SIZE = 100000
BUFFER_MIN_SIZE = 1000
PERCENT_LATEST = 0.5