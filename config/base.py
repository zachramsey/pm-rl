
DEBUG = True
SEED = 42

# {model} = model name | {type} = log/plot/report | {date} = current date
SAVE_PATTERN = "{model}_{type}_{date:%Y-%m-%d_%H-%M-%S}"

# Directories
DATA_DIR = "data/stock_data/"
PICKLE_DIR = "data/pickles/"
MODEL_DIR = "data/models/"
LOG_DIR = "data/logs/"
PLOT_DIR = "data/plots/"
REPORT_DIR = "data/reports/"

# Logging
EVAL_FREQ = 1
PLOT_FREQ = 1
SAVE_FREQ = 25

# Console
COLLECT_PRINT_FREQ = 250
TRAIN_PRINT_FREQ = 1
EVAL_PRINT_FREQ = 125

# Data
TICKERS = "sp500"       # "sp500"
NORM = "diff"           # "diff"/"traj"
MIN_VOLUME = 100000
TRAIN_RATIO = 0.8
BATCH_SIZE = 128
WINDOW_SIZE = 50
NUM_ASSETS = 32
INDICATORS = False

# Training
MODEL = "DSAC"          # "DSAC"/"LSRE-CANN"/"TD3"/"DreamerV3"
NUM_EPOCHS = 1000
SEED_EPOCHS = 0
UPDATE_STEPS = 50

# Environment
INITIAL_CASH = 25000
COMISSION = 0.0

# Reward
REWARD = "log_returns"  # "returns"/"log_returns"/"sharpe_ratio"
REWARD_SCALE = 1
RISK_FREE_RATE = 0.04

# Experience Replay Buffer
BUFFER_SIZE = 425000
BUFFER_MIN_SIZE = 1000
PERCENT_LATEST = 0.125