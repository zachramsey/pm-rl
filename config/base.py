
LOAD_LOCAL = True
SEED = 42
ALGORITHM = "PG"          # "DreamerV3"/"DSAC"/"PG"/"SAC"/"TD3"

# {model} = model name | {type} = log/plot/report | {date} = current date
SAVE_PATTERN = "{model}_{type}_{date:%Y-%m-%d_%H-%M-%S}"

# Directories
DATA_DIR = "_local/stock_data/"
PICKLE_DIR = "_local/pickles/"
MODEL_DIR = "_local/models/"
LOG_DIR = "_local/logs/"
PLOT_DIR = "_local/plots/"
REPORT_DIR = "_local/reports/"

# Logging
EVAL_FREQ = 1
PLOT_FREQ = 1
SAVE_FREQ = 25

# Data
TARGETS = "sp500"               # "sp500"/"sp100"/"djia"
CONTEXT = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
SCALER = "minmax"               # "standard"/"minmax"
MIN_VOLUME = 100000
TRAIN_RATIO = 0.8
WINDOW_SIZE = 32
NUM_ASSETS = 32
INDICATORS = {
    # "ema": {"timeperiod": 30}, 
    # "ema": {"timeperiod": 60}, 
    # "ema": {"timeperiod": 90}, 
    # "bbands": {"timeperiod": 20}, 
    # "macd": {}, 
    # "atr": {}, 
    # "rsi": {"timeperiod": 30}, 
    # "adx": {"timeperiod": 30}, 
    # "dx": {"timeperiod": 30}, 
    # "stoch": {}, 
    # "cci": {}, 
    # "obv": {}, 
    # "adosc": {}
}

# Environment
INITIAL_CASH = 25000
COMISSION = 0.0

# Reward
REWARD = "log_returns"  # "returns"/"log_returns"/"sharpe_ratio"
REWARD_SCALE = 1
RISK_FREE_RATE = 0.04

# Model-Specific Configurations
if ALGORITHM == "DreamerV3":
    TRAINER = "Off-Policy"
    from config.dreamer import *
elif ALGORITHM == "DSAC":
    TRAINER = "Off-Policy"
    from config.dsac import *
elif ALGORITHM == "PG":
    TRAINER = "On-Policy"
    from config.pg import *
elif ALGORITHM == "SAC":
    TRAINER = "Off-Policy"
    from config.sac import *
elif ALGORITHM == "TD3":
    TRAINER = "Off-Policy"
    from config.td3 import *
else:
    raise NotImplementedError(f"Model {ALGORITHM} not implemented")
