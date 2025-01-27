from config.base import LOG_DIR, PLOT_DIR, MODEL, SAVE_PATTERN

import os
from datetime import datetime as dt
from train import TrainOffPolicy

if __name__ == "__main__":
    # Clear the latest log file
    with open(LOG_DIR + "latest.log", "w") as f:
        f.write("")
    
    # Clear the latest plot file
    with open(PLOT_DIR + "latest.png", "w") as f:
        f.write("")

    try:
        # Off-Policy Training
        trader = TrainOffPolicy()
        trader.train()
    except KeyboardInterrupt:
        # Save the latest log and plot files
        os.rename(LOG_DIR + "latest.log", LOG_DIR + SAVE_PATTERN.format(model=MODEL, type="log", date=dt.now()) + ".log")
        os.rename(PLOT_DIR + "latest.png", PLOT_DIR + SAVE_PATTERN.format(model=MODEL, type="plot", date=dt.now()) + ".png")
