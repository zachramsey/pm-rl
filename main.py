from config.base import TRAINER, LOG_DIR, PLOT_DIR, ALGORITHM, SAVE_PATTERN

import pandas as pd
import sys, traceback
import shutil
from datetime import datetime as dt
from util.logger import TrainingLogger

if TRAINER == "On-Policy":
    from train.on_policy import Train
else:
    from train.off_policy import Train

if __name__ == "__main__":
    pd.set_option('future.no_silent_downcasting', True)
    try:
        logger = TrainingLogger(LOG_DIR)
        logger.log_config()
        trader = Train(logger)
        trader.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted... exiting.\n" + "="*50)
    except Exception as e:
        with open(LOG_DIR + "latest.log", "a") as f:
            f.write("\n" + "="*50 + f"\nError:\n")
            traceback.print_exc(file=f)
            f.write("\n" + "="*50 + "\n")
        traceback.print_exc(file=sys.stdout)
    finally:
        shutil.copyfile(PLOT_DIR + "latest.png", PLOT_DIR + SAVE_PATTERN.format(model=ALGORITHM, type="plot", date=dt.now()) + ".png")
        # shutil.copyfile(REPORT_DIR + "latest.html", REPORT_DIR + SAVE_PATTERN.format(model=ALGORITHM, type="report", date=dt.now()) + ".html")
