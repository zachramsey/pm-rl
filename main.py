from config.base import LOG_DIR, PLOT_DIR, REPORT_DIR, MODEL, SAVE_PATTERN

import shutil
from util.util import write_config
from datetime import datetime as dt
from train import TrainOffPolicy

if __name__ == "__main__":
    try:
        write_config()
        trader = TrainOffPolicy()
        trader.train()
    except Exception as e:
        with open(LOG_DIR + "latest.log", "a") as f:
            f.write("\n" + "="*50 + f"\nError: {e}\n" + "="*50 + "\n")
        raise e
    finally:
        shutil.copyfile(LOG_DIR + "latest.log", LOG_DIR + SAVE_PATTERN.format(model=MODEL, type="log", date=dt.now()) + ".log")
        shutil.copyfile(PLOT_DIR + "latest.png", PLOT_DIR + SAVE_PATTERN.format(model=MODEL, type="plot", date=dt.now()) + ".png")
        # shutil.copyfile(REPORT_DIR + "latest.html", REPORT_DIR + SAVE_PATTERN.format(model=MODEL, type="report", date=dt.now()) + ".html")
