import os
import yaml
from datetime import datetime as dt
from train import TrainOffPolicy

if __name__ == "__main__":

    # Load config file
    cfg_file = "configs/base.yaml"
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    # Clear the latest log file
    with open(cfg["log_dir"] + "latest.log", "w") as f:
        f.write("")
    
    # Clear the latest plot file
    with open(cfg["plot_dir"] + "latest.png", "w") as f:
        f.write("")

    try:
        # Off-Policy Training
        trader = TrainOffPolicy(cfg)
        trader.train()
    except KeyboardInterrupt:
        # Save the latest log and plot files
        datetime_str = dt.now().strftime("%y-%m-%d_%H-%M-%S")
        os.rename(cfg["log_dir"] + "latest.log", cfg["log_dir"] + f"{datetime_str}.log")
        os.rename(cfg["plot_dir"] + "latest.png", cfg["plot_dir"] + f"{datetime_str}.png")
