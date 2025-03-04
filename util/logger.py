import logging
import textwrap
import sys
import time
from datetime import datetime as dt
import shutil

from config.base import ALGORITHM, TARGETS, CONTEXT, INDICATORS, SCALER, WINDOW_SIZE, MIN_VOLUME, NUM_ASSETS, INITIAL_CASH, COMISSION, REWARD, REWARD_SCALE, RISK_FREE_RATE
from config.base import POLICY, BATCH_SIZE, NUM_EPOCHS, UPDATE_STEPS, EPSILON, LR
from config.base import DEPTH, NUM_LATENTS, LATENT_DIM, NUM_CROSS_HEADS, CROSS_HEAD_DIM, NUM_SELF_HEADS, SELF_HEAD_DIM, DROPOUT
from config.base import ROLLOUT_PRINT_FREQ, TRAIN_PRINT_FREQ, EVAL_PRINT_FREQ

class TrainingLogger:
    def __init__(self, log_dir: str, table_width: int = 80):
        self.log_dir = log_dir
        self.table_width = table_width

        self.save_file = f"{ALGORITHM}-{dt.now().strftime('%Y%m%d-%H%M%S')}.log"
        logging.basicConfig(filename=log_dir + 'latest.log', level=logging.INFO, format='%(message)s')

        self.train_len = None
        self.eval_len = None
        self.epoch_start_time = None
        self.rollout_rew_total = 0
        self.eval_rew_total = 0

    @staticmethod
    def is_done_wrapper(msg: str, operation, *args, **kwargs):
        print(f"{msg}...", end='\r')
        output = operation(*args, **kwargs)
        print(f"{msg}... Done!")
        return output
    
    @staticmethod
    def print_inline_every(iter, freq, term, msg):
        if iter % freq == 0 or iter == term - 1:
            if iter > 0:
                sys.stdout.write("\033[F\033[K")
            print(msg)

    @staticmethod
    def progress_logger(message: str, curr: int, final: int):
        sys.stdout.write(f"\r{message} | {curr}/{final} | {(curr/final):.2%}     ")
        sys.stdout.flush()
        if curr == final: print("")  # Move to the next line

    @staticmethod
    def _format_indicators(indicators: dict) -> list:
        indicator_list = []
        for name, params in indicators.items():
            indicator = name
            if len(params) > 0:
                indicator += '_' + '_'.join(f'{v}' for _, v in params.items())
            indicator_list.append(indicator)
        return indicator_list
    
    def format_banner(self, title: str):
        lines = []
        lines.append(f"+{'-' * (self.table_width - 2)}+")
        lines.append(f"| {title.center(self.table_width - 2)} |")
        lines.append(f"+{'-' * (self.table_width - 2)}+")
        return '\n'+'\n'.join(lines)+'\n'

    def format_table(self, data: dict, title: str = None):
        # Convert list values to comma-separated strings
        formatted_data = {k: ', '.join(map(str, v)) if isinstance(v, list) else str(v) for k, v in data.items()}
        
        # Determine the max key width
        max_key_length = max(len(str(k)) for k in formatted_data.keys())
        key_width = max_key_length + 2  # Adding padding
        value_width = self.table_width - key_width - 5  # Account for '|', spacing, and structure
        
        # Create the table
        table_lines = []
        border = f"+{'-' * key_width}+{'-' * (2 + value_width)}+"
        
        # Add title row if provided
        if title:
            table_lines.append(f"+{'-' * (key_width + 3 + value_width)}+")
            table_lines.append(f"| {title.center(key_width + (value_width + 1))} |")
            table_lines.append(border)
        
        for key, value in formatted_data.items():
            wrapped_value = textwrap.wrap(value, width=value_width)
            table_lines.append(f"| {key.ljust(key_width - 2)} | {wrapped_value[0].ljust(value_width)} |")
            for line in wrapped_value[1:]:
                table_lines.append(f"| {' ' * (key_width - 2)} | {line.ljust(value_width)} |")
        
        table_lines.append(border)
        return '\n'+'\n'.join(table_lines)+'\n'
    
    def log_config(self):
        has_indicators = INDICATORS is not None and len(INDICATORS) > 0
        base_title = "BASE CONFIGURATION"
        base_config = {
            "algorithm": ALGORITHM,
            "targets": TARGETS,
            "context": CONTEXT,
            "indicators": self._format_indicators(INDICATORS) if has_indicators else "None",
            "scaler": SCALER,
            "window_size": WINDOW_SIZE,
            "min_volume": MIN_VOLUME,
            "num_assets": NUM_ASSETS,
            "initial_cash": INITIAL_CASH,
            "comission": COMISSION,
            "reward": REWARD,
            "reward_scale": REWARD_SCALE,
            "risk_free_rate": RISK_FREE_RATE
        }

        train_title = "TRAINING CONFIGURATION"
        if ALGORITHM == "PG":
            train_config = {
                "policy": POLICY,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "update_steps": UPDATE_STEPS,
                "epsilon": EPSILON,
                "lr": LR
            }
            if POLICY == "LSRE-CANN":
                train_config.update({
                    "depth": DEPTH,
                    "num_latents": NUM_LATENTS,
                    "latent_dim": LATENT_DIM,
                    "num_cross_heads": NUM_CROSS_HEADS,
                    "cross_head_dim": CROSS_HEAD_DIM,
                    "num_self_heads": NUM_SELF_HEADS,
                    "self_head_dim": SELF_HEAD_DIM,
                    "dropout": DROPOUT
                })
        else:
            raise NotImplementedError(f"Algorithm {ALGORITHM} config logging not implemented")
        
        logging.info(self.format_table(base_config, base_title))
        logging.info(self.format_table(train_config, train_title))

    def epoch_start(self, epoch: int, train_len: int, eval_len: int):
        self.train_len = train_len
        self.eval_len = eval_len
        self.epoch_start_time = time.time()
        string = "-" * self.table_width + f"\nEpoch {epoch}:"
        print(string)
        logging.info(string+"\n")

    def log_rollout(self, step: int, datetime: str, reward: float, value: float):
        self.rollout_rew_total += reward
        string = (
            f"  Rollout"
            f" | Step: {step}"
            f" | Date: {datetime.strftime('%d/%m/%y')}"
            f" | Reward: {self.rollout_rew_total:.6f}"
            f" | Value: {value:.2f}"
        )
        self.print_inline_every(step, ROLLOUT_PRINT_FREQ, self.train_len, string)
        if step == self.train_len - 1:
            logging.info(string+"\n")

    def log_update(self, step: int, info: dict):
        loss = info["loss"][-1]
        string = (
            f"  Update"
            f" | Step: {step}"
            f" | Loss: {loss:.12f}"
        )
        self.print_inline_every(step, TRAIN_PRINT_FREQ, self.train_len // BATCH_SIZE, string)
        if step == self.train_len // BATCH_SIZE - 1:
            sys.stdout.write("\033[F\033[K")
            string += f" | Avg Loss: {sum(info['loss']) / len(info['loss']):.12f}"
            print(string)
            logging.info(string+"\n")

    def log_eval(self, step: int, datetime: str, reward: float, value: float):
        self.eval_rew_total += reward
        string = (
            f"  Evaluate"
            f" | Step: {step}"
            f" | Date: {datetime.strftime('%d/%m/%y')}"
            f" | Reward: {self.eval_rew_total:.6}"
            f" | Value: {value:.2f}"
        )
        self.print_inline_every(step, EVAL_PRINT_FREQ, self.eval_len, string)
        if step == self.eval_len - 1:
            logging.info(string+"\n")

    def epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        string = f"Epoch Time: {epoch_time:.2f}s"
        print(string)
        logging.info(string+"\n")
        self.rollout_rew_total = 0
        self.eval_rew_total = 0

    def close(self):
        logging.shutdown()
        shutil.copyfile(self.log_dir + 'latest.log', self.log_dir + self.save_file)
