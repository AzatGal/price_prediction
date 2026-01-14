import os
import json
import logging
import matplotlib.pyplot as plt
# from typing import Dict, Any, Optional

from easydict import EasyDict

WHITE = '\033[97m'
RESET = '\033[0m'


def to_json(obj):
    if hasattr(obj, 'json'):
        return obj.json
    return obj

class Logger:
    def __init__(self, cfg: EasyDict) -> None:
        self.log_dir = cfg.exp_dir
        os.makedirs(self.log_dir, exist_ok=True)

        log_format = f"{WHITE}%(message)s{RESET}"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.save_config(cfg)
        self.num_epoch = cfg.num_epoch

        self.metrics_file = os.path.join(self.log_dir, "metrics.jsonl")

    def save_config(self, cfg: EasyDict) -> None:
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False, default=to_json)

    def print(self, msg: str) -> None:
        self.logger.info(msg)

    def log_metrics(self, epoch: int, train: dict, valid: dict) -> None:
        self.print(f"Epoch {epoch}/{self.num_epoch}")
        record = {
            "epoch": epoch,
            **{'train_' + k: v for k, v in train.items()},
            **{'valid_' + k: v for k, v in valid.items()},
        }
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.print(
            "Train | " + " | ".join(f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in train.items())
        )
        self.print(
            "Valid | " + " | ".join(f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in valid.items())
        )

    def plot(self, name_values: str, show: bool = False) -> None:
        steps = []
        metric_dict = {}

        with open(self.metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                steps.append(record.pop("epoch"))
                for key, value in record.items():
                    if name_values in key:
                        if key not in metric_dict.keys():
                            metric_dict[key] = []
                        metric_dict[key].append(value)

        plt.figure(figsize=(10, 6))
        for key, values in metric_dict.items():
            plt.plot(steps, values, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(name_values)
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.log_dir, f"{name_values}.png")
        plt.savefig(plot_path)

        if show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    config = EasyDict({
        'exp_dir': "runs/exp_01",
        "model": "ResNet18",
        "num_epoch": 50,
        "lr": 0.001,
        "batch_size": 32
    })

    logger = Logger(config)

    for epoch in range(1, 6):
        # Simulate training
        train_loss = 0.5 / epoch
        val_acc = 0.8 + 0.1 * epoch / 5

        logger.log_metrics({
            "train_loss": train_loss,
            "val_acc": val_acc,

        }, epoch)

    logger.plot('loss')
