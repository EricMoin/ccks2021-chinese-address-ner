import os
import numpy as np
import torch
import random
from config import Config
from trainer import KFoldTrainer, SingleTrainer  # Updated import
# For reading train/dev data from dataset import NERDataset # For creating datasets for training
# For creating DataLoaders for training

# Setup logging
from logger import logger


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logger.info(f"Seed set to {seed_value}")


def main():
    # Load main configuration
    # The Config class now loads from YAML within its __init__
    config = Config(config_path='config.yaml')

    set_seed(config.seed)
    os.makedirs(config.work_dir, exist_ok=True)  # Main work directory

    # trainer = KFoldTrainer(config=config)
    # trainer.kfold_train()
    trainer = SingleTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
