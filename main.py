import os
import numpy as np
import torch
import random
from config import Config
from trainer import KFoldTrainer, SingleTrainer  # 更新的导入
# 用于读取训练/验证数据，从dataset导入NERDataset用于创建训练数据集
# 用于创建训练的DataLoaders

# 设置日志
from logger import logger


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logger.info(f"Seed set to {seed_value}")


def main():
    # 加载主配置
    # Config类现在在其__init__方法中从YAML加载配置
    config = Config(config_path='config.yaml')

    set_seed(config.seed)
    os.makedirs(config.work_dir, exist_ok=True)  # 主工作目录

    # trainer = KFoldTrainer(config=config)
    # trainer.kfold_train()
    trainer = SingleTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
