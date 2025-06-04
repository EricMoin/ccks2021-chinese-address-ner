import os
import sys
from config import Config
from trainer import BiaffineSpanTrainer
from logger import logger
import torch
import random
import numpy as np


def set_seed(seed: int):
    """设置随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def check_gpu_memory():
    """检查GPU内存状态"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024 / 1024 / 1024  # GB
            logger.info(
                f"GPU {i}: {props.name}, Total Memory: {total_memory:.1f} GB")

            # 清理缓存
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
            cached = torch.cuda.memory_reserved(i) / 1024 / 1024
            logger.info(
                f"GPU {i} Memory - Allocated: {allocated:.1f} MB, Cached: {cached:.1f} MB")
    else:
        logger.info("CUDA is not available, using CPU")


def main():
    # 加载配置
    try:
        config = Config('config.yaml')
    except FileNotFoundError:
        logger.error("Configuration file 'config.yaml' not found")
        logger.info("Please make sure the config.yaml file exists.")
        return 1

    # 设置设备
    if config.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(0)
    else:
        device = 'cpu'
        if config.device == 'cuda':
            logger.warning("CUDA not available, falling back to CPU")

    logger.info(f"Using device: {device}")

    # 检查GPU状态
    if device.startswith('cuda'):
        check_gpu_memory()

    # 根据可用显存自动调整批次大小
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(
            0).total_memory / 1024 / 1024 / 1024
        if total_memory < 10:  # 小于10GB显存
            if config.batch_size > 8:
                logger.warning(
                    f"Reducing batch size from {config.batch_size} to 8 for memory optimization")
                config.batch_size = 8

    # 创建工作目录
    os.makedirs(config.work_dir, exist_ok=True)
    logger.info(f"Work directory: {config.work_dir}")

    # 设置随机种子
    set_seed(config.seed)

    # 记录重要配置信息

    try:
        # 开始Biaffine+Span模型训练
        trainer = BiaffineSpanTrainer(config, device)
        best_f1 = trainer.train()
        logger.info(f"Training completed successfully! Best F1: {best_f1:.4f}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    logger.info("All done! 🎉")
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
