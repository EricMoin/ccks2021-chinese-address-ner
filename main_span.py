#!/usr/bin/env python3
"""
Main training script for span-based and biaffine NER models
"""

import os
import sys
import torch
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import Config
from span_biaffine_model import SpanBiaffineNER, HybridNER
from span_converter import SpanDataset, SpanConverter, span_collate_fn
from span_trainer import SpanTrainer, HybridTrainer, SpanKFoldTrainer
from conll_reader import ConllReader
from logger import logger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config: Config):
    """Create model based on configuration"""
    if config.model_type == 'span':
        # Pure span-based model
        # Use entity types from label map
        num_labels = len(config.label_map.entity_types)  # Includes "O"
        model = SpanBiaffineNER(num_labels, config)
        logger.info(
            f"Created SpanBiaffineNER model with {num_labels} span labels")
    elif config.model_type == 'hybrid':
        # Hybrid model (sequence + span)
        sequence_num_labels = len(
            config.label_map.labels)  # Full BIOES label set
        model = HybridNER(sequence_num_labels, config)
        logger.info(
            f"Created HybridNER model with {sequence_num_labels} sequence labels")
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    return model


def create_datasets(config: Config):
    """Create training and validation datasets"""
    conll_reader = ConllReader()

    # Load data
    train_data = list(conll_reader.read(config.train_file))

    val_data = list(conll_reader.read(config.dev_file))

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Create datasets
    train_dataset = SpanDataset(train_data, tokenizer, config.label_map)
    val_dataset = SpanDataset(val_data, tokenizer, config.label_map)

    logger.info(
        f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config: Config):
    """Create data loaders"""
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        collate_fn=span_collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=span_collate_fn
    )

    return train_dataloader, val_dataloader


def single_train(config: Config):
    """Train a single model"""
    logger.info("Starting single model training")

    # Set seed
    set_seed(config.seed)

    # Create datasets and dataloaders
    train_dataset, val_dataset = create_datasets(config)
    train_dataloader, val_dataloader = create_dataloaders(
        train_dataset, val_dataset, config)

    # Create model
    model = create_model(config)

    # Create trainer
    if config.model_type == 'span':
        trainer = SpanTrainer(config, model, train_dataloader,
                              val_dataloader, config.device)
    elif config.model_type == 'hybrid':
        trainer = HybridTrainer(
            config, model, train_dataloader, val_dataloader, config.device)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    # Train
    best_f1 = trainer.train()

    logger.info(f"Single training completed. Best F1: {best_f1:.4f}")
    return best_f1


def kfold_train(config: Config):
    """Train with K-fold cross-validation"""
    logger.info("Starting K-fold cross-validation training")

    # Set seed
    set_seed(config.seed)

    # Create K-fold trainer
    kfold_trainer = SpanKFoldTrainer(config, model_type=config.model_type)

    # Train
    results = kfold_trainer.kfold_train()

    logger.info(f"K-fold training completed.")
    logger.info(
        f"Average F1: {results['average_f1']:.4f} ± {results['std_f1']:.4f}")
    logger.info(f"Individual fold results: {results['fold_results']}")

    return results


def main():
    """Main function"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train span-based NER models')
    parser.add_argument('--config', type=str, default='config_span.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['single', 'kfold'], default='single',
                        help='Training mode: single or k-fold cross-validation')
    parser.add_argument('--model_type', type=str, choices=['span', 'hybrid'], default=None,
                        help='Model type (overrides config file)')

    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Override model type if specified
    if args.model_type:
        config.model_type = args.model_type
        logger.info(f"Model type overridden to: {config.model_type}")

    # Log configuration
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Model type: {config.model_type}")
    logger.info(f"Training mode: {args.mode}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Work directory: {config.work_dir}")

    # Create work directory
    os.makedirs(config.work_dir, exist_ok=True)

    # Train based on mode
    if args.mode == 'single':
        results = single_train(config)
    elif args.mode == 'kfold':
        results = kfold_train(config)
    else:
        raise ValueError(f"Unknown training mode: {args.mode}")

    logger.info("Training completed successfully!")
    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
