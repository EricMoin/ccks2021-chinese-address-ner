import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import copy
import logging
from sklearn.metrics import classification_report
from typing import Dict, List, Optional

from config import Config
from span_biaffine_model import SpanBiaffineNER, HybridNER, SpatialDropout
from span_converter import SpanConverter, SpanDataset, SpanMetrics
from model import FreeLB
from conll_reader import ConllReader
from label import LabelMap

logger = logging.getLogger(__name__)


class SpanTrainer:
    """
    Trainer for span-based NER models
    """

    def __init__(self, config: Config, model: nn.Module, train_dataloader: DataLoader,
                 val_dataloader: DataLoader, device: str):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(device)
        self.scheduler = None

        # Initialize span converter and metrics
        self.span_converter = SpanConverter(config.label_map)
        self.span_metrics = SpanMetrics(config.label_map)

        # Move model to device
        self.model.to(self.device)

        logger.info(
            f"SpanTrainer initialized. Training on {self.device}. Work directory: {self.config.work_dir}")

    def train(self):
        """Train the span-based model"""
        self.model.train()

        # Optimizer with different learning rates for different components
        optimizer = torch.optim.AdamW([
            {'params': self.model.bert.embeddings.parameters(
            ), 'lr': self.config.learning_rate * 5},
            {'params': self.model.lstm.parameters(
            ), 'lr': self.config.learning_rate * 25},
            {'params': self.model.span_classifier.parameters(
            ), 'lr': self.config.learning_rate * 25}
        ], lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        # Learning rate scheduler
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )

        best_val_f1 = 0
        os.makedirs(self.config.work_dir, exist_ok=True)

        # Initialize FreeLB if enabled
        freelb = None
        if hasattr(self.config, 'use_freelb') and self.config.use_freelb:
            freelb = FreeLB(
                self.model,
                adv_lr=self.config.freelb_adv_lr,
                adv_steps=self.config.freelb_adv_steps,
                adv_init_mag=self.config.freelb_adv_init_mag,
                adv_max_norm=self.config.freelb_adv_max_norm,
                adv_norm_type=self.config.freelb_adv_norm_type,
                base_model=self.config.freelb_base_model
            )
            logger.info(
                "FreeLB adversarial training configured for span model.")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train] ({os.path.basename(self.config.work_dir)})"
            )

            for batch in train_pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                span_labels = batch["span_labels"].to(self.device)

                optimizer.zero_grad()

                if freelb and epoch >= getattr(self.config, 'adversarial_training_start_epoch', 0):
                    # FreeLB adversarial training
                    original_embeddings = self.model.bert.embeddings.word_embeddings(
                        input_ids)
                    adv_loss = freelb.attack(
                        original_embeddings.detach(), attention_mask, span_labels)
                    current_loss = adv_loss
                else:
                    # Standard training
                    loss = self.model(input_ids, attention_mask,
                                      span_labels=span_labels)
                    loss.backward()
                    current_loss = loss.item()

                # Gradient clipping and optimizer step
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                train_loss += current_loss
                train_pbar.set_postfix({"loss": f"{current_loss:.4f}"})

            avg_train_loss = train_loss / len(self.train_dataloader)
            logger.info(
                f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")

            # Evaluation
            val_metrics = self.evaluate()
            val_f1 = val_metrics['f1']

            logger.info(f"Epoch {epoch+1} validation F1: {val_f1:.4f}")

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_path = os.path.join(
                    self.config.work_dir, 'best_span_model.pt')
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"New best model saved with F1: {best_val_f1:.4f}")

        logger.info(
            f"Training completed. Best validation F1: {best_val_f1:.4f}")
        return best_val_f1

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the span-based model"""
        self.model.eval()
        all_true_spans = []
        all_pred_spans = []
        eval_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                span_labels = batch["span_labels"].to(self.device)
                tokens = batch["tokens"]
                original_labels = batch["original_labels"]

                # Forward pass
                span_scores = self.model(input_ids, attention_mask)

                # Compute loss
                loss = self.model.compute_span_loss(
                    span_scores, span_labels, attention_mask)
                eval_loss += loss.item()

                # Decode predictions
                pred_spans = self.span_converter.decode_span_labels(
                    span_scores, attention_mask)

                # Convert ground truth to spans
                true_spans = []
                for i, (token_list, label_list) in enumerate(zip(tokens, original_labels)):
                    spans = self.span_converter.bioes_to_spans(
                        token_list, label_list)
                    true_spans.append(spans)

                all_true_spans.extend(true_spans)
                all_pred_spans.extend(pred_spans)

        # Compute metrics
        metrics = self.span_metrics.compute_span_f1(
            all_true_spans, all_pred_spans)
        entity_metrics = self.span_metrics.compute_entity_type_f1(
            all_true_spans, all_pred_spans)

        avg_eval_loss = eval_loss / len(self.val_dataloader)
        metrics['loss'] = avg_eval_loss

        # Log entity-specific metrics
        for entity_type, entity_metric in entity_metrics.items():
            logger.info(f"Entity {entity_type}: P={entity_metric['precision']:.4f}, "
                        f"R={entity_metric['recall']:.4f}, F1={entity_metric['f1']:.4f}")

        return metrics


class HybridTrainer:
    """
    Trainer for hybrid models (sequence + span)
    """

    def __init__(self, config: Config, model: nn.Module, train_dataloader: DataLoader,
                 val_dataloader: DataLoader, device: str):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(device)
        self.scheduler = None

        # Initialize span converter and metrics
        self.span_converter = SpanConverter(config.label_map)
        self.span_metrics = SpanMetrics(config.label_map)

        # Move model to device
        self.model.to(self.device)

        logger.info(
            f"HybridTrainer initialized. Training on {self.device}. Work directory: {self.config.work_dir}")

    def train(self):
        """Train the hybrid model"""
        self.model.train()

        # Optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': self.model.bert.embeddings.parameters(
            ), 'lr': self.config.learning_rate * 5},
            {'params': self.model.lstm.parameters(
            ), 'lr': self.config.learning_rate * 25},
            {'params': self.model.sequence_classifier.parameters(
            ), 'lr': self.config.learning_rate * 25},
            {'params': self.model.crf.parameters(
            ), 'lr': self.config.learning_rate * 50},
            {'params': self.model.span_classifier.parameters(
            ), 'lr': self.config.learning_rate * 25}
        ], lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        # Learning rate scheduler
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )

        best_val_f1 = 0
        os.makedirs(self.config.work_dir, exist_ok=True)

        # Initialize FreeLB if enabled
        freelb = None
        if hasattr(self.config, 'use_freelb') and self.config.use_freelb:
            freelb = FreeLB(
                self.model,
                adv_lr=self.config.freelb_adv_lr,
                adv_steps=self.config.freelb_adv_steps,
                adv_init_mag=self.config.freelb_adv_init_mag,
                adv_max_norm=self.config.freelb_adv_max_norm,
                adv_norm_type=self.config.freelb_adv_norm_type,
                base_model=self.config.freelb_base_model
            )
            logger.info(
                "FreeLB adversarial training configured for hybrid model.")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train] ({os.path.basename(self.config.work_dir)})"
            )

            for batch in train_pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sequence_labels = batch["sequence_labels"].to(self.device)
                span_labels = batch["span_labels"].to(self.device)

                optimizer.zero_grad()

                if freelb and epoch >= getattr(self.config, 'adversarial_training_start_epoch', 0):
                    # FreeLB adversarial training
                    original_embeddings = self.model.bert.embeddings.word_embeddings(
                        input_ids)
                    # For hybrid model, we need to pass both types of labels
                    adv_loss = freelb.attack(original_embeddings.detach(), attention_mask,
                                             {'sequence_labels': sequence_labels, 'span_labels': span_labels})
                    current_loss = adv_loss
                else:
                    # Standard training
                    loss = self.model(input_ids, attention_mask,
                                      sequence_labels=sequence_labels, span_labels=span_labels)
                    loss.backward()
                    current_loss = loss.item()

                # Gradient clipping and optimizer step
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                train_loss += current_loss
                train_pbar.set_postfix({"loss": f"{current_loss:.4f}"})

            avg_train_loss = train_loss / len(self.train_dataloader)
            logger.info(
                f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")

            # Evaluation
            val_metrics = self.evaluate()
            val_f1 = val_metrics['combined_f1']

            logger.info(
                f"Epoch {epoch+1} validation combined F1: {val_f1:.4f}")

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_path = os.path.join(
                    self.config.work_dir, 'best_hybrid_model.pt')
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(
                    f"New best hybrid model saved with F1: {best_val_f1:.4f}")

        logger.info(
            f"Training completed. Best validation F1: {best_val_f1:.4f}")
        return best_val_f1

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the hybrid model"""
        self.model.eval()
        all_true_spans = []
        all_pred_spans_sequence = []
        all_pred_spans_span = []
        all_pred_spans_combined = []
        all_tokens = []
        eval_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sequence_labels = batch["sequence_labels"].to(self.device)
                span_labels = batch["span_labels"].to(self.device)
                tokens = batch["tokens"]
                original_labels = batch["original_labels"]

                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                sequence_predictions = outputs['sequence_predictions']
                span_scores = outputs['span_scores']

                # Compute loss (approximate)
                loss = self.model(input_ids, attention_mask,
                                  sequence_labels=sequence_labels, span_labels=span_labels)
                eval_loss += loss.item()

                # Decode span predictions
                pred_spans_span = self.span_converter.decode_span_labels(
                    span_scores, attention_mask)

                # Convert sequence predictions to spans
                pred_spans_sequence = []
                for i, (seq_pred, token_list) in enumerate(zip(sequence_predictions, tokens)):
                    # Convert sequence prediction indices to labels
                    seq_labels = [self.config.label_map.id2label[idx]
                                  for idx in seq_pred[:len(token_list)]]
                    spans = self.span_converter.bioes_to_spans(
                        token_list, seq_labels)
                    pred_spans_sequence.append(spans)

                # Combine predictions
                pred_spans_combined = self.span_converter.combine_sequence_and_span_predictions(
                    [[self.config.label_map.id2label[idx] for idx in seq_pred[:len(token_list)]]
                     for seq_pred, token_list in zip(sequence_predictions, tokens)],
                    pred_spans_span,
                    tokens,
                    sequence_weight=getattr(
                        self.config, 'sequence_loss_weight', 0.5),
                    span_weight=getattr(self.config, 'span_loss_weight', 0.5)
                )

                # Convert combined predictions to spans
                combined_spans = []
                for combined_pred, token_list in zip(pred_spans_combined, tokens):
                    spans = self.span_converter.bioes_to_spans(
                        token_list, combined_pred)
                    combined_spans.append(spans)

                # Convert ground truth to spans
                true_spans = []
                for i, (token_list, label_list) in enumerate(zip(tokens, original_labels)):
                    spans = self.span_converter.bioes_to_spans(
                        token_list, label_list)
                    true_spans.append(spans)

                all_true_spans.extend(true_spans)
                all_pred_spans_sequence.extend(pred_spans_sequence)
                all_pred_spans_span.extend(pred_spans_span)
                all_pred_spans_combined.extend(combined_spans)
                all_tokens.extend(tokens)

        # Compute metrics for all approaches
        sequence_metrics = self.span_metrics.compute_span_f1(
            all_true_spans, all_pred_spans_sequence)
        span_metrics = self.span_metrics.compute_span_f1(
            all_true_spans, all_pred_spans_span)
        combined_metrics = self.span_metrics.compute_span_f1(
            all_true_spans, all_pred_spans_combined)

        avg_eval_loss = eval_loss / len(self.val_dataloader)

        # Log all metrics
        logger.info(f"Sequence F1: {sequence_metrics['f1']:.4f}")
        logger.info(f"Span F1: {span_metrics['f1']:.4f}")
        logger.info(f"Combined F1: {combined_metrics['f1']:.4f}")

        return {
            'loss': avg_eval_loss,
            'sequence_f1': sequence_metrics['f1'],
            'span_f1': span_metrics['f1'],
            'combined_f1': combined_metrics['f1'],
            'sequence_precision': sequence_metrics['precision'],
            'sequence_recall': sequence_metrics['recall'],
            'span_precision': span_metrics['precision'],
            'span_recall': span_metrics['recall'],
            'combined_precision': combined_metrics['precision'],
            'combined_recall': combined_metrics['recall']
        }


class SpanKFoldTrainer:
    """
    K-Fold cross-validation trainer for span-based models
    """

    def __init__(self, config: Config, model_type: str = 'span'):
        self.config = config
        self.model_type = model_type  # 'span' or 'hybrid'

    def kfold_train(self):
        """Perform K-fold cross-validation training"""
        from sklearn.model_selection import KFold

        # Load data
        reader = ConllReader(self.config.train_file)
        all_data = reader.read()

        kfold = KFold(n_splits=self.config.k_folds, shuffle=True,
                      random_state=self.config.seed)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_data)):
            logger.info(f"Starting fold {fold + 1}/{self.config.k_folds}")

            # Split data
            train_data = [all_data[i] for i in train_idx]
            val_data = [all_data[i] for i in val_idx]

            # Create datasets
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            train_dataset = SpanDataset(
                train_data, tokenizer, self.config.label_map)
            val_dataset = SpanDataset(
                val_data, tokenizer, self.config.label_map)

            # Create data loaders
            from span_converter import span_collate_fn
            train_dataloader = DataLoader(
                train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=span_collate_fn)
            val_dataloader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=span_collate_fn)

            # Create model
            if self.model_type == 'span':
                # For span model, use entity types from label map
                # Includes "O"
                num_labels = len(self.config.label_map.entity_types)
                model = SpanBiaffineNER(num_labels, self.config)
                trainer = SpanTrainer(
                    self.config, model, train_dataloader, val_dataloader, self.config.device)
            elif self.model_type == 'hybrid':
                # For hybrid model, we need the full label set for sequence labeling
                sequence_num_labels = len(self.config.label_map.labels)
                model = HybridNER(sequence_num_labels, self.config)
                trainer = HybridTrainer(
                    self.config, model, train_dataloader, val_dataloader, self.config.device)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Set fold-specific work directory
            fold_work_dir = os.path.join(
                self.config.work_dir, f'fold_{fold + 1}')
            trainer.config.work_dir = fold_work_dir

            # Train
            best_f1 = trainer.train()
            fold_results.append(best_f1)

            logger.info(f"Fold {fold + 1} completed with F1: {best_f1:.4f}")

        # Compute average results
        avg_f1 = sum(fold_results) / len(fold_results)
        std_f1 = (sum((f1 - avg_f1) ** 2 for f1 in fold_results) /
                  len(fold_results)) ** 0.5

        logger.info(
            f"K-Fold results: Average F1 = {avg_f1:.4f} ± {std_f1:.4f}")
        logger.info(f"Individual fold results: {fold_results}")

        return {
            'average_f1': avg_f1,
            'std_f1': std_f1,
            'fold_results': fold_results
        }
