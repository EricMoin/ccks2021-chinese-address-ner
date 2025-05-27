import os
from sklearn.model_selection import KFold
import torch
from config import Config
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import copy
import logging
from sklearn.metrics import classification_report

from conll_reader import ConllReader, MultiConllReader
from dataset import NERDataset, NERTestDataset
from model import FreeLB, AddressNER
from label import LabelMap

logger = logging.getLogger(__name__)


class StochasticWeightAveraging:
    def __init__(self, model, swa_start_epoch, swa_lr=None, swa_freq=5):
        """
        Implements Stochastic Weight Averaging (SWA)
        Args:
            model: The model to apply SWA to
            swa_start_epoch: The epoch to start averaging weights (0-indexed)
            swa_lr: The learning rate to use during SWA (currently not used for optimizer re-init)
            swa_freq: How frequently (in epochs) to update the SWA model
        """
        self.model_base = model  # Keep a reference to the original model structure for deepcopy
        self.swa_start_epoch = swa_start_epoch
        # Note: SWA LR is often handled by scheduler or fixed small LR during SWA phase
        self.swa_lr = swa_lr
        self.swa_freq = swa_freq
        self.swa_model = None
        self.n_averaged = 0
        logger.info(
            f"SWA initialized: start_epoch={swa_start_epoch}, lr={swa_lr}, freq={swa_freq}")

    def update(self, epoch, model_current_state):
        """Update the SWA model by averaging with current model weights"""
        if epoch < self.swa_start_epoch:
            return

        if (epoch - self.swa_start_epoch) % self.swa_freq != 0:
            return

        logger.info(f"Updating SWA model at epoch {epoch}")
        if self.swa_model is None:
            # Create SWA model from base structure
            self.swa_model = copy.deepcopy(self.model_base)
            self.swa_model.load_state_dict(copy.deepcopy(model_current_state))
            logger.info("SWA model initialized with current model weights.")
        else:
            # Update running average of parameters
            current_params = dict(model_current_state)
            for name, swa_param in self.swa_model.named_parameters():
                if swa_param.requires_grad:
                    model_param = current_params[name]
                    swa_param.data.mul_(
                        self.n_averaged / (self.n_averaged + 1))
                    swa_param.data.add_(
                        model_param.data / (self.n_averaged + 1))
            logger.info(
                f"SWA model updated. n_averaged became {self.n_averaged + 1}")
        self.n_averaged += 1

    def get_final_model_state_dict(self):
        """Return the SWA model's state_dict with averaged weights"""
        if self.swa_model is None:
            logger.warning(
                "SWA was enabled, but no averaging was done. Returning None for SWA model state_dict.")
            return None
        logger.info("Final SWA model state_dict retrieved.")
        return self.swa_model.state_dict()


class Trainer:
    config: Config
    model: nn.Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    device: torch.device

    def __init__(self, config: Config, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, device: str):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(device)
        self.scheduler = None  # Initialize scheduler attribute

        # Initialize SWA if enabled
        if self.config.use_swa:
            self.swa = StochasticWeightAveraging(
                model=self.model,  # Pass the model instance for deepcopy base
                swa_start_epoch=self.config.swa_start_epoch,
                swa_lr=self.config.swa_lr,
                swa_freq=self.config.swa_freq
            )
        else:
            self.swa = None

        self.model.to(self.device)
        logger.info(
            f"Trainer initialized for training. Training on {self.device}. Work directory: {self.config.work_dir}")

    def train(self):
        self.model.train()  # Ensure model is in training mode
        # Optimizer
        optimizer = torch.optim.AdamW([
            {'params': self.model.bert.embeddings.parameters(
            ), 'lr': self.config.learning_rate * 5},
            {'params': self.model.lstm.parameters(
            ), 'lr': self.config.learning_rate * 25},
            {'params': self.model.classifier.parameters(
            ), 'lr': self.config.learning_rate * 25},
            {'params': self.model.crf.parameters(
            ), 'lr': self.config.learning_rate * 50}
        ], lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )

        best_val_metric = 0
        # Fold specific work_dir
        os.makedirs(self.config.work_dir, exist_ok=True)

        freelb = None
        if hasattr(self.config, 'adversarial_training_start_epoch') and self.config.adversarial_training_start_epoch >= 0 and self.config.use_freelb:
            freelb = FreeLB(
                self.model,
                adv_lr=self.config.freelb_adv_lr,
                adv_steps=self.config.freelb_adv_steps,
                adv_init_mag=self.config.freelb_adv_init_mag,
                adv_max_norm=self.config.freelb_adv_max_norm,
                adv_norm_type=self.config.freelb_adv_norm_type,
                base_model=self.config.freelb_base_model
            )
            logger.info("FreeLB adversarial training is configured.")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(
                self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train] ({os.path.basename(self.config.work_dir)})")

            for batch in train_pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                if freelb and epoch >= self.config.adversarial_training_start_epoch:
                    # Get original embeddings for FreeLB
                    original_embeddings = self.model.bert.embeddings.word_embeddings(
                        input_ids)
                    # FreeLB's .attack() method will handle its own gradient accumulation
                    # and self.model.zero_grad() internally before its loop.
                    adv_loss = freelb.attack(
                        original_embeddings.detach(), attention_mask, labels)
                    # The gradients are now accumulated in model.parameters() from FreeLB's attack.
                    # We use adv_loss for logging, but the gradients for optimizer.step() are from FreeLB.
                    current_loss = adv_loss  # For logging purposes
                else:
                    # Standard forward and backward pass if FreeLB is not active
                    loss = self.model(input_ids, attention_mask, labels)
                    loss.backward()
                    current_loss = loss.item()

                # Gradient clipping and optimizer step (applies to gradients from either standard pass or FreeLB)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                train_loss += current_loss  # Accumulate the loss for the epoch average
                train_pbar.set_postfix({"loss": f"{current_loss:.4f}"})

            avg_train_loss = train_loss / \
                len(self.train_dataloader) if len(
                    self.train_dataloader) > 0 else 0
            logger.info(
                f"Epoch {epoch+1} ({os.path.basename(self.config.work_dir)}) avg training loss: {avg_train_loss:.4f}")

            # Evaluation
            self.model.eval()
            all_preds_eval = []
            all_labels_eval = []
            eval_loss = 0

            with torch.no_grad():
                for batch in tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Eval] ({os.path.basename(self.config.work_dir)})"):
                    input_ids_eval = batch["input_ids"].to(self.device)
                    attention_mask_eval = batch["attention_mask"].to(
                        self.device)
                    labels_eval = batch["labels"].to(self.device)

                    loss_eval_batch = self.model(
                        input_ids_eval, attention_mask_eval, labels_eval)
                    eval_loss += loss_eval_batch.item()

                    predictions_eval = self.model(
                        input_ids_eval, attention_mask_eval)

                    for pred_seq, mask_seq, label_seq in zip(predictions_eval, attention_mask_eval, labels_eval):
                        true_length = mask_seq.sum().item()
                        all_preds_eval.extend(pred_seq[:true_length])
                        all_labels_eval.extend(
                            label_seq[:true_length].cpu().numpy())

            avg_eval_loss = eval_loss / \
                len(self.val_dataloader) if len(self.val_dataloader) > 0 else 0
            correct_eval = sum(p == l for p, l in zip(
                all_preds_eval, all_labels_eval))
            total_eval = len(all_preds_eval) if len(
                all_preds_eval) > 0 else 1
            current_val_metric = correct_eval / total_eval
            logger.info(
                f"Epoch {epoch+1} ({os.path.basename(self.config.work_dir)}) Val Loss: {avg_eval_loss:.4f}, Val Acc: {current_val_metric:.4f}")

            if total_eval > 0:
                # Generate and log classification report
                # Ensure all_preds_eval and all_labels_eval are flat lists of integers (label IDs)
                # Get label names from id
                target_names = [self.config.label_map.id2label[i] for i in sorted(
                    list(set(all_labels_eval + all_preds_eval)))]
                # Filter out OOD labels if any from target_names before passing to classification_report
                # This assumes your label_map.id2label correctly maps all occurring IDs.
                # It's also important that all_preds_eval and all_labels_eval contain numerical IDs.

                # Handle cases where some labels might only appear in preds or true labels
                # and might not be in the initial set of labels (if id2label is not exhaustive)
                # We'll use labels present in either preds or true, and map them.
                present_label_ids = sorted(
                    list(set(all_labels_eval).union(set(all_preds_eval))))

                # Ensure all these IDs have a mapping in id2label
                valid_target_names = []
                valid_label_ids_for_report = []

                for label_id in present_label_ids:
                    if label_id in self.config.label_map.id2label:
                        valid_target_names.append(
                            self.config.label_map.id2label[label_id])
                        valid_label_ids_for_report.append(label_id)
                    else:
                        logger.warning(
                            f"Label ID {label_id} found in predictions/gold labels but not in id2label map. Skipping for report.")

                if valid_label_ids_for_report:  # Proceed only if there are valid labels to report
                    try:
                        report = classification_report(
                            all_labels_eval,
                            all_preds_eval,
                            labels=valid_label_ids_for_report,  # Use only IDs that have a name
                            target_names=valid_target_names,   # Corresponding names
                            digits=4,
                            zero_division=0  # Avoids warnings when a class has no predictions or no true samples
                        )
                        logger.info(
                            f"Classification Report for Epoch {epoch+1} ({os.path.basename(self.config.work_dir)}):\n{report}")
                    except ValueError as e:
                        logger.error(
                            f"Could not generate classification report: {e}. Preds: {set(all_preds_eval)}, Labels: {set(all_labels_eval)}")
                else:
                    logger.warning(
                        "No valid labels found to generate classification report (all predicted/gold labels were unmappable).")

            if self.swa is not None:
                self.swa.update(epoch, self.model.state_dict())
            report_dict = classification_report(
                all_labels_eval,
                all_preds_eval,
                labels=valid_label_ids_for_report,
                target_names=valid_target_names,
                digits=4,
                zero_division=0,
                output_dict=True
            )
            micro_f1 = report_dict['weighted avg']['f1-score']
            current_val_metric = micro_f1  # 使用 Micro-F1 作为 val metric
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                saved_path = os.path.join(
                    self.config.work_dir, "best_model.pt")  # Saved in fold-specific dir
                torch.save(self.model.state_dict(), saved_path)
                logger.info(
                    f"Saved new best model ({os.path.basename(self.config.work_dir)}) with val metric: {current_val_metric:.4f} to {saved_path}")

        if self.swa is not None:
            swa_model_state_dict = self.swa.get_final_model_state_dict()
            if swa_model_state_dict is not None:
                swa_save_path = os.path.join(
                    self.config.work_dir, "swa_model.pt")  # Saved in fold-specific dir
                torch.save(swa_model_state_dict, swa_save_path)
                logger.info(
                    f"Saved final SWA model ({os.path.basename(self.config.work_dir)}) to {swa_save_path}")
        logger.info(
            f"Training finished for work directory: {self.config.work_dir}")


class KFoldTrainer:
    def __init__(self, config: Config):
        self.config = config

    def kfold_train(self):
        logger.info("--- Starting K-Fold Training Pipeline ---")
        folds_base_dir = os.path.join(
            self.config.work_dir, self.config.model_name)
        os.makedirs(folds_base_dir, exist_ok=True)
        logger.info(
            f"Base directory for K-Folds of model '{self.config.model_name}': {folds_base_dir}")

        multi_reader = MultiConllReader()
        full_data_conll = list(multi_reader.read(
            # List of Sentence objects
            [self.config.train_file, self.config.dev_file]))

        kf = KFold(n_splits=self.config.k_folds, shuffle=True,
                   random_state=self.config.seed)

        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(full_data_conll)):
            fold_num = fold_idx + 1
            logger.info(
                f"--- Processing Fold {fold_num}/{self.config.k_folds} for model '{self.config.model_name}' ---")

            fold_train_data = [full_data_conll[i] for i in train_indices]
            fold_val_data = [full_data_conll[i] for i in val_indices]

            # Create a specific config for this fold
            # Start with a copy of the main config
            fold_cfg = copy.deepcopy(self.config)
            fold_cfg.work_dir = os.path.join(
                folds_base_dir, f"fold_{fold_num}")
            # model_name for AddressNER should be the adapted model path being k-folded
            fold_cfg.model_name = self.config.model_name
            os.makedirs(fold_cfg.work_dir, exist_ok=True)
            logger.info(
                f"Fold {fold_num} config: work_dir='{fold_cfg.work_dir}', model_name_for_tokenizer='{fold_cfg.model_name}'")

            # Instantiate model for this fold (uses fold_cfg.model_name for tokenizer)
            model = AddressNER(num_labels=len(
                fold_cfg.label_map.labels), config=fold_cfg)

            # Create datasets and dataloaders for this fold
            train_dataset = NERDataset(
                fold_train_data, model.tokenizer, fold_cfg.label_map.label2id)
            val_dataset = NERDataset(
                fold_val_data, model.tokenizer, fold_cfg.label_map.label2id)
            train_loader = DataLoader(
                train_dataset, batch_size=fold_cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=fold_cfg.batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True)

            # Instantiate and run trainer for this fold
            trainer_fold = Trainer(config=fold_cfg,
                                   model=model,
                                   train_dataloader=train_loader,
                                   val_dataloader=val_loader,
                                   device=fold_cfg.device)
            trainer_fold.train()  # This will save best_model.pt and swa_model.pt in fold_cfg.work_dir
        logger.info("--- K-Fold Training Pipeline Finished ---")


class SingleTrainer:
    def __init__(self, config: Config):
        self.config = config

    def train(self):
        model = AddressNER(num_labels=len(
            self.config.label_map.labels), config=self.config)
        self.config.work_dir = os.path.join(
            self.config.work_dir, self.config.model_name)
        os.makedirs(self.config.work_dir, exist_ok=True)
        conll_reader = ConllReader()
        train_data = list(conll_reader.read(self.config.train_file))
        val_data = list(conll_reader.read(self.config.dev_file))

        train_dataset = NERDataset(
            train_data, model.tokenizer, self.config.label_map.label2id)
        val_dataset = NERDataset(
            val_data, model.tokenizer, self.config.label_map.label2id)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

        # Instantiate and run trainer for this fold
        trainer = Trainer(config=self.config,
                          model=model,
                          train_dataloader=train_loader,
                          val_dataloader=val_loader,
                          device=self.config.device)
        trainer.train()
