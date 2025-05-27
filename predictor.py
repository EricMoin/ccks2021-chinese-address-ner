import logging
import os
import re
import torch
from config import Config
from conll_reader import ConllEntity
from dataset import NERDataset
from label import LabelMap
from model import AddressNER
from torch.utils.data import DataLoader
from tqdm import tqdm
from logger import logger


class Predictor:
    def __init__(self, model_init_config: Config):
        """
        Initializes the Predictor.
        Args:
            model_init_config (Config): Configuration object used to initialize the AddressNER model,
                                       primarily for tokenizer and model architecture.
                                       Its model_name should point to the base adapted model.
        """
        self.model_config = model_init_config  # Config for AddressNER instantiation
        self.device = torch.device(self.model_config.device)

    def get_predictions_for_fold(self,
                                 fold_work_dir: str,
                                 test_file_path: str,
                                 label_map: LabelMap,  # Pass the main label_map
                                 batch_size: int,
                                 use_swa_if_available: bool) -> list[list[str]]:
        """
        Loads a trained model from a specific fold directory and generates predictions.
        Args:
            fold_work_dir (str): Path to the working directory of a trained fold 
                                 (e.g., result/model_xyz/fold_1).
            test_file_path (str): Path to the raw test file (e.g., data/final_test.txt).
            label_map (LabelMap): The global LabelMap object.
            batch_size (int): Batch size for prediction.
            use_swa_if_available (bool): Whether to prefer SWA model if available.

        Returns:
            list[list[str]]: A list of predicted label sequences for the test set.
        """
        model_to_load_path = None
        swa_model_path = os.path.join(fold_work_dir, "swa_model.pt")
        best_model_path = os.path.join(fold_work_dir, "best_model.pt")

        if use_swa_if_available and os.path.exists(swa_model_path):
            logger.info(
                f"Predictor: Using SWA model for inference from: {swa_model_path}")
            model_to_load_path = swa_model_path
        elif os.path.exists(best_model_path):
            logger.info(
                f"Predictor: Using best_model.pt for inference from: {best_model_path}")
            model_to_load_path = best_model_path
        else:
            logger.error(
                f"Predictor: No SWA or best model found in {fold_work_dir}. Cannot generate predictions for this fold.")
            return []  # Return empty list for this fold

        try:
            model_name = re.sub(r'(/swa_model\.pt)|(/best_model\.pt)',
                                r'', model_to_load_path)
            model_name = re.sub(r'result/pretrained/',
                                r'', model_name)
            model_name = re.sub(r'_adapted_ep[\d]+_seed[\d]+',
                                r'', model_name)
            model_name = model_name.replace(r"_", '/')
            self.model_config.model_name = model_name
            self.model = AddressNER(num_labels=len(self.model_config.label_map.labels),
                                    config=self.model_config)
            self.model.load_state_dict(torch.load(
                model_to_load_path, map_location=self.device))
            self.model.to(self.device)
            logger.info(
                f"Predictor: Successfully loaded model weights from {model_to_load_path}")
        except Exception as e:
            logger.error(
                f"Predictor: Error loading model state_dict from {model_to_load_path}: {e}")
            return []

        self.model.eval()

        # --- Test Data Preparation ---
        test_sentences_char_tokens = []
        if not os.path.exists(test_file_path):
            logger.error(
                f"Predictor: Test file not found at: {test_file_path}")
            return []

        with open(test_file_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if line:
                    try:
                        # Assuming format: guid<SEP>text
                        text_part = line.split('\u0001')[1]
                        test_sentences_char_tokens.append(list(text_part))
                    except IndexError:
                        logger.warning(
                            f"Predictor: Skipping malformed line in test file {test_file_path}: {line}")
                        # Keep example count consistent
                        test_sentences_char_tokens.append([])

        if not test_sentences_char_tokens:
            logger.warning(
                f"Predictor: No character tokens extracted from test file {test_file_path}.")
            return []

        test_conll_examples = [ConllEntity(chars, ['O'] * len(chars))
                               for chars in test_sentences_char_tokens]

        # Tokenizer comes from self.model (AddressNER instance initialized with model_init_config)
        # label_map.label2id is used by NERDataset
        test_dataset = NERDataset(
            data=test_conll_examples,
            tokenizer=self.model.tokenizer,
            label_map=label_map.label2id  # Use label_map from main_cfg
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        all_preds_sequences = []
        with torch.no_grad():
            pbar_desc = f"Predicting with fold model: {os.path.basename(fold_work_dir)}"
            for batch in tqdm(test_dataloader, desc=pbar_desc):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Model's forward pass for prediction (without labels)
                batch_pred_indices = self.model(input_ids, attention_mask)

                # Iterate through examples in batch
                for i in range(len(batch_pred_indices)):
                    # batch_pred_indices[i] is List[int] for one example
                    pred_indices_for_example = batch_pred_indices[i]

                    # Convert indices to label strings using label_map from main_cfg
                    pred_labels = [label_map.id2label.get(
                        p_idx, 'O') for p_idx in pred_indices_for_example]
                    all_preds_sequences.append(pred_labels)

        logger.info(
            f"Predictor: Generated {len(all_preds_sequences)} prediction sequences for test set using model from {fold_work_dir}.")
        return all_preds_sequences
