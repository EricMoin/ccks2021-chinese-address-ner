import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
import logging

from config import Config
from span_biaffine_model import SpanBiaffineNER, HybridNER
from span_converter import SpanConverter, SpanDataset
from dataset import NERTestDataset
from conll_reader import ConllEntity

logger = logging.getLogger(__name__)


class SpanPredictor:
    """
    Predictor for span-based NER models
    """

    def __init__(self, config: Config, model_path: str):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize span converter
        self.span_converter = SpanConverter(config.label_map)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Create and load model
        self.model = self._create_model()
        self._load_model(model_path)

        logger.info(f"SpanPredictor initialized with model from: {model_path}")

    def _create_model(self):
        """Create model based on configuration"""
        if self.config.model_type == 'span':
            num_labels = len(self.config.label_map.entity_types) + 1
            model = SpanBiaffineNER(num_labels, self.config)
        elif self.config.model_type == 'hybrid':
            sequence_num_labels = len(self.config.label_map.labels)
            model = HybridNER(sequence_num_labels, self.config)
        else:
            raise ValueError(
                f"Unsupported model type: {self.config.model_type}")

        return model

    def _load_model(self, model_path: str):
        """Load model weights"""
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from: {model_path}")

    def predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """
        Predict entities for a batch of texts

        Args:
            texts: List of input texts

        Returns:
            List of entity predictions for each text
        """
        # Tokenize texts
        batch_tokens = []
        batch_encodings = []

        for text in texts:
            tokens = list(text)  # Character-level tokenization for Chinese
            encoding = self.tokenizer(
                tokens,
                truncation=True,
                padding="max_length",
                max_length=150,
                is_split_into_words=True,
                return_tensors="pt",
            )
            batch_tokens.append(tokens)
            batch_encodings.append(encoding)

        # Stack encodings
        input_ids = torch.cat([enc["input_ids"]
                              for enc in batch_encodings], dim=0).to(self.device)
        attention_mask = torch.cat(
            [enc["attention_mask"] for enc in batch_encodings], dim=0).to(self.device)

        # Predict
        with torch.no_grad():
            if self.config.model_type == 'span':
                span_scores = self.model(input_ids, attention_mask)
                predictions = self.span_converter.decode_span_labels(
                    span_scores, attention_mask, threshold=self.config.span_threshold
                )
            elif self.config.model_type == 'hybrid':
                outputs = self.model(input_ids, attention_mask)
                sequence_predictions = outputs['sequence_predictions']
                span_scores = outputs['span_scores']

                # Decode span predictions
                span_predictions = self.span_converter.decode_span_labels(
                    span_scores, attention_mask, threshold=self.config.span_threshold
                )

                # Convert sequence predictions to spans
                sequence_span_predictions = []
                for i, (seq_pred, tokens) in enumerate(zip(sequence_predictions, batch_tokens)):
                    seq_labels = [self.config.label_map.id2label[idx]
                                  for idx in seq_pred[:len(tokens)]]
                    spans = self.span_converter.bioes_to_spans(
                        tokens, seq_labels)
                    sequence_span_predictions.append(spans)

                # Combine predictions
                combined_predictions = self.span_converter.combine_sequence_and_span_predictions(
                    [[self.config.label_map.id2label[idx] for idx in seq_pred[:len(tokens)]]
                     for seq_pred, tokens in zip(sequence_predictions, batch_tokens)],
                    span_predictions,
                    batch_tokens,
                    sequence_weight=self.config.sequence_loss_weight,
                    span_weight=self.config.span_loss_weight
                )

                # Convert combined predictions to spans
                predictions = []
                for combined_pred, tokens in zip(combined_predictions, batch_tokens):
                    spans = self.span_converter.bioes_to_spans(
                        tokens, combined_pred)
                    predictions.append(spans)
            else:
                raise ValueError(
                    f"Unsupported model type: {self.config.model_type}")

        return predictions

    def predict_single(self, text: str) -> List[Dict]:
        """
        Predict entities for a single text

        Args:
            text: Input text

        Returns:
            List of entity predictions
        """
        predictions = self.predict_batch([text])
        return predictions[0]

    def predict_file(self, test_file: str, output_file: str, batch_size: int = 16):
        """
        Predict entities for a test file and save results

        Args:
            test_file: Path to test file
            output_file: Path to output file
            batch_size: Batch size for prediction
        """
        # Load test data
        test_dataset = NERTestDataset(
            test_file, self.tokenizer, self.config.label_map.label2id)
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        all_texts = []

        logger.info(f"Starting prediction on {len(test_dataset)} examples")

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                texts = batch["text"]
                tokens_list = batch["tokens"]

                # Predict
                if self.config.model_type == 'span':
                    span_scores = self.model(input_ids, attention_mask)
                    batch_predictions = self.span_converter.decode_span_labels(
                        span_scores, attention_mask, threshold=self.config.span_threshold
                    )
                elif self.config.model_type == 'hybrid':
                    outputs = self.model(input_ids, attention_mask)
                    sequence_predictions = outputs['sequence_predictions']
                    span_scores = outputs['span_scores']

                    # Decode span predictions
                    span_predictions = self.span_converter.decode_span_labels(
                        span_scores, attention_mask, threshold=self.config.span_threshold
                    )

                    # Convert sequence predictions to spans
                    sequence_span_predictions = []
                    for i, (seq_pred, tokens) in enumerate(zip(sequence_predictions, tokens_list)):
                        seq_labels = [self.config.label_map.id2label[idx]
                                      for idx in seq_pred[:len(tokens)]]
                        spans = self.span_converter.bioes_to_spans(
                            tokens, seq_labels)
                        sequence_span_predictions.append(spans)

                    # Combine predictions
                    combined_predictions = self.span_converter.combine_sequence_and_span_predictions(
                        [[self.config.label_map.id2label[idx] for idx in seq_pred[:len(tokens)]]
                         for seq_pred, tokens in zip(sequence_predictions, tokens_list)],
                        span_predictions,
                        tokens_list,
                        sequence_weight=self.config.sequence_loss_weight,
                        span_weight=self.config.span_loss_weight
                    )

                    # Convert combined predictions to spans
                    batch_predictions = []
                    for combined_pred, tokens in zip(combined_predictions, tokens_list):
                        spans = self.span_converter.bioes_to_spans(
                            tokens, combined_pred)
                        batch_predictions.append(spans)
                else:
                    raise ValueError(
                        f"Unsupported model type: {self.config.model_type}")

                all_predictions.extend(batch_predictions)
                all_texts.extend(texts)

        # Save predictions
        self._save_predictions(all_texts, all_predictions, output_file)
        logger.info(f"Predictions saved to: {output_file}")

    def _save_predictions(self, texts: List[str], predictions: List[List[Dict]], output_file: str):
        """Save predictions to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (text, pred_spans) in enumerate(zip(texts, predictions)):
                # Sort spans by start position
                pred_spans = sorted(pred_spans, key=lambda x: x['start'])

                # Create output line
                line_parts = []
                current_pos = 0

                for span in pred_spans:
                    start, end, label = span['start'], span['end'], span['label']

                    # Add text before span
                    if start > current_pos:
                        line_parts.append(text[current_pos:start])

                    # Add span with label
                    span_text = text[start:end+1]
                    line_parts.append(f"<{label}>{span_text}</{label}>")

                    current_pos = end + 1

                # Add remaining text
                if current_pos < len(text):
                    line_parts.append(text[current_pos:])

                # Write line
                f.write(f"{i+1}{''.join(line_parts)}\n")


class EnsembleSpanPredictor:
    """
    Ensemble predictor for multiple span-based models
    """

    def __init__(self, config: Config, model_paths: List[str]):
        self.config = config
        self.device = torch.device(config.device)
        self.span_converter = SpanConverter(config.label_map)

        # Load multiple predictors
        self.predictors = []
        for model_path in model_paths:
            predictor = SpanPredictor(config, model_path)
            self.predictors.append(predictor)

        logger.info(
            f"EnsembleSpanPredictor initialized with {len(self.predictors)} models")

    def predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """
        Predict entities using ensemble of models

        Args:
            texts: List of input texts

        Returns:
            List of ensemble entity predictions
        """
        # Get predictions from all models
        all_predictions = []
        for predictor in self.predictors:
            predictions = predictor.predict_batch(texts)
            all_predictions.append(predictions)

        # Ensemble predictions
        ensemble_predictions = []
        for i in range(len(texts)):
            text_predictions = [pred[i] for pred in all_predictions]
            ensemble_pred = self._ensemble_spans(text_predictions)
            ensemble_predictions.append(ensemble_pred)

        return ensemble_predictions

    def _ensemble_spans(self, span_lists: List[List[Dict]]) -> List[Dict]:
        """
        Ensemble multiple span predictions using voting

        Args:
            span_lists: List of span predictions from different models

        Returns:
            Ensembled span predictions
        """
        # Collect all unique spans
        span_votes = {}

        for spans in span_lists:
            for span in spans:
                key = (span['start'], span['end'], span['label'])
                if key not in span_votes:
                    span_votes[key] = {
                        'count': 0,
                        'confidence_sum': 0.0,
                        'span': span
                    }
                span_votes[key]['count'] += 1
                span_votes[key]['confidence_sum'] += span.get(
                    'confidence', 1.0)

        # Filter spans by vote threshold
        vote_threshold = len(self.predictors) // 2 + 1  # Majority vote
        ensemble_spans = []

        for key, vote_info in span_votes.items():
            if vote_info['count'] >= vote_threshold:
                span = vote_info['span'].copy()
                span['confidence'] = vote_info['confidence_sum'] / \
                    vote_info['count']
                span['votes'] = vote_info['count']
                ensemble_spans.append(span)

        # Resolve conflicts
        ensemble_spans = self.span_converter.resolve_span_conflicts(
            ensemble_spans)

        return ensemble_spans

    def predict_file(self, test_file: str, output_file: str, batch_size: int = 16):
        """Predict entities for a test file using ensemble"""
        # Load test data
        test_dataset = NERTestDataset(
            test_file, self.predictors[0].tokenizer, self.config.label_map.label2id)
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        all_texts = []

        logger.info(
            f"Starting ensemble prediction on {len(test_dataset)} examples")

        for batch in test_dataloader:
            texts = batch["text"]

            # Get predictions from all models
            batch_predictions = self.predict_batch(texts)

            all_predictions.extend(batch_predictions)
            all_texts.extend(texts)

        # Save predictions
        self.predictors[0]._save_predictions(
            all_texts, all_predictions, output_file)
        logger.info(f"Ensemble predictions saved to: {output_file}")
