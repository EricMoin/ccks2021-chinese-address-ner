import torch
from typing import List, Tuple, Dict, Optional
from conll_reader import ConllEntity
from label import LabelMap
import numpy as np


def span_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for span dataset
    """
    # Extract all items
    input_ids = []
    attention_mask = []
    sequence_labels = []
    span_labels = []
    tokens = []
    original_labels = []

    for item in batch:
        input_ids.append(item["input_ids"])
        attention_mask.append(item["attention_mask"])
        sequence_labels.append(item["sequence_labels"])
        span_labels.append(item["span_labels"])
        tokens.append(item["tokens"])
        original_labels.append(item["original_labels"])

    # Stack tensors
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "sequence_labels": torch.stack(sequence_labels),
        "span_labels": torch.stack(span_labels),
        "tokens": tokens,  # Keep as list
        "original_labels": original_labels  # Keep as list
    }


class SpanConverter:
    """
    Convert between sequence labeling (BIOES) and span-based representations
    """

    def __init__(self, label_map: LabelMap):
        self.label_map = label_map
        self.label2id = label_map.label2id
        self.id2label = label_map.id2label

        # Use entity types directly from label_map
        self.entity_types = [et for et in label_map.entity_types if et != 'O']

        # Create mapping from entity type to label indices
        # 0 is reserved for "O", other entity types start from 1
        self.entity_type2id = {'O': 0}
        for i, entity_type in enumerate(self.entity_types):
            self.entity_type2id[entity_type] = i + 1

        self.id2entity_type = {v: k for k, v in self.entity_type2id.items()}

    def bioes_to_spans(self, tokens: List[str], labels: List[str]) -> List[Dict]:
        """
        Convert BIOES sequence labels to spans

        Args:
            tokens: List of tokens
            labels: List of BIOES labels

        Returns:
            List of span dictionaries with keys: start, end, label, text
        """
        spans = []
        current_span = None

        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label == 'O':
                # End current span if exists
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None
            elif label.startswith('B-'):
                # Begin new span
                if current_span is not None:
                    spans.append(current_span)
                entity_type = label[2:]
                current_span = {
                    'start': i,
                    'end': i,
                    'label': entity_type,
                    'text': token
                }
            elif label.startswith('I-'):
                # Continue current span
                if current_span is not None:
                    entity_type = label[2:]
                    if current_span['label'] == entity_type:
                        current_span['end'] = i
                        current_span['text'] += token
                    else:
                        # Label mismatch, end current span and start new one
                        spans.append(current_span)
                        current_span = {
                            'start': i,
                            'end': i,
                            'label': entity_type,
                            'text': token
                        }
                else:
                    # I- without B-, treat as B-
                    entity_type = label[2:]
                    current_span = {
                        'start': i,
                        'end': i,
                        'label': entity_type,
                        'text': token
                    }
            elif label.startswith('E-'):
                # End current span
                if current_span is not None:
                    entity_type = label[2:]
                    if current_span['label'] == entity_type:
                        current_span['end'] = i
                        current_span['text'] += token
                        spans.append(current_span)
                        current_span = None
                    else:
                        # Label mismatch, end current span and create single-token span
                        spans.append(current_span)
                        spans.append({
                            'start': i,
                            'end': i,
                            'label': entity_type,
                            'text': token
                        })
                        current_span = None
                else:
                    # E- without B-, treat as single token span
                    entity_type = label[2:]
                    spans.append({
                        'start': i,
                        'end': i,
                        'label': entity_type,
                        'text': token
                    })
            elif label.startswith('S-'):
                # Single token span
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None
                entity_type = label[2:]
                spans.append({
                    'start': i,
                    'end': i,
                    'label': entity_type,
                    'text': token
                })

        # Add final span if exists
        if current_span is not None:
            spans.append(current_span)

        return spans

    def spans_to_bioes(self, tokens: List[str], spans: List[Dict]) -> List[str]:
        """
        Convert spans to BIOES sequence labels

        Args:
            tokens: List of tokens
            spans: List of span dictionaries

        Returns:
            List of BIOES labels
        """
        labels = ['O'] * len(tokens)

        for span in spans:
            start = span['start']
            end = span['end']
            entity_type = span['label']

            if start == end:
                # Single token span
                labels[start] = f'S-{entity_type}'
            else:
                # Multi-token span
                labels[start] = f'B-{entity_type}'
                for i in range(start + 1, end):
                    labels[i] = f'I-{entity_type}'
                labels[end] = f'E-{entity_type}'

        return labels

    def create_span_labels(self, tokens: List[str], labels: List[str], max_length: int) -> torch.Tensor:
        """
        Create span label matrix from BIOES labels

        Args:
            tokens: List of tokens
            labels: List of BIOES labels
            max_length: Maximum sequence length

        Returns:
            Tensor of shape [max_length, max_length] with span labels
        """
        # Convert BIOES to spans
        spans = self.bioes_to_spans(tokens, labels)

        # Create span label matrix
        span_labels = torch.zeros(max_length, max_length, dtype=torch.long)

        for span in spans:
            start = min(span['start'], max_length - 1)
            end = min(span['end'], max_length - 1)
            entity_type = span['label']

            if entity_type in self.entity_type2id:
                label_id = self.entity_type2id[entity_type]
                span_labels[start, end] = label_id

        return span_labels

    def decode_span_labels(self, span_scores: torch.Tensor, attention_mask: torch.Tensor,
                           threshold: float = 0.5) -> List[List[Dict]]:
        """
        Decode span predictions from scores

        Args:
            span_scores: [batch_size, seq_len, seq_len, num_labels]
            attention_mask: [batch_size, seq_len]
            threshold: Confidence threshold

        Returns:
            List of span lists for each example in batch
        """
        batch_size, seq_len, _, num_labels = span_scores.size()
        batch_spans = []

        # Get probabilities
        probs = torch.softmax(span_scores, dim=-1)

        for b in range(batch_size):
            spans = []
            seq_length = attention_mask[b].sum().item()

            for start in range(seq_length):
                for end in range(start, seq_length):
                    # Get best label for this span
                    best_label_idx = torch.argmax(probs[b, start, end]).item()
                    best_prob = probs[b, start, end, best_label_idx].item()

                    # Skip "O" label and low confidence spans
                    if best_label_idx > 0 and best_prob > threshold:
                        entity_type = self.id2entity_type[best_label_idx]
                        spans.append({
                            'start': start,
                            'end': end,
                            'label': entity_type,
                            'confidence': best_prob
                        })

            batch_spans.append(spans)

        return batch_spans

    def resolve_span_conflicts(self, spans: List[Dict]) -> List[Dict]:
        """
        Resolve overlapping spans by keeping the one with highest confidence

        Args:
            spans: List of span dictionaries

        Returns:
            List of non-overlapping spans
        """
        if not spans:
            return spans

        # Sort spans by confidence (descending)
        spans = sorted(spans, key=lambda x: x['confidence'], reverse=True)

        resolved_spans = []
        used_positions = set()

        for span in spans:
            start, end = span['start'], span['end']

            # Check if this span overlaps with any already selected span
            overlap = False
            for pos in range(start, end + 1):
                if pos in used_positions:
                    overlap = True
                    break

            if not overlap:
                resolved_spans.append(span)
                for pos in range(start, end + 1):
                    used_positions.add(pos)

        # Sort by start position
        resolved_spans = sorted(resolved_spans, key=lambda x: x['start'])

        return resolved_spans

    def combine_sequence_and_span_predictions(self, sequence_preds: List[List[str]],
                                              span_preds: List[List[Dict]],
                                              tokens_list: List[List[str]],
                                              sequence_weight: float = 0.5,
                                              span_weight: float = 0.5) -> List[List[str]]:
        """
        Combine predictions from sequence labeling and span-based models

        Args:
            sequence_preds: Sequence labeling predictions
            span_preds: Span-based predictions
            tokens_list: List of token lists
            sequence_weight: Weight for sequence predictions
            span_weight: Weight for span predictions

        Returns:
            Combined predictions in BIOES format
        """
        combined_preds = []

        for seq_pred, span_pred, tokens in zip(sequence_preds, span_preds, tokens_list):
            # Resolve span conflicts
            resolved_spans = self.resolve_span_conflicts(span_pred)

            # Convert spans to BIOES
            span_bioes = self.spans_to_bioes(tokens, resolved_spans)

            # Simple voting: if both models agree, use that; otherwise use sequence model
            combined = []
            for i, (seq_label, span_label) in enumerate(zip(seq_pred, span_bioes)):
                if seq_label == span_label:
                    combined.append(seq_label)
                else:
                    # Use confidence-based combination or default to sequence model
                    if sequence_weight >= span_weight:
                        combined.append(seq_label)
                    else:
                        combined.append(span_label)

            combined_preds.append(combined)

        return combined_preds


class SpanDataset:
    """
    Dataset class for span-based NER training
    """

    def __init__(self, data: List[ConllEntity], tokenizer, label_map: LabelMap, max_length: int = 150):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length
        self.span_converter = SpanConverter(label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        entity = self.data[index]

        # Process tokens and labels (skip prompt part)
        tokens = []
        labels = []
        is_prompt = False

        for token, label in zip(entity.tokens, entity.labels):
            if token == "<EOS>":
                is_prompt = True
                continue
            if not is_prompt:
                tokens.append(token)
                labels.append(label)

        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            labels = labels[:self.max_length]

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors="pt",
        )

        # Create sequence labels
        label_ids = []
        for label in labels:
            if label in self.label_map.label2id:
                label_ids.append(self.label_map.label2id[label])
            else:
                label_ids.append(self.label_map.label2id["O"])

        # Ensure label_ids doesn't exceed max_length
        if len(label_ids) > self.max_length:
            label_ids = label_ids[:self.max_length]

        # Pad sequence labels to exactly max_length
        padded_sequence_labels = label_ids + \
            [self.label_map.label2id["O"]] * (self.max_length - len(label_ids))

        # Ensure exactly max_length
        padded_sequence_labels = padded_sequence_labels[:self.max_length]

        # Create span labels
        span_labels = self.span_converter.create_span_labels(
            tokens, labels, self.max_length)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sequence_labels": torch.tensor(padded_sequence_labels, dtype=torch.long),
            "span_labels": span_labels,
            "tokens": tokens,
            "original_labels": labels
        }


class SpanMetrics:
    """
    Evaluation metrics for span-based NER
    """

    def __init__(self, label_map: LabelMap):
        self.label_map = label_map
        self.span_converter = SpanConverter(label_map)

    def compute_span_f1(self, true_spans: List[List[Dict]], pred_spans: List[List[Dict]]) -> Dict[str, float]:
        """
        Compute span-level F1 score

        Args:
            true_spans: Ground truth spans
            pred_spans: Predicted spans

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        true_spans_set = set()
        pred_spans_set = set()

        # Convert spans to tuples for set operations
        for i, (true_batch, pred_batch) in enumerate(zip(true_spans, pred_spans)):
            for span in true_batch:
                true_spans_set.add(
                    (i, span['start'], span['end'], span['label']))
            for span in pred_batch:
                pred_spans_set.add(
                    (i, span['start'], span['end'], span['label']))

        # Compute metrics
        tp = len(true_spans_set & pred_spans_set)
        fp = len(pred_spans_set - true_spans_set)
        fn = len(true_spans_set - pred_spans_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def compute_entity_type_f1(self, true_spans: List[List[Dict]], pred_spans: List[List[Dict]]) -> Dict[str, Dict[str, float]]:
        """
        Compute F1 score for each entity type

        Args:
            true_spans: Ground truth spans
            pred_spans: Predicted spans

        Returns:
            Dictionary with F1 scores for each entity type
        """
        entity_types = set()

        # Collect all entity types
        for true_batch, pred_batch in zip(true_spans, pred_spans):
            for span in true_batch:
                entity_types.add(span['label'])
            for span in pred_batch:
                entity_types.add(span['label'])

        results = {}

        for entity_type in entity_types:
            true_type_spans = set()
            pred_type_spans = set()

            for i, (true_batch, pred_batch) in enumerate(zip(true_spans, pred_spans)):
                for span in true_batch:
                    if span['label'] == entity_type:
                        true_type_spans.add((i, span['start'], span['end']))
                for span in pred_batch:
                    if span['label'] == entity_type:
                        pred_type_spans.add((i, span['start'], span['end']))

            tp = len(true_type_spans & pred_type_spans)
            fp = len(pred_type_spans - true_type_spans)
            fn = len(true_type_spans - pred_type_spans)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / \
                (precision + recall) if (precision + recall) > 0 else 0.0

            results[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }

        return results
