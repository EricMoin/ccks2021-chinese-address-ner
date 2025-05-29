import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional, Dict
import math
from config import Config
from span_losses import SpanFocalLoss, SpanCombinedLoss, SpanDiceLoss, SpanBoundaryLoss, SpanLabelSmoothingLoss


class BiaffineAttention(nn.Module):
    """
    Biaffine attention mechanism for span classification
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super(BiaffineAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Biaffine transformation: U1 * x1 * W * x2^T + U2 * [x1; x2] + b
        self.W = nn.Parameter(torch.randn(input_dim, output_dim, input_dim))
        self.U1 = nn.Linear(input_dim, output_dim, bias=False)
        self.U2 = nn.Linear(2 * input_dim, output_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.U1.weight)
        nn.init.xavier_uniform_(self.U2.weight)
        if self.U2.bias is not None:
            nn.init.zeros_(self.U2.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: [batch_size, seq_len, input_dim] - start representations
            x2: [batch_size, seq_len, input_dim] - end representations

        Returns:
            [batch_size, seq_len, seq_len, output_dim] - biaffine scores
        """
        batch_size, seq_len, input_dim = x1.size()

        # Biaffine term: x1 * W * x2^T
        # x1: [batch, seq_len, input_dim]
        # W: [input_dim, output_dim, input_dim]
        # x2: [batch, seq_len, input_dim]

        # Reshape for batch matrix multiplication
        x1_expanded = x1.unsqueeze(2)  # [batch, seq_len, 1, input_dim]
        # [batch, input_dim, output_dim, input_dim]
        W_expanded = self.W.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Compute x1 * W
        # [batch, seq_len, output_dim, input_dim]
        x1_W = torch.einsum('bsid,biod->bsod', x1_expanded, W_expanded)

        # Compute (x1 * W) * x2^T
        # [batch, seq_len, seq_len, output_dim]
        biaffine_scores = torch.einsum('bsod,btd->bsto', x1_W, x2)

        # Linear terms
        u1_scores = self.U1(x1).unsqueeze(2)  # [batch, seq_len, 1, output_dim]
        # [batch, seq_len, seq_len, output_dim]
        u1_scores = u1_scores.expand(-1, -1, seq_len, -1)

        # Concatenate x1 and x2 for each pair
        # [batch, seq_len, seq_len, input_dim]
        x1_expanded = x1.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, seq_len, seq_len, input_dim]
        x2_expanded = x2.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, 2*input_dim]
        x_concat = torch.cat([x1_expanded, x2_expanded], dim=-1)

        u2_scores = self.U2(x_concat)  # [batch, seq_len, seq_len, output_dim]

        # Combine all terms
        scores = biaffine_scores + u1_scores + u2_scores

        return scores


class SpanClassifier(nn.Module):
    """
    Span-based classifier using biaffine attention
    """

    def __init__(self, hidden_dim: int, num_labels: int, dropout: float = 0.1):
        super(SpanClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        # Separate representations for span start and end
        self.start_projection = nn.Linear(hidden_dim, hidden_dim)
        self.end_projection = nn.Linear(hidden_dim, hidden_dim)

        # Biaffine attention for span classification
        self.biaffine = BiaffineAttention(hidden_dim, num_labels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]

        Returns:
            span_scores: [batch_size, seq_len, seq_len, num_labels]
        """
        # Apply dropout
        hidden_states = self.dropout(hidden_states)

        # Project to start and end representations
        start_repr = torch.relu(self.start_projection(hidden_states))
        end_repr = torch.relu(self.end_projection(hidden_states))

        # Apply dropout to projections
        start_repr = self.dropout(start_repr)
        end_repr = self.dropout(end_repr)

        # Compute biaffine scores
        span_scores = self.biaffine(start_repr, end_repr)

        # Mask invalid spans (where start > end or outside sequence)
        batch_size, seq_len = attention_mask.size()

        # Create mask for valid spans
        span_mask = torch.zeros(batch_size, seq_len,
                                seq_len, device=hidden_states.device)
        for i in range(seq_len):
            for j in range(i, seq_len):
                span_mask[:, i, j] = attention_mask[:, i] * \
                    attention_mask[:, j]

        # Apply mask to scores
        span_mask = span_mask.unsqueeze(-1)  # [batch, seq_len, seq_len, 1]
        span_scores = span_scores * span_mask + (1 - span_mask) * (-1e9)

        return span_scores


class SpanBiaffineNER(nn.Module):
    """
    Span-based NER model with biaffine attention
    """

    def __init__(self, num_labels: int, config: Config):
        super(SpanBiaffineNER, self).__init__()
        self.config = config
        self.num_labels = num_labels

        # BERT encoder
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Dropout layers
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.spatial_dropout = SpatialDropout(config.spatial_dropout)

        # BiLSTM for contextual representation
        self.lstm = nn.LSTM(
            input_size=768,  # BERT hidden size
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )

        # Span classifier with biaffine attention
        self.span_classifier = SpanClassifier(
            hidden_dim=512,  # BiLSTM output size
            num_labels=num_labels,
            dropout=0.1
        )

        # Loss function - use combined loss for better performance
        loss_type = getattr(config, 'span_loss_type', 'combined')
        if loss_type == 'focal':
            self.criterion = SpanFocalLoss(alpha=0.25, gamma=2.0)
        elif loss_type == 'dice':
            self.criterion = SpanDiceLoss(smooth=1.0)
        elif loss_type == 'boundary':
            self.criterion = SpanBoundaryLoss(boundary_weight=2.0)
        elif loss_type == 'label_smoothing':
            self.criterion = SpanLabelSmoothingLoss(smoothing=0.1)
        elif loss_type == 'combined':
            self.criterion = SpanCombinedLoss(
                focal_weight=0.4,
                dice_weight=0.3,
                boundary_weight=0.3
            )
        else:
            # Fallback to cross entropy
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
            self.use_custom_loss = False

        # Freeze BERT layers if specified
        self._freeze_bert_layers(config.freeze_bert_layers)

    def _freeze_bert_layers(self, num_layers_to_freeze):
        """Freeze BERT layers"""
        if num_layers_to_freeze <= 0:
            return

        # Always freeze embedding layer
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze first n encoder layers
        for layer_idx in range(min(num_layers_to_freeze, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, span_labels=None, labels=None, inputs_embeds=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            span_labels: [batch_size, seq_len, seq_len] - span labels
            labels: alias for span_labels (for compatibility with FreeLB)
            inputs_embeds: [batch_size, seq_len, hidden_dim] - for adversarial training

        Returns:
            loss or span_scores
        """
        # Handle labels parameter (alias for span_labels)
        if labels is not None and span_labels is None:
            span_labels = labels
        # BERT encoding
        if inputs_embeds is not None:
            if self.training and self.embedding_dropout.p > 0:
                pass  # FreeLB handles embedding perturbation
            outputs = self.bert(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask)
        elif input_ids is not None:
            if self.training and self.embedding_dropout.p > 0:
                current_embeddings = self.bert.embeddings.word_embeddings(
                    input_ids)
                current_embeddings = self.embedding_dropout(current_embeddings)
                outputs = self.bert(
                    inputs_embeds=current_embeddings, attention_mask=attention_mask)
            else:
                outputs = self.bert(input_ids=input_ids,
                                    attention_mask=attention_mask)
        else:
            raise ValueError("Must provide either input_ids or inputs_embeds")

        bert_output = outputs.last_hidden_state  # [batch, seq_len, 768]

        # Apply spatial dropout
        bert_output = self.spatial_dropout(bert_output)

        # BiLSTM
        lstm_output, _ = self.lstm(bert_output)  # [batch, seq_len, 512]

        # Span classification
        # [batch, seq_len, seq_len, num_labels]
        span_scores = self.span_classifier(lstm_output, attention_mask)

        if span_labels is not None:
            # Training mode - compute loss
            loss = self.compute_span_loss(
                span_scores, span_labels, attention_mask)
            return loss
        else:
            # Inference mode - return scores
            return span_scores

    def compute_span_loss(self, span_scores, span_labels, attention_mask):
        """
        Compute span-based loss

        Args:
            span_scores: [batch_size, seq_len, seq_len, num_labels]
            span_labels: [batch_size, seq_len, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        # Use custom loss functions that handle masking internally
        if hasattr(self.criterion, 'forward') and len(self.criterion.forward.__code__.co_varnames) > 3:
            # Custom loss function that accepts attention_mask
            loss = self.criterion(span_scores, span_labels, attention_mask)
        else:
            # Fallback to original cross entropy approach
            batch_size, seq_len, _, num_labels = span_scores.size()

            # Create mask for valid spans
            span_mask = torch.zeros(batch_size, seq_len,
                                    seq_len, device=span_scores.device)
            for i in range(seq_len):
                for j in range(i, seq_len):
                    span_mask[:, i, j] = attention_mask[:, i] * \
                        attention_mask[:, j]

            # Flatten for loss computation
            span_scores_flat = span_scores.contiguous().view(-1, num_labels)
            span_labels_flat = span_labels.contiguous().view(-1)
            span_mask_flat = span_mask.contiguous().view(-1)

            # Set invalid spans to ignore_index
            span_labels_flat = span_labels_flat * \
                span_mask_flat.long() + (1 - span_mask_flat.long()) * (-100)

            # Compute loss
            loss = self.criterion(span_scores_flat, span_labels_flat)

        return loss

    def decode_spans(self, span_scores, attention_mask, threshold=0.5):
        """
        Decode spans from scores

        Args:
            span_scores: [batch_size, seq_len, seq_len, num_labels]
            attention_mask: [batch_size, seq_len]
            threshold: confidence threshold for span extraction

        Returns:
            List of predicted spans for each example in batch
        """
        batch_size, seq_len, _, num_labels = span_scores.size()
        batch_spans = []

        for b in range(batch_size):
            spans = []
            seq_length = attention_mask[b].sum().item()

            # Get probabilities
            # [seq_len, seq_len, num_labels]
            probs = F.softmax(span_scores[b], dim=-1)

            # Extract spans
            for start in range(seq_length):
                for end in range(start, seq_length):
                    # Get best label for this span
                    best_label_idx = torch.argmax(probs[start, end]).item()
                    best_prob = probs[start, end, best_label_idx].item()

                    # Skip "O" label (assuming it's index 0) and low confidence spans
                    if best_label_idx > 0 and best_prob > threshold:
                        spans.append({
                            'start': start,
                            'end': end,
                            'label': best_label_idx,
                            'confidence': best_prob
                        })

            batch_spans.append(spans)

        return batch_spans


class SpatialDropout(nn.Module):
    """Spatial dropout implementation"""

    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        if not self.training or self.drop_prob == 0:
            return inputs

        batch_size, seq_len, hidden_dim = inputs.shape
        mask = torch.rand(batch_size, 1, hidden_dim,
                          device=inputs.device) > self.drop_prob
        mask = mask.float() / (1 - self.drop_prob)
        return inputs * mask


class HybridNER(nn.Module):
    """
    Hybrid model combining sequence labeling and span-based approaches
    """

    def __init__(self, num_labels: int, config: Config):
        super(HybridNER, self).__init__()
        self.config = config
        self.num_labels = num_labels

        # Shared BERT encoder
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Dropout layers
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.spatial_dropout = SpatialDropout(config.spatial_dropout)

        # Shared BiLSTM
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )

        # Sequence labeling head (original CRF-based)
        from TorchCRF import CRF
        self.sequence_classifier = nn.Linear(512, num_labels)
        self.crf = CRF(num_labels=num_labels)

        # Span-based head (use entity types count, not full BIOES labels)
        span_num_labels = len(config.label_map.entity_types)  # Includes "O"
        self.span_classifier = SpanClassifier(
            hidden_dim=512,
            num_labels=span_num_labels,
            dropout=0.1
        )

        # Loss weights
        self.sequence_weight = getattr(config, 'sequence_loss_weight', 0.5)
        self.span_weight = getattr(config, 'span_loss_weight', 0.5)

        # Loss functions
        loss_type = getattr(config, 'span_loss_type', 'combined')
        if loss_type == 'focal':
            self.span_criterion = SpanFocalLoss(alpha=0.25, gamma=2.0)
        elif loss_type == 'dice':
            self.span_criterion = SpanDiceLoss(smooth=1.0)
        elif loss_type == 'boundary':
            self.span_criterion = SpanBoundaryLoss(boundary_weight=2.0)
        elif loss_type == 'label_smoothing':
            self.span_criterion = SpanLabelSmoothingLoss(smoothing=0.1)
        elif loss_type == 'combined':
            self.span_criterion = SpanCombinedLoss(
                focal_weight=0.4,
                dice_weight=0.3,
                boundary_weight=0.3
            )
        else:
            self.span_criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Freeze BERT layers
        self._freeze_bert_layers(config.freeze_bert_layers)

    def _freeze_bert_layers(self, num_layers_to_freeze):
        """Freeze BERT layers"""
        if num_layers_to_freeze <= 0:
            return

        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        for layer_idx in range(min(num_layers_to_freeze, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, sequence_labels=None,
                span_labels=None, labels=None, inputs_embeds=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            sequence_labels: [batch_size, seq_len] - BIO/BIOES labels
            span_labels: [batch_size, seq_len, seq_len] - span labels
            labels: dict or tensor (for compatibility with FreeLB)
            inputs_embeds: [batch_size, seq_len, hidden_dim]
        """
        # Handle labels parameter (for FreeLB compatibility)
        if labels is not None:
            if isinstance(labels, dict):
                # FreeLB passes a dict with both label types
                if 'sequence_labels' in labels and sequence_labels is None:
                    sequence_labels = labels['sequence_labels']
                if 'span_labels' in labels and span_labels is None:
                    span_labels = labels['span_labels']
            else:
                # If labels is a tensor, assume it's span_labels
                if span_labels is None:
                    span_labels = labels
        # BERT encoding
        if inputs_embeds is not None:
            outputs = self.bert(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask)
        elif input_ids is not None:
            if self.training and self.embedding_dropout.p > 0:
                current_embeddings = self.bert.embeddings.word_embeddings(
                    input_ids)
                current_embeddings = self.embedding_dropout(current_embeddings)
                outputs = self.bert(
                    inputs_embeds=current_embeddings, attention_mask=attention_mask)
            else:
                outputs = self.bert(input_ids=input_ids,
                                    attention_mask=attention_mask)
        else:
            raise ValueError("Must provide either input_ids or inputs_embeds")

        bert_output = outputs.last_hidden_state
        bert_output = self.spatial_dropout(bert_output)

        # Shared BiLSTM
        lstm_output, _ = self.lstm(bert_output)

        # Sequence labeling
        sequence_logits = self.sequence_classifier(lstm_output)

        # Span classification
        span_scores = self.span_classifier(lstm_output, attention_mask)

        if sequence_labels is not None or span_labels is not None:
            # Training mode
            total_loss = 0

            if sequence_labels is not None:
                # CRF loss for sequence labeling
                sequence_loss = - \
                    self.crf(sequence_logits, sequence_labels,
                             mask=attention_mask.bool())
                total_loss += self.sequence_weight * sequence_loss.mean()

            if span_labels is not None:
                # Span loss
                span_loss = self.compute_span_loss(
                    span_scores, span_labels, attention_mask)
                total_loss += self.span_weight * span_loss

            return total_loss
        else:
            # Inference mode
            sequence_predictions = self.crf.viterbi_decode(
                sequence_logits, mask=attention_mask.bool())
            return {
                'sequence_predictions': sequence_predictions,
                'span_scores': span_scores
            }

    def compute_span_loss(self, span_scores, span_labels, attention_mask):
        """Compute span loss"""
        # Use custom loss functions that handle masking internally
        if hasattr(self.span_criterion, 'forward') and len(self.span_criterion.forward.__code__.co_varnames) > 3:
            # Custom loss function that accepts attention_mask
            loss = self.span_criterion(
                span_scores, span_labels, attention_mask)
        else:
            # Fallback to original cross entropy approach
            batch_size, seq_len, _, num_labels = span_scores.size()

            # Create mask for valid spans
            span_mask = torch.zeros(batch_size, seq_len,
                                    seq_len, device=span_scores.device)
            for i in range(seq_len):
                for j in range(i, seq_len):
                    span_mask[:, i, j] = attention_mask[:, i] * \
                        attention_mask[:, j]

            # Flatten for loss computation
            span_scores_flat = span_scores.contiguous().view(-1, num_labels)
            span_labels_flat = span_labels.contiguous().view(-1)
            span_mask_flat = span_mask.contiguous().view(-1)

            # Set invalid spans to ignore_index
            span_labels_flat = span_labels_flat * \
                span_mask_flat.long() + (1 - span_mask_flat.long()) * (-100)

            loss = self.span_criterion(span_scores_flat, span_labels_flat)
        return loss
