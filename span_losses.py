import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SpanFocalLoss(nn.Module):
    """
    Focal Loss for span-based NER
    解决类别不平衡问题，更关注难样本
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -100):
        super(SpanFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, span_scores: torch.Tensor, span_labels: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            span_scores: [batch_size, seq_len, seq_len, num_labels]
            span_labels: [batch_size, seq_len, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        batch_size, seq_len, _, num_labels = span_scores.size()

        # Create span mask
        span_mask = torch.zeros(batch_size, seq_len,
                                seq_len, device=span_scores.device)
        for i in range(seq_len):
            for j in range(i, seq_len):
                span_mask[:, i, j] = attention_mask[:, i] * \
                    attention_mask[:, j]

        # Flatten tensors
        span_scores_flat = span_scores.contiguous().view(-1, num_labels)
        span_labels_flat = span_labels.contiguous().view(-1)
        span_mask_flat = span_mask.contiguous().view(-1)

        # Apply mask to labels
        valid_positions = span_mask_flat.bool()
        valid_scores = span_scores_flat[valid_positions]
        valid_labels = span_labels_flat[valid_positions]

        # Calculate probabilities
        log_probs = F.log_softmax(valid_scores, dim=-1)
        probs = F.softmax(valid_scores, dim=-1)

        # Get target probabilities
        target_probs = probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)

        # Calculate focal weight
        focal_weight = (1 - target_probs) ** self.gamma

        # Calculate alpha weight
        alpha_weight = torch.where(
            valid_labels == 0, 1 - self.alpha, self.alpha)

        # Calculate cross entropy
        ce_loss = F.nll_loss(log_probs, valid_labels, reduction='none')

        # Apply focal loss formula
        focal_loss = alpha_weight * focal_weight * ce_loss

        return focal_loss.mean()


class SpanDiceLoss(nn.Module):
    """
    Dice Loss for span-based NER
    更关注实体的精确边界识别
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super(SpanDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, span_scores: torch.Tensor, span_labels: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            span_scores: [batch_size, seq_len, seq_len, num_labels]
            span_labels: [batch_size, seq_len, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        batch_size, seq_len, _, num_labels = span_scores.size()

        # Get probabilities
        probs = F.softmax(span_scores, dim=-1)

        # Create one-hot encoding for labels
        labels_one_hot = F.one_hot(span_labels, num_labels).float()

        # Create span mask
        span_mask = torch.zeros(batch_size, seq_len,
                                seq_len, device=span_scores.device)
        for i in range(seq_len):
            for j in range(i, seq_len):
                span_mask[:, i, j] = attention_mask[:, i] * \
                    attention_mask[:, j]

        # Apply mask
        mask_expanded = span_mask.unsqueeze(-1).expand_as(probs)
        probs = probs * mask_expanded
        labels_one_hot = labels_one_hot * mask_expanded

        # Calculate dice coefficient for each class
        dice_losses = []
        for class_idx in range(num_labels):
            pred_class = probs[:, :, :, class_idx]
            true_class = labels_one_hot[:, :, :, class_idx]

            intersection = (pred_class * true_class).sum()
            union = pred_class.sum() + true_class.sum()

            dice_coeff = (2 * intersection + self.smooth) / \
                (union + self.smooth)
            dice_loss = 1 - dice_coeff

            # Weight non-O classes more heavily
            weight = 1.0 if class_idx == 0 else 2.0
            dice_losses.append(weight * dice_loss)

        return torch.stack(dice_losses).mean()


class SpanContrastiveLoss(nn.Module):
    """
    Contrastive Loss for span-based NER
    通过对比学习提高实体边界的判别能力
    """

    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        super(SpanContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, span_scores: torch.Tensor, span_labels: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            span_scores: [batch_size, seq_len, seq_len, num_labels]
            span_labels: [batch_size, seq_len, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        batch_size, seq_len, _, num_labels = span_scores.size()

        # Create span mask
        span_mask = torch.zeros(batch_size, seq_len,
                                seq_len, device=span_scores.device)
        for i in range(seq_len):
            for j in range(i, seq_len):
                span_mask[:, i, j] = attention_mask[:, i] * \
                    attention_mask[:, j]

        # Get valid spans
        valid_mask = span_mask.bool()

        # Extract features and labels for valid spans
        # [num_valid_spans, num_labels]
        span_features = span_scores[valid_mask]
        span_targets = span_labels[valid_mask]   # [num_valid_spans]

        if len(span_features) == 0:
            return torch.tensor(0.0, device=span_scores.device, requires_grad=True)

        # Normalize features
        span_features = F.normalize(span_features, p=2, dim=1)

        # Calculate similarity matrix
        similarity_matrix = torch.matmul(
            span_features, span_features.T) / self.temperature

        # Create positive and negative masks
        label_matrix = span_targets.unsqueeze(0) == span_targets.unsqueeze(1)
        positive_mask = label_matrix & (
            span_targets.unsqueeze(1) != 0)  # Exclude O class
        negative_mask = ~label_matrix

        # Calculate contrastive loss
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=span_scores.device, requires_grad=True)

        # Positive pairs
        positive_similarities = similarity_matrix[positive_mask]

        # Negative pairs
        negative_similarities = similarity_matrix[negative_mask]

        # Calculate loss
        positive_loss = -torch.log(torch.sigmoid(positive_similarities)).mean()
        negative_loss = - \
            torch.log(torch.sigmoid(-negative_similarities + self.margin)).mean()

        return positive_loss + negative_loss


class SpanBoundaryLoss(nn.Module):
    """
    Boundary-aware Loss for span-based NER
    专门优化实体边界的准确性
    """

    def __init__(self, boundary_weight: float = 2.0):
        super(SpanBoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, span_scores: torch.Tensor, span_labels: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            span_scores: [batch_size, seq_len, seq_len, num_labels]
            span_labels: [batch_size, seq_len, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        batch_size, seq_len, _, num_labels = span_scores.size()

        # Create span mask
        span_mask = torch.zeros(batch_size, seq_len,
                                seq_len, device=span_scores.device)
        for i in range(seq_len):
            for j in range(i, seq_len):
                span_mask[:, i, j] = attention_mask[:, i] * \
                    attention_mask[:, j]

        # Flatten tensors
        span_scores_flat = span_scores.contiguous().view(-1, num_labels)
        span_labels_flat = span_labels.contiguous().view(-1)
        span_mask_flat = span_mask.contiguous().view(-1)

        # Calculate base cross entropy loss
        ce_losses = self.ce_loss(span_scores_flat, span_labels_flat)

        # Create boundary weight mask
        boundary_weights = torch.ones_like(span_labels_flat, dtype=torch.float)

        # Identify boundary spans (start and end positions of entities)
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(i, seq_len):
                    flat_idx = b * seq_len * seq_len + i * seq_len + j
                    if span_labels_flat[flat_idx] != 0:  # Non-O label
                        # Check if this is a boundary span
                        is_boundary = (i == 0 or j == seq_len - 1 or
                                       (i > 0 and span_labels[b, i-1, j] == 0) or
                                       (j < seq_len - 1 and span_labels[b, i, j+1] == 0))
                        if is_boundary:
                            boundary_weights[flat_idx] = self.boundary_weight

        # Apply weights and mask
        weighted_losses = ce_losses * boundary_weights * span_mask_flat

        return weighted_losses.sum() / (span_mask_flat.sum() + 1e-8)


class SpanCombinedLoss(nn.Module):
    """
    Combined Loss for span-based NER
    结合多种损失函数的优势
    """

    def __init__(self,
                 focal_weight: float = 0.4,
                 dice_weight: float = 0.3,
                 boundary_weight: float = 0.3,
                 **kwargs):
        super(SpanCombinedLoss, self).__init__()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

        self.focal_loss = SpanFocalLoss(**kwargs)
        self.dice_loss = SpanDiceLoss(**kwargs)
        self.boundary_loss = SpanBoundaryLoss(**kwargs)

    def forward(self, span_scores: torch.Tensor, span_labels: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            span_scores: [batch_size, seq_len, seq_len, num_labels]
            span_labels: [batch_size, seq_len, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        focal_loss = self.focal_loss(span_scores, span_labels, attention_mask)
        dice_loss = self.dice_loss(span_scores, span_labels, attention_mask)
        boundary_loss = self.boundary_loss(
            span_scores, span_labels, attention_mask)

        total_loss = (self.focal_weight * focal_loss +
                      self.dice_weight * dice_loss +
                      self.boundary_weight * boundary_loss)

        return total_loss


class SpanLabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for span-based NER
    通过标签平滑减少过拟合，提高泛化能力
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super(SpanLabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, span_scores: torch.Tensor, span_labels: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            span_scores: [batch_size, seq_len, seq_len, num_labels]
            span_labels: [batch_size, seq_len, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        batch_size, seq_len, _, num_labels = span_scores.size()

        # Create span mask
        span_mask = torch.zeros(batch_size, seq_len,
                                seq_len, device=span_scores.device)
        for i in range(seq_len):
            for j in range(i, seq_len):
                span_mask[:, i, j] = attention_mask[:, i] * \
                    attention_mask[:, j]

        # Flatten tensors
        span_scores_flat = span_scores.contiguous().view(-1, num_labels)
        span_labels_flat = span_labels.contiguous().view(-1)
        span_mask_flat = span_mask.contiguous().view(-1)

        # Apply mask
        valid_positions = span_mask_flat.bool()
        valid_scores = span_scores_flat[valid_positions]
        valid_labels = span_labels_flat[valid_positions]

        # Create smoothed labels
        num_valid = len(valid_labels)
        smoothed_labels = torch.full_like(
            valid_scores, self.smoothing / (num_labels - 1))
        smoothed_labels.scatter_(
            1, valid_labels.unsqueeze(1), 1.0 - self.smoothing)

        # Calculate loss
        log_probs = F.log_softmax(valid_scores, dim=-1)
        loss = -(smoothed_labels * log_probs).sum(dim=-1).mean()

        return loss
