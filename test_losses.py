#!/usr/bin/env python3
"""
Test script to compare different loss functions for span-based NER
"""

import torch
import numpy as np
from span_losses import (
    SpanFocalLoss, SpanDiceLoss, SpanBoundaryLoss,
    SpanCombinedLoss, SpanLabelSmoothingLoss
)
import torch.nn.functional as F


def create_test_data():
    """Create synthetic test data"""
    batch_size, seq_len, num_labels = 2, 10, 22  # 21 entity types + 1 for O

    # Create random span scores
    span_scores = torch.randn(batch_size, seq_len, seq_len, num_labels)

    # Create attention mask
    attention_mask = torch.ones(batch_size, seq_len)

    # Create span labels with some entities
    span_labels = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.long)

    # Add some positive examples
    span_labels[0, 1, 3] = 5   # Entity type 5 from position 1 to 3
    span_labels[0, 5, 5] = 10  # Entity type 10 at position 5
    span_labels[1, 2, 4] = 3   # Entity type 3 from position 2 to 4
    span_labels[1, 7, 9] = 15  # Entity type 15 from position 7 to 9

    return span_scores, span_labels, attention_mask


def test_loss_functions():
    """Test different loss functions"""

    print("🧪 Testing Span Loss Functions")
    print("=" * 50)

    # Create test data
    span_scores, span_labels, attention_mask = create_test_data()

    # Initialize loss functions
    loss_functions = {
        'Cross Entropy': torch.nn.CrossEntropyLoss(),
        'Focal Loss': SpanFocalLoss(alpha=0.25, gamma=2.0),
        'Dice Loss': SpanDiceLoss(smooth=1.0),
        'Boundary Loss': SpanBoundaryLoss(boundary_weight=2.0),
        'Label Smoothing': SpanLabelSmoothingLoss(smoothing=0.1),
        'Combined Loss': SpanCombinedLoss(
            focal_weight=0.4,
            dice_weight=0.3,
            boundary_weight=0.3
        )
    }

    results = {}

    for name, loss_fn in loss_functions.items():
        try:
            if name == 'Cross Entropy':
                # Handle cross entropy separately (needs flattening)
                batch_size, seq_len, _, num_labels = span_scores.size()

                # Create mask for valid spans
                span_mask = torch.zeros(batch_size, seq_len, seq_len)
                for i in range(seq_len):
                    for j in range(i, seq_len):
                        span_mask[:, i, j] = attention_mask[:, i] * \
                            attention_mask[:, j]

                # Flatten tensors
                span_scores_flat = span_scores.contiguous().view(-1, num_labels)
                span_labels_flat = span_labels.contiguous().view(-1)
                span_mask_flat = span_mask.contiguous().view(-1)

                # Set invalid spans to ignore_index
                span_labels_flat = span_labels_flat * \
                    span_mask_flat.long() + (1 - span_mask_flat.long()) * (-100)

                loss = loss_fn(span_scores_flat, span_labels_flat)
            else:
                # Custom loss functions
                loss = loss_fn(span_scores, span_labels, attention_mask)

            results[name] = loss.item()
            print(f"✅ {name:18}: {loss.item():.4f}")

        except Exception as e:
            print(f"❌ {name:18}: Error - {str(e)}")
            results[name] = float('inf')

    print("\n📊 Loss Comparison")
    print("=" * 50)

    # Sort by loss value
    sorted_results = sorted(results.items(), key=lambda x: x[1])

    for i, (name, loss_val) in enumerate(sorted_results):
        if loss_val != float('inf'):
            status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{status} {name:18}: {loss_val:.4f}")

    print("\n💡 Recommendations:")
    print("=" * 50)
    print("1. 📈 For class imbalance: Use Focal Loss or Combined Loss")
    print("2. 🎯 For boundary accuracy: Use Boundary Loss or Combined Loss")
    print("3. 🔄 For generalization: Use Label Smoothing or Combined Loss")
    print("4. ⚖️  For balanced approach: Use Combined Loss (recommended)")
    print("5. 🚀 For fast training: Use Cross Entropy or Focal Loss")


def test_gradient_magnitudes():
    """Test gradient magnitudes for different loss functions"""

    print("\n🔬 Testing Gradient Magnitudes")
    print("=" * 50)

    # Create test data with gradients enabled
    span_scores, span_labels, attention_mask = create_test_data()

    loss_functions = {
        'Focal Loss': SpanFocalLoss(alpha=0.25, gamma=2.0),
        'Combined Loss': SpanCombinedLoss(),
    }

    for name, loss_fn in loss_functions.items():
        # Create a simple model
        test_scores = span_scores.clone().requires_grad_(True)

        # Calculate loss
        loss = loss_fn(test_scores, span_labels, attention_mask)

        # Backward pass
        loss.backward()

        # Check gradient magnitude
        grad_norm = test_scores.grad.norm().item()

        print(f"{name:18}: Loss={loss.item():.4f}, Grad Norm={grad_norm:.4f}")


if __name__ == "__main__":
    test_loss_functions()
    test_gradient_magnitudes()

    print("\n🎉 Testing completed!")
    print("💡 Consider using 'combined' loss type for best results.")
