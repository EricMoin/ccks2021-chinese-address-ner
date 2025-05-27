import torch.nn as nn
from torch.utils.data import Dataset
from TorchCRF import CRF
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

from config import Config


class SpatialDropout(nn.Module):
    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        """
        Spatial dropout: drops entire feature maps/channels rather than individual elements
        """
        if not self.training or self.drop_prob == 0:
            return inputs

        # inputs shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = inputs.shape

        # Create mask that drops the same channels for all sequence positions
        # Shape: [batch_size, 1, hidden_dim]
        mask = torch.rand(batch_size, 1, hidden_dim,
                          device=inputs.device) > self.drop_prob
        mask = mask.float() / (1 - self.drop_prob)  # Scale to maintain expected value

        # Broadcast mask along sequence dimension without adding a new dimension
        # Final shape remains [batch_size, seq_len, hidden_dim]
        return inputs * mask


class AddressNER(nn.Module):
    def __init__(self, num_labels: int, config: Config):
        super(AddressNER, self).__init__()
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Add embedding dropout
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)

        # Add spatial dropout
        self.spatial_dropout = SpatialDropout(config.spatial_dropout)

        self.lstm = nn.LSTM(
            input_size=768,  # BERT hidden size
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(512, num_labels)
        self.crf = CRF(num_labels=num_labels)

        # Initialize losses
        self.crf_loss = CRFAwareLoss(
            self.crf, transition_penalty=config.crf_transition_penalty)
        self.focal_loss = FocalLoss(
            num_labels, alpha=config.focal_loss_alpha, gamma=config.focal_loss_gamma)
        self.hybrid_loss = HybridLoss(
            self.crf_loss,
            self.focal_loss,
            crf_weight=config.hybrid_loss_weight_crf,
            focal_weight=config.hybrid_loss_weight_focal
        )

        # Freeze BERT layers
        self._freeze_bert_layers(config.freeze_bert_layers)

    def _freeze_bert_layers(self, num_layers_to_freeze):
        """Freeze the first num_layers_to_freeze layers of BERT"""
        if num_layers_to_freeze <= 0:
            return

        # Always freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze the first n encoder layers
        for layer_idx in range(min(num_layers_to_freeze, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None):
        if inputs_embeds is not None:
            # If inputs_embeds are provided, use them directly
            # Potentially apply embedding_dropout if it makes sense here, though FreeLB perturbs after initial embedding
            if self.training and self.embedding_dropout.p > 0:
                # It's a bit unusual to apply dropout again if FreeLB already works on embeddings.
                # However, if the original intent was to dropout embeddings *before* BERT encoder,
                # this could be a place. For FreeLB, usually the attack is on the clean embeddings.
                # Let's assume FreeLB provides embeddings that are ready for BERT encoder.
                # Or apply self.embedding_dropout(inputs_embeds) if deemed necessary.
                pass
            outputs = self.bert(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask)
        elif input_ids is not None:
            # Original path: Apply embedding dropout directly to the input embeddings from input_ids
            if self.training and self.embedding_dropout.p > 0:
                # Get the embeddings
                current_embeddings = self.bert.embeddings.word_embeddings(
                    input_ids)
                # Apply dropout to embeddings
                current_embeddings = self.embedding_dropout(current_embeddings)
                # Pass modified embeddings through BERT
                outputs = self.bert(
                    inputs_embeds=current_embeddings, attention_mask=attention_mask)
            else:
                outputs = self.bert(input_ids=input_ids,
                                    attention_mask=attention_mask)
        else:
            raise ValueError(
                "Either input_ids or inputs_embeds must be provided.")

        bert_output = outputs.last_hidden_state  # [batch, seq_len, 768]

        # Apply spatial dropout to BERT output
        bert_output = self.spatial_dropout(bert_output)

        # Pass BERT output directly to LSTM
        lstm_output, _ = self.lstm(bert_output)

        logits = self.classifier(lstm_output)

        if labels is not None:
            # During training, calculate the hybrid loss
            loss = self.hybrid_loss(logits, labels, attention_mask.bool())
            return loss.mean()
        else:
            # During inference, decode the best path
            return self.crf.viterbi_decode(logits, mask=attention_mask.bool())

    def __len__(self):
        return len(self.data)


class FreeLB:
    def __init__(self, model, adv_lr=1e-1, adv_steps=3, adv_init_mag=2e-2, adv_max_norm=0.0, adv_norm_type='l2', base_model='bert'):
        self.model = model
        self.adv_lr = adv_lr
        self.adv_steps = adv_steps
        self.adv_init_mag = adv_init_mag
        self.adv_max_norm = adv_max_norm    # if 0, use adv_init_mag as the constraint
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model  # Currently unused but kept for signature consistency

    def attack(self, inputs_embeds, attention_mask, labels):
        """
        Performs FreeLB attack and accumulates gradients on model parameters.
        inputs_embeds: The original embeddings of the input (detached).
        """
        # Initialize adversarial perturbation delta on the same device as inputs_embeds
        delta = torch.zeros_like(inputs_embeds, device=inputs_embeds.device)
        if self.adv_init_mag > 0:  # Only apply uniform noise if adv_init_mag is positive
            delta.uniform_(-self.adv_init_mag, self.adv_init_mag)
        delta.requires_grad = True

        accumulated_loss_for_log = 0.0

        # Zero out model gradients once before starting the adversarial steps.
        # This ensures that the gradients accumulated are solely from the adversarial perturbations.
        self.model.zero_grad()

        for i in range(self.adv_steps):
            perturbed_embeds = inputs_embeds + delta

            loss_adv = self.model(
                inputs_embeds=perturbed_embeds, attention_mask=attention_mask, labels=labels)
            accumulated_loss_for_log += loss_adv.item()  # Log the raw loss for this step

            # Normalize loss for backward pass to average gradients across steps
            # as specified in FreeLB algorithm (gradient is mean of grads from K steps)
            loss_adv_normalized = loss_adv / self.adv_steps

            # Zero out delta gradients for its own update step
            if delta.grad is not None:
                delta.grad.data.zero_()

            # Accumulates gradients in self.model.parameters from this adversarial step
            loss_adv_normalized.backward()

            # Update perturbation delta
            if delta.grad is None:  # Should not happen if loss_adv depends on delta and model has trainable params
                # If it does, means no gradient flowed to delta, perhaps an issue with model structure or adv_steps=0
                if self.adv_steps > 0:  # only break if we expected steps
                    logger.warning(
                        "Delta gradient is None during FreeLB attack, breaking early.")
                break

            # Normalize the gradient of delta before applying the learning rate
            # Flatten to calculate norm across embedding dimensions, then unsqueeze to match delta shape
            flat_delta_grad = delta.grad.data.flatten(1)
            if self.adv_norm_type == 'l2':
                delta_grad_norm = torch.norm(
                    flat_delta_grad, p=2, dim=1, keepdim=True)
            elif self.adv_norm_type == 'linf':  # Technically, for L-inf, update is often just sign based
                # but FreeLB paper uses g_t / ||g_t|| for L2, let's adapt for L-inf norm constraint
                delta_grad_norm = torch.norm(
                    flat_delta_grad, p=float('inf'), dim=1, keepdim=True)
            else:
                raise ValueError("adv_norm_type must be 'l2' or 'linf'")

            # Unsqueeze grad_norm to match dimensions of delta [B, S, H] from [B, 1]
            for _ in range(len(inputs_embeds.shape) - len(delta_grad_norm.shape)):
                delta_grad_norm = delta_grad_norm.unsqueeze(-1)

            # Update delta, adding small epsilon to denominator for stability
            delta.data = delta.data + self.adv_lr * \
                (delta.grad.data / (delta_grad_norm + 1e-12))

            # Project delta back to the norm ball
            # Use adv_max_norm if specified, otherwise use adv_init_mag as the constraint radius
            effective_constraint_norm = self.adv_max_norm if self.adv_max_norm > 0 else self.adv_init_mag

            if effective_constraint_norm > 0:  # Only project if a positive constraint is given
                flat_delta = delta.data.flatten(1)
                if self.adv_norm_type == 'l2':
                    current_delta_norm = torch.norm(
                        flat_delta, p=2, dim=1, keepdim=True)
                elif self.adv_norm_type == 'linf':
                    current_delta_norm = torch.norm(
                        flat_delta, p=float('inf'), dim=1, keepdim=True)

                for _ in range(len(inputs_embeds.shape) - len(current_delta_norm.shape)):
                    current_delta_norm = current_delta_norm.unsqueeze(-1)

                # Calculate clipping coefficient: min(1, constraint_norm / current_norm)
                clip_coef = (effective_constraint_norm /
                             (current_delta_norm + 1e-12))
                clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
                delta.data = delta.data * clip_coef

        # Gradients are now accumulated in self.model.parameters().
        # The trainer will call optimizer.step() using these gradients.
        return accumulated_loss_for_log / self.adv_steps if self.adv_steps > 0 else 0


class CRFAwareLoss(nn.Module):
    def __init__(self, crf: CRF, transition_penalty=0.175):
        super().__init__()
        self.crf = crf
        self.transition_probs = F.softmax(
            self.crf.trans_matrix, dim=1).detach()
        self.transition_penalty = transition_penalty
        # 获取 CRF 的转移矩阵（形状: [num_tags, num_tags]）
        self.transition_matrix = crf.trans_matrix.detach()

    def forward(self, emissions, tags, mask):
        # 常规 CRF 负对数似然损失
        crf_loss = -self.crf(emissions, tags, mask=mask)

        # 计算标签转移的不合理性惩罚
        batch_size, seq_len = tags.shape
        penalty = 0.0
        for i in range(seq_len - 1):
            current_tags = tags[:, i].to(self.transition_probs.device)
            next_tags = tags[:, i + 1].to(self.transition_probs.device)
            # 对每对连续标签计算转移概率的负值（越小越合理）
            invalid_transitions = - \
                torch.log(
                    self.transition_probs[current_tags, next_tags] + 1e-8)
            penalty += invalid_transitions.mean()

        # 总损失 = CRF 损失 + 惩罚项
        total_loss = crf_loss + self.transition_penalty * penalty
        return total_loss


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        """
        计算Focal Loss

        Args:
            logits: 模型输出 [batch_size, seq_len, num_classes]
            targets: 目标标签 [batch_size, seq_len]
            mask: 有效位置掩码 [batch_size, seq_len]

        Returns:
            loss: 标量损失值
        """
        # 将logits转换为概率
        probs = F.softmax(logits, dim=-1)

        # 获取目标类别的概率
        batch_size, seq_len, _ = logits.shape

        # 创建one-hot编码
        target_one_hot = F.one_hot(targets, self.num_classes).float()

        # 计算每个位置的概率
        pt = (probs * target_one_hot).sum(dim=-1)  # [batch_size, seq_len]

        # 计算focal loss
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.ones_like(pt) * self.alpha
        alpha_weight = torch.where(
            targets > 0, alpha_weight, 1 - alpha_weight)  # 对背景类使用1-alpha

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(
            logits.view(-1, self.num_classes),
            targets.view(-1),
            reduction='none'
        ).view(batch_size, seq_len)

        # 应用权重
        loss = alpha_weight * focal_weight * ce_loss

        # 应用掩码并执行reduction
        loss = loss * mask.float()

        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-10)
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class HybridLoss(nn.Module):
    def __init__(self, crf_loss, focal_loss, crf_weight=0.5, focal_weight=0.5):
        super(HybridLoss, self).__init__()
        self.crf_loss = crf_loss
        self.focal_loss = focal_loss
        self.crf_weight = crf_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets, mask):
        """
        计算混合损失

        Args:
            logits: 模型输出 [batch_size, seq_len, num_classes]
            targets: 目标标签 [batch_size, seq_len]
            mask: 有效位置掩码 [batch_size, seq_len]

        Returns:
            loss: 标量损失值
        """
        crf_loss_val = self.crf_loss(logits, targets, mask)
        focal_loss_val = self.focal_loss(logits, targets, mask)

        return self.crf_weight * crf_loss_val + self.focal_weight * focal_loss_val
