import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict
import logging

from config import Config

logger = logging.getLogger(__name__)


class SpatialDropout(nn.Module):
    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        """
        空间Dropout：丢弃整个特征图/通道而不是单个元素
        """
        if not self.training or self.drop_prob == 0:
            return inputs

        # 输入形状: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = inputs.shape

        # 创建掩码，对所有序列位置丢弃相同的通道
        # 形状: [batch_size, 1, hidden_dim]
        mask = torch.rand(batch_size, 1, hidden_dim,
                          device=inputs.device) > self.drop_prob
        mask = mask.float() / (1 - self.drop_prob)  # 缩放以保持期望值

        # 沿序列维度广播掩码，不添加新维度
        # 最终形状保持为 [batch_size, seq_len, hidden_dim]
        return inputs * mask


class FreeLB:
    def __init__(self, model, adv_lr=1e-1, adv_steps=3, adv_init_mag=2e-2, adv_max_norm=0.0, adv_norm_type='l2', base_model='bert'):
        self.model = model
        self.adv_lr = adv_lr
        self.adv_steps = adv_steps
        self.adv_init_mag = adv_init_mag
        self.adv_max_norm = adv_max_norm    # 如果为0，使用adv_init_mag作为约束
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model  # 现在正式使用这个参数

        # 根据base_model设置不同的攻击策略
        if base_model == 'bert':
            self.attack_embedding_layers = True
            self.attack_encoder_layers = True
        elif base_model == 'embed':
            self.attack_embedding_layers = True
            self.attack_encoder_layers = False
        else:
            # 默认攻击embedding层
            self.attack_embedding_layers = True
            self.attack_encoder_layers = False

    def attack_span(self, inputs_embeds, attention_mask, span_labels):
        """
        为span模型执行FreeLB攻击
        """
        # 在与inputs_embeds相同的设备上初始化对抗扰动delta
        delta = torch.zeros_like(inputs_embeds, device=inputs_embeds.device)
        if self.adv_init_mag > 0:
            delta.uniform_(-self.adv_init_mag, self.adv_init_mag)
        delta.requires_grad = True

        accumulated_loss_for_log = 0.0

        # 在开始对抗步骤之前，一次性清零模型梯度
        self.model.zero_grad()

        for i in range(self.adv_steps):
            perturbed_embeds = inputs_embeds + delta

            loss_adv = self.model(
                inputs_embeds=perturbed_embeds,
                attention_mask=attention_mask,
                span_labels=span_labels
            )
            accumulated_loss_for_log += loss_adv.item()

            # 标准化损失以进行反向传播
            loss_adv_normalized = loss_adv / self.adv_steps

            # 清零delta梯度
            if delta.grad is not None:
                delta.grad.data.zero_()

            # 在模型参数中累积梯度
            loss_adv_normalized.backward()

            # 更新扰动delta
            if delta.grad is None:
                if self.adv_steps > 0:
                    logger.warning("FreeLB span攻击期间Delta梯度为None，提前中断。")
                break

            # 标准化delta的梯度
            flat_delta_grad = delta.grad.data.flatten(1)
            if self.adv_norm_type == 'l2':
                delta_grad_norm = torch.norm(
                    flat_delta_grad, p=2, dim=1, keepdim=True)
            elif self.adv_norm_type == 'linf':
                delta_grad_norm = torch.norm(
                    flat_delta_grad, p=float('inf'), dim=1, keepdim=True)
            else:
                delta_grad_norm = torch.norm(
                    flat_delta_grad, p=2, dim=1, keepdim=True)

                delta_grad_norm = delta_grad_norm.unsqueeze(-1)

            # 更新delta
            normalized_grad = flat_delta_grad / (delta_grad_norm + 1e-8)
            delta.data += self.adv_lr * normalized_grad.view_as(delta)

            # 约束扰动大小
            if self.adv_max_norm > 0:
                delta_norm = torch.norm(delta.reshape(
                    delta.size(0), -1), dim=1, keepdim=True)
                delta_norm = delta_norm.unsqueeze(-1)
                delta.data = delta.data * torch.min(
                    torch.ones_like(delta_norm),
                    self.adv_max_norm / (delta_norm + 1e-8)
                )

        return accumulated_loss_for_log / self.adv_steps


class SpanFocalLoss(nn.Module):
    """
    专门为Span模型设计的Focal Loss
    可以处理展平后的2D输入
    """

    def __init__(self, num_classes, alpha=0.25, gamma=2.0, reduction='mean'):
        super(SpanFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward_span(self, span_scores, span_labels, valid_span_mask):
        """
        为span模型特制的前向函数

        Args:
            span_scores: [batch, seq_len, seq_len, num_classes]
            span_labels: [batch, seq_len, seq_len] 
            valid_span_mask: [batch, seq_len, seq_len]
        """
        # 展平并过滤有效span
        span_scores_flat = span_scores.reshape(-1, span_scores.size(-1))
        span_labels_flat = span_labels.reshape(-1)
        valid_mask_flat = valid_span_mask.reshape(-1)

        valid_indices = valid_mask_flat > 0
        if valid_indices.sum() > 0:
            valid_scores = span_scores_flat[valid_indices]
            valid_labels = span_labels_flat[valid_indices]

            # 计算概率
            probs = F.softmax(valid_scores, dim=-1)

            # 获取目标类别的概率
            targets_one_hot = F.one_hot(valid_labels, self.num_classes).float()
            pt = (probs * targets_one_hot).sum(dim=-1)

            # 计算focal weight
            focal_weight = (1 - pt) ** self.gamma

            # 计算alpha weight
            alpha_weight = torch.where(
                valid_labels > 0,
                torch.tensor(self.alpha, device=valid_labels.device),
                torch.tensor(1 - self.alpha, device=valid_labels.device)
            )

            # 计算交叉熵损失
            ce_loss = F.cross_entropy(
                valid_scores, valid_labels, reduction='none')

            # 应用权重
            focal_loss = alpha_weight * focal_weight * ce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
        else:
            return torch.tensor(0.0, device=span_scores.device, requires_grad=True)


class EfficientBiaffineAttention(nn.Module):
    """
    显存优化的Biaffine attention机制
    使用更少的参数和更高效的计算
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super(EfficientBiaffineAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 使用分解的biaffine变换减少参数量
        # 原始: input_dim x output_dim x input_dim
        # 分解: 2 x (input_dim x rank) + (rank x output_dim x rank)
        self.rank = min(64, input_dim // 4)  # 低秩分解

        # 分解的权重矩阵
        self.W1 = nn.Parameter(torch.randn(
            input_dim, self.rank))      # [input_dim, rank]
        # [rank, output_dim, rank]
        self.W2 = nn.Parameter(torch.randn(self.rank, output_dim, self.rank))
        # [input_dim, rank] - 修复形状
        self.W3 = nn.Parameter(torch.randn(input_dim, self.rank))

        # 线性变换
        self.U1 = nn.Linear(input_dim, output_dim, bias=False)
        self.U2 = nn.Linear(2 * input_dim, output_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3)
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

        # 低秩Biaffine计算
        # x1 -> W1: [batch, seq_len, rank]
        x1_proj = torch.matmul(x1, self.W1)

        # x2 -> W3: [batch, seq_len, rank]
        x2_proj = torch.matmul(x2, self.W3)

        # 计算biaffine scores使用爱因斯坦求和简化
        # [batch, seq_len, rank] x [rank, output_dim, rank] x [batch, seq_len, rank]
        # -> [batch, seq_len, seq_len, output_dim]
        biaffine_scores = torch.einsum(
            'bsr,rod,btr->bsto', x1_proj, self.W2, x2_proj)

        # 线性项
        u1_scores = self.U1(x1).unsqueeze(2)  # [batch, seq_len, 1, output_dim]
        u1_scores = u1_scores.expand(-1, -1, seq_len, -1)

        # 拼接特征进行线性变换 - 使用chunk操作减少临时内存
        u2_scores = torch.zeros(batch_size, seq_len, seq_len, self.output_dim,
                                device=x1.device, dtype=x1.dtype)

        # 分块处理以减少内存使用
        chunk_size = min(32, seq_len)  # 根据显存调整
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            x1_chunk = x1[:, i:end_i].unsqueeze(2).expand(-1, -1, seq_len, -1)
            x2_chunk = x2.unsqueeze(1).expand(-1, end_i - i, -1, -1)
            x_concat = torch.cat([x1_chunk, x2_chunk], dim=-1)
            u2_scores[:, i:end_i] = self.U2(x_concat)

        # 组合所有项
        scores = biaffine_scores + u1_scores + u2_scores

        return scores


class EfficientSpanClassifier(nn.Module):
    """
    显存优化的Span分类器
    """

    def __init__(self, hidden_dim: int, num_labels: int, dropout: float = 0.1, biaffine_hidden_dim: int = 512, use_biaffine: bool = True):
        super(EfficientSpanClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.use_biaffine = use_biaffine

        # 使用配置中的biaffine隐藏维度
        self.projection_dim = biaffine_hidden_dim

        if self.use_biaffine:
            # 分别的开始和结束表示投影
            self.start_projection = nn.Sequential(
                nn.Linear(hidden_dim, self.projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            self.end_projection = nn.Sequential(
                nn.Linear(hidden_dim, self.projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # Biaffine attention用于span分类
            self.biaffine = EfficientBiaffineAttention(
                self.projection_dim, num_labels)
        else:
            # 简单的线性分类器（不使用biaffine）
            self.span_classifier = nn.Sequential(
                # 拼接start和end表示
                nn.Linear(hidden_dim * 2, self.projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.projection_dim, num_labels)
            )

        # 额外的dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]

        Returns:
            span_scores: [batch_size, seq_len, seq_len, num_labels]
        """
        # 应用dropout
        hidden_states = self.dropout(hidden_states)

        batch_size, seq_len, hidden_dim = hidden_states.size()

        if self.use_biaffine:
            # 使用biaffine attention
            # 投影到开始和结束表示
            start_repr = self.start_projection(hidden_states)
            end_repr = self.end_projection(hidden_states)

            # 计算biaffine scores
            span_scores = self.biaffine(start_repr, end_repr)
        else:
            # 使用简单的线性分类器
            # 创建所有可能的span组合
            span_scores = torch.zeros(batch_size, seq_len, seq_len, self.num_labels,
                                      device=hidden_states.device, dtype=hidden_states.dtype)

            for start in range(seq_len):
                for end in range(start, seq_len):
                    # 拼接start和end的表示
                    # [batch_size, hidden_dim]
                    start_repr = hidden_states[:, start, :]
                    # [batch_size, hidden_dim]
                    end_repr = hidden_states[:, end, :]
                    # [batch_size, hidden_dim*2]
                    span_repr = torch.cat([start_repr, end_repr], dim=-1)

                    # 通过分类器得到span分数
                    span_logits = self.span_classifier(
                        span_repr)  # [batch_size, num_labels]
                    span_scores[:, start, end, :] = span_logits

        # 创建有效span的掩码（start <= end 且在序列内）
        device = hidden_states.device

        # 高效创建span掩码
        span_mask = torch.zeros(batch_size, seq_len, seq_len, device=device)

        # 使用广播创建位置掩码
        positions = torch.arange(seq_len, device=device)
        start_positions = positions.reshape(1, -1, 1)  # [1, seq_len, 1]
        end_positions = positions.reshape(1, 1, -1)    # [1, 1, seq_len]
        # [1, seq_len, seq_len]
        position_mask = (start_positions <= end_positions).float()

        # 结合attention mask
        attention_start = attention_mask.unsqueeze(2)  # [batch, seq_len, 1]
        attention_end = attention_mask.unsqueeze(1)    # [batch, 1, seq_len]
        attention_span_mask = attention_start * \
            attention_end  # [batch, seq_len, seq_len]

        # 最终的span掩码
        span_mask = position_mask * attention_span_mask

        # 应用掩码到分数
        span_mask = span_mask.unsqueeze(-1)  # [batch, seq_len, seq_len, 1]
        span_scores = span_scores * span_mask + (1 - span_mask) * (-1e9)

        return span_scores


class BertBiaffineSpanNER(nn.Module):
    """
    BERT + Biaffine + Span-based NER模型
    针对8GB显存优化
    """

    def __init__(self, num_labels: int, config: Config):
        super(BertBiaffineSpanNER, self).__init__()
        self.num_labels = num_labels
        self.config = config
        self.hidden_dim = 768

        # BERT编码器
        self.bert = AutoModel.from_pretrained(config.model_name)

        # 应用dropout
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.spatial_dropout = SpatialDropout(config.spatial_dropout)

        # 可选的BERT投影层，用于维度对齐
        if hasattr(config, 'use_bert_projection') and config.use_bert_projection:
            self.use_bert_projection = True
            self.bert_projection = nn.Linear(
                self.hidden_dim, config.hidden_dim)
        else:
            self.use_bert_projection = False

        # Span分类器
        self.classifier = EfficientSpanClassifier(
            hidden_dim=self.hidden_dim,
            num_labels=num_labels,
            dropout=config.span_dropout,
            biaffine_hidden_dim=config.biaffine_hidden_dim,
            use_biaffine=config.use_biaffine
        )

        # 初始化损失函数
        self._init_loss_function(config.span_loss_type, config)

        # 初始化权重
        self._init_weights()

        # 冻结BERT层
        self._freeze_bert_layers(config.freeze_bert_layers)

        logger.info(
            f"BertBiaffineSpanNER initialized with {num_labels} labels")
        logger.info(
            f"Model size: {sum(p.numel() for p in self.parameters()):,} parameters")

    def _init_weights(self):
        """初始化权重"""
        if self.use_bert_projection:
            nn.init.xavier_uniform_(self.bert_projection.weight)
            nn.init.zeros_(self.bert_projection.bias)

    def _init_loss_function(self, loss_type: str, config: Config):
        """初始化损失函数"""
        if loss_type == 'focal':
            self.loss_fn = SpanFocalLoss(
                num_classes=self.num_labels,
                alpha=config.focal_loss_alpha,
                gamma=config.focal_loss_gamma
            )
        elif loss_type == 'combined':
            self.loss_fn = HybridSpanLoss(
                num_classes=self.num_labels,
                alpha=config.focal_loss_alpha,
                gamma=config.focal_loss_gamma
            )
        else:
            # 默认使用交叉熵
            self.loss_fn = None

        logger.info(f"Loss function initialized: {loss_type}")
        if loss_type in ['focal', 'combined']:
            logger.info(
                f"Focal loss parameters: alpha={config.focal_loss_alpha}, gamma={config.focal_loss_gamma}")

    def _freeze_bert_layers(self, num_layers_to_freeze: int):
        """冻结BERT的前num_layers_to_freeze层"""
        if num_layers_to_freeze <= 0:
            return

        # 总是冻结嵌入层
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # 冻结前n个编码器层
        for layer_idx in range(min(num_layers_to_freeze, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, span_labels=None, inputs_embeds=None):
        """
        前向传播

        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            span_labels: span标签矩阵 [batch_size, seq_len, seq_len]，可选
            inputs_embeds: 输入嵌入，可选

        Returns:
            训练时返回loss，推理时返回span预测列表
        """
        # 获取BERT输出
        if inputs_embeds is not None:
            outputs = self.bert(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state

        # 应用BERT投影（如果启用）
        if self.use_bert_projection:
            sequence_output = self.bert_projection(sequence_output)

        # 分类器计算span分数
        span_scores = self.classifier(sequence_output, attention_mask)

        if span_labels is not None:
            # 训练模式：计算损失
            loss = self.compute_span_loss(
                span_scores, span_labels, attention_mask)
            return loss
        else:
            # 推理模式：返回span预测，使用config中的threshold
            return self.decode_spans(span_scores, attention_mask, threshold=self.config.span_threshold)

    def compute_span_loss(self, span_scores, span_labels, attention_mask):
        """计算span损失"""
        batch_size, seq_len, _, num_labels = span_scores.size()

        # 创建有效span的掩码
        device = span_scores.device
        positions = torch.arange(seq_len, device=device)
        start_positions = positions.reshape(1, -1, 1)
        end_positions = positions.reshape(1, 1, -1)
        position_mask = (start_positions <= end_positions).float()

        attention_start = attention_mask.unsqueeze(2).float()
        attention_end = attention_mask.unsqueeze(1).float()
        attention_span_mask = attention_start * attention_end

        valid_span_mask = position_mask * attention_span_mask

        # 展平为[batch * seq_len * seq_len, num_labels]
        span_scores_flat = span_scores.reshape(-1, num_labels)
        span_labels_flat = span_labels.reshape(-1)
        valid_mask_flat = valid_span_mask.reshape(-1)

        # 只计算有效span的损失
        valid_indices = valid_mask_flat > 0
        valid_count = valid_indices.sum().item()

        if valid_count > 0:
            valid_scores = span_scores_flat[valid_indices]
            valid_labels = span_labels_flat[valid_indices]

            # 计算类别权重 - 使用更温和的平衡策略
            label_counts = torch.bincount(
                valid_labels, minlength=num_labels)
            total_samples = valid_labels.numel()

            # 使用温和的权重平衡
            class_weights = torch.ones(num_labels, device=device)
            for i in range(num_labels):
                if label_counts[i] > 0:
                    if i == 0:  # O标签，给予较小的权重
                        class_weights[i] = 0.5  # 不要太小，避免梯度问题
                    else:  # 实体标签，给予较大的权重
                        # 计算合理的权重：实体标签应该得到更多关注
                        weight = min(5.0, total_samples /
                                     (label_counts[i] * 5))
                        class_weights[i] = max(2.0, weight)  # 确保最小权重为2
                else:
                    class_weights[i] = 1.0

            # 使用权重计算损失
            criterion_weighted = nn.CrossEntropyLoss(
                weight=class_weights, reduction='mean')
            loss = criterion_weighted(valid_scores, valid_labels)

            # 确保损失是有意义的数值
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0:
                # fallback到简单的交叉熵
                loss = F.cross_entropy(
                    valid_scores, valid_labels, reduction='mean')

        else:
            # 创建一个小的、有梯度的损失
            loss = span_scores.mean() * 0.0 + 1e-8
            loss.requires_grad_(True)

        # 确保损失是标量且有梯度
        if loss.dim() > 0:
            loss = loss.mean()

        # 应用span损失权重
        loss = loss * self.config.span_loss_weight

        return loss

    def decode_spans(self, span_scores, attention_mask, threshold=0.1):
        """
        标准的span解码 - 遵循span-based NER最佳实践

        Args:
            span_scores: [batch_size, seq_len, seq_len, num_labels]
            attention_mask: [batch_size, seq_len]
            threshold: 置信度阈值（只使用一个阈值）
        """
        batch_size, seq_len, _, num_labels = span_scores.size()
        predictions = []

        for batch_idx in range(batch_size):
            batch_predictions = []
            seq_length = attention_mask[batch_idx].sum().item()

            # 获取当前序列的span分数
            batch_span_scores = span_scores[batch_idx,
                                            :seq_length, :seq_length]

            # 使用softmax获取概率
            span_probs = F.softmax(batch_span_scores, dim=-1)

            # 获取最高概率的标签和概率值
            max_probs, span_preds = torch.max(span_probs, dim=-1)

            # 标准span提取：只要预测不是O标签且置信度足够
            for start in range(seq_length):
                for end in range(start, seq_length):
                    pred_label_id = span_preds[start, end].item()
                    if pred_label_id > 0:  # 不是O标签(0)
                        confidence = max_probs[start, end].item()

                        # 只用一个置信度阈值
                        if confidence > threshold:
                            # 关键修复：将label ID转换为label字符串
                            # 需要从span_converter获取ID到标签的映射
                            # 但这里没有直接访问span_converter，所以保存原始ID
                            # 由外部调用者处理转换，或者传入span_converter
                            batch_predictions.append({
                                'start': start,
                                'end': end,
                                'label': pred_label_id,  # 保持原有格式，由外部转换
                                'confidence': confidence
                            })

            predictions.append(batch_predictions)

        return predictions


class HybridSpanLoss(nn.Module):
    """
    用于span模型的混合损失函数
    结合交叉熵和Focal Loss
    """

    def __init__(self, focal_weight=0.6, ce_weight=0.4, num_classes=21, alpha=0.25, gamma=2.0):
        super(HybridSpanLoss, self).__init__()
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def focal_loss_2d(self, logits, targets):
        """
        计算2D输入的Focal Loss

        Args:
            logits: [num_samples, num_classes]
            targets: [num_samples]

        Returns:
            loss: [num_samples]
        """
        # 计算概率
        probs = F.softmax(logits, dim=-1)

        # 获取目标类别的概率
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        pt = (probs * targets_one_hot).sum(dim=-1)  # [num_samples]

        # 计算focal weight
        focal_weight = (1 - pt) ** self.gamma

        # 计算alpha weight
        alpha_weight = torch.where(
            targets > 0,
            torch.tensor(self.alpha, device=targets.device),
            torch.tensor(1 - self.alpha, device=targets.device)
        )

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # 应用权重
        focal_loss = alpha_weight * focal_weight * ce_loss

        return focal_loss

    def forward_span(self, span_scores, span_labels, valid_span_mask):
        """为span模型特制的前向函数"""
        # 展平为[batch * seq_len * seq_len, num_labels]
        span_scores_flat = span_scores.reshape(-1, span_scores.size(-1))
        span_labels_flat = span_labels.reshape(-1)
        valid_mask_flat = valid_span_mask.reshape(-1)

        # 只计算有效span的损失
        valid_indices = valid_mask_flat > 0
        valid_count = valid_indices.sum().item()

        if valid_count > 0:
            valid_scores = span_scores_flat[valid_indices]
            valid_labels = span_labels_flat[valid_indices]

            # 计算类别权重 - 强烈偏向实体标签
            label_counts = torch.bincount(
                valid_labels, minlength=self.num_classes)
            total_samples = valid_labels.numel()

            # 使用更强的不平衡权重策略
            class_weights = torch.ones(
                self.num_classes, device=valid_scores.device)
            for i in range(self.num_classes):
                if label_counts[i] > 0:
                    if i == 0:  # O标签，给予很小的权重
                        class_weights[i] = 0.05  # 降低到0.05
                    else:  # 实体标签，给予很大的权重
                        class_weights[i] = min(
                            # 提高到20
                            20.0, total_samples / (label_counts[i] * 1.5))
                else:
                    class_weights[i] = 1.0

            # 计算两种损失
            try:
                focal_loss = self.focal_loss_2d(valid_scores, valid_labels)
                if len(focal_loss.shape) > 0:
                    focal_loss = focal_loss.mean()

                # 使用权重的交叉熵损失
                ce_loss = F.cross_entropy(
                    valid_scores, valid_labels, weight=class_weights, reduction='mean')

                # 组合损失 - 增加focal loss的权重
                total_loss = self.focal_weight * focal_loss * 2.0 + self.ce_weight * ce_loss

                # 检查损失有效性
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    # fallback到加权交叉熵
                    total_loss = F.cross_entropy(
                        valid_scores, valid_labels, weight=class_weights, reduction='mean')

            except Exception as e:
                print(f"Loss calculation error: {e}")
                # fallback到加权交叉熵
                total_loss = F.cross_entropy(
                    valid_scores, valid_labels, weight=class_weights, reduction='mean')

        else:
            # 创建一个小的损失以保持梯度流
            total_loss = span_scores.mean() * 0.0 + 1e-8
            total_loss.requires_grad_(True)

        return total_loss
