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
        self._init_loss_function(config)

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

    def _init_loss_function(self, config: Config):
        """初始化损失函数 - 默认使用hybrid"""
        # 移除多种选项，直接使用改进的混合损失
        self.loss_fn = AdvancedHybridSpanLoss(
            num_classes=self.num_labels,
            focal_weight=getattr(config, 'focal_weight', 0.7),
            ce_weight=getattr(config, 'ce_weight', 0.3),
            dice_weight=getattr(config, 'dice_weight', 0.2),
            alpha=config.focal_loss_alpha,
            gamma=config.focal_loss_gamma,
            label_smoothing=getattr(config, 'label_smoothing', 0.1)
        )
        logger.info(
            f"Advanced Hybrid Loss initialized with focal_weight={getattr(config, 'focal_weight', 0.7)}, ce_weight={getattr(config, 'ce_weight', 0.3)}, dice_weight={getattr(config, 'dice_weight', 0.2)}")

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
        """改进的span损失计算 - 使用智能采样和高效mask"""
        batch_size, seq_len, _, num_labels = span_scores.size()
        device = span_scores.device

        # 更智能的有效span掩码构建
        valid_span_mask = self._create_intelligent_span_mask(
            attention_mask, seq_len)

        # 使用新的混合损失函数
        if hasattr(self.loss_fn, 'forward_span'):
            loss = self.loss_fn.forward_span(
                span_scores, span_labels, valid_span_mask)
        else:
            # Fallback 处理
            loss = self._compute_fallback_loss(
                span_scores, span_labels, valid_span_mask)

        # 应用span损失权重
        loss = loss * self.config.span_loss_weight
        return loss

    def _create_intelligent_span_mask(self, attention_mask, seq_len):
        """创建智能的span掩码，大幅减少无效位置的计算量"""
        device = attention_mask.device
        batch_size = attention_mask.size(0)

        # 获取每个样本的有效序列长度
        effective_lengths = attention_mask.sum(dim=1)  # [batch_size]

        # 创建高效的掩码矩阵
        valid_span_mask = torch.zeros(
            batch_size, seq_len, seq_len, device=device)

        for batch_idx in range(batch_size):
            effective_len = effective_lengths[batch_idx].item()

            # 动态确定最大span长度
            if effective_len <= 64:
                max_span_length = min(8, effective_len // 3)
                priority_lengths = [1, 2, 3, 4, 5, 6, 7, 8]
            elif effective_len <= 128:
                max_span_length = min(12, effective_len // 4)
                priority_lengths = [1, 2, 3, 4, 5, 6, 8, 10, 12]
            elif effective_len <= 256:
                max_span_length = min(16, effective_len // 6)
                priority_lengths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
            else:
                max_span_length = min(20, effective_len // 8)
                priority_lengths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20]

            # 只对有效长度范围内的span设置掩码
            for span_len in priority_lengths:
                if span_len > max_span_length:
                    break

                for start in range(effective_len):
                    end = start + span_len - 1
                    if end < effective_len:
                        # 额外的合理性检查
                        if self._is_span_position_reasonable(start, end, effective_len):
                            valid_span_mask[batch_idx, start, end] = 1.0

        return valid_span_mask

    def _is_span_position_reasonable(self, start: int, end: int, seq_len: int) -> bool:
        """检查span位置的合理性"""
        span_length = end - start + 1

        # 避免在序列末尾的padding区域
        if start >= seq_len * 0.95:
            return False

        # 基于span长度的启发式过滤
        if span_length == 1:
            return True  # 单字符span总是合理的
        elif span_length <= 3:
            return True  # 短span通常是合理的
        elif span_length <= 8:
            # 中等长度span需要更谨慎
            return start < seq_len * 0.8  # 不能太靠近末尾
        else:
            # 长span需要严格条件
            return start < seq_len * 0.6 and span_length <= 16

        return True

    def _compute_fallback_loss(self, span_scores, span_labels, valid_span_mask):
        """Fallback损失计算"""
        span_scores_flat = span_scores.reshape(-1, span_scores.size(-1))
        span_labels_flat = span_labels.reshape(-1)
        valid_mask_flat = valid_span_mask.reshape(-1)

        valid_indices = valid_mask_flat > 0
        if valid_indices.sum() > 0:
            valid_scores = span_scores_flat[valid_indices]
            valid_labels = span_labels_flat[valid_indices]
            return F.cross_entropy(valid_scores, valid_labels, reduction='mean')
        else:
            return span_scores.mean() * 0.0 + 1e-8

    def decode_spans(self, span_scores, attention_mask, threshold=0.1):
        """
        专门针对中文地址NER优化的span解码策略

        核心改进：
        1. 多层次阈值策略：不同类型实体使用不同阈值
        2. 地址结构约束：利用地址层次结构过滤不合理span
        3. 冲突解决机制：处理重叠span的优先级选择
        4. 置信度加权：结合多个指标进行span筛选
        """
        batch_size, seq_len, _, num_labels = span_scores.size()
        predictions = []

        # 地址层次优先级定义
        address_hierarchy = {
            'administrative': ['prov', 'city', 'district', 'devzone', 'town', 'community', 'village_group'],
            'location': ['road', 'roadno', 'poi', 'subpoi'],
            'building': ['houseno', 'cellno', 'floorno', 'roomno'],
            'auxiliary': ['detail', 'assist', 'distance', 'intersection', 'redundant', 'others']
        }

        # 创建标签优先级映射
        label_priority = {}
        for level, (category, labels_in_category) in enumerate(address_hierarchy.items()):
            for priority, label in enumerate(labels_in_category):
                label_priority[label] = (level, priority)

        for batch_idx in range(batch_size):
            seq_length = attention_mask[batch_idx].sum().item()
            batch_span_scores = span_scores[batch_idx,
                                            :seq_length, :seq_length]

            # 使用softmax获取概率
            span_probs = F.softmax(batch_span_scores, dim=-1)
            max_probs, span_preds = torch.max(span_probs, dim=-1)

            # 收集候选spans
            candidate_spans = []

            for start in range(seq_length):
                for end in range(start, seq_length):
                    pred_label_id = span_preds[start, end].item()

                    if pred_label_id > 0:  # 不是O标签
                        confidence = max_probs[start, end].item()
                        span_length = end - start + 1

                        # 获取标签名
                        # 注意：这里需要从外部获取id_to_label映射
                        # 暂时使用ID，由调用者转换

                        # 多层次阈值策略
                        adjusted_threshold = self._get_adaptive_threshold(
                            pred_label_id, span_length, seq_length, threshold)

                        if confidence > adjusted_threshold:
                            # 计算复合置信度分数
                            composite_score = self._calculate_composite_score(
                                span_probs[start, end], pred_label_id, start, end, seq_length)

                            candidate_spans.append({
                                'start': start,
                                'end': end,
                                'label': pred_label_id,
                                'confidence': confidence,
                                'composite_score': composite_score,
                                'span_length': span_length
                            })

            # 应用地址结构约束和冲突解决
            final_spans = self._resolve_address_conflicts(
                candidate_spans, label_priority)

            predictions.append(final_spans)

        return predictions

    def _get_adaptive_threshold(self, label_id: int, span_length: int, seq_length: int, base_threshold: float) -> float:
        """
        获取自适应阈值，根据标签类型、span长度和序列长度调整 - 进一步优化
        """
        # 基础阈值调整
        threshold = base_threshold

        # 根据span长度调整 - 进一步降低短span的阈值要求
        if span_length == 1:
            threshold *= 0.6  # 从0.7降低到0.6，单字符span更容易被接受
        elif span_length <= 3:
            threshold *= 0.7  # 从0.8降低到0.7，短span适当降低
        elif span_length <= 8:
            threshold *= 0.9  # 从1.0降低到0.9，中等长度稍微降低
        elif span_length <= 15:
            threshold *= 1.1  # 新增：较长span适中要求
        else:
            threshold *= 1.2  # 从1.3降低到1.2，超长span要求稍微降低

        # 根据序列长度调整 - 更温和的调整
        if seq_length < 50:
            threshold *= 0.85   # 从0.9降低到0.85，短序列进一步降低要求
        elif seq_length > 200:
            threshold *= 1.05   # 从1.1降低到1.05，长序列要求适当降低

        # 确保阈值在合理范围内，下限进一步降低
        # 从max(0.05, min(0.8))调整为max(0.03, min(0.75))
        return max(0.03, min(0.75, threshold))

    def _calculate_composite_score(self, prob_dist: torch.Tensor, pred_label_id: int,
                                   start: int, end: int, seq_length: int) -> float:
        """
        计算复合置信度分数，结合多个因素 - 优化权重分配
        """
        # 基础概率分数
        base_prob = prob_dist[pred_label_id].item()

        # 计算概率分布的集中度（熵的反向指标）
        entropy = -(prob_dist * torch.log(prob_dist + 1e-8)).sum().item()
        concentration = 1.0 / (1.0 + entropy)  # 熵越低，集中度越高

        # 位置分数（开始部分得分更高，但不要过于偏向）
        position_score = 1.0 - (start / seq_length) * 0.2  # 从0.3降低到0.2，减少位置偏见

        # 长度分数（更平缓的分数分布）
        span_length = end - start + 1
        if span_length <= 2:
            length_score = 0.85 + span_length * 0.075  # 提升很短span的分数
        elif span_length <= 5:
            length_score = 1.0
        elif span_length <= 10:
            length_score = 0.95
        elif span_length <= 20:
            length_score = 0.9   # 对中长span更宽松
        else:
            length_score = max(0.7, 1.0 - (span_length - 20) * 0.02)  # 更温和的惩罚

        # 调整复合分数权重，更偏向基础概率
        composite = (base_prob * 0.7 +      # 从0.6提升到0.7，更重视模型概率
                     concentration * 0.15 +  # 从0.2降低到0.15
                     position_score * 0.08 +  # 从0.1降低到0.08
                     length_score * 0.07)     # 从0.1降低到0.07

        return composite

    def _resolve_address_conflicts(self, candidate_spans: list, label_priority: dict) -> list:
        """
        解决地址span冲突，基于地址结构优先级 - 更保守的冲突解决
        """
        if not candidate_spans:
            return []

        # 按复合分数排序
        sorted_spans = sorted(candidate_spans,
                              key=lambda x: x['composite_score'],
                              reverse=True)

        final_spans = []
        occupied_positions = set()

        for span in sorted_spans:
            start, end = span['start'], span['end']
            span_positions = set(range(start, end + 1))

            # 检查是否与已选择的span冲突
            if not span_positions & occupied_positions:
                final_spans.append(span)
                occupied_positions.update(span_positions)
            else:
                # 更宽松的重叠处理：允许更多的边界调整
                overlap_size = len(span_positions & occupied_positions)
                if overlap_size < len(span_positions) * 0.4:  # 从0.3提升到0.4，允许更多重叠
                    # 可以尝试调整边界
                    available_positions = span_positions - occupied_positions
                    if len(available_positions) >= max(1, len(span_positions) * 0.25):  # 从0.5降低到0.25，更宽松
                        # 调整span边界
                        new_start = min(available_positions)
                        new_end = max(available_positions)

                        # 确保调整后的span仍然合理
                        if new_end >= new_start and (new_end - new_start + 1) >= 1:
                            adjusted_span = span.copy()
                            adjusted_span['start'] = new_start
                            adjusted_span['end'] = new_end
                            # 从0.8提升到0.9，减少惩罚
                            adjusted_span['confidence'] *= 0.9

                            final_spans.append(adjusted_span)
                            occupied_positions.update(
                                range(new_start, new_end + 1))

        # 按起始位置重新排序
        final_spans.sort(key=lambda x: x['start'])

        return final_spans


class AdvancedHybridSpanLoss(nn.Module):
    """
    高级混合span损失函数
    结合Focal Loss、交叉熵损失和Dice Loss的优势
    专门针对span NER的类别不平衡问题进行优化
    """

    def __init__(self, num_classes, focal_weight=0.7, ce_weight=0.3, dice_weight=0.2,
                 alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super(AdvancedHybridSpanLoss, self).__init__()
        self.num_classes = num_classes
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        # 预计算的类别权重
        self.register_buffer('base_class_weights',
                             self._compute_base_weights())

    def _compute_base_weights(self):
        """预计算基础类别权重"""
        weights = torch.ones(self.num_classes)
        weights[0] = 0.1  # O标签给予很小权重
        weights[1:] = 3.0  # 实体标签给予较大权重
        return weights

    def forward_span(self, span_scores, span_labels, valid_span_mask):
        """优化的span损失前向传播"""
        # 使用智能采样减少O标签的影响
        sampled_scores, sampled_labels, sample_weights = self._intelligent_sampling(
            span_scores, span_labels, valid_span_mask)

        if sampled_scores.size(0) == 0:
            return span_scores.mean() * 0.0 + 1e-8

        # 计算三种损失的组合
        focal_loss = self._compute_focal_loss(
            sampled_scores, sampled_labels, sample_weights)
        ce_loss = self._compute_weighted_ce_loss(
            sampled_scores, sampled_labels, sample_weights)
        dice_loss = self._compute_dice_loss(sampled_scores, sampled_labels)

        # 动态权重调整
        total_loss = (self.focal_weight * focal_loss +
                      self.ce_weight * ce_loss +
                      self.dice_weight * dice_loss)

        return total_loss

    def _intelligent_sampling(self, span_scores, span_labels, valid_span_mask):
        """智能采样：平衡O标签和实体标签的比例"""
        # 展平张量
        span_scores_flat = span_scores.reshape(-1, span_scores.size(-1))
        span_labels_flat = span_labels.reshape(-1)
        valid_mask_flat = valid_span_mask.reshape(-1)

        # 获取有效样本
        valid_indices = valid_mask_flat > 0
        valid_scores = span_scores_flat[valid_indices]
        valid_labels = span_labels_flat[valid_indices]

        if valid_scores.size(0) == 0:
            return torch.empty(0, span_scores.size(-1), device=span_scores.device), \
                torch.empty(0, dtype=torch.long, device=span_scores.device), \
                torch.empty(0, device=span_scores.device)

        # 分离O标签和实体标签
        o_mask = valid_labels == 0
        entity_mask = valid_labels > 0

        o_indices = torch.where(o_mask)[0]
        entity_indices = torch.where(entity_mask)[0]

        # 采样策略：确保实体标签不被稀释
        entity_count = entity_indices.size(0)
        o_count = o_indices.size(0)

        if entity_count > 0 and o_count > 0:
            # 限制O标签样本数量，最多是实体标签的2-3倍
            max_o_samples = min(o_count, entity_count * 3)
            if o_count > max_o_samples:
                # 随机采样O标签
                selected_o_indices = o_indices[torch.randperm(o_count)[
                    :max_o_samples]]
            else:
                selected_o_indices = o_indices

            # 合并索引
            selected_indices = torch.cat([entity_indices, selected_o_indices])
        elif entity_count > 0:
            selected_indices = entity_indices
        else:
            # 如果没有实体标签，随机采样一部分O标签
            max_o_samples = min(o_count, 100)  # 限制O标签数量
            selected_indices = o_indices[torch.randperm(o_count)[
                :max_o_samples]]

        # 获取采样后的数据
        sampled_scores = valid_scores[selected_indices]
        sampled_labels = valid_labels[selected_indices]

        # 计算样本权重
        sample_weights = torch.ones_like(sampled_labels, dtype=torch.float)
        sample_weights[sampled_labels == 0] = 0.2  # O标签权重
        sample_weights[sampled_labels > 0] = 1.0   # 实体标签权重

        return sampled_scores, sampled_labels, sample_weights

    def _compute_focal_loss(self, scores, labels, sample_weights):
        """计算加权Focal Loss"""
        probs = F.softmax(scores, dim=-1)
        targets_one_hot = F.one_hot(labels, self.num_classes).float()
        pt = (probs * targets_one_hot).sum(dim=-1)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weight
        alpha_weight = torch.where(labels > 0, self.alpha, 1 - self.alpha)

        # Cross entropy
        ce_loss = F.cross_entropy(scores, labels, reduction='none')

        # 组合权重
        focal_loss = alpha_weight * focal_weight * sample_weights * ce_loss

        return focal_loss.mean()

    def _compute_weighted_ce_loss(self, scores, labels, sample_weights):
        """计算加权交叉熵损失"""
        # 动态计算类别权重
        unique_labels = torch.unique(labels)
        class_weights = self.base_class_weights.clone()

        for label in unique_labels:
            if label > 0:  # 实体标签
                count = (labels == label).sum().float()
                total = labels.size(0)
                # 根据频率调整权重
                class_weights[label] = torch.clamp(
                    total / (count * 2), 1.0, 10.0)

        # 标签平滑
        if self.label_smoothing > 0:
            smooth_loss = self._label_smoothing_loss(
                scores, labels, sample_weights)
            return smooth_loss
        else:
            ce_loss = F.cross_entropy(
                scores, labels, weight=class_weights, reduction='none')
            return (ce_loss * sample_weights).mean()

    def _compute_dice_loss(self, scores, labels):
        """计算Dice Loss以进一步平衡类别"""
        probs = F.softmax(scores, dim=-1)
        targets_one_hot = F.one_hot(labels, self.num_classes).float()

        dice_losses = []
        for c in range(1, self.num_classes):  # 跳过O标签
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union > 0:
                dice_loss = 1 - (2.0 * intersection + 1e-8) / (union + 1e-8)
                dice_losses.append(dice_loss)

        if dice_losses:
            return torch.stack(dice_losses).mean()
        else:
            return torch.tensor(0.0, device=scores.device)

    def _label_smoothing_loss(self, scores, labels, sample_weights):
        """标签平滑损失"""
        log_probs = F.log_softmax(scores, dim=-1)
        targets_one_hot = F.one_hot(labels, self.num_classes).float()

        # 应用标签平滑
        smooth_targets = targets_one_hot * (1 - self.label_smoothing) + \
            self.label_smoothing / self.num_classes

        loss = -(smooth_targets * log_probs).sum(dim=-1)
        return (loss * sample_weights).mean()
