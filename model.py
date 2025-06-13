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


class MultiScaleFeatureFusion(nn.Module):
    """
    多尺度特征融合模块，融合BERT不同层的特征
    """

    def __init__(self, hidden_size, num_layers=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 不同层的权重学习
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

        # 特征变换层
        self.feature_transform = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        # 融合后的特征变换
        self.fusion_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, layer_outputs):
        """
        Args:
            layer_outputs: List of tensors from different BERT layers
                          Each tensor: [batch_size, seq_len, hidden_size]
        """
        # 确保我们有足够的层
        if len(layer_outputs) < self.num_layers:
            # 如果层数不够，重复最后一层
            while len(layer_outputs) < self.num_layers:
                layer_outputs.append(layer_outputs[-1])

        # 只取最后num_layers层
        layer_outputs = layer_outputs[-self.num_layers:]

        # 对每一层进行变换
        transformed_layers = []
        for i, layer_output in enumerate(layer_outputs):
            transformed = self.feature_transform[i](layer_output)
            transformed_layers.append(transformed)

        # 计算加权和
        weights = torch.softmax(self.layer_weights, dim=0)
        fused_features = sum(
            w * layer for w, layer in zip(weights, transformed_layers))

        # 最终变换
        output = self.fusion_transform(fused_features)

        return output


class MultiHeadBiaffineAttention(nn.Module):
    """
    多头双仿射注意力机制，不同头专注于不同类型的span关系
    """

    def __init__(self, hidden_size, num_labels, num_heads=4, head_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads

        # 确保维度正确
        assert self.head_dim * num_heads <= hidden_size

        # 每个头的投影层
        self.start_projections = nn.ModuleList([
            nn.Linear(hidden_size, self.head_dim) for _ in range(num_heads)
        ])
        self.end_projections = nn.ModuleList([
            nn.Linear(hidden_size, self.head_dim) for _ in range(num_heads)
        ])

        # 每个头的双仿射层 - 修复：直接作为参数而不是放在ModuleList中
        self.biaffine_layers = nn.ParameterList([
            nn.Parameter(torch.randn(self.head_dim, num_labels, self.head_dim))
            for _ in range(num_heads)
        ])

        # 头融合层
        self.head_fusion = nn.Linear(num_heads * num_labels, num_labels)

        # 层归一化
        self.layer_norm = nn.LayerNorm(num_labels)

        self.dropout = nn.Dropout(0.1)

    def forward(self, sequence_output, attention_mask=None):
        """
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] 注意力掩码（可选，用于保持接口一致性）
        Returns:
            span_logits: [batch_size, seq_len, seq_len, num_labels]
        """
        batch_size, seq_len, _ = sequence_output.size()

        # 多头计算
        head_outputs = []

        for head_idx in range(self.num_heads):
            # 投影到头维度
            start_repr = self.start_projections[head_idx](
                sequence_output)  # [batch, seq, head_dim]
            end_repr = self.end_projections[head_idx](
                sequence_output)      # [batch, seq, head_dim]

            # 双仿射注意力
            # [head_dim, num_labels, head_dim]
            biaffine_weight = self.biaffine_layers[head_idx]

            # 计算双仿射分数 - 修复维度问题
            # 方法1：使用torch.einsum正确处理3D biaffine weight
            # start_repr: [batch, seq, head_dim]
            # biaffine_weight: [head_dim, num_labels, head_dim]
            # 输出: [batch, seq, num_labels, head_dim]
            temp = torch.einsum('bsh,hlk->bslk', start_repr, biaffine_weight)

            # temp: [batch, seq, num_labels, head_dim] @ end_repr: [batch, seq, head_dim]
            # 输出: [batch, seq, num_labels, seq] -> 需要转换为 [batch, seq, seq, num_labels]
            head_logits = torch.einsum('bslk,btk->bslt', temp, end_repr)

            # 转换维度：[batch, seq, num_labels, seq] -> [batch, seq, seq, num_labels]
            # [batch, seq, seq, num_labels]
            head_logits = head_logits.permute(0, 1, 3, 2)

            head_outputs.append(head_logits)

        # 融合所有头的输出 [batch, seq, seq, num_heads * num_labels]
        combined = torch.cat(head_outputs, dim=-1)

        # 投影到最终标签维度
        # [batch, seq, seq, num_labels]
        span_logits = self.head_fusion(combined)
        span_logits = self.layer_norm(span_logits)
        span_logits = self.dropout(span_logits)

        return span_logits


class HierarchicalSpanClassifier(nn.Module):
    """
    层次化span分类器，根据地址结构层次分别处理
    """

    def __init__(self, hidden_size, address_hierarchy, total_labels):
        super().__init__()
        self.address_hierarchy = address_hierarchy
        self.total_labels = total_labels

        # 为每个层次创建专门的分类器
        self.hierarchy_classifiers = nn.ModuleDict()
        self.hierarchy_label_maps = {}

        label_offset = 1  # 0 是 'O' 标签

        for level_name, labels in address_hierarchy.items():
            num_labels_in_level = len(labels) + 1  # +1 for 'O'

            # 创建分类器
            self.hierarchy_classifiers[level_name] = MultiHeadBiaffineAttention(
                hidden_size, num_labels_in_level, num_heads=2
            )

            # 创建标签映射
            level_map = {'O': 0}
            for i, label in enumerate(labels):
                level_map[label] = i + 1

            self.hierarchy_label_maps[level_name] = level_map
            label_offset += len(labels)

        # 最终融合层 - 修复维度计算
        # 计算每个层次分类器的输出维度（每个层次输出的实际是num_labels，不是num_labels * num_heads）
        # MultiHeadBiaffineAttention在最后会通过head_fusion投影回num_labels
        total_hierarchy_labels = 0
        for level_name, labels in address_hierarchy.items():
            num_labels_in_level = len(labels) + 1  # +1 for 'O'
            total_hierarchy_labels += num_labels_in_level  # 每个层次输出num_labels_in_level维

        self.final_fusion = nn.Sequential(
            nn.Linear(total_hierarchy_labels, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, total_labels)
        )

    def forward(self, sequence_output, hierarchy_targets=None):
        """
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            hierarchy_targets: Dict of targets for each hierarchy level (training only)
        """
        hierarchy_outputs = {}
        hierarchy_logits = []

        # 为每个层次计算预测
        for level_name, classifier in self.hierarchy_classifiers.items():
            # MultiHeadBiaffineAttention 输出 [batch, seq, seq, level_labels]
            level_logits = classifier(sequence_output)
            hierarchy_outputs[level_name] = level_logits
            hierarchy_logits.append(level_logits)

        # 拼接所有层次的输出
        # [batch, seq, seq, total_hierarchy_labels]
        combined_logits = torch.cat(hierarchy_logits, dim=-1)

        # 检查维度，确保final_fusion能够处理
        batch_size, seq_len, seq_len2, total_dim = combined_logits.shape

        # 将span矩阵展平处理
        # [batch * seq * seq, total_hierarchy_labels]
        flattened = combined_logits.view(-1, total_dim)

        # 通过fusion层
        # [batch * seq * seq, total_labels]
        fused = self.final_fusion(flattened)

        # 恢复原始形状
        # [batch, seq, seq, total_labels]
        final_logits = fused.view(
            batch_size, seq_len, seq_len2, self.total_labels)

        return final_logits, hierarchy_outputs


class SpanBoundaryDetector(nn.Module):
    """
    专门的span边界检测模块
    """

    def __init__(self, hidden_size):
        super().__init__()

        # 开始边界检测
        self.start_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

        # 结束边界检测
        self.end_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

        # 边界一致性检测
        self.boundary_consistency = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, sequence_output):
        """
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
        Returns:
            start_logits: [batch_size, seq_len]
            end_logits: [batch_size, seq_len]
            consistency_matrix: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, hidden_size = sequence_output.size()

        # 边界检测
        start_logits = self.start_detector(
            sequence_output).squeeze(-1)  # [batch, seq]
        end_logits = self.end_detector(
            sequence_output).squeeze(-1)      # [batch, seq]

        # 一致性检测 - 计算每对位置的一致性
        consistency_scores = []
        for i in range(seq_len):
            # [batch, seq, hidden]
            start_repr = sequence_output[:, i:i+1, :].expand(-1, seq_len, -1)
            end_repr = sequence_output  # [batch, seq, hidden]

            # 拼接start和end表示
            # [batch, seq, hidden*2]
            combined = torch.cat([start_repr, end_repr], dim=-1)
            consistency = self.boundary_consistency(
                combined).squeeze(-1)  # [batch, seq]
            consistency_scores.append(consistency)

        consistency_matrix = torch.stack(
            consistency_scores, dim=1)  # [batch, seq, seq]

        return start_logits, end_logits, consistency_matrix


class FreeLB:
    """
    FreeLB对抗训练实现，保持原有逻辑
    """

    def __init__(self, model, adv_lr=0.05, adv_steps=3, adv_init_mag=0.05,
                 adv_max_norm=0.07, adv_norm_type='l2', base_model='bert'):
        self.model = model
        self.adv_lr = adv_lr
        self.adv_steps = adv_steps
        self.adv_init_mag = adv_init_mag
        self.adv_max_norm = adv_max_norm
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model

    def attack_span(self, inputs_embeds, attention_mask, span_labels):
        """针对span模型的对抗训练"""
        # 初始化扰动
        if self.adv_init_mag > 0:
            noise = torch.zeros_like(inputs_embeds).uniform_(
                -self.adv_init_mag, self.adv_init_mag)
            noise.requires_grad_()
        else:
            noise = torch.zeros_like(inputs_embeds)
            noise.requires_grad_()

        total_loss = 0

        for step in range(self.adv_steps):
            # 添加扰动
            perturbed_embeds = inputs_embeds + noise

            # 前向传播
            loss = self.model(
                inputs_embeds=perturbed_embeds,
                attention_mask=attention_mask,
                span_labels=span_labels
            )

            total_loss += loss / self.adv_steps

            # 最后一步不需要计算梯度
            if step == self.adv_steps - 1:
                break

            # 计算梯度
            loss.backward(retain_graph=True)

            # 更新扰动
            if noise.grad is not None:
                if self.adv_norm_type == 'l2':
                    noise_norm = torch.norm(noise.grad, dim=-1, keepdim=True)
                    noise_grad_normalized = noise.grad / (noise_norm + 1e-8)
                else:  # linf
                    noise_grad_normalized = torch.sign(noise.grad)

                noise = noise + self.adv_lr * noise_grad_normalized

                # 限制扰动大小
                if self.adv_norm_type == 'l2':
                    noise_norm = torch.norm(noise, dim=-1, keepdim=True)
                    noise = noise / torch.max(
                        noise_norm / self.adv_max_norm,
                        torch.ones_like(noise_norm)
                    )
                else:  # linf
                    noise = torch.clamp(
                        noise, -self.adv_max_norm, self.adv_max_norm)

                noise = noise.detach()
                noise.requires_grad_()

        return total_loss


class BertBiaffineSpanNER(nn.Module):
    """
    基于BERT的双仿射网络跨度命名实体识别模型

    该模型结合BERT编码器和双仿射注意力机制来进行基于跨度的命名实体识别。
    模型架构包括：
    1. BERT编码器用于序列表示学习
    2. 多层特征融合
    3. 双仿射注意力用于跨度分类
    4. 可选的层次化分类和边界检测
    """

    def __init__(self, num_labels: int, config: Config):
        super(BertBiaffineSpanNER, self).__init__()

        self.config = config
        self.num_labels = num_labels

        # BERT编码器
        self.bert = AutoModel.from_pretrained(
            config.model_name,
            hidden_dropout_prob=getattr(config, 'hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=getattr(
                config, 'attention_probs_dropout_prob', 0.1)
        )

        # 获取BERT隐藏层维度
        self.hidden_size = self.bert.config.hidden_size

        # 冻结BERT的前N层
        if getattr(config, 'freeze_bert_layers', 0) > 0:
            self._freeze_bert_layers(config.freeze_bert_layers)

        # 空间Dropout层
        self.spatial_dropout = SpatialDropout(
            drop_prob=getattr(config, 'spatial_dropout', 0.15)
        )

        # 嵌入层Dropout
        self.embedding_dropout = nn.Dropout(
            p=getattr(config, 'embedding_dropout', 0.15)
        )

        # 多尺度特征融合（如果启用）
        if getattr(config, 'use_hierarchical', False):
            fusion_layers = getattr(config, 'fusion_layers', 4)
            self.feature_fusion = MultiScaleFeatureFusion(
                hidden_size=self.hidden_size,
                num_layers=fusion_layers
            )
        else:
            self.feature_fusion = None

        # 主要的跨度分类器
        biaffine_hidden_dim = getattr(config, 'biaffine_hidden_dim', 512)
        span_dropout = getattr(config, 'span_dropout', 0.1)
        use_biaffine = getattr(config, 'use_biaffine', True)

        if getattr(config, 'use_hierarchical', False):
            # 使用多头双仿射注意力进行层次化分类
            biaffine_heads = getattr(config, 'biaffine_heads', 4)
            self.span_classifier = MultiHeadBiaffineAttention(
                hidden_size=self.hidden_size,
                num_labels=num_labels,
                num_heads=biaffine_heads
            )
        else:
            # 使用标准的高效跨度分类器
            self.span_classifier = EfficientSpanClassifier(
                hidden_dim=self.hidden_size,
                num_labels=num_labels,
                dropout=span_dropout,
                biaffine_hidden_dim=biaffine_hidden_dim,
                use_biaffine=use_biaffine
            )

        # 边界检测模块（可选）
        if getattr(config, 'use_boundary_detection', False):
            self.boundary_detector = SpanBoundaryDetector(self.hidden_size)
        else:
            self.boundary_detector = None

        # 损失函数
        if getattr(config, 'use_focal_loss', False):
            focal_alpha = getattr(config, 'focal_loss_alpha', 0.25)
            focal_gamma = getattr(config, 'focal_loss_gamma', 2.0)
            self.span_loss_fn = SpanFocalLoss(
                num_classes=num_labels,
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction='mean'
            )
        else:
            # 使用标签平滑的交叉熵损失
            label_smoothing = getattr(config, 'label_smoothing', 0.0)
            self.span_loss_fn = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=label_smoothing
            )

        # 损失权重
        self.span_loss_weight = getattr(config, 'span_loss_weight', 1.0)
        self.boundary_loss_weight = getattr(
            config, 'boundary_loss_weight', 0.3)
        self.hierarchy_loss_weight = getattr(
            config, 'hierarchy_loss_weight', 0.2)

        logger.info(
            f"BertBiaffineSpanNER initialized with {num_labels} labels")
        logger.info(f"Model configuration: hidden_size={self.hidden_size}, "
                    f"biaffine_hidden_dim={biaffine_hidden_dim}, "
                    f"use_biaffine={use_biaffine}, "
                    f"use_hierarchical={getattr(config, 'use_hierarchical', False)}, "
                    f"use_boundary_detection={getattr(config, 'use_boundary_detection', False)}")

    def _freeze_bert_layers(self, num_layers: int):
        """冻结BERT的前N层"""
        # 冻结嵌入层
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # 冻结前N个transformer层
        if num_layers > 0:
            for layer_idx in range(min(num_layers, len(self.bert.encoder.layer))):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False

        logger.info(f"Frozen first {num_layers} BERT layers")

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, span_labels=None):
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len] 输入token id
            attention_mask: [batch_size, seq_len] 注意力掩码
            inputs_embeds: [batch_size, seq_len, hidden_size] 输入嵌入（用于对抗训练）
            span_labels: [batch_size, seq_len, seq_len] 跨度标签矩阵（训练时提供）

        Returns:
            如果提供span_labels，返回损失值；否则返回预测logits
        """

        # BERT编码
        if inputs_embeds is not None:
            # 对抗训练模式，使用扰动后的嵌入
            bert_outputs = self.bert(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        else:
            # 正常训练/推理模式
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        # 获取序列表示
        # [batch, seq_len, hidden]
        sequence_output = bert_outputs.last_hidden_state

        # 应用嵌入层dropout
        sequence_output = self.embedding_dropout(sequence_output)

        # 多尺度特征融合（如果启用）
        if self.feature_fusion is not None:
            # 使用所有隐藏层进行特征融合
            all_hidden_states = bert_outputs.hidden_states
            sequence_output = self.feature_fusion(all_hidden_states)

        # 应用空间dropout
        sequence_output = self.spatial_dropout(sequence_output)

        # 主要跨度分类
        span_logits = self.span_classifier(sequence_output, attention_mask)

        # 边界检测（如果启用）
        boundary_outputs = None
        if self.boundary_detector is not None:
            start_logits, end_logits, consistency_matrix = self.boundary_detector(
                sequence_output)
            boundary_outputs = {
                'start_logits': start_logits,
                'end_logits': end_logits,
                'consistency_matrix': consistency_matrix
            }

            # 如果提供了标签，计算损失
        if span_labels is not None:
            return self._calculate_loss(span_logits, span_labels, attention_mask, boundary_outputs)

        # 推理模式，返回span_logits供后续处理
        # trainer.py期望能够通过索引访问，所以直接返回span_logits
        return span_logits

    def _calculate_loss(self, span_logits, span_labels, attention_mask, boundary_outputs=None):
        """
        计算训练损失

        Args:
            span_logits: [batch, seq_len, seq_len, num_labels] 跨度分类logits
            span_labels: [batch, seq_len, seq_len] 跨度标签
            attention_mask: [batch, seq_len] 注意力掩码
            boundary_outputs: 边界检测输出（可选）

        Returns:
            总损失值
        """

        # 创建有效跨度掩码
        batch_size, seq_len = attention_mask.shape

        # 基本的有效跨度掩码（基于attention mask）
        valid_span_mask = self._create_valid_span_mask(attention_mask)

        # 主要跨度分类损失
        if isinstance(self.span_loss_fn, SpanFocalLoss):
            # 使用Focal Loss
            span_loss = self.span_loss_fn.forward_span(
                span_logits, span_labels, valid_span_mask
            )
        else:
            # 使用交叉熵损失
            span_loss = self._compute_span_ce_loss(
                span_logits, span_labels, valid_span_mask
            )

        total_loss = self.span_loss_weight * span_loss

        # 边界检测损失（如果启用）
        if boundary_outputs is not None and self.boundary_loss_weight > 0:
            boundary_loss = self._compute_boundary_loss(
                boundary_outputs, span_labels, attention_mask
            )
            total_loss += self.boundary_loss_weight * boundary_loss

        return total_loss

    def _create_valid_span_mask(self, attention_mask):
        """
        创建有效跨度掩码

        Args:
            attention_mask: [batch, seq_len]

        Returns:
            valid_span_mask: [batch, seq_len, seq_len]
        """
        batch_size, seq_len = attention_mask.shape

        # 创建上三角掩码（只考虑start <= end的跨度）
        triu_mask = torch.triu(torch.ones(
            seq_len, seq_len, device=attention_mask.device))

        # 扩展到batch维度
        triu_mask = triu_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # 结合attention mask
        # valid_span[i, j, k] = 1 当且仅当 j和k都是有效位置且j <= k
        attention_expanded_i = attention_mask.unsqueeze(
            2).expand(-1, -1, seq_len)
        attention_expanded_j = attention_mask.unsqueeze(
            1).expand(-1, seq_len, -1)

        valid_span_mask = triu_mask * attention_expanded_i * attention_expanded_j

        return valid_span_mask

    def _compute_span_ce_loss(self, span_logits, span_labels, valid_span_mask):
        """
        计算跨度分类的交叉熵损失

        Args:
            span_logits: [batch, seq_len, seq_len, num_labels]
            span_labels: [batch, seq_len, seq_len]
            valid_span_mask: [batch, seq_len, seq_len]

        Returns:
            损失值
        """
        # 展平处理
        span_logits_flat = span_logits.reshape(-1, span_logits.size(-1))
        span_labels_flat = span_labels.reshape(-1)
        valid_mask_flat = valid_span_mask.reshape(-1)

        # 只计算有效跨度的损失
        valid_indices = valid_mask_flat > 0

        if valid_indices.sum() > 0:
            valid_logits = span_logits_flat[valid_indices]
            valid_labels = span_labels_flat[valid_indices]

            loss = F.cross_entropy(
                valid_logits, valid_labels, reduction='mean')
            return loss
        else:
            return torch.tensor(0.0, device=span_logits.device, requires_grad=True)

    def _compute_boundary_loss(self, boundary_outputs, span_labels, attention_mask):
        """
        计算边界检测损失

        Args:
            boundary_outputs: 边界检测输出
            span_labels: [batch, seq_len, seq_len] 跨度标签
            attention_mask: [batch, seq_len] 注意力掩码

        Returns:
            边界损失值
        """
        start_logits = boundary_outputs['start_logits']  # [batch, seq_len]
        end_logits = boundary_outputs['end_logits']      # [batch, seq_len]

        # 从跨度标签中提取边界标签
        start_labels, end_labels = self._extract_boundary_labels(span_labels)

        # 计算边界检测损失
        start_loss = F.binary_cross_entropy_with_logits(
            start_logits, start_labels.float(),
            weight=attention_mask.float(),
            reduction='mean'
        )

        end_loss = F.binary_cross_entropy_with_logits(
            end_logits, end_labels.float(),
            weight=attention_mask.float(),
            reduction='mean'
        )

        return (start_loss + end_loss) / 2

    def _extract_boundary_labels(self, span_labels):
        """
        从跨度标签中提取边界标签

        Args:
            span_labels: [batch, seq_len, seq_len]

        Returns:
            start_labels: [batch, seq_len] 开始位置标签
            end_labels: [batch, seq_len] 结束位置标签
        """
        batch_size, seq_len, _ = span_labels.shape

        start_labels = torch.zeros(
            batch_size, seq_len, device=span_labels.device)
        end_labels = torch.zeros(
            batch_size, seq_len, device=span_labels.device)

        # 从跨度矩阵中提取边界信息
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(i, seq_len):
                    if span_labels[b, i, j] > 0:  # 非O标签
                        start_labels[b, i] = 1
                        end_labels[b, j] = 1

        return start_labels, end_labels

    def predict_spans(self, input_ids, attention_mask, span_threshold=None):
        """
        预测跨度实体

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            span_threshold: 置信度阈值

        Returns:
            预测的跨度列表
        """
        if span_threshold is None:
            span_threshold = getattr(self.config, 'span_threshold', 0.5)

        # 前向传播获取预测结果
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids,
                                   attention_mask=attention_mask)
            span_logits = outputs['span_logits']

            # 转换为概率
            span_probs = F.softmax(span_logits, dim=-1)

            # 提取跨度
            batch_spans = []

            for batch_idx in range(span_probs.shape[0]):
                spans = []
                seq_len = attention_mask[batch_idx].sum().item()

                for start in range(seq_len):
                    for end in range(start, seq_len):
                        # 获取最大概率的标签
                        max_prob, pred_label = span_probs[batch_idx, start, end].max(
                            dim=-1)

                        # 如果不是O标签且超过阈值
                        if pred_label.item() > 0 and max_prob.item() > span_threshold:
                            spans.append({
                                'start': start,
                                'end': end,
                                'label': pred_label.item(),
                                'confidence': max_prob.item()
                            })

                batch_spans.append(spans)

            return batch_spans

    def decode_spans(self, span_logits, attention_mask, threshold=0.3):
        """
        从span logits中解码出预测的跨度

        Args:
            span_logits: [batch, seq_len, seq_len, num_labels] 跨度预测logits
            attention_mask: [batch, seq_len] 注意力掩码
            threshold: float 基础置信度阈值

        Returns:
            List[List[Dict]]: 每个样本的预测跨度列表
        """
        # 转换为概率
        span_probs = F.softmax(span_logits, dim=-1)
        batch_size, seq_len, _, num_labels = span_probs.shape

        batch_predictions = []

        for batch_idx in range(batch_size):
            # 获取有效序列长度
            valid_len = attention_mask[batch_idx].sum().item()

            # 候选跨度列表
            candidate_spans = []

            # 遍历所有可能的跨度位置
            for start in range(valid_len):
                for end in range(start, min(start + self._get_max_span_length(start, valid_len), valid_len)):
                    # 获取当前位置的概率分布
                    prob_dist = span_probs[batch_idx, start, end]

                    # 找到最大概率的标签（排除O标签，索引0）
                    non_o_probs = prob_dist[1:]  # 排除O标签
                    if len(non_o_probs) == 0:
                        continue

                    max_prob, max_label_idx = non_o_probs.max(dim=0)
                    actual_label_id = max_label_idx.item() + 1  # 加1因为排除了O标签

                    # 计算自适应阈值
                    adaptive_threshold = self._get_adaptive_threshold(
                        actual_label_id, end - start + 1, valid_len, threshold
                    )

                    # 检查是否超过阈值
                    if max_prob.item() > adaptive_threshold:
                        # 计算复合置信度分数
                        composite_score = self._calculate_composite_score(
                            prob_dist, actual_label_id, start, end, valid_len
                        )

                        candidate_spans.append({
                            'start': start,
                            'end': end,
                            'label': actual_label_id,
                            'confidence': max_prob.item(),
                            'composite_score': composite_score,
                            'span_length': end - start + 1
                        })

            # 解决跨度冲突并选择最终预测
            final_spans = self._resolve_span_conflicts_decode(candidate_spans)

            batch_predictions.append(final_spans)

        return batch_predictions

    def _get_max_span_length(self, start_pos, seq_len):
        """
        根据起始位置和序列长度获取最大跨度长度

        Args:
            start_pos: 起始位置
            seq_len: 序列长度

        Returns:
            最大跨度长度
        """
        # 基础最大长度限制
        max_length = getattr(self.config, 'max_span_length', 10)

        # 根据剩余序列长度调整
        remaining_length = seq_len - start_pos

        return min(max_length, remaining_length)

    def _get_adaptive_threshold(self, label_id, span_length, seq_length, base_threshold):
        """
        计算自适应阈值

        Args:
            label_id: 标签ID
            span_length: 跨度长度
            seq_length: 序列长度
            base_threshold: 基础阈值

        Returns:
            调整后的阈值
        """
        # 基础阈值
        threshold = base_threshold

        # 根据跨度长度调整：短跨度使用较低阈值，长跨度使用较高阈值
        if span_length == 1:
            threshold *= 0.8  # 单字符实体较容易识别
        elif span_length <= 3:
            threshold *= 0.9  # 短跨度
        elif span_length <= 6:
            threshold *= 1.0  # 中等长度
        else:
            threshold *= 1.2  # 长跨度需要更高置信度

        # 根据位置调整：序列开头和结尾的实体通常更可靠
        # 这里可以根据需要添加位置相关的调整

        # 确保阈值在合理范围内
        threshold = max(0.1, min(0.9, threshold))

        return threshold

    def _calculate_composite_score(self, prob_dist, label_id, start, end, seq_length):
        """
        计算复合置信度分数

        Args:
            prob_dist: [num_labels] 概率分布
            label_id: 预测标签ID
            start: 起始位置
            end: 结束位置
            seq_length: 序列长度

        Returns:
            复合置信度分数
        """
        # 基础概率分数
        base_score = prob_dist[label_id].item()

        # 熵惩罚：概率分布越分散，置信度越低
        entropy = -(prob_dist * torch.log(prob_dist + 1e-8)).sum().item()
        max_entropy = math.log(len(prob_dist))
        entropy_penalty = entropy / max_entropy  # 归一化到[0,1]

        # 长度奖励/惩罚
        span_length = end - start + 1
        if span_length <= 3:
            length_bonus = 0.1  # 短跨度奖励
        elif span_length <= 6:
            length_bonus = 0.0  # 中等长度无调整
        else:
            length_bonus = -0.1  # 长跨度惩罚

        # 位置奖励：开头和结尾的实体通常更可靠
        position_bonus = 0.0
        relative_start = start / seq_length
        relative_end = end / seq_length

        if relative_start < 0.2 or relative_end > 0.8:
            position_bonus = 0.05

        # 计算最终复合分数
        composite_score = (
            base_score * (1.0 - 0.3 * entropy_penalty) +
            length_bonus +
            position_bonus
        )

        return max(0.0, min(1.0, composite_score))

    def _resolve_span_conflicts_decode(self, candidate_spans):
        """
        解决候选跨度之间的冲突

        Args:
            candidate_spans: 候选跨度列表

        Returns:
            解决冲突后的最终跨度列表
        """
        if not candidate_spans:
            return []

        # 按复合分数排序
        sorted_spans = sorted(
            candidate_spans, key=lambda x: x['composite_score'], reverse=True)

        final_spans = []
        used_positions = set()

        for span in sorted_spans:
            start, end = span['start'], span['end']

            # 检查是否与已选择的跨度冲突
            conflict = False
            for pos in range(start, end + 1):
                if pos in used_positions:
                    conflict = True
                    break

            if not conflict:
                # 选择这个跨度
                final_spans.append({
                    'start': start,
                    'end': end,
                    'label': span['label'],
                    'confidence': span['confidence']
                })

                # 标记已使用的位置
                for pos in range(start, end + 1):
                    used_positions.add(pos)

        # 按起始位置排序
        final_spans.sort(key=lambda x: x['start'])

        return final_spans

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        return {
            'model_type': 'BertBiaffineSpanNER',
            'num_labels': self.num_labels,
            'hidden_size': self.hidden_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'use_biaffine': getattr(self.config, 'use_biaffine', True),
            'use_hierarchical': getattr(self.config, 'use_hierarchical', False),
            'use_boundary_detection': getattr(self.config, 'use_boundary_detection', False)
        }
