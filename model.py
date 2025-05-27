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


class AddressNER(nn.Module):
    def __init__(self, num_labels: int, config: Config):
        super(AddressNER, self).__init__()
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # 添加嵌入层dropout
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)

        # 添加空间dropout
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

        # 初始化损失函数
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

        # 冻结BERT层
        self._freeze_bert_layers(config.freeze_bert_layers)

    def _freeze_bert_layers(self, num_layers_to_freeze):
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

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None):
        if inputs_embeds is not None:
            # 如果提供了inputs_embeds，直接使用它们
            # 如果有意义的话，可能会应用embedding_dropout，尽管FreeLB在初始嵌入后进行扰动
            if self.training and self.embedding_dropout.p > 0:
                # 如果FreeLB已经在嵌入上工作，再次应用dropout有点不寻常
                # 但是，如果原始意图是在BERT编码器*之前*对嵌入进行dropout，
                # 这可能是一个地方。对于FreeLB，通常攻击是在干净的嵌入上进行的。
                # 让我们假设FreeLB提供的嵌入已经准备好用于BERT编码器。
                # 或者如果认为有必要，应用self.embedding_dropout(inputs_embeds)。
                pass
            outputs = self.bert(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask)
        elif input_ids is not None:
            # 原始路径：直接对来自input_ids的输入嵌入应用嵌入dropout
            if self.training and self.embedding_dropout.p > 0:
                # 获取嵌入
                current_embeddings = self.bert.embeddings.word_embeddings(
                    input_ids)
                # 对嵌入应用dropout
                current_embeddings = self.embedding_dropout(current_embeddings)
                # 将修改后的嵌入传递给BERT
                outputs = self.bert(
                    inputs_embeds=current_embeddings, attention_mask=attention_mask)
            else:
                outputs = self.bert(input_ids=input_ids,
                                    attention_mask=attention_mask)
        else:
            raise ValueError(
                "必须提供input_ids或inputs_embeds中的一个。")

        bert_output = outputs.last_hidden_state  # [batch, seq_len, 768]

        # 对BERT输出应用空间dropout
        bert_output = self.spatial_dropout(bert_output)

        # 将BERT输出直接传递给LSTM
        lstm_output, _ = self.lstm(bert_output)

        logits = self.classifier(lstm_output)

        if labels is not None:
            # 训练期间，计算混合损失
            loss = self.hybrid_loss(logits, labels, attention_mask.bool())
            return loss.mean()
        else:
            # 推理期间，解码最佳路径
            return self.crf.viterbi_decode(logits, mask=attention_mask.bool())

    def __len__(self):
        return len(self.data)


class FreeLB:
    def __init__(self, model, adv_lr=1e-1, adv_steps=3, adv_init_mag=2e-2, adv_max_norm=0.0, adv_norm_type='l2', base_model='bert'):
        self.model = model
        self.adv_lr = adv_lr
        self.adv_steps = adv_steps
        self.adv_init_mag = adv_init_mag
        self.adv_max_norm = adv_max_norm    # 如果为0，使用adv_init_mag作为约束
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model  # 目前未使用，但保留以保持签名一致性

    def attack(self, inputs_embeds, attention_mask, labels):
        """
        执行FreeLB攻击并在模型参数上累积梯度。
        inputs_embeds: 输入的原始嵌入（已分离）。
        """
        # 在与inputs_embeds相同的设备上初始化对抗扰动delta
        delta = torch.zeros_like(inputs_embeds, device=inputs_embeds.device)
        if self.adv_init_mag > 0:  # 只有当adv_init_mag为正时才应用均匀噪声
            delta.uniform_(-self.adv_init_mag, self.adv_init_mag)
        delta.requires_grad = True

        accumulated_loss_for_log = 0.0

        # 在开始对抗步骤之前，一次性清零模型梯度。
        # 这确保累积的梯度仅来自对抗扰动。
        self.model.zero_grad()

        for i in range(self.adv_steps):
            perturbed_embeds = inputs_embeds + delta

            loss_adv = self.model(
                inputs_embeds=perturbed_embeds, attention_mask=attention_mask, labels=labels)
            accumulated_loss_for_log += loss_adv.item()  # 记录此步骤的原始损失

            # 标准化损失以进行反向传播，以平均各步骤的梯度
            # 如FreeLB算法中指定的（梯度是K步梯度的平均值）
            loss_adv_normalized = loss_adv / self.adv_steps

            # 为其自身更新步骤清零delta梯度
            if delta.grad is not None:
                delta.grad.data.zero_()

            # 从此对抗步骤在self.model.parameters中累积梯度
            loss_adv_normalized.backward()

            # 更新扰动delta
            if delta.grad is None:  # 如果loss_adv依赖于delta且模型有可训练参数，这不应该发生
                # 如果确实发生了，意味着没有梯度流向delta，可能是模型结构问题或adv_steps=0
                if self.adv_steps > 0:  # 只有在我们期望步骤时才中断
                    logger.warning(
                        "FreeLB攻击期间Delta梯度为None，提前中断。")
                break

            # 在应用学习率之前标准化delta的梯度
            # 展平以计算嵌入维度上的范数，然后unsqueeze以匹配delta形状
            flat_delta_grad = delta.grad.data.flatten(1)
            if self.adv_norm_type == 'l2':
                delta_grad_norm = torch.norm(
                    flat_delta_grad, p=2, dim=1, keepdim=True)
            elif self.adv_norm_type == 'linf':  # 技术上，对于L-inf，更新通常只是基于符号
                # 但FreeLB论文对L2使用g_t / ||g_t||，让我们适应L-inf范数约束
                delta_grad_norm = torch.norm(
                    flat_delta_grad, p=float('inf'), dim=1, keepdim=True)
            else:
                raise ValueError("adv_norm_type必须是'l2'或'linf'")

            # Unsqueeze grad_norm以匹配delta的维度[B, S, H]从[B, 1]
            for _ in range(len(inputs_embeds.shape) - len(delta_grad_norm.shape)):
                delta_grad_norm = delta_grad_norm.unsqueeze(-1)

            # 更新delta，在分母中添加小的epsilon以保持稳定性
            delta.data = delta.data + self.adv_lr * \
                (delta.grad.data / (delta_grad_norm + 1e-12))

            # 将delta投影回范数球
            # 如果指定了adv_max_norm则使用它，否则使用adv_init_mag作为约束半径
            effective_constraint_norm = self.adv_max_norm if self.adv_max_norm > 0 else self.adv_init_mag

            if effective_constraint_norm > 0:  # 只有在给定正约束时才投影
                flat_delta = delta.data.flatten(1)
                if self.adv_norm_type == 'l2':
                    current_delta_norm = torch.norm(
                        flat_delta, p=2, dim=1, keepdim=True)
                elif self.adv_norm_type == 'linf':
                    current_delta_norm = torch.norm(
                        flat_delta, p=float('inf'), dim=1, keepdim=True)

                for _ in range(len(inputs_embeds.shape) - len(current_delta_norm.shape)):
                    current_delta_norm = current_delta_norm.unsqueeze(-1)

                # 计算裁剪系数：min(1, constraint_norm / current_norm)
                clip_coef = (effective_constraint_norm /
                             (current_delta_norm + 1e-12))
                clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
                delta.data = delta.data * clip_coef

        # 梯度现在已在self.model.parameters()中累积。
        # 训练器将使用这些梯度调用optimizer.step()。
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
