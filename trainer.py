from model import FreeLB, BertBiaffineSpanNER
from conll_reader import ConllReader
import torch.nn.functional as F
import json
import numpy as np
from sklearn.metrics import classification_report
import logging
import copy
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from config import Config
import torch
from typing import Dict
import os
import re


logger = logging.getLogger(__name__)


class StochasticWeightAveraging:
    def __init__(self, model, swa_start_epoch, swa_lr=None, swa_freq=5):
        """
        实现随机权重平均（SWA）
        Args:
            model: 要应用SWA的模型
            swa_start_epoch: 开始平均权重的轮次（从0开始索引）
            swa_lr: SWA期间使用的学习率（目前不用于优化器重新初始化）
            swa_freq: 更新SWA模型的频率（以轮次为单位）
        """
        self.model_base = model  # 保持对原始模型结构的引用以进行深拷贝
        self.swa_start_epoch = swa_start_epoch
        # 注意：SWA学习率通常由调度器处理或在SWA阶段使用固定的小学习率
        self.swa_lr = swa_lr
        self.swa_freq = swa_freq
        self.swa_model = None
        self.n_averaged = 0
        logger.info(
            f"SWA已初始化: start_epoch={swa_start_epoch}, lr={swa_lr}, freq={swa_freq}")

    def update(self, epoch, model_current_state):
        """通过与当前模型权重平均来更新SWA模型"""
        if epoch < self.swa_start_epoch:
            return

        if (epoch - self.swa_start_epoch) % self.swa_freq != 0:
            return

        logger.info(f"在第{epoch}轮更新SWA模型")
        if self.swa_model is None:
            # 从基础结构创建SWA模型
            self.swa_model = copy.deepcopy(self.model_base)
            self.swa_model.load_state_dict(copy.deepcopy(model_current_state))
            logger.info("SWA模型已用当前模型权重初始化。")
        else:
            # 更新参数的运行平均值
            current_params = dict(model_current_state)
            for name, swa_param in self.swa_model.named_parameters():
                if swa_param.requires_grad:
                    model_param = current_params[name]
                    swa_param.data.mul_(
                        self.n_averaged / (self.n_averaged + 1))
                    swa_param.data.add_(
                        model_param.data / (self.n_averaged + 1))
            logger.info(
                f"SWA模型已更新。n_averaged变为{self.n_averaged + 1}")
        self.n_averaged += 1

    def get_final_model_state_dict(self):
        """返回具有平均权重的SWA模型的state_dict"""
        if self.swa_model is None:
            logger.warning(
                "SWA已启用，但未进行平均。为SWA模型state_dict返回None。")
            return None
        logger.info("已检索最终SWA模型state_dict。")
        return self.swa_model.state_dict()


class SpanConverter:
    """
    处理BIOES格式到span格式的转换
    """

    def __init__(self, labels: list, label_scheme: str = 'BIOES'):
        self.labels = labels
        self.label_scheme = label_scheme

        # 构建span标签映射 - 用于span矩阵
        # 0: O标签, 1+: 实体类型标签
        self.label_to_id = {'O': 0}
        self.id_to_label = {0: 'O'}

        # 为每个实体类型分配ID（不包含BIOES前缀）
        label_id = 1
        for label in labels:
            self.label_to_id[label] = label_id
            self.id_to_label[label_id] = label
            label_id += 1

        self.num_labels = len(self.label_to_id)

        # 构建BIOES标签映射（仅用于序列标注兼容性，不用于span矩阵）
        self.bioes_to_id = {'O': 0}
        self.id_to_bioes = {0: 'O'}

        idx = 1
        for label in labels:
            if label_scheme == 'BIOES':
                self.bioes_to_id[f'B-{label}'] = idx
                self.bioes_to_id[f'I-{label}'] = idx + 1
                self.bioes_to_id[f'E-{label}'] = idx + 2
                self.bioes_to_id[f'S-{label}'] = idx + 3

                self.id_to_bioes[idx] = f'B-{label}'
                self.id_to_bioes[idx + 1] = f'I-{label}'
                self.id_to_bioes[idx + 2] = f'E-{label}'
                self.id_to_bioes[idx + 3] = f'S-{label}'
                idx += 4

        logger.info(
            f"SpanConverter初始化: 实体类型={len(labels)}个, 总标签数={self.num_labels}")

    def bioes_to_spans(self, bioes_sequence: list, text_tokens: list = None) -> list:
        """
        将BIOES序列转换为span格式

        Args:
            bioes_sequence: BIOES标签序列
            text_tokens: 对应的文本token序列（用于调试）

        Returns:
            list: span列表，每个span为(start, end, label)
        """
        spans = []
        current_span = None

        for i, tag in enumerate(bioes_sequence):
            if tag == 'O':
                # 结束当前span（如果有）
                if current_span is not None:
                    logger.warning(f"Incomplete span found: {current_span}")
                    current_span = None
                continue

            if tag.startswith('B-'):
                # 开始新span
                if current_span is not None:
                    logger.warning(f"Incomplete span found: {current_span}")
                current_span = {'start': i, 'label': tag[2:]}

            elif tag.startswith('I-'):
                # 继续当前span
                if current_span is None or current_span['label'] != tag[2:]:
                    logger.warning(f"Orphaned I- tag at position {i}: {tag}")
                    current_span = {'start': i, 'label': tag[2:]}

            elif tag.startswith('E-'):
                # 结束当前span
                if current_span is None or current_span['label'] != tag[2:]:
                    logger.warning(f"Orphaned E- tag at position {i}: {tag}")
                else:
                    spans.append(
                        (current_span['start'], i, current_span['label']))
                current_span = None

            elif tag.startswith('S-'):
                # 单token span
                if current_span is not None:
                    logger.warning(f"Incomplete span found: {current_span}")
                spans.append((i, i, tag[2:]))
                current_span = None

        # 处理未完成的span
        if current_span is not None:
            logger.warning(f"Incomplete span at end: {current_span}")

        return spans

    def spans_to_matrix(self, spans: list, seq_length: int) -> torch.Tensor:
        """
        将span列表转换为span标签矩阵

        Args:
            spans: span列表，每个span为(start, end, label)
            seq_length: 序列长度

        Returns:
            torch.Tensor: [seq_length, seq_length]的标签矩阵
        """
        span_matrix = torch.zeros(seq_length, seq_length, dtype=torch.long)

        for start, end, label in spans:
            if start < seq_length and end < seq_length and start <= end:
                label_id = self.label_to_id.get(label, 0)  # 默认为O标签
                if label_id > 0:  # 只设置非O标签
                    span_matrix[start, end] = label_id
                else:
                    # 记录无法找到的标签（可能的数据问题）
                    if label != 'O':
                        logger.warning(f"无法找到标签 '{label}' 在label_to_id映射中")

        return span_matrix

    def tokens_to_spans_batch(self, batch_bioes_sequences: list, batch_tokens: list = None) -> list:
        """
        批量转换BIOES序列到span格式
        """
        batch_spans = []
        for i, bioes_seq in enumerate(batch_bioes_sequences):
            tokens = batch_tokens[i] if batch_tokens else None
            spans = self.bioes_to_spans(bioes_seq, tokens)
            batch_spans.append(spans)
        return batch_spans


class SpanNERDataset:
    """
    Span-based NER数据集类
    """

    def __init__(self, examples: list, tokenizer_name_or_path, max_length: int, span_converter: SpanConverter):
        self.examples = examples
        self.tokenizer_name_or_path = tokenizer_name_or_path  # 存储路径而不是tokenizer对象
        self.max_length = max_length
        self.span_converter = span_converter
        self._tokenizer = None  # 懒加载的tokenizer

    @property
    def tokenizer(self):
        """懒加载tokenizer，避免在fork之前初始化"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path)
        return self._tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # 统一处理字典和对象格式
        if isinstance(example, dict):
            tokens = example['tokens']
            labels = example.get('labels', [])
        else:
            tokens = example.tokens
            labels = getattr(example, 'labels', [])

        # 分词和编码
        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors='pt'
        )

        # 处理标签对齐
        word_ids = encoding.word_ids()
        aligned_labels = ['O'] * len(word_ids)

        # 从标签获取BIOES格式标签
        if labels:
            for i, word_id in enumerate(word_ids):
                if word_id is not None and word_id < len(labels):
                    # 直接使用原始标签（应该已经是BIOES格式的字符串）
                    aligned_labels[i] = labels[word_id]

        # 转换为span格式
        spans = self.span_converter.bioes_to_spans(aligned_labels)
        seq_length = encoding['input_ids'].shape[1]
        span_labels = self.span_converter.spans_to_matrix(spans, seq_length)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'span_labels': span_labels
        }


class SpanEvaluator:
    """
    Span模型评估器
    """

    def __init__(self, span_converter: SpanConverter):
        self.span_converter = span_converter

    def extract_spans_from_matrix(self, span_matrix: torch.Tensor, attention_mask: torch.Tensor) -> list:
        """
        从span标签矩阵中提取span列表

        Args:
            span_matrix: [seq_len, seq_len] 的标签矩阵
            attention_mask: [seq_len] 的注意力掩码

        Returns:
            list: span列表，每个span为(start, end, label)
        """
        spans = []
        seq_len = attention_mask.sum().item()

        for start in range(seq_len):
            for end in range(start, seq_len):
                label_id = span_matrix[start, end].item()
                if label_id > 0:  # 0是O标签
                    label = self.span_converter.id_to_label.get(label_id, 'O')
                    if label != 'O':
                        spans.append((start, end, label))

        return spans

    def evaluate_spans(self, predictions: list, gold_spans: list) -> dict:
        """
        评估span预测结果

        Args:
            predictions: 模型预测的span列表
            gold_spans: 真实的span列表

        Returns:
            dict: 包含precision, recall, f1的评估结果
        """
        true_positives = 0
        predicted_spans = 0
        gold_spans_count = 0

        for pred_spans, gold_spans_batch in zip(predictions, gold_spans):
            pred_set = set()
            gold_set = set()

            # 转换预测结果
            if isinstance(pred_spans, list):
                for span in pred_spans:
                    if isinstance(span, dict):
                        # 关键修复：将预测的label ID转换为字符串标签
                        label_id = span['label']
                        if isinstance(label_id, int):
                            # 使用span_converter将ID转换为标签字符串
                            label_str = self.span_converter.id_to_label.get(
                                label_id, 'O')
                            if label_str != 'O':  # 只添加非O标签的span
                                pred_set.add(
                                    (span['start'], span['end'], label_str))
                        else:
                            # 如果已经是字符串，直接使用
                            pred_set.add(
                                (span['start'], span['end'], label_id))
                    else:
                        pred_set.add(span)  # 已经是tuple格式

            # 转换金标准
            for span in gold_spans_batch:
                gold_set.add(span)

            # 计算指标
            predicted_spans += len(pred_set)
            gold_spans_count += len(gold_set)
            true_positives += len(pred_set & gold_set)

        # 计算precision, recall, f1
        precision = true_positives / predicted_spans if predicted_spans > 0 else 0.0
        recall = true_positives / gold_spans_count if gold_spans_count > 0 else 0.0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'predicted_spans': predicted_spans,
            'gold_spans': gold_spans_count
        }


class BiaffineSpanTrainer:
    """
    BERT + Biaffine + Span-based NER训练器
    针对8GB显存优化
    """

    def __init__(self, config: Config, device: str = 'cuda'):
        self.config = config
        self.device = device

        # 设置随机种子
        self._set_seed(config.seed)

        # 初始化数据
        self._load_data()

        # 初始化模型
        self._init_model()

        # 初始化优化器和调度器
        self._init_optimizer()

        # 初始化对抗训练
        self._init_adversarial_training()

        # 初始化SWA
        self._init_swa()

        # 初始化评估器
        self.evaluator = SpanEvaluator(self.span_converter)

        # 训练统计
        self.global_step = 0
        self.best_f1 = 0.0
        self.train_losses = []
        self.val_metrics = []

    def _set_seed(self, seed: int):
        """设置随机种子"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _load_data(self):
        """加载和预处理数据"""
        logger.info("Loading span data...")

        # 读取数据
        conll_reader = ConllReader()
        train_data = list(conll_reader.read(self.config.train_file))
        val_data = list(conll_reader.read(self.config.dev_file))

        logger.info(f"Raw train data samples: {len(train_data)}")
        logger.info(f"Raw val data samples: {len(val_data)}")

        # 关键修复：从config中获取正确的实体类型列表
        if hasattr(self.config, 'label_map') and hasattr(self.config.label_map, 'labels'):
            # 从完整的BIOES标签列表中提取纯实体类型
            entity_labels = set()
            for label in self.config.label_map.labels:
                if label != 'O' and '-' in label:
                    # 从BIOES标签中提取实体类型 (如 'B-city' -> 'city')
                    entity_type = label.split('-', 1)[1]
                    entity_labels.add(entity_type)
            entity_labels = sorted(list(entity_labels))
            logger.info(
                f"从config.label_map.labels提取实体类型: {len(entity_labels)}个")
        else:
            # fallback：从数据中提取实体类型
            entity_types = set()
            for sample in train_data[:100]:  # 检查前100个样本
                labels = getattr(sample, 'labels', []) if hasattr(
                    sample, 'labels') else sample.get('labels', [])
                for label in labels:
                    if label != 'O' and '-' in label:
                        entity_type = label.split('-', 1)[1]  # 提取实体类型
                        entity_types.add(entity_type)
            entity_labels = sorted(list(entity_types))
            logger.info(f"从数据中提取的实体类型: {len(entity_labels)}个")

        # 验证实体标签列表
        if not entity_labels:
            raise ValueError("未找到任何实体类型！请检查config.label_map或数据格式")

        # 创建span转换器 - 传入纯实体类型
        self.span_converter = SpanConverter(
            # 这里应该是['prov', 'city', 'district', ...]而不是BIOES格式
            labels=entity_labels,
            label_scheme=getattr(self.config.label_map, 'type', 'BIOES') if hasattr(
                self.config, 'label_map') else 'BIOES'
        )

        # 创建数据集
        # tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)  # 注释掉，改为懒加载

        # 针对显存优化的最大长度
        max_length = getattr(self.config, 'max_sequence_length', 384)

        self.train_dataset = SpanNERDataset(
            train_data, self.config.model_name, max_length, self.span_converter)  # 传入模型名称
        self.val_dataset = SpanNERDataset(
            val_data, self.config.model_name, max_length, self.span_converter)  # 传入模型名称

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True if self.device == 'cuda' else False
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True if self.device == 'cuda' else False
        )

        logger.info(f"Train span samples: {len(self.train_dataset)}")
        logger.info(f"Val span samples: {len(self.val_dataset)}")
        logger.info(f"Span labels: {self.span_converter.num_labels}")

        # 快速验证数据转换
        total_spans = 0
        total_non_zero = 0

        for i in range(min(10, len(self.train_dataset))):  # 只检查前10个样本
            try:
                sample_data = self.train_dataset[i]
                span_labels = sample_data['span_labels']
                total_spans += span_labels.numel()
                non_zero_count = (span_labels > 0).sum().item()
                total_non_zero += non_zero_count
            except Exception as e:
                logger.error(f"样本{i}转换失败: {e}")

        if total_non_zero == 0:
            logger.error("严重错误：没有找到任何非零span标签！数据转换有问题。")
            logger.error(f"检查的样本数: {min(10, len(self.train_dataset))}")
            raise ValueError("Span标签转换失败，所有标签都是0")
        else:
            logger.info(
                f"数据转换验证通过: 非零span标签比例={total_non_zero/total_spans:.6f}")
            logger.info(f"总span位置数: {total_spans}, 非零span数: {total_non_zero}")

    def _init_model(self):
        """初始化模型"""
        num_labels = self.span_converter.num_labels
        self.model = BertBiaffineSpanNER(num_labels, self.config)
        self.model.to(self.device)

        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        logger.info(
            f"Span model loaded. Total params: {total_params:,}, Trainable: {trainable_params:,}")

        # 检查显存使用
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            logger.info(f"GPU memory allocated: {allocated:.1f} MB")

    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        # 分离BERT和其他参数的学习率
        bert_params = []
        span_classifier_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bert' in name:
                    bert_params.append(param)
                elif 'span_classifier' in name or 'biaffine' in name:
                    span_classifier_params.append(param)
                else:
                    other_params.append(param)

        # 设置不同的学习率
        bert_lr = self.config.learning_rate
        span_lr = self.config.learning_rate * 5  # span分类器使用更高的学习率
        other_lr = self.config.learning_rate * 2  # 其他层使用中等学习率

        optimizer_grouped_parameters = [
            {'params': bert_params, 'lr': bert_lr,
                'weight_decay': self.config.weight_decay},
            {'params': span_classifier_params, 'lr': span_lr,
                'weight_decay': self.config.weight_decay},
            {'params': other_params, 'lr': other_lr,
                'weight_decay': self.config.weight_decay}
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        # 学习率调度器
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        logger.info(
            f"Optimizer initialized for span model. Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        logger.info(
            f"Parameter groups: BERT={len(bert_params)}, Span={len(span_classifier_params)}, Other={len(other_params)}")

    def _init_adversarial_training(self):
        """初始化对抗训练"""
        self.use_adversarial = getattr(self.config, 'use_freelb', False)
        if self.use_adversarial:
            self.freelb = FreeLB(
                model=self.model,
                adv_lr=getattr(self.config, 'freelb_adv_lr', 0.05),
                adv_steps=getattr(self.config, 'freelb_adv_steps', 3),
                adv_init_mag=getattr(self.config, 'freelb_adv_init_mag', 0.05),
                adv_max_norm=getattr(self.config, 'freelb_adv_max_norm', 0.07),
                adv_norm_type=getattr(
                    self.config, 'freelb_adv_norm_type', 'l2'),
                base_model=getattr(self.config, 'freelb_base_model', 'bert')
            )
            self.adv_start_epoch = getattr(
                self.config, 'adversarial_training_start_epoch', 3)
            logger.info(
                f"FreeLB adversarial training enabled for span model, starting from epoch {self.adv_start_epoch}, base_model={self.config.freelb_base_model}")
        else:
            logger.info("FreeLB adversarial training disabled")

    def _init_swa(self):
        """初始化SWA"""
        self.use_swa = getattr(self.config, 'use_swa', False)
        if self.use_swa:
            self.swa = StochasticWeightAveraging(
                model=self.model,
                swa_start_epoch=getattr(self.config, 'swa_start_epoch', 0),
                swa_lr=getattr(self.config, 'swa_lr', 1e-5),
                swa_freq=getattr(self.config, 'swa_freq', 2)
            )
            logger.info("SWA enabled for span model")

    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)

        # 是否使用对抗训练
        use_adv_this_epoch = self.use_adversarial and epoch >= self.adv_start_epoch

        progress_bar = tqdm(self.train_dataloader, desc=f"Span Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            span_labels = batch['span_labels'].to(self.device)

            if use_adv_this_epoch:
                # 对抗训练
                inputs_embeds = self.model.bert.embeddings.word_embeddings(
                    input_ids).detach()
                inputs_embeds.requires_grad_(True)

                loss = self.freelb.attack_span(
                    inputs_embeds, attention_mask, span_labels)

            else:
                # 常规训练
                self.optimizer.zero_grad()

                try:
                    loss = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        span_labels=span_labels
                    )

                    # 检查损失值
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Invalid loss detected: {loss}")
                        continue

                    # 确保损失有梯度
                    if not loss.requires_grad:
                        logger.warning("Loss does not require grad!")
                        continue

                    loss.backward()

                except Exception as e:
                    logger.error(f"Error during forward pass: {e}")
                    continue

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            # 优化器步骤
            self.optimizer.step()
            self.scheduler.step()

            # 统计
            loss_value = loss
            total_loss += loss_value
            self.global_step += 1

            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'adv': 'ON' if use_adv_this_epoch else 'OFF'
            })

            # 显存清理
            if batch_idx % 100 == 0 and self.device == 'cuda':
                torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        # SWA更新
        if self.use_swa:
            self.swa.update(epoch, self.model.state_dict())

        return avg_loss

    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_gold_standards = []
        total_loss = 0.0
        num_batches = len(self.val_dataloader)

        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc="Span Evaluating")

            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                span_labels = batch['span_labels'].to(self.device)

                # 计算损失
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    span_labels=span_labels
                )
                total_loss += loss

                # 获取预测
                predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # 提取gold spans
                batch_size = span_labels.size(0)
                for i in range(batch_size):
                    # 预测spans - 使用config中的threshold
                    pred_spans = predictions[i] if isinstance(predictions, list) else \
                        self.model.decode_spans(
                            predictions[i:i+1], attention_mask[i:i+1], threshold=self.config.span_threshold)[0]
                    all_predictions.append(pred_spans)

                    # Gold spans
                    gold_spans = self.evaluator.extract_spans_from_matrix(
                        span_labels[i], attention_mask[i]
                    )
                    all_gold_standards.append(gold_spans)

        # 计算评估指标
        metrics = self.evaluator.evaluate_spans(
            all_predictions, all_gold_standards)
        metrics['val_loss'] = total_loss / num_batches

        return metrics

    def train(self) -> float:
        """完整训练流程"""
        logger.info("Starting span training...")

        best_f1 = 0.0
        patience = self.config.early_stopping_patience
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)

            # 评估
            metrics = self.evaluate()
            val_f1 = metrics['f1']
            val_loss = metrics['val_loss']

            # 记录
            self.val_metrics.append(metrics)

            logger.info(
                f"Span Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_f1={val_f1:.4f}, "
                f"val_precision={metrics['precision']:.4f}, "
                f"val_recall={metrics['recall']:.4f}"
            )

            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.best_f1 = best_f1
                patience_counter = 0
                self._save_model('best_model.pt')
                logger.info(f"New best span F1: {best_f1:.4f}")
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered after {patience} epochs without improvement")
                break

            # 显存清理
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # 使用SWA权重进行最终评估
        if self.use_swa:
            logger.info("Evaluating with SWA weights...")
            swa_state_dict = self.swa.get_final_model_state_dict()
            if swa_state_dict is not None:
                self.model.load_state_dict(swa_state_dict)
                swa_metrics = self.evaluate()
                logger.info(f"SWA Span F1: {swa_metrics['f1']:.4f}")

                if swa_metrics['f1'] > best_f1:
                    best_f1 = swa_metrics['f1']
                    self._save_model('swa_model.pt')
                    logger.info(f"SWA achieved better span F1: {best_f1:.4f}")

        logger.info(f"Span training completed. Best F1: {best_f1:.4f}")
        # self._save_training_log()  # 注释掉训练日志保存

        return best_f1

    def _save_model(self, filename: str):
        """保存模型"""
        # 提取模型名称（处理不同格式的模型名称）
        model_name = self.config.model_name

        # 如果是路径格式，提取最后一部分作为模型名称
        if '/' in model_name:
            model_name = model_name.split('/')[-1]

        # 如果包含特殊字符，进行清理以确保文件系统兼容性
        model_name = re.sub(r'[<>:"|?*]', '_', model_name)

        # 创建新的保存路径：result/pretrained/[model_name]/
        save_dir = os.path.join('result', 'pretrained', model_name)
        model_path = os.path.join(save_dir, filename)

        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_f1': self.best_f1,
            'span_converter': self.span_converter
        }, model_path)

        logger.info(f"Span model saved to {model_path}")

    def _save_training_log(self):
        """保存训练日志 - 已禁用"""
        logger.info("Training log saving disabled")
        return

    def train_with_kfold_cv(self) -> Dict[str, float]:
        """
        使用K折交叉验证进行训练

        Returns:
            dict: 包含平均性能指标的字典
        """
        logger.info(f"Starting {self.config.k_folds}-fold cross validation...")

        from sklearn.model_selection import KFold
        import numpy as np

        # 合并训练和验证数据进行K折分割
        all_data = list(self.train_dataset.examples) + \
            list(self.val_dataset.examples)

        kf = KFold(n_splits=self.config.k_folds, shuffle=True,
                   random_state=self.config.seed)
        fold_results = []

        for fold, (train_indices, val_indices) in enumerate(kf.split(all_data)):
            logger.info(f"Training fold {fold + 1}/{self.config.k_folds}")

            # 分割数据
            fold_train_data = [all_data[i] for i in train_indices]
            fold_val_data = [all_data[i] for i in val_indices]

            # 创建新的数据集
            fold_train_dataset = SpanNERDataset(
                fold_train_data, self.config.model_name,
                getattr(self.config, 'max_sequence_length', 384),
                self.span_converter
            )
            fold_val_dataset = SpanNERDataset(
                fold_val_data, self.config.model_name,
                getattr(self.config, 'max_sequence_length', 384),
                self.span_converter
            )

            # 创建数据加载器
            fold_train_dataloader = DataLoader(
                fold_train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                pin_memory=True if self.device == 'cuda' else False
            )
            fold_val_dataloader = DataLoader(
                fold_val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                pin_memory=True if self.device == 'cuda' else False
            )

            # 重新初始化模型
            num_labels = self.span_converter.num_labels
            self.model = BertBiaffineSpanNER(num_labels, self.config)
            self.model.to(self.device)

            # 重新初始化优化器
            self._init_optimizer()

            # 临时替换数据加载器
            original_train_dataloader = self.train_dataloader
            original_val_dataloader = self.val_dataloader
            self.train_dataloader = fold_train_dataloader
            self.val_dataloader = fold_val_dataloader

            # 训练当前fold
            fold_best_f1 = self.train()
            fold_results.append(fold_best_f1)

            # 恢复原始数据加载器
            self.train_dataloader = original_train_dataloader
            self.val_dataloader = original_val_dataloader

            logger.info(
                f"Fold {fold + 1} completed with F1: {fold_best_f1:.4f}")

        # 计算平均性能
        mean_f1 = np.mean(fold_results)
        std_f1 = np.std(fold_results)

        logger.info(
            f"K-fold CV completed. Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
        logger.info(f"Individual fold results: {fold_results}")

        # 保存交叉验证结果
        cv_results = {
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'fold_results': fold_results,
            'k_folds': self.config.k_folds
        }

        # cv_log_path = os.path.join(
        #     self.config.work_dir, 'kfold_cv_results.json')
        # os.makedirs(self.config.work_dir, exist_ok=True)

        # import json
        # with open(cv_log_path, 'w', encoding='utf-8') as f:
        #     json.dump(cv_results, f, indent=2, ensure_ascii=False)

        # logger.info(f"K-fold CV results saved to {cv_log_path}")

        return cv_results
