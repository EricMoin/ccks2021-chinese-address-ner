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


class AddressSpanConverter:
    """
    专门针对中文地址NER任务优化的Span转换器

    核心改进：
    1. 基于地址结构层次的span表示
    2. 避免span重叠冲突的智能span选择
    3. 针对中文地址特点的span过滤策略
    4. 更精确的阈值和后处理机制
    """

    def __init__(self, labels: list, label_scheme: str = 'BIOES'):
        self.labels = labels
        self.label_scheme = label_scheme

        # 构建标签映射
        self.label_to_id = {'O': 0}
        self.id_to_label = {0: 'O'}

        label_id = 1
        for label in labels:
            self.label_to_id[label] = label_id
            self.id_to_label[label_id] = label
            label_id += 1

        self.num_labels = len(self.label_to_id)

        # 中文地址结构层次定义
        self.address_hierarchy = {
            'administrative': ['prov', 'city', 'district', 'devzone', 'town', 'community', 'village_group'],
            'location': ['road', 'roadno', 'poi', 'subpoi'],
            'building': ['houseno', 'cellno', 'floorno', 'roomno'],
            'auxiliary': ['detail', 'assist', 'distance', 'intersection', 'redundant', 'others']
        }

        # 为每个标签分配层次级别和优先级
        self.label_priority = {}
        self.label_hierarchy_level = {}

        for level, (category, labels_in_category) in enumerate(self.address_hierarchy.items()):
            for priority, label in enumerate(labels_in_category):
                if label in self.label_to_id:
                    self.label_hierarchy_level[label] = level
                    self.label_priority[label] = priority

        logger.info(
            f"AddressSpanConverter初始化: {len(labels)}个标签, {len(self.address_hierarchy)}个层次")

    def bioes_to_spans(self, bioes_sequence: list, text_tokens: list = None) -> list:
        """
        增强的BIOES到span转换，专门处理中文地址特点
        """
        spans = []
        current_span = None

        for i, tag in enumerate(bioes_sequence):
            if tag == 'O':
                if current_span is not None:
                    spans.append(
                        (current_span['start'], i - 1, current_span['label']))
                    current_span = None
                continue

            if '-' not in tag:
                continue

            prefix, entity_type = tag.split('-', 1)

            if prefix == 'B':
                if current_span is not None:
                    spans.append(
                        (current_span['start'], i - 1, current_span['label']))
                current_span = {'start': i, 'label': entity_type}

            elif prefix == 'I':
                if current_span is None or current_span['label'] != entity_type:
                    if current_span is not None:
                        spans.append(
                            (current_span['start'], i - 1, current_span['label']))
                    current_span = {'start': i, 'label': entity_type}

            elif prefix == 'E':
                if current_span is None:
                    spans.append((i, i, entity_type))
                elif current_span['label'] == entity_type:
                    spans.append(
                        (current_span['start'], i, current_span['label']))
                else:
                    spans.append(
                        (current_span['start'], i - 1, current_span['label']))
                    spans.append((i, i, entity_type))
                current_span = None

            elif prefix == 'S':
                if current_span is not None:
                    spans.append(
                        (current_span['start'], i - 1, current_span['label']))
                spans.append((i, i, entity_type))
                current_span = None

        if current_span is not None:
            spans.append((current_span['start'], len(
                bioes_sequence) - 1, current_span['label']))

        return self._resolve_span_conflicts(spans)

    def _resolve_span_conflicts(self, spans: list) -> list:
        """
        解决span冲突，基于中文地址结构的优先级 - 更宽松的冲突解决
        """
        if not spans:
            return spans

        # 按层次级别和优先级排序
        sorted_spans = sorted(spans, key=lambda x: (
            self.label_hierarchy_level.get(x[2], 999),  # 层次级别
            self.label_priority.get(x[2], 999),         # 同层次内优先级
            x[0],                                       # 起始位置
            -(x[1] - x[0])                             # 长度（长的优先）
        ))

        resolved_spans = []
        occupied_positions = set()

        for start, end, label in sorted_spans:
            span_positions = set(range(start, end + 1))

            # 检查是否与已选择的span冲突
            if not span_positions & occupied_positions:
                resolved_spans.append((start, end, label))
                occupied_positions.update(span_positions)
            else:
                # 更宽松的部分重叠处理：从0.5降低到0.3
                available_positions = span_positions - occupied_positions
                if len(available_positions) >= max(1, len(span_positions) * 0.3):  # 从0.5降低到0.3
                    # 如果超过30%位置可用，调整span边界（之前是50%）
                    new_start = min(available_positions)
                    new_end = max(available_positions)
                    if new_end >= new_start:
                        resolved_spans.append((new_start, new_end, label))
                        occupied_positions.update(
                            range(new_start, new_end + 1))

        return resolved_spans

    def spans_to_matrix(self, spans: list, seq_length: int) -> torch.Tensor:
        """
        优化的span矩阵构建，专门针对中文地址NER - 进一步提高成功率

        关键改进：
        1. 基于地址结构的智能span过滤
        2. 更合理的长度限制策略  
        3. 层次化的span优先级处理
        4. 针对中文地址的语言学约束
        """
        span_matrix = torch.zeros(seq_length, seq_length, dtype=torch.long)

        # 解决span冲突
        resolved_spans = self._resolve_span_conflicts(spans)

        # 统计信息
        total_spans = len(spans)
        successful_spans = 0
        filtered_by_length = 0
        filtered_by_position = 0

        for start, end, label in resolved_spans:
            span_len = end - start + 1

            # 基本边界检查
            if not (0 <= start < seq_length and 0 <= end < seq_length and start <= end):
                continue

            # 基于地址结构的长度限制
            max_length = self._get_max_length_for_label(label, seq_length)
            if span_len > max_length:
                filtered_by_length += 1
                # 对于超长span，尝试截断而不是直接丢弃
                if span_len <= max_length * 1.5:  # 如果不是特别长，尝试截断
                    # 保留前部分
                    new_end = start + max_length - 1
                    if new_end < seq_length:
                        end = new_end
                        span_len = end - start + 1
                    else:
                        continue
                else:
                    continue

            # 位置合理性检查
            if not self._is_position_reasonable(start, end, seq_length):
                filtered_by_position += 1
                continue

            # 设置标签
            label_id = self.label_to_id.get(label, 0)
            if label_id > 0:
                span_matrix[start, end] = label_id
                successful_spans += 1

        # 统计报告 - 调整成功率阈值
        if total_spans > 0:
            success_rate = successful_spans / total_spans
            logger.debug(f"地址Span矩阵构建: 总数={total_spans}, 成功={successful_spans}, "
                         f"成功率={success_rate:.3f}, 长度过滤={filtered_by_length}, "
                         f"位置过滤={filtered_by_position}")

            # if success_rate < 0.8:  # 从0.7提升到0.8，期望更高成功率
            #     logger.warning(f"地址span成功率偏低 ({success_rate:.3f})，检查数据或参数设置")
            # elif success_rate >= 0.8:
            #     logger.info(f"地址span成功率良好 ({success_rate:.3f})")

        return span_matrix

    def _get_max_length_for_label(self, label: str, seq_length: int) -> int:
        """
        根据地址要素类型返回合理的最大长度 - 进一步放宽限制
        """
        # 进一步放宽的长度限制，基于真实中文地址数据分析
        length_limits = {
            'prov': 8,        # 省份：新疆维吾尔自治区 (8字符)
            'city': 15,       # 城市：内蒙古自治区呼和浩特市 (15字符)
            'district': 18,   # 区县：经济技术开发区管委会 (18字符)
            'devzone': 25,    # 开发区：国家级经济技术开发区 (25字符)
            'town': 20,       # 乡镇：某某街道办事处社区 (20字符)
            'community': 25,  # 社区：某某社区居民委员会 (25字符)
            'village_group': 15,  # 村组：某某村民小组 (15字符)
            'road': 30,       # 道路：人民大道中山北路延长线 (30字符)
            'roadno': 12,     # 路号：12345-6789号 (12字符)
            'poi': 40,        # POI：某某大学某某学院某某楼 (40字符)
            'subpoi': 50,     # 子POI：某某商场某某专柜某某品牌店 (50字符)
            'houseno': 12,    # 门牌号：123栋456单元 (12字符)
            'cellno': 10,     # 单元号：第3单元A座 (10字符)
            'floorno': 8,     # 楼层：地下2层 (8字符)
            'roomno': 10,     # 房间号：1502室A (10字符)
            'detail': 35,     # 详细信息：靠近某某路口往南50米 (35字符)
            'assist': 25,     # 辅助信息：红色大门旁边小巷内 (25字符)
            'distance': 20,   # 距离信息：距离地铁站500米 (20字符)
            'intersection': 25,  # 交叉口：人民路与中山路交叉口 (25字符)
            'redundant': 20,  # 冗余信息 (20字符)
            'others': 25      # 其他 (25字符)
        }

        base_limit = length_limits.get(label, 25)  # 默认上限从15提升到25

        # 根据序列长度的动态调整也更宽松
        if seq_length < 100:
            return min(base_limit, seq_length // 3)    # 从//4改为//3
        elif seq_length < 200:
            return min(base_limit, seq_length // 4)    # 从//6改为//4
        else:
            return min(base_limit, seq_length // 6)    # 从//8改为//6

    def _is_position_reasonable(self, start: int, end: int, seq_length: int) -> bool:
        """
        检查span位置是否合理 - 进一步放宽限制
        """
        span_length = end - start + 1

        # 放宽序列末尾限制，从0.9提升到0.95
        if end >= seq_length * 0.95:
            return False

        # 放宽长度限制，允许更长的span
        if span_length < 1 or span_length > seq_length // 2:  # 从//3改为//2
            return False

        return True

    def create_hierarchical_span_mask(self, seq_length: int, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        创建基于地址层次的span掩码
        """
        mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)

        effective_seq_len = attention_mask.sum().item(
        ) if attention_mask is not None else seq_length

        # 根据地址层次设置不同的span长度优先级
        for length in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20]:
            if length > effective_seq_len // 3:
                break

            for start in range(effective_seq_len - length + 1):
                end = start + length - 1
                if attention_mask is None or (attention_mask[start] > 0 and attention_mask[end] > 0):
                    mask[start, end] = True

        return mask


class AddressSpanNERDataset:
    """
    专门针对中文地址NER优化的数据集
    """

    def __init__(self, examples: list, tokenizer_name_or_path, max_length: int, span_converter: AddressSpanConverter):
        self.examples = examples
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.max_length = max_length
        self.span_converter = span_converter
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path)
        return self._tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        if isinstance(example, dict):
            tokens = example['tokens']
            labels = example.get('labels', [])
        else:
            tokens = example.tokens
            labels = getattr(example, 'labels', [])

        # 编码
        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors='pt'
        )

        # 标签对齐
        word_ids = encoding.word_ids()
        aligned_labels = ['O'] * len(word_ids)

        if labels:
            for i, word_id in enumerate(word_ids):
                if word_id is not None and word_id < len(labels):
                    aligned_labels[i] = labels[word_id]

        # 转换为span
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

    def __init__(self, span_converter: AddressSpanConverter):
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
        self.span_converter = AddressSpanConverter(
            labels=entity_labels,
            label_scheme=getattr(self.config.label_map, 'type', 'BIOES') if hasattr(
                self.config, 'label_map') else 'BIOES'
        )

        # 创建数据集
        # tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)  # 注释掉，改为懒加载

        # 针对显存优化的最大长度
        max_length = getattr(self.config, 'max_sequence_length', 384)

        self.train_dataset = AddressSpanNERDataset(
            train_data, self.config.model_name, max_length, self.span_converter)  # 传入模型名称
        self.val_dataset = AddressSpanNERDataset(
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
        sample_spans_analysis = []

        for i in range(min(10, len(self.train_dataset))):  # 只检查前10个样本
            try:
                sample_data = self.train_dataset[i]
                span_labels = sample_data['span_labels']
                attention_mask = sample_data['attention_mask']

                # 获取有效序列长度
                effective_seq_len = attention_mask.sum().item()

                total_spans += span_labels.numel()
                non_zero_count = (span_labels > 0).sum().item()
                total_non_zero += non_zero_count

                # 分析每个样本的span分布
                sample_analysis = {
                    'sample_id': i,
                    'seq_length': span_labels.shape[0],
                    'effective_seq_len': effective_seq_len,
                    'total_positions': span_labels.numel(),
                    'non_zero_spans': non_zero_count,
                    'span_density': non_zero_count / span_labels.numel() if span_labels.numel() > 0 else 0
                }

                # 提取实际的span位置
                actual_spans = []
                for start in range(span_labels.shape[0]):
                    for end in range(start, span_labels.shape[1]):
                        if span_labels[start, end] > 0:
                            label_id = span_labels[start, end].item()
                            label_str = self.span_converter.id_to_label.get(
                                label_id, f'ID{label_id}')
                            actual_spans.append((start, end, label_str))

                sample_analysis['actual_spans'] = actual_spans
                sample_spans_analysis.append(sample_analysis)

                logger.debug(f"样本{i}: 有效长度={effective_seq_len}, "
                             f"非零span={non_zero_count}, 密度={sample_analysis['span_density']:.6f}, "
                             f"span详情={actual_spans[:3]}{'...' if len(actual_spans) > 3 else ''}")

            except Exception as e:
                logger.error(f"样本{i}转换失败: {e}")

        if total_non_zero == 0:
            logger.error("严重错误：没有找到任何非零span标签！")
            logger.error("详细分析:")
            for analysis in sample_spans_analysis:
                logger.error(f"  样本{analysis['sample_id']}: "
                             f"矩阵大小{analysis['seq_length']}x{analysis['seq_length']}, "
                             f"有效长度{analysis['effective_seq_len']}, "
                             f"非零span数{analysis['non_zero_spans']}")

            # 检查原始数据格式
            if len(self.train_dataset.examples) > 0:
                sample_example = self.train_dataset.examples[0]
                logger.error(f"原始数据样本格式检查:")
                if isinstance(sample_example, dict):
                    logger.error(f"  字典格式，键: {list(sample_example.keys())}")
                    if 'labels' in sample_example:
                        labels = sample_example['labels'][:10]  # 只显示前10个标签
                        logger.error(f"  前10个标签: {labels}")
                else:
                    logger.error(f"  对象格式，属性: {dir(sample_example)}")
                    if hasattr(sample_example, 'labels'):
                        labels = sample_example.labels[:10]  # 只显示前10个标签
                        logger.error(f"  前10个标签: {labels}")

            # 检查span转换器配置
            logger.error(f"AddressSpanConverter配置:")
            logger.error(
                f"  实体标签: {list(self.span_converter.label_to_id.keys())}")
            logger.error(f"  标签映射: {self.span_converter.label_to_id}")

            raise ValueError("Span标签转换失败，所有标签都是0")
        else:
            logger.info(
                f"数据转换验证通过: 非零span标签比例={total_non_zero/total_spans:.6f}")
            logger.info(f"总span位置数: {total_spans}, 非零span数: {total_non_zero}")

            # 统计span长度分布
            span_lengths = []
            for analysis in sample_spans_analysis:
                for start, end, label in analysis['actual_spans']:
                    span_lengths.append(end - start + 1)

            if span_lengths:
                logger.info(f"Span长度统计: 平均={np.mean(span_lengths):.2f}, "
                            f"最小={min(span_lengths)}, 最大={max(span_lengths)}, "
                            f"中位数={np.median(span_lengths):.2f}")

                # 按长度分组统计
                length_counts = {}
                for length in span_lengths:
                    length_counts[length] = length_counts.get(length, 0) + 1
                # 只显示前10个
                logger.info(
                    f"长度分布: {dict(sorted(length_counts.items())[:10])}")

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
                    # 预测spans - 使用优化的阈值策略
                    if isinstance(predictions, list):
                        pred_spans = predictions[i]
                    else:
                        # 使用配置中的base阈值，让模型内部进行动态调整
                        base_threshold = getattr(
                            self.config, 'span_threshold', 0.3)

                        pred_spans = self.model.decode_spans(
                            predictions[i:i+1],
                            attention_mask[i:i+1],
                            threshold=base_threshold
                        )[0]

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
            fold_train_dataset = AddressSpanNERDataset(
                fold_train_data, self.config.model_name,
                getattr(self.config, 'max_sequence_length', 384),
                self.span_converter
            )
            fold_val_dataset = AddressSpanNERDataset(
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
