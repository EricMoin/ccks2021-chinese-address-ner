import os
import torch
import json
from typing import List, Dict, Optional, Tuple
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer
from model import BertBiaffineSpanNER
from trainer import AddressSpanConverter, SpanEvaluator, AddressSpanNERDataset
from conll_reader import ConllReader
from sentence_reader import SentenceReader
from logger import logger
from config import Config


class BiaffineSpanPredictor:
    """
    Biaffine+Span模型预测器
    支持多模型集成和延迟加载
    """

    def __init__(self, config: Config = None, config_path: str = 'config.yaml'):
        """
        初始化span预测器

        Args:
            config: 配置对象（可选）
            config_path: 配置文件路径（当config为None时使用）
        """
        if config is not None:
            self.config = config
        else:
            self.config = Config(config_path)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # 提取实体类型列表（从完整的BIOES标签列表中）
        if hasattr(self.config, 'label_map') and hasattr(self.config.label_map, 'labels'):
            entity_labels = set()
            for label in self.config.label_map.labels:
                if label != 'O' and '-' in label:
                    entity_type = label.split('-', 1)[1]
                    entity_labels.add(entity_type)
            entity_labels = sorted(list(entity_labels))
        else:
            entity_labels = []

        # 初始化span转换器
        self.span_converter = AddressSpanConverter(
            labels=entity_labels,
            label_scheme=getattr(self.config.label_map, 'type', 'BIOES')
        )

        # 初始化评估器
        self.evaluator = SpanEvaluator(self.span_converter)

        # 延迟加载相关
        self.model = None
        self.tokenizer = None
        self.current_model_path = None

        logger.info("BiaffineSpanPredictor initialized with lazy loading")
        logger.info(f"Device: {self.device}")
        logger.info(f"Entity types: {len(entity_labels)}")

    def _ensure_tokenizer_loaded(self):
        """确保tokenizer已加载"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name)
            logger.info(f"Tokenizer loaded: {self.config.model_name}")

    def _load_model(self, model_path: str) -> BertBiaffineSpanNER:
        """加载训练好的模型"""
        # 创建模型实例
        model = BertBiaffineSpanNER(
            num_labels=self.span_converter.num_labels,
            config=self.config
        )

        # 加载权重
        if os.path.exists(model_path):
            logger.info(f"Loading model weights from: {model_path}")
            checkpoint = None

            try:
                # 首先尝试使用weights_only=False加载完整检查点（包含config等对象）
                logger.debug(
                    "Attempting to load checkpoint with weights_only=False")
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=False)
                logger.info("Successfully loaded checkpoint with full objects")
            except Exception as e:
                logger.warning(f"Failed to load with weights_only=False: {e}")
                try:
                    # 如果失败，尝试只加载权重
                    logger.debug(
                        "Attempting to load checkpoint with weights_only=True")
                    checkpoint = torch.load(
                        model_path, map_location=self.device, weights_only=True)
                    logger.info(
                        "Successfully loaded weights only (no additional objects)")
                except Exception as e2:
                    logger.error(
                        f"Failed to load model weights with both methods: {e2}")
                    logger.error(f"Original error: {e}")
                    raise RuntimeError(f"Could not load model from {model_path}. "
                                       f"Please check if the file is corrupted or incompatible.") from e2

            # 处理不同的保存格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    logger.debug("Loading from 'model_state_dict' key")
                    model.load_state_dict(checkpoint['model_state_dict'])

                    # 如果保存的检查点包含span_converter，使用它来确保兼容性
                    if 'span_converter' in checkpoint:
                        try:
                            self.span_converter = checkpoint['span_converter']
                            logger.info("Using span_converter from checkpoint")
                        except Exception as e:
                            logger.warning(
                                f"Could not load span_converter from checkpoint: {e}")
                            logger.info(
                                "Using current span_converter configuration")

                    # 记录其他可用信息
                    if 'best_f1' in checkpoint:
                        logger.info(
                            f"Model was saved with best F1: {checkpoint['best_f1']:.4f}")

                elif 'state_dict' in checkpoint:
                    logger.debug("Loading from 'state_dict' key")
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    logger.warning(
                        "Checkpoint format not recognized, trying direct state_dict load")
                    model.load_state_dict(checkpoint)
            else:
                # 直接是state_dict格式
                logger.debug("Loading checkpoint as direct state_dict")
                model.load_state_dict(checkpoint)

        else:
            logger.warning(
                f"Model path {model_path} not found, using random initialization")

        model.to(self.device)
        model.eval()

        # 验证模型加载
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Model loaded successfully with {total_params:,} parameters")

        return model

    def _ensure_model_loaded(self, model_path: str):
        """确保指定的模型已加载"""
        if self.model is None or self.current_model_path != model_path:
            logger.info(f"Loading model for prediction: {model_path}")
            self.model = self._load_model(model_path)
            self.current_model_path = model_path
            # 清理之前的GPU内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def get_predictions_for_fold(self, fold_work_dir: str, test_file_path: str,
                                 label_map, batch_size: int = 8,
                                 use_swa_if_available: bool = True) -> Optional[List[List[str]]]:
        """
        为指定fold目录生成预测结果

        Args:
            fold_work_dir: fold工作目录路径
            test_file_path: 测试文件路径
            label_map: 标签映射（保持兼容性，实际使用self.span_converter）
            batch_size: 批次大小
            use_swa_if_available: 是否优先使用SWA模型

        Returns:
            List[List[str]]: 每个测试样本的标签序列列表，如果失败返回None
        """
        # 确定使用哪个模型文件
        model_path = None
        if use_swa_if_available:
            swa_path = os.path.join(fold_work_dir, 'swa_model.pt')
            best_path = os.path.join(fold_work_dir, 'best_model.pt')

            if os.path.exists(swa_path):
                model_path = swa_path
                logger.info(f"Using SWA model: {swa_path}")
            elif os.path.exists(best_path):
                model_path = best_path
                logger.info(
                    f"SWA model not found, using best model: {best_path}")
        else:
            best_path = os.path.join(fold_work_dir, 'best_model.pt')
            if os.path.exists(best_path):
                model_path = best_path
                logger.info(f"Using best model: {best_path}")

        if model_path is None:
            logger.error(f"No valid model found in {fold_work_dir}")
            return None

        # 确保模型和tokenizer已加载
        self._ensure_model_loaded(model_path)
        self._ensure_tokenizer_loaded()

        # 读取测试数据
        sentence_reader = SentenceReader()
        test_char_sequences = sentence_reader.read_tokens(test_file_path)

        if not test_char_sequences:
            logger.error(f"No test data found in {test_file_path}")
            return None

        logger.info(
            f"Loaded {len(test_char_sequences)} test samples from {test_file_path}")

        # 批量预测
        all_predictions = []
        max_length = getattr(self.config, 'max_sequence_length', 384)

        for i in range(0, len(test_char_sequences), batch_size):
            batch_char_sequences = test_char_sequences[i:i + batch_size]
            batch_predictions = self._predict_batch_sequences(
                batch_char_sequences, max_length)
            all_predictions.extend(batch_predictions)

            # 显示进度
            if (i // batch_size + 1) % 50 == 0:
                logger.info(
                    f"Processed {i + len(batch_char_sequences)}/{len(test_char_sequences)} samples")

        logger.info(
            f"Prediction completed for {fold_work_dir}: {len(all_predictions)} samples")
        return all_predictions

    def _predict_batch_sequences(self, char_sequences: List[List[str]], max_length: int) -> List[List[str]]:
        """
        预测一批字符序列

        Args:
            char_sequences: 字符序列列表
            max_length: 最大序列长度

        Returns:
            List[List[str]]: 每个序列的标签列表
        """
        batch_results = []

        for char_seq in char_sequences:
            # 编码序列
            encoding = self.tokenizer(
                char_seq,
                is_split_into_words=True,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=max_length
            )

            # 移动到设备
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # 预测
            with torch.no_grad():
                # 确保模型处于评估模式
                self.model.eval()

                # 获取span logits
                span_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # 使用模型的decode_spans方法解码span
                threshold = getattr(self.config, 'span_threshold', 0.15)
                span_predictions = self.model.decode_spans(
                    span_logits,  # span_logits已经包含batch维度
                    attention_mask,
                    threshold=threshold
                )

                # 将span预测转换为序列标签
                sequence_labels = self._spans_to_bio_labels(
                    span_predictions[0] if isinstance(
                        span_predictions, list) else [],
                    char_seq,
                    encoding
                )

            batch_results.append(sequence_labels)

        return batch_results

    def _spans_to_bio_labels(self, spans: List[Dict], char_seq: List[str], encoding) -> List[str]:
        """
        将span预测结果转换为BIOES标签序列

        Args:
            spans: span预测结果
            char_seq: 原始字符序列
            encoding: tokenizer编码结果

        Returns:
            List[str]: BIOES标签序列
        """
        # 初始化标签序列
        labels = ['O'] * len(char_seq)

        # 获取word_ids映射
        word_ids = encoding.word_ids()

        # 处理每个预测的span
        for span in spans:
            start_token = span.get('start', -1)
            end_token = span.get('end', -1)
            label_id = span.get('label', 0)

            # 转换标签ID为标签名
            if isinstance(label_id, int):
                label_name = self.span_converter.id_to_label.get(label_id, 'O')
            else:
                label_name = str(label_id)

            if label_name == 'O':
                continue

            # 将token位置映射到word位置
            start_word = None
            end_word = None

            if word_ids is not None:
                for i, word_id in enumerate(word_ids):
                    if word_id is not None:
                        if i == start_token:
                            start_word = word_id
                        if i == end_token:
                            end_word = word_id

            # 设置BIOES标签
            if start_word is not None and end_word is not None and start_word < len(labels) and end_word < len(labels):
                if start_word == end_word:
                    # 单字符实体
                    labels[start_word] = f'S-{label_name}'
                else:
                    # 多字符实体
                    labels[start_word] = f'B-{label_name}'
                    for j in range(start_word + 1, end_word):
                        if j < len(labels):
                            labels[j] = f'I-{label_name}'
                    if end_word < len(labels):
                        labels[end_word] = f'E-{label_name}'

        return labels

    def predict_single(self, text: str, model_path: str, max_length: int = None) -> List[Dict]:
        """
        预测单个文本的实体

        Args:
            text: 输入文本
            model_path: 模型路径
            max_length: 最大序列长度

        Returns:
            List[Dict]: 预测的实体列表，每个元素包含start, end, label, text, confidence
        """
        if max_length is None:
            max_length = getattr(self.config, 'max_predict_length', 384)

        # 确保模型和tokenizer已加载
        self._ensure_model_loaded(model_path)
        self._ensure_tokenizer_loaded()

        # 简单分词（可以根据需要改进）
        tokens = list(text)

        # 编码
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

        # 移动到设备
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 预测
        with torch.no_grad():
            predictions = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # 处理预测结果
        if isinstance(predictions, list) and len(predictions) > 0:
            spans = predictions[0]  # 取第一个batch的结果
        else:
            spans = []

        # 转换为结果格式
        results = []
        word_ids = encoding.word_ids()

        for span in spans:
            start_token = span['start']
            end_token = span['end']
            label_id = span['label']
            confidence = span.get('confidence', 1.0)

            # 将token位置映射回字符位置
            if word_ids is not None:
                # 找到对应的word位置
                start_word = None
                end_word = None

                for i, word_id in enumerate(word_ids):
                    if word_id is not None:
                        if i == start_token:
                            start_word = word_id
                        if i == end_token:
                            end_word = word_id

                if start_word is not None and end_word is not None:
                    # 获取实体文本
                    entity_text = ''.join(tokens[start_word:end_word + 1])

                    # 获取标签名
                    label_name = self.span_converter.id_to_label.get(
                        label_id, 'O')

                    if label_name != 'O':
                        results.append({
                            'start': start_word,
                            'end': end_word,
                            'label': label_name,
                            'text': entity_text,
                            'confidence': confidence
                        })

        return results

    def predict_batch(self, texts: List[str], model_path: str, batch_size: int = 8, max_length: int = None) -> List[List[Dict]]:
        """
        批量预测文本实体

        Args:
            texts: 文本列表
            model_path: 模型路径
            batch_size: 批次大小
            max_length: 最大序列长度

        Returns:
            List[List[Dict]]: 每个文本的预测结果列表
        """
        if max_length is None:
            max_length = getattr(self.config, 'max_predict_length', 384)

        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []

            for text in batch_texts:
                predictions = self.predict_single(text, model_path, max_length)
                batch_results.append(predictions)

            results.extend(batch_results)

        return results

    def predict_from_file(self, input_file: str, output_file: str, model_path: str, max_length: int = None):
        """
        从文件读取文本并预测，保存结果到文件

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            model_path: 模型路径
            max_length: 最大序列长度
        """
        if max_length is None:
            max_length = getattr(self.config, 'max_predict_length', 384)

        logger.info(f"Reading from: {input_file}")

        # 确保模型和tokenizer已加载
        self._ensure_model_loaded(model_path)
        self._ensure_tokenizer_loaded()

        results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                text = line.strip()
                if text:
                    predictions = self.predict_single(
                        text, model_path, max_length)
                    results.append({
                        'line': line_idx,
                        'text': text,
                        'entities': predictions
                    })

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Predictions saved to: {output_file}")
        logger.info(f"Processed {len(results)} texts")

    def evaluate_on_dataset(self, test_file: str, model_path: str, max_length: int = None) -> Dict[str, float]:
        """
        在测试集上评估模型性能

        Args:
            test_file: 测试文件路径（CoNLL格式）
            model_path: 模型路径
            max_length: 最大序列长度

        Returns:
            Dict[str, float]: 评估指标
        """
        if max_length is None:
            max_length = getattr(self.config, 'max_predict_length', 384)

        logger.info(f"Evaluating on: {test_file}")

        # 确保模型和tokenizer已加载
        self._ensure_model_loaded(model_path)
        self._ensure_tokenizer_loaded()

        # 读取测试数据
        conll_reader = ConllReader()
        test_data = list(conll_reader.read(test_file))

        # 转换为span格式数据
        span_examples = []
        for example in test_data:
            tokens = example.tokens
            labels = [self.config.label_map.id2label[label_id]
                      for label_id in example.labels]

            span_examples.append({
                'tokens': tokens,
                'labels': labels
            })

        # 创建数据集和数据加载器
        dataset = AddressSpanNERDataset(
            span_examples,
            self.config.model_name,  # 传递模型名称而非tokenizer对象
            max_length,
            self.span_converter
        )

        dataloader = DataLoader(
            dataset,
            batch_size=getattr(self.config, 'inference_batch_size', 16),
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        # 评估
        all_predictions = []
        all_gold_spans = []
        total_loss = 0.0
        num_batches = 0

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # 移动到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                span_labels = batch['span_labels'].to(self.device)

                # 计算损失
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    span_labels=span_labels
                )
                total_loss += loss.item()

                # 获取预测
                predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                all_predictions.extend(predictions)

                # 提取真实spans
                for i in range(span_labels.size(0)):
                    gold_spans = []
                    seq_len = attention_mask[i].sum().item()

                    for start in range(seq_len):
                        for end in range(start, seq_len):
                            label_id = span_labels[i, start, end].item()
                            if label_id > 0:
                                label = self.span_converter.id_to_label[label_id]
                                gold_spans.append((start, end, label))

                    all_gold_spans.append(gold_spans)

                num_batches += 1

        # 计算评估指标
        metrics = self.evaluator.evaluate_spans(
            all_predictions, all_gold_spans)
        metrics['val_loss'] = total_loss / num_batches

        logger.info("Evaluation Results:")
        logger.info(f"  Loss: {metrics['val_loss']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")

        return metrics

    def interactive_predict(self, model_path: str):
        """
        交互式预测模式

        Args:
            model_path: 模型路径
        """
        logger.info("Entering interactive prediction mode...")
        logger.info("Type 'quit' to exit")

        # 确保模型和tokenizer已加载
        self._ensure_model_loaded(model_path)
        self._ensure_tokenizer_loaded()

        while True:
            try:
                text = input("\nEnter text to predict: ").strip()

                if text.lower() in ['quit', 'exit', 'q']:
                    break

                if not text:
                    continue

                # 预测
                entities = self.predict_single(text, model_path)

                # 显示结果
                if entities:
                    print(f"\nFound {len(entities)} entities:")
                    for i, entity in enumerate(entities, 1):
                        print(f"  {i}. {entity['text']} ({entity['label']}) "
                              f"[{entity['start']}:{entity['end']}] "
                              f"confidence: {entity['confidence']:.3f}")
                else:
                    print("  No entities found.")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Prediction error: {e}")

        logger.info("Interactive mode ended.")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if self.model is None:
            return {
                'model_type': 'BertBiaffineSpanNER',
                'status': 'No model loaded',
                'config': {
                    'max_sequence_length': self.config.max_sequence_length,
                    'num_labels': self.span_converter.num_labels,
                }
            }

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        return {
            'model_type': 'BertBiaffineSpanNER',
            'current_model_path': self.current_model_path,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': {
                'max_sequence_length': self.config.max_sequence_length,
                'num_labels': self.span_converter.num_labels,
                'use_bert_projection': getattr(self.config, 'use_bert_projection', True),
                'bert_projection_dim': getattr(self.config, 'bert_projection_dim', 512),
            }
        }
