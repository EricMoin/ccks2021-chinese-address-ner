import logging
import os
import re
import torch
from config import Config
from conll_reader import ConllEntity
from dataset import NERDataset
from label import LabelMap
from model import AddressNER
from torch.utils.data import DataLoader
from tqdm import tqdm
from logger import logger


class Predictor:
    def __init__(self, model_init_config: Config):
        """
        初始化预测器。
        Args:
            model_init_config (Config): 用于初始化AddressNER模型的配置对象，
                                       主要用于分词器和模型架构。
                                       其model_name应指向基础适配模型。
        """
        self.model_config = model_init_config  # 用于AddressNER实例化的配置
        self.device = torch.device(self.model_config.device)

    def get_predictions_for_fold(self,
                                 fold_work_dir: str,
                                 test_file_path: str,
                                 label_map: LabelMap,  # 传递主要的label_map
                                 batch_size: int,
                                 use_swa_if_available: bool) -> list[list[str]]:
        """
        从特定折叠目录加载训练好的模型并生成预测。
        Args:
            fold_work_dir (str): 训练折叠的工作目录路径
                                 (例如, result/model_xyz/fold_1)。
            test_file_path (str): 原始测试文件的路径 (例如, data/final_test.txt)。
            label_map (LabelMap): 全局LabelMap对象。
            batch_size (int): 预测的批次大小。
            use_swa_if_available (bool): 如果可用是否优先使用SWA模型。

        Returns:
            list[list[str]]: 测试集的预测标签序列列表。
        """
        model_to_load_path = None
        swa_model_path = os.path.join(fold_work_dir, "swa_model.pt")
        best_model_path = os.path.join(fold_work_dir, "best_model.pt")

        if use_swa_if_available and os.path.exists(swa_model_path):
            logger.info(
                f"预测器: 使用SWA模型进行推理，来源: {swa_model_path}")
            model_to_load_path = swa_model_path
        elif os.path.exists(best_model_path):
            logger.info(
                f"预测器: 使用best_model.pt进行推理，来源: {best_model_path}")
            model_to_load_path = best_model_path
        else:
            logger.error(
                f"预测器: 在{fold_work_dir}中未找到SWA或最佳模型。无法为此折生成预测。")
            return []  # 为此折返回空列表

        try:
            model_name = re.sub(r'(/swa_model\.pt)|(/best_model\.pt)',
                                r'', model_to_load_path)
            model_name = re.sub(r'result/pretrained/',
                                r'', model_name)
            model_name = re.sub(r'_adapted_ep[\d]+_seed[\d]+',
                                r'', model_name)
            model_name = model_name.replace(r"_", '/')
            self.model_config.model_name = model_name
            self.model = AddressNER(num_labels=len(self.model_config.label_map.labels),
                                    config=self.model_config)
            self.model.load_state_dict(torch.load(
                model_to_load_path, map_location=self.device))
            self.model.to(self.device)
            logger.info(
                f"Predictor: Successfully loaded model weights from {model_to_load_path}")
        except Exception as e:
            logger.error(
                f"Predictor: Error loading model state_dict from {model_to_load_path}: {e}")
            return []

        self.model.eval()

        # --- Test Data Preparation ---
        test_sentences_char_tokens = []
        if not os.path.exists(test_file_path):
            logger.error(
                f"Predictor: Test file not found at: {test_file_path}")
            return []

        with open(test_file_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if line:
                    try:
                        # 假设格式: guid<SEP>text
                        text_part = line.split('\u0001')[1]
                        test_sentences_char_tokens.append(list(text_part))
                    except IndexError:
                        logger.warning(
                            f"预测器: 跳过测试文件{test_file_path}中格式错误的行: {line}")
                        # 保持示例计数一致
                        test_sentences_char_tokens.append([])

        if not test_sentences_char_tokens:
            logger.warning(
                f"Predictor: No character tokens extracted from test file {test_file_path}.")
            return []

        test_conll_examples = [ConllEntity(chars, ['O'] * len(chars))
                               for chars in test_sentences_char_tokens]

        # 分词器来自self.model（用model_init_config初始化的AddressNER实例）
        # NERDataset使用label_map.label2id
        test_dataset = NERDataset(
            data=test_conll_examples,
            tokenizer=self.model.tokenizer,
            label_map=label_map.label2id  # 使用main_cfg的label_map
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        all_preds_sequences = []
        with torch.no_grad():
            pbar_desc = f"使用折叠模型进行预测: {os.path.basename(fold_work_dir)}"
            for batch in tqdm(test_dataloader, desc=pbar_desc):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # 模型的前向传播进行预测（不使用标签）
                batch_pred_indices = self.model(input_ids, attention_mask)

                # 遍历批次中的示例
                for i in range(len(batch_pred_indices)):
                    # batch_pred_indices[i]是一个示例的List[int]
                    pred_indices_for_example = batch_pred_indices[i]

                    # 使用main_cfg的label_map将索引转换为标签字符串
                    pred_labels = [label_map.id2label.get(
                        p_idx, 'O') for p_idx in pred_indices_for_example]
                    all_preds_sequences.append(pred_labels)

        logger.info(
            f"预测器: 使用来自{fold_work_dir}的模型为测试集生成了{len(all_preds_sequences)}个预测序列。")
        return all_preds_sequences

    def get_predictions_for_conll_data(self,
                                       fold_work_dir: str,
                                       test_conll_examples: list,
                                       label_map: LabelMap,
                                       batch_size: int,
                                       use_swa_if_available: bool) -> tuple[list[list[str]], list]:
        """
        从特定折叠目录加载训练好的模型并对ConLL数据生成预测。
        Args:
            fold_work_dir (str): 训练折叠的工作目录路径
            test_conll_examples (list): ConLL实体列表
            label_map (LabelMap): 全局LabelMap对象
            batch_size (int): 预测的批次大小
            use_swa_if_available (bool): 如果可用是否优先使用SWA模型

        Returns:
            tuple[list[list[str]], list]: (预测标签序列列表, logits列表)
        """
        model_to_load_path = None
        swa_model_path = os.path.join(fold_work_dir, "swa_model.pt")
        best_model_path = os.path.join(fold_work_dir, "best_model.pt")

        if use_swa_if_available and os.path.exists(swa_model_path):
            logger.info(
                f"预测器: 使用SWA模型进行推理，来源: {swa_model_path}")
            model_to_load_path = swa_model_path
        elif os.path.exists(best_model_path):
            logger.info(
                f"预测器: 使用best_model.pt进行推理，来源: {best_model_path}")
            model_to_load_path = best_model_path
        else:
            logger.error(
                f"预测器: 在{fold_work_dir}中未找到SWA或最佳模型。无法为此折生成预测。")
            return [], []

        try:
            model_name = re.sub(r'(/swa_model\.pt)|(/best_model\.pt)',
                                r'', model_to_load_path)
            model_name = re.sub(r'result/pretrained/',
                                r'', model_name)
            model_name = re.sub(r'_adapted_ep[\d]+_seed[\d]+',
                                r'', model_name)
            model_name = model_name.replace(r"_", '/')
            self.model_config.model_name = model_name
            self.model = AddressNER(num_labels=len(self.model_config.label_map.labels),
                                    config=self.model_config)
            self.model.load_state_dict(torch.load(
                model_to_load_path, map_location=self.device))
            self.model.to(self.device)
            logger.info(
                f"Predictor: Successfully loaded model weights from {model_to_load_path}")
        except Exception as e:
            logger.error(
                f"Predictor: Error loading model state_dict from {model_to_load_path}: {e}")
            return [], []

        self.model.eval()

        # 使用传入的ConLL实体创建数据集
        test_dataset = NERDataset(
            data=test_conll_examples,
            tokenizer=self.model.tokenizer,
            label_map=label_map.label2id
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        all_preds_sequences = []
        all_logits = []

        with torch.no_grad():
            pbar_desc = f"使用折叠模型进行ConLL数据预测: {os.path.basename(fold_work_dir)}"
            for batch in tqdm(test_dataloader, desc=pbar_desc):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # 获取BERT输出
                if self.model.training and self.model.embedding_dropout.p > 0:
                    current_embeddings = self.model.bert.embeddings.word_embeddings(
                        input_ids)
                    current_embeddings = self.model.embedding_dropout(
                        current_embeddings)
                    outputs = self.model.bert(
                        inputs_embeds=current_embeddings, attention_mask=attention_mask)
                else:
                    outputs = self.model.bert(
                        input_ids=input_ids, attention_mask=attention_mask)

                bert_output = outputs.last_hidden_state
                bert_output = self.model.spatial_dropout(bert_output)
                lstm_output, _ = self.model.lstm(bert_output)
                logits = self.model.classifier(lstm_output)

                # 使用CRF解码获取预测
                batch_pred_indices = self.model.crf.viterbi_decode(
                    logits, mask=attention_mask.bool())

                # 处理预测结果
                for i in range(len(batch_pred_indices)):
                    pred_indices_for_example = batch_pred_indices[i]
                    logits_for_example = logits[i]

                    # 将索引转换为标签字符串
                    pred_labels = [label_map.id2label.get(
                        p_idx, 'O') for p_idx in pred_indices_for_example]

                    # 修复：移除特殊标记的标签，只保留与原始tokens对应的标签
                    # tokenizer添加了[CLS]和[SEP]，所以预测标签需要去掉首尾
                    if len(pred_labels) >= 2:
                        # 去掉[CLS]（第一个）和[SEP]（找到第一个[SEP]的位置）
                        # 对于padding的情况，我们需要找到实际的序列结束位置
                        original_tokens = test_conll_examples[len(
                            all_preds_sequences)].tokens
                        actual_length = len(original_tokens)

                        # 预测标签格式：[CLS] + token_labels + [SEP] + padding
                        # 我们只需要token_labels部分
                        # +1 for [CLS]
                        if len(pred_labels) > actual_length + 1:
                            # 跳过[CLS]，取actual_length个
                            aligned_pred_labels = pred_labels[1:actual_length+1]
                        else:
                            # 如果预测标签不够长，使用可用的部分
                            aligned_pred_labels = pred_labels[1:] if len(
                                pred_labels) > 1 else pred_labels

                        # 确保长度匹配
                        if len(aligned_pred_labels) > actual_length:
                            aligned_pred_labels = aligned_pred_labels[:actual_length]
                        elif len(aligned_pred_labels) < actual_length:
                            # 如果还是不够，用'O'填充
                            aligned_pred_labels.extend(
                                ['O'] * (actual_length - len(aligned_pred_labels)))
                    else:
                        aligned_pred_labels = pred_labels

                    all_preds_sequences.append(aligned_pred_labels)

                    # 对logits也进行相应的对齐
                    if len(aligned_pred_labels) <= logits_for_example.shape[0]:
                        # 同样跳过[CLS]位置的logits
                        if logits_for_example.shape[0] > len(aligned_pred_labels):
                            aligned_logits = logits_for_example[1:len(
                                aligned_pred_labels)+1]
                        else:
                            aligned_logits = logits_for_example[:len(
                                aligned_pred_labels)]
                    else:
                        aligned_logits = logits_for_example

                    all_logits.append(aligned_logits)

        logger.info(
            f"预测器: 使用来自{fold_work_dir}的模型为ConLL数据生成了{len(all_preds_sequences)}个预测序列。")
        return all_preds_sequences, all_logits
