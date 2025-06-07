from collections import Counter
import glob
import os
import copy
import re
import torch
from typing import Dict, List, Optional

from config import Config
from predictor import BiaffineSpanPredictor
from sentence_reader import SentenceReader
from logger import logger


class ResultWriter:
    def __init__(self, config: Config, model_weights: Optional[Dict[str, float]] = None):
        self.config = config
        # model_weights 格式: {"model_name": weight}
        self.model_weights = model_weights or {}
        self.use_soft_voting = bool(model_weights)

        if self.use_soft_voting:
            logger.info(f"使用软投票，模型权重: {self.model_weights}")
        else:
            logger.info("使用硬投票（多数投票）")

    def _get_model_name_from_path(self, fold_dir_path: str) -> str:
        """从fold目录路径中提取模型名称"""
        model_name = re.sub(r'(/swa_model\.pt)|(/best_model\.pt)',
                            r'', fold_dir_path)
        model_name = re.sub(r'result/pretrained/',
                            r'', model_name)
        model_name = re.sub(r'_adapted_ep[\d]+_seed[\d]+',
                            r'', model_name)
        model_name = model_name.replace(r"_", '/')
        return model_name

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """归一化权重，使其总和为1"""
        total_weight = sum(weights.values())
        if total_weight == 0:
            logger.warning("权重总和为0，使用均匀权重")
            return {k: 1.0 / len(weights) for k in weights.keys()}
        return {k: v / total_weight for k, v in weights.items()}

    def _ensemble_predictions_soft_voting(self,
                                          all_predictions: List[List[List[str]]],
                                          model_names: List[str],
                                          num_test_examples: int,
                                          original_test_char_sequences: List[List[str]]) -> List[List[str]]:
        """
        使用软投票进行模型集成

        Args:
            all_predictions: 所有模型的预测结果
            model_names: 模型名称列表
            num_test_examples: 测试样本数量
            original_test_char_sequences: 原始测试字符序列

        Returns:
            集成后的预测标签
        """
        logger.info("开始软投票集成...")

        # 准备模型权重
        active_weights = {}
        for model_name in model_names:
            if model_name in self.model_weights:
                active_weights[model_name] = self.model_weights[model_name]
            else:
                # 如果模型不在权重字典中，给予默认权重
                active_weights[model_name] = 1.0
                logger.warning(f"模型 {model_name} 不在权重配置中，使用默认权重 1.0")

        # 归一化权重
        active_weights = self._normalize_weights(active_weights)
        logger.info(f"归一化后的模型权重: {active_weights}")

        ensembled_final_labels = []

        for example_idx in range(num_test_examples):
            char_seq_len = len(original_test_char_sequences[example_idx])
            example_ensembled_labels = []

            for token_idx in range(char_seq_len):
                # 收集所有模型对该位置的预测
                label_scores = {}  # {label: weighted_score}

                for model_idx, (predictions, model_name) in enumerate(zip(all_predictions, model_names)):
                    if example_idx < len(predictions) and token_idx < len(predictions[example_idx]):
                        predicted_label = predictions[example_idx][token_idx]
                        weight = active_weights.get(model_name, 0.0)

                        if predicted_label not in label_scores:
                            label_scores[predicted_label] = 0.0
                        label_scores[predicted_label] += weight
                    else:
                        # 处理缺失预测的情况
                        weight = active_weights.get(model_name, 0.0)
                        if 'O' not in label_scores:
                            label_scores['O'] = 0.0
                        label_scores['O'] += weight
                        logger.debug(
                            f"预测缺失 example {example_idx}, token {token_idx} from model {model_name}. 默认为 'O'.")

                # 选择得分最高的标签
                if label_scores:
                    best_label = max(label_scores.keys(),
                                     key=lambda x: label_scores[x])
                    example_ensembled_labels.append(best_label)
                else:
                    logger.warning(
                        f"没有找到有效预测 example {example_idx}, token {token_idx}. 默认为 'O'.")
                    example_ensembled_labels.append('O')

            ensembled_final_labels.append(example_ensembled_labels)

        logger.info("软投票集成完成")
        return ensembled_final_labels

    def _ensemble_predictions_hard_voting(self,
                                          all_predictions: List[List[List[str]]],
                                          num_test_examples: int,
                                          original_test_char_sequences: List[List[str]]) -> List[List[str]]:
        """
        使用硬投票进行模型集成（原有逻辑）

        Args:
            all_predictions: 所有模型的预测结果
            num_test_examples: 测试样本数量
            original_test_char_sequences: 原始测试字符序列

        Returns:
            集成后的预测标签
        """
        logger.info("开始硬投票集成...")

        ensembled_final_labels = []

        for example_idx in range(num_test_examples):
            char_seq_len = len(original_test_char_sequences[example_idx])
            example_ensembled_labels_for_tokens = []

            for token_idx in range(char_seq_len):
                votes_for_this_token = []

                for source_idx, source_predictions in enumerate(all_predictions):
                    if example_idx < len(source_predictions) and token_idx < len(source_predictions[example_idx]):
                        votes_for_this_token.append(
                            source_predictions[example_idx][token_idx])
                    else:
                        logger.debug(
                            f"Vote missing for example {example_idx}, token {token_idx} from source {source_idx}. Defaulting to 'O'.")
                        votes_for_this_token.append('O')

                if not votes_for_this_token:
                    logger.warning(
                        f"No votes for example {example_idx}, token {token_idx}. Defaulting to 'O'.")
                    majority_label = 'O'
                else:
                    vote_counts = Counter(votes_for_this_token)
                    majority_label = vote_counts.most_common(1)[0][0]

                example_ensembled_labels_for_tokens.append(majority_label)
            ensembled_final_labels.append(example_ensembled_labels_for_tokens)

        logger.info("硬投票集成完成")
        return ensembled_final_labels

    def discover_work_dirs(self, main_work_dir: str, model_name_pattern: str = "*_adapted_ep*_seed*") -> list[str]:
        """
        发现每个适配模型的每个折叠目录
        示例: main_work_dir/some_adapted_model_name/fold_1, main_work_dir/some_adapted_model_name/fold_2
        """
        discovered_paths = []  # 存储发现的路径列表
        # 适配模型目录的路径模式，例如: result/hfl_chinese-roberta-wwm-ext_adapted_ep3_seed2024
        model_dirs_pattern = os.path.join(
            main_work_dir, model_name_pattern)
        for model_dir_path in glob.glob(model_dirs_pattern):  # 遍历匹配的模型目录
            if os.path.isdir(model_dir_path):  # 确认是目录
                if os.path.exists(os.path.join(model_dir_path, "best_model.pt")) or \
                        os.path.exists(os.path.join(model_dir_path, "swa_model.pt")):
                    discovered_paths.append(
                        model_dir_path)  # 添加到发现的路径列表
                logger.info(
                    f"发现有效的预测目录: {model_dir_path}")
        if not discovered_paths:  # 如果没有找到任何目录
            logger.warning(
                f"在 {main_work_dir} 中未找到匹配模式的目录。")
        return sorted(discovered_paths)  # 返回排序后的路径列表

    def discover_fold_work_dirs(self, main_work_dir: str, adapted_model_name_pattern: str = "*_adapted_ep*_seed*", fold_pattern: str = "fold_*") -> list[str]:
        """
        发现每个适配模型的每个折叠目录
        示例: main_work_dir/some_adapted_model_name/fold_1, main_work_dir/some_adapted_model_name/fold_2
        """
        discovered_paths = []  # 存储发现的路径列表
        # 适配模型目录的路径模式，例如: result/hfl_chinese-roberta-wwm-ext_adapted_ep3_seed2024
        adapted_model_dirs_pattern = os.path.join(
            main_work_dir, adapted_model_name_pattern)
        for model_dir_path in glob.glob(adapted_model_dirs_pattern):  # 遍历匹配的模型目录
            if os.path.isdir(model_dir_path):  # 确认是目录
                fold_dirs_pattern = os.path.join(
                    model_dir_path, fold_pattern)  # 构建折叠目录模式
                for fold_dir_path in glob.glob(fold_dirs_pattern):  # 遍历匹配的折叠目录
                    if os.path.isdir(fold_dir_path):  # 确认是目录
                        # 检查是否存在模型文件以确认是有效的折叠目录
                        if os.path.exists(os.path.join(fold_dir_path, "best_model.pt")) or \
                                os.path.exists(os.path.join(fold_dir_path, "swa_model.pt")):
                            discovered_paths.append(
                                fold_dir_path)  # 添加到发现的路径列表
                            logger.info(
                                f"发现有效的预测折叠目录: {fold_dir_path}")
        if not discovered_paths:  # 如果没有找到任何目录
            logger.warning(
                f"在 {main_work_dir} 中未找到匹配模式的折叠目录。")
        return sorted(discovered_paths)  # 返回排序后的路径列表

    def write_ensembled_predictions_conll_format(
        self,
        output_path: str,
        original_char_sequences: list[list[str]],
        ensembled_labels: list[list[str]]
    ):
        """
        将集成预测的标签以CoNLL格式写入指定文件路径

        参数:
        - output_path: 输出文件的路径
        - original_char_sequences: 原始字符序列列表，每个序列是一个字符列表
        - ensembled_labels: 预测标签列表，每组标签对应一个字符序列

        返回:
        - 成功写入返回True，否则返回False
        """
        logger.info(
            f"正在将集成预测写入CoNLL格式: {output_path}")
        try:
            with open(output_path, "w", encoding="utf8") as f:  # 以UTF-8编码打开输出文件
                for i, chars in enumerate(original_char_sequences):  # 遍历每个字符序列
                    labels_for_example = ensembled_labels[i]  # 获取对应的标签
                    min_len = min(len(chars), len(
                        labels_for_example))  # 取字符和标签长度的较小值
                    for j in range(min_len):  # 写入字符和标签
                        char_token = chars[j]
                        label_to_write = labels_for_example[j]
                        f.write(f"{char_token}\t{label_to_write}\n")
                    if len(chars) > min_len:  # 如果原始文本更长，用'O'填充剩余部分
                        for j in range(min_len, len(chars)):
                            f.write(f"{chars[j]}\tO\n")
                    elif len(labels_for_example) > min_len:  # 如果预测更长（正常情况下不应该发生）
                        logger.warning(
                            f"示例 {i}: 预测长度超过原始文本。截断预测。")
                    f.write("\n")  # 写入空行分隔每组数据
                logger.info(
                    f"集成预测已以CoNLL格式写入 {output_path}")
                return True
        except IOError as e:  # 处理写入失败的情况
            logger.error(f"写入CoNLL输出到 {output_path} 失败: {e}")
            return False

    def process_final_submission_output(self, ensembled_conll_pred_file: str, final_submission_file: str, original_test_file_path: str):
        logger.info(
            f"正在处理 {ensembled_conll_pred_file} 为最终提交格式 {final_submission_file}")
        original_test_data = []  # 存储原始测试数据
        if not os.path.exists(original_test_file_path):  # 检查原始测试文件是否存在
            logger.error(
                f"未找到原始测试文件 {original_test_file_path}。无法创建提交。")
            return

        test_sentence_reader = SentenceReader()  # 创建句子读取器
        original_test_data = test_sentence_reader.read_parts(  # 读取原始测试数据
            file_path=original_test_file_path)

        predicted_labels_per_example = []  # 存储每个示例的预测标签
        current_example_labels = []  # 当前示例的标签列表
        if not os.path.exists(ensembled_conll_pred_file):  # 检查集成预测文件是否存在
            logger.error(
                f"未找到集成CoNLL预测文件 {ensembled_conll_pred_file}。无法创建提交。")
            return

        with open(ensembled_conll_pred_file, 'r', encoding='utf-8') as f_conll:  # 读取集成预测文件
            for line_conll in f_conll:
                line_conll = line_conll.strip()
                if not line_conll:  # 处理空行
                    if current_example_labels:
                        predicted_labels_per_example.append(
                            current_example_labels)
                        current_example_labels = []
                else:
                    parts = line_conll.split('\t')
                    if len(parts) == 2:  # 处理有效行
                        current_example_labels.append(parts[1])
                    else:
                        logger.warning(
                            f"CoNLL预测文件中的格式错误行 '{ensembled_conll_pred_file}': '{line_conll}'。忽略该标记。")
            if current_example_labels:  # 处理最后一个示例
                predicted_labels_per_example.append(current_example_labels)

        if len(original_test_data) != len(predicted_labels_per_example):  # 检查示例数量是否匹配
            logger.error(
                f"示例数量不匹配！原始: {len(original_test_data)}, 预测: {len(predicted_labels_per_example)}。")
            num_examples_to_process = min(
                len(original_test_data), len(predicted_labels_per_example))
        else:
            num_examples_to_process = len(original_test_data)

        try:
            with open(final_submission_file, 'w', encoding='utf8') as fout:  # 写入最终提交文件
                written_count = 0
                for i in range(num_examples_to_process):
                    guid, original_text = original_test_data[i]  # 获取示例ID和原始文本
                    # 获取预测标签
                    labels_for_this_example = predicted_labels_per_example[i]
                    original_text_char_list = list(original_text)  # 将文本转换为字符列表

                    if len(original_text_char_list) != len(labels_for_this_example):  # 处理长度不匹配的情况
                        logger.warning(
                            f"GUID {guid} (示例 {i}): 长度不匹配！文本: {len(original_text_char_list)} 字符, 标签: {len(labels_for_this_example)}。填充/截断标签。"
                        )
                        final_labels_list = (labels_for_this_example + ['O'] * len(
                            original_text_char_list))[:len(original_text_char_list)]
                    else:
                        final_labels_list = labels_for_this_example

                    labels_str = ' '.join(final_labels_list)  # 将标签列表转换为字符串
                    fout.write(
                        f"{guid}\u0001{original_text}\u0001{labels_str}\n")  # 写入最终格式
                    written_count += 1
                logger.info(
                    f"已写入 {written_count} 行到最终提交文件: {final_submission_file}")
        except IOError as e:  # 处理写入失败的情况
            logger.error(
                f"写入最终提交文件到 {final_submission_file} 失败: {e}")

    def run_prediction_and_ensembling_pipeline(self, fold_work_dirs: list[str]):
        # 记录开始预测和集成流程的日志
        logger.info("--- 开始预测和集成流程 ---")
        # 检查是否提供了折叠工作目录列表
        if not fold_work_dirs:
            logger.error(
                "No fold work directories provided or discovered. Cannot run prediction.")
            return

        # 存储所有折叠的预测结果和对应的模型名称
        all_test_predictions_sources = []
        model_names = []

        # 遍历每个折叠目录进行预测
        for fold_dir_path in fold_work_dirs:
            logger.info(
                f"--- Generating predictions for fold: {fold_dir_path} ---")

            # 提取模型名称
            model_name = self._get_model_name_from_path(fold_dir_path)

            # 为当前模型创建特定的配置和预测器
            config = copy.deepcopy(self.config)
            config.model_name = model_name
            predictor = BiaffineSpanPredictor(config=config)

            logger.info(f"为模型 {model_name} 创建专用预测器")

            # 获取当前折叠的预测结果
            fold_predictions = predictor.get_predictions_for_fold(
                fold_work_dir=fold_dir_path,
                test_file_path=self.config.test_file,  # 测试文件路径
                label_map=self.config.label_map,     # 标签映射
                batch_size=self.config.batch_size,   # 批处理大小
                use_swa_if_available=self.config.use_swa  # 是否使用SWA模型
            )
            # 如果成功生成预测，则添加到结果列表中
            if fold_predictions:
                all_test_predictions_sources.append(fold_predictions)
                model_names.append(model_name)
                logger.info(f"成功收集模型 {model_name} 的预测结果")
            else:
                logger.warning(
                    f"No predictions returned from fold: {fold_dir_path}. Skipping for ensembling.")

            # 清理当前预测器以释放显存
            del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"已清理模型 {model_name} 的显存")

        # 检查是否收集到任何预测结果
        if not all_test_predictions_sources:
            logger.error(
                "No predictions collected from any fold. Cannot ensemble.")
            return

        # 记录收集到的预测源数量
        logger.info(
            f"Collected predictions from {len(all_test_predictions_sources)} sources for ensembling.")
        logger.info(f"模型列表: {model_names}")

        # 读取原始测试数据
        sentence_reader = SentenceReader()
        original_test_char_sequences = sentence_reader.read_tokens(
            file_path=self.config.test_file)
        num_test_examples = len(original_test_char_sequences)

        # 根据配置选择投票方式
        if self.use_soft_voting:
            ensembled_final_labels = self._ensemble_predictions_soft_voting(
                all_predictions=all_test_predictions_sources,
                model_names=model_names,
                num_test_examples=num_test_examples,
                original_test_char_sequences=original_test_char_sequences
            )
        else:
            ensembled_final_labels = self._ensemble_predictions_hard_voting(
                all_predictions=all_test_predictions_sources,
                num_test_examples=num_test_examples,
                original_test_char_sequences=original_test_char_sequences
            )

        os.makedirs(self.config.work_dir, exist_ok=True)

        # 根据投票方式生成不同的输出文件名
        voting_type = "soft" if self.use_soft_voting else "hard"
        ensembled_conll_output_path = os.path.join(
            self.config.work_dir, f"ensembled_predictions_pipeline_{voting_type}.conll")

        # 写入集成预测结果
        write_ok = self.write_ensembled_predictions_conll_format(
            output_path=ensembled_conll_output_path,
            original_char_sequences=original_test_char_sequences,
            ensembled_labels=ensembled_final_labels
        )

        # 如果成功写入，则处理最终提交输出
        if write_ok:
            # 生成对应的提交文件名
            output_file_parts = os.path.splitext(self.config.output_file)
            final_output_file = f"{output_file_parts[0]}_{voting_type}{output_file_parts[1]}"

            self.process_final_submission_output(
                ensembled_conll_pred_file=ensembled_conll_output_path,
                final_submission_file=final_output_file,
                original_test_file_path=self.config.test_file
            )
            logger.info(f"最终提交文件已生成: {final_output_file}")

        # 记录流程完成日志
        logger.info("--- Prediction and Ensembling Pipeline Finished ---")
