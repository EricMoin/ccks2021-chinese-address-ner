import os
from sklearn.model_selection import KFold
import torch
from config import Config
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import copy
import logging
from sklearn.metrics import classification_report

from conll_reader import ConllReader, MultiConllReader
from dataset import NERDataset, NERTestDataset
from model import FreeLB, AddressNER
from label import LabelMap
from visualization import TrainingVisualizer

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


class Trainer:
    config: Config
    model: nn.Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    device: torch.device

    def __init__(self, config: Config, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, device: str):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(device)
        self.scheduler = None  # 初始化调度器属性

        # 初始化训练可视化工具
        self.visualizer = TrainingVisualizer(self.config.work_dir)

        # 如果启用则初始化SWA
        if self.config.use_swa:
            self.swa = StochasticWeightAveraging(
                model=self.model,  # 传递模型实例用于深拷贝基础
                swa_start_epoch=self.config.swa_start_epoch,
                swa_lr=self.config.swa_lr,
                swa_freq=self.config.swa_freq
            )
        else:
            self.swa = None

        self.model.to(self.device)
        logger.info(
            f"训练器已初始化。在{self.device}上训练。工作目录：{self.config.work_dir}")

    def train(self):
        self.model.train()  # 确保模型处于训练模式
        # 优化器
        optimizer = torch.optim.AdamW([
            {'params': self.model.bert.embeddings.parameters(
            ), 'lr': self.config.learning_rate * 5},
            {'params': self.model.lstm.parameters(
            ), 'lr': self.config.learning_rate * 25},
            {'params': self.model.classifier.parameters(
            ), 'lr': self.config.learning_rate * 25},
            {'params': self.model.crf.parameters(
            ), 'lr': self.config.learning_rate * 50}
        ], lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )

        best_val_metric = 0
        # 折叠特定的工作目录
        os.makedirs(self.config.work_dir, exist_ok=True)

        freelb = None
        if hasattr(self.config, 'adversarial_training_start_epoch') and self.config.adversarial_training_start_epoch >= 0 and self.config.use_freelb:
            freelb = FreeLB(
                self.model,
                adv_lr=self.config.freelb_adv_lr,
                adv_steps=self.config.freelb_adv_steps,
                adv_init_mag=self.config.freelb_adv_init_mag,
                adv_max_norm=self.config.freelb_adv_max_norm,
                adv_norm_type=self.config.freelb_adv_norm_type,
                base_model=self.config.freelb_base_model
            )
            logger.info("FreeLB对抗训练已配置。")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(
                self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train] ({os.path.basename(self.config.work_dir)})")

            for batch in train_pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                if freelb and epoch >= self.config.adversarial_training_start_epoch:
                    # 获取FreeLB的原始嵌入
                    original_embeddings = self.model.bert.embeddings.word_embeddings(
                        input_ids)
                    # FreeLB的.attack()方法将处理自己的梯度累积
                    # 并在其循环之前内部调用self.model.zero_grad()。
                    adv_loss = freelb.attack(
                        original_embeddings.detach(), attention_mask, labels)
                    # 梯度现在已从FreeLB的攻击中在model.parameters()中累积。
                    # 我们使用adv_loss进行日志记录，但optimizer.step()的梯度来自FreeLB。
                    current_loss = adv_loss  # 用于日志记录目的
                else:
                    # 如果FreeLB未激活，则进行标准前向和后向传播
                    loss = self.model(input_ids, attention_mask, labels)
                    loss.backward()
                    current_loss = loss.item()

                # 梯度裁剪和优化器步骤（适用于标准传播或FreeLB的梯度）
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                train_loss += current_loss  # 累积损失以计算轮次平均值
                train_pbar.set_postfix({"loss": f"{current_loss:.4f}"})

            avg_train_loss = train_loss / \
                len(self.train_dataloader) if len(
                    self.train_dataloader) > 0 else 0
            logger.info(
                f"第{epoch+1}轮 ({os.path.basename(self.config.work_dir)}) 平均训练损失: {avg_train_loss:.4f}")

            # 评估
            self.model.eval()
            all_preds_eval = []
            all_labels_eval = []
            eval_loss = 0

            with torch.no_grad():
                for batch in tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Eval] ({os.path.basename(self.config.work_dir)})"):
                    input_ids_eval = batch["input_ids"].to(self.device)
                    attention_mask_eval = batch["attention_mask"].to(
                        self.device)
                    labels_eval = batch["labels"].to(self.device)

                    loss_eval_batch = self.model(
                        input_ids_eval, attention_mask_eval, labels_eval)
                    eval_loss += loss_eval_batch.item()

                    predictions_eval = self.model(
                        input_ids_eval, attention_mask_eval)

                    for pred_seq, mask_seq, label_seq in zip(predictions_eval, attention_mask_eval, labels_eval):
                        true_length = mask_seq.sum().item()
                        all_preds_eval.extend(pred_seq[:true_length])
                        all_labels_eval.extend(
                            label_seq[:true_length].cpu().numpy())

            avg_eval_loss = eval_loss / \
                len(self.val_dataloader) if len(self.val_dataloader) > 0 else 0
            correct_eval = sum(p == l for p, l in zip(
                all_preds_eval, all_labels_eval))
            total_eval = len(all_preds_eval) if len(
                all_preds_eval) > 0 else 1
            current_val_metric = correct_eval / total_eval
            logger.info(
                f"第{epoch+1}轮 ({os.path.basename(self.config.work_dir)}) 验证损失: {avg_eval_loss:.4f}, 验证准确率: {current_val_metric:.4f}")

            if total_eval > 0:
                # 生成并记录分类报告
                # 确保all_preds_eval和all_labels_eval是整数（标签ID）的扁平列表
                # 从id获取标签名称
                target_names = [self.config.label_map.id2label[i] for i in sorted(
                    list(set(all_labels_eval + all_preds_eval)))]
                # 在传递给classification_report之前从target_names中过滤掉任何OOD标签
                # 这假设您的label_map.id2label正确映射所有出现的ID。
                # all_preds_eval和all_labels_eval包含数字ID也很重要。

                # 处理某些标签可能只出现在预测或真实标签中的情况
                # 并且可能不在初始标签集中（如果id2label不详尽）
                # 我们将使用预测或真实中存在的标签，并映射它们。
                present_label_ids = sorted(
                    list(set(all_labels_eval).union(set(all_preds_eval))))

                # 确保所有这些ID在id2label中都有映射
                valid_target_names = []
                valid_label_ids_for_report = []

                for label_id in present_label_ids:
                    if label_id in self.config.label_map.id2label:
                        valid_target_names.append(
                            self.config.label_map.id2label[label_id])
                        valid_label_ids_for_report.append(label_id)
                    else:
                        logger.warning(
                            f"标签ID {label_id} 在预测/金标准标签中找到，但不在id2label映射中。跳过报告。")

                if valid_label_ids_for_report:  # 仅在有有效标签要报告时继续
                    try:
                        report = classification_report(
                            all_labels_eval,
                            all_preds_eval,
                            labels=valid_label_ids_for_report,  # 仅使用有名称的ID
                            target_names=valid_target_names,   # 对应的名称
                            digits=4,
                            zero_division=0  # 避免当某个类别没有预测或没有真实样本时的警告
                        )
                        logger.info(
                            f"第{epoch+1}轮分类报告 ({os.path.basename(self.config.work_dir)}):\n{report}")
                    except ValueError as e:
                        logger.error(
                            f"无法生成分类报告: {e}. 预测: {set(all_preds_eval)}, 标签: {set(all_labels_eval)}")
                else:
                    logger.warning(
                        "未找到有效标签来生成分类报告（所有预测/金标准标签都无法映射）。")

            if self.swa is not None:
                self.swa.update(epoch, self.model.state_dict())
            report_dict = classification_report(
                all_labels_eval,
                all_preds_eval,
                labels=valid_label_ids_for_report,
                target_names=valid_target_names,
                digits=4,
                zero_division=0,
                output_dict=True
            )
            micro_f1 = report_dict['weighted avg']['f1-score']
            current_val_metric = micro_f1  # 使用 Micro-F1 作为 val metric

            # 计算真正的验证准确率（之前的current_val_metric实际上是正确率）
            val_accuracy = correct_eval / total_eval

            # 记录当前epoch的训练指标到可视化工具
            self.visualizer.record_epoch(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_eval_loss,
                val_f1=micro_f1,
                val_accuracy=val_accuracy
            )

            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                saved_path = os.path.join(
                    self.config.work_dir, "best_model.pt")  # 保存在折叠特定目录中
                torch.save(self.model.state_dict(), saved_path)
                logger.info(
                    f"保存新的最佳模型 ({os.path.basename(self.config.work_dir)}) 验证指标: {current_val_metric:.4f} 到 {saved_path}")

        if self.swa is not None:
            swa_model_state_dict = self.swa.get_final_model_state_dict()
            if swa_model_state_dict is not None:
                swa_save_path = os.path.join(
                    self.config.work_dir, "swa_model.pt")  # 保存在折叠特定目录中
                torch.save(swa_model_state_dict, swa_save_path)
                logger.info(
                    f"保存最终SWA模型 ({os.path.basename(self.config.work_dir)}) 到 {swa_save_path}")

        # 完成训练可视化，生成并保存图表
        self.visualizer.finalize_training_visualization()

        logger.info(
            f"工作目录训练完成: {self.config.work_dir}")


class KFoldTrainer:
    def __init__(self, config: Config):
        self.config = config

    def kfold_train(self):
        logger.info("--- 开始K折训练流水线 ---")
        folds_base_dir = os.path.join(
            self.config.work_dir, self.config.model_name)
        os.makedirs(folds_base_dir, exist_ok=True)
        logger.info(
            f"模型 '{self.config.model_name}' 的K折基础目录: {folds_base_dir}")

        multi_reader = MultiConllReader()
        full_data_conll = list(multi_reader.read(
            # 句子对象列表
            [self.config.train_file, self.config.dev_file]))

        kf = KFold(n_splits=self.config.k_folds, shuffle=True,
                   random_state=self.config.seed)

        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(full_data_conll)):
            fold_num = fold_idx + 1
            logger.info(
                f"--- 处理模型 '{self.config.model_name}' 的第 {fold_num}/{self.config.k_folds} 折 ---")

            fold_train_data = [full_data_conll[i] for i in train_indices]
            fold_val_data = [full_data_conll[i] for i in val_indices]

            # 为此折创建特定配置
            # 从主配置开始复制
            fold_cfg = copy.deepcopy(self.config)
            fold_cfg.work_dir = os.path.join(
                folds_base_dir, f"fold_{fold_num}")
            # AddressNER的model_name应该是正在进行k折的适配模型路径
            fold_cfg.model_name = self.config.model_name
            os.makedirs(fold_cfg.work_dir, exist_ok=True)
            logger.info(
                f"第{fold_num}折配置: work_dir='{fold_cfg.work_dir}', model_name_for_tokenizer='{fold_cfg.model_name}'")

            # 为此折实例化模型（使用fold_cfg.model_name作为分词器）
            model = AddressNER(num_labels=len(
                fold_cfg.label_map.labels), config=fold_cfg)

            # 为此折创建数据集和数据加载器
            train_dataset = NERDataset(
                fold_train_data, model.tokenizer, fold_cfg.label_map.label2id)
            val_dataset = NERDataset(
                fold_val_data, model.tokenizer, fold_cfg.label_map.label2id)
            train_loader = DataLoader(
                train_dataset, batch_size=fold_cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=fold_cfg.batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True)

            # 为此折实例化并运行训练器
            trainer_fold = Trainer(config=fold_cfg,
                                   model=model,
                                   train_dataloader=train_loader,
                                   val_dataloader=val_loader,
                                   device=fold_cfg.device)
            trainer_fold.train()  # 这将在fold_cfg.work_dir中保存best_model.pt和swa_model.pt
        logger.info("--- K折训练流水线完成 ---")


class SingleTrainer:
    def __init__(self, config: Config):
        self.config = config

    def train(self):
        model = AddressNER(num_labels=len(
            self.config.label_map.labels), config=self.config)
        self.config.work_dir = os.path.join(
            self.config.work_dir, self.config.model_name)
        os.makedirs(self.config.work_dir, exist_ok=True)
        conll_reader = ConllReader()
        train_data = list(conll_reader.read(self.config.train_file))
        val_data = list(conll_reader.read(self.config.dev_file))

        train_dataset = NERDataset(
            train_data, model.tokenizer, self.config.label_map.label2id)
        val_dataset = NERDataset(
            val_data, model.tokenizer, self.config.label_map.label2id)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

        # 实例化并运行训练器
        trainer = Trainer(config=self.config,
                          model=model,
                          train_dataloader=train_loader,
                          val_dataloader=val_loader,
                          device=self.config.device)
        trainer.train()
