import yaml
import os
from types import SimpleNamespace
from label import LabelMap
import torch
from logger import logger


class Config:
    # Define attributes with type hints for static analysis and autocompletion
    # 数据文件路径
    train_file: str  # 训练数据文件路径
    dev_file: str    # 验证数据文件路径
    test_file: str   # 测试数据文件路径
    output_file: str  # 输出文件路径

    # 模型相关配置
    model_name: str  # 预训练模型名称或路径
    batch_size: int  # 批次大小
    num_epochs: int  # 训练轮数
    learning_rate: float  # 学习率
    weight_decay: float   # 权重衰减
    device: str          # 训练设备 ('cuda' 或 'cpu')
    work_dir: str        # 工作目录
    freeze_bert_layers: int  # 冻结的BERT层数

    # 标签映射
    label_map: LabelMap  # 标签映射对象

    # 训练策略相关
    adversarial_training_start_epoch: int  # 开始对抗训练的轮数
    use_freelb: bool                       # 是否使用FreeLB对抗训练
    freelb_adv_lr: float                   # FreeLB对抗学习率
    freelb_adv_steps: int                  # FreeLB对抗步数
    freelb_adv_init_mag: float             # FreeLB扰动初始大小
    freelb_adv_max_norm: float             # FreeLB扰动最大范数
    freelb_adv_norm_type: str              # FreeLB扰动范数类型 ('l2' or 'linf')
    # FreeLB攻击的基础模型部分 ('bert' or 'embed')
    freelb_base_model: str

    # 损失函数相关
    crf_transition_penalty: float  # CRF转移惩罚
    focal_loss_alpha: float       # Focal Loss的alpha参数
    focal_loss_gamma: float       # Focal Loss的gamma参数
    hybrid_loss_weight_crf: float     # 混合损失中CRF的权重
    hybrid_loss_weight_focal: float   # 混合损失中Focal Loss的权重

    # Dropout相关
    spatial_dropout: float    # 空间dropout率
    embedding_dropout: float  # 嵌入层dropout率

    # SWA (Stochastic Weight Averaging) 相关
    use_swa: bool            # 是否使用SWA
    swa_start_epoch: int     # 开始SWA的轮数
    swa_lr: float           # SWA的学习率
    swa_freq: int           # SWA更新频率

    # 其他配置
    seed: int    # 随机种子
    k_folds: int  # K折交叉验证的折数

    def __init__(self, config_path: str):
        # 从yaml文件加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            raise ValueError(f"YAML file '{config_path}' is empty or invalid.")

        # 设置数据文件路径
        self.train_file = config_dict.get('train_file')
        self.dev_file = config_dict.get('dev_file')
        self.test_file = config_dict.get('test_file')
        self.output_file = config_dict.get('output_file')

        # 设置模型相关配置
        self.model_name = config_dict.get('model_name')
        self.batch_size = config_dict.get('batch_size', 16)
        self.num_epochs = config_dict.get('num_epochs', 2)
        self.learning_rate = config_dict.get('learning_rate', 2.0e-5)
        self.weight_decay = config_dict.get('weight_decay', 0.01)
        self.device = config_dict.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.work_dir = config_dict.get('work_dir', 'result')
        self.freeze_bert_layers = config_dict.get('freeze_bert_layers', 0)

        # 设置训练策略相关
        self.adversarial_training_start_epoch = config_dict.get(
            'adversarial_training_start_epoch', 0)
        self.use_freelb = config_dict.get(
            'use_freelb', False)  # Default to False
        self.freelb_adv_lr = config_dict.get('freelb_adv_lr', 0.03)
        self.freelb_adv_steps = config_dict.get('freelb_adv_steps', 3)
        self.freelb_adv_init_mag = config_dict.get('freelb_adv_init_mag', 0.05)
        self.freelb_adv_max_norm = config_dict.get('freelb_adv_max_norm', 0.0)
        self.freelb_adv_norm_type = config_dict.get(
            'freelb_adv_norm_type', 'l2')
        self.freelb_base_model = config_dict.get('freelb_base_model', 'bert')

        # 设置损失函数相关
        self.crf_transition_penalty = config_dict.get(
            'crf_transition_penalty', 0.175)
        self.focal_loss_alpha = config_dict.get('focal_loss_alpha', 0.25)
        self.focal_loss_gamma = config_dict.get('focal_loss_gamma', 1.5)
        self.hybrid_loss_weight_crf = config_dict.get(
            'hybrid_loss_weight_crf', 0.5)
        self.hybrid_loss_weight_focal = config_dict.get(
            'hybrid_loss_weight_focal', 0.5)

        # 设置Dropout相关
        self.spatial_dropout = config_dict.get('spatial_dropout', 0.15)
        self.embedding_dropout = config_dict.get('embedding_dropout', 0.15)

        # 设置SWA相关
        self.use_swa = config_dict.get('use_swa', True)
        self.swa_start_epoch = config_dict.get('swa_start_epoch', 0)
        self.swa_lr = config_dict.get('swa_lr', 1.0e-5)
        self.swa_freq = config_dict.get('swa_freq', 1)

        # 设置其他配置
        self.seed = config_dict.get('seed', 2024)
        self.k_folds = config_dict.get('k_folds', 5)

        # 处理标签映射
        label_map_dict = config_dict.get('label_map', {})
        if not label_map_dict:
            raise ValueError(
                "label_map configuration is missing in the YAML file")

        self.label_map = LabelMap(
            labels=label_map_dict.get('labels', []),
            type=label_map_dict.get('type', 'BIOES')
        )


class AdaptationConfig():
    seed: int

    # 预训练模型名称
    model_name: str
    generator_model_name_or_path: str  # Generator model path
    discriminator_model_name_or_path: str  # Discriminator model path
    # 领域语料文件路径
    corpus_file: str
    # 训练轮数
    num_epochs: int
    # 批次大小
    batch_size: int
    # 最大序列长度
    max_length: int
    # 预热步数
    warmup_steps: int
    # 学习率
    learning_rate: float
    # 权重衰减系数
    weight_decay: float
    # Adam优化器的epsilon参数
    adam_epsilon: float
    # 梯度裁剪的最大范数
    max_grad_norm: float
    # 掩码概率
    mask_probability: float
    # ELECTRA specific loss weights
    generator_loss_weight: float
    discriminator_loss_weight: float
    # Optional: for DataLoader
    num_workers: int

    def __init__(self, config_path: str):
        # 从yaml文件加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            raise ValueError(f"YAML file '{config_path}' is empty or invalid.")

        logger.info(f"load config from {config_dict}")
        # 从配置字典中读取适配相关的参数
        self.seed = config_dict.get('seed', 2025)

        # 设置配置参数
        self.model_name = config_dict.get(
            'model_name', 'microsoft/deberta-v3-base')
        self.generator_model_name_or_path = config_dict.get(
            # Default to a small generator
            'generator_model_name_or_path', 'prajjwal1/bert-tiny')
        self.discriminator_model_name_or_path = config_dict.get(
            'discriminator_model_name_or_path', 'hfl/chinese-roberta-wwm-ext')
        self.corpus_file = config_dict.get(
            'corpus_file', 'data/address.txt')
        self.num_epochs = config_dict.get('num_epochs', 3)
        self.batch_size = config_dict.get('batch_size', 16)
        self.max_length = config_dict.get('max_length', 128)
        self.learning_rate = config_dict.get('learning_rate', 5.0e-5)
        self.weight_decay = config_dict.get('weight_decay', 0.01)
        self.adam_epsilon = config_dict.get('adam_epsilon', 1.0e-8)
        self.warmup_steps = config_dict.get('warmup_steps', 0)
        self.max_grad_norm = config_dict.get('max_grad_norm', 1.0)
        self.mask_probability = config_dict.get('mask_probability', 0.15)

        self.generator_loss_weight = config_dict.get(
            'generator_loss_weight', 1.0)
        self.discriminator_loss_weight = config_dict.get(
            'discriminator_loss_weight', 50.0)
        self.num_workers = config_dict.get('num_workers', 0)
