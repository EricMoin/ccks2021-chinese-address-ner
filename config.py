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
    focal_loss_alpha: float       # Focal Loss的alpha参数
    focal_loss_gamma: float       # Focal Loss的gamma参数

    # Dropout相关
    spatial_dropout: float    # 空间dropout率
    embedding_dropout: float  # 嵌入层dropout率

    # SWA (Stochastic Weight Averaging) 相关
    use_swa: bool            # 是否使用SWA
    swa_start_epoch: int     # 开始SWA的轮数
    swa_lr: float           # SWA的学习率
    swa_freq: int           # SWA更新频率

    # Span-based和Biaffine相关配置
    model_type: str          # 模型类型 ('span')
    use_biaffine: bool       # 是否使用biaffine attention
    span_threshold: float    # span预测的置信度阈值
    span_loss_weight: float      # span损失的权重
    biaffine_hidden_dim: int     # biaffine attention的隐藏维度
    span_dropout: float          # span分类器的dropout率
    span_loss_type: str          # span损失函数类型

    # 其他配置
    seed: int    # 随机种子
    k_folds: int  # K折交叉验证的折数
    max_sequence_length: int  # 最大序列长度
    early_stopping_patience: int  # 早停的耐心值

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
        self.focal_loss_alpha = config_dict.get('focal_loss_alpha', 0.25)
        self.focal_loss_gamma = config_dict.get('focal_loss_gamma', 1.5)

        # 设置Dropout相关
        self.spatial_dropout = config_dict.get('spatial_dropout', 0.15)
        self.embedding_dropout = config_dict.get('embedding_dropout', 0.15)

        # 设置SWA相关
        self.use_swa = config_dict.get('use_swa', True)
        self.swa_start_epoch = config_dict.get('swa_start_epoch', 0)
        self.swa_lr = config_dict.get('swa_lr', 1.0e-5)
        self.swa_freq = config_dict.get('swa_freq', 1)

        # 设置Span-based和Biaffine相关配置
        self.model_type = config_dict.get(
            'model_type', 'span')  # 默认为span
        self.use_biaffine = config_dict.get('use_biaffine', True)
        self.span_threshold = config_dict.get('span_threshold', 0.5)
        self.span_loss_weight = config_dict.get('span_loss_weight', 1.0)
        self.biaffine_hidden_dim = config_dict.get('biaffine_hidden_dim', 512)
        self.span_dropout = config_dict.get('span_dropout', 0.1)
        self.span_loss_type = config_dict.get('span_loss_type', 'combined')

        # 设置其他配置
        self.seed = config_dict.get('seed', 2024)
        self.k_folds = config_dict.get('k_folds', 5)
        self.max_sequence_length = config_dict.get('max_sequence_length', 384)
        self.early_stopping_patience = config_dict.get(
            'early_stopping_patience', 5)

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

    adapted_model_dir: str

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
        self.adapted_model_dir = os.path.join(
            "pretrained", f"{self.discriminator_model_name_or_path.replace('/', '_')}_electra_adapted_ep{self.num_epochs}_seed{self.seed}"
        )

        self.generator_loss_weight = config_dict.get(
            'generator_loss_weight', 1.0)
        self.discriminator_loss_weight = config_dict.get(
            'discriminator_loss_weight', 50.0)
        self.num_workers = config_dict.get('num_workers', 0)


class AugmentConfig(Config):
    """数据增强配置类，用于处理数据增强相关的配置参数"""

    # 文件路径
    train_file: str          # 训练数据文件路径
    dev_file: str           # 验证数据文件路径
    test_file: str          # 测试数据文件路径（用于可选的伪标签生成）
    work_dir: str           # 工作目录，用于保存增强文件和加载模型

    # 模型和预测设置（用于第二阶段 - 基于置信度的增强）
    model_name: str         # 基础模型名称（如来自Hugging Face）
    device: str             # 训练设备 ('cuda' 或 'cpu')
    batch_size: int         # 批次大小
    use_swa: bool          # 是否使用SWA（随机权重平均）模型

    # 标签映射
    label_map: LabelMap     # 标签映射对象

    # 数据增强特定参数
    augmentation_min_pattern_freq: int      # 实体类型模式被考虑用于增强的最小频率
    augmentation_factor: int                # 对于每个匹配频繁模式的句子，尝试生成的新句子数量
    augmentation_max_sentences: int         # 每个阶段总共生成的新句子的最大数量
    augmentation_confidence_threshold: float  # 模型预测的实体被包含在新词典中的最小平均置信度（第二阶段）

    # 伪标签策略（第二阶段）
    use_test_for_pseudo_labeling: bool      # 是否使用测试集进行伪标签生成（默认：False，使用dev集）
    validate_pseudo_labels: bool           # 当使用dev集时，是否验证伪标签与真实标签的一致性

    def __init__(self, config_path: str):
        """
        从YAML配置文件初始化数据增强配置

        Args:
            config_path: 配置文件路径
        """
        # 先调用父类构造函数，获取基础配置
        super().__init__(config_path)

        # 重新读取配置文件以获取增强特定的参数
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            raise ValueError(f"YAML file '{config_path}' is empty or invalid.")

        logger.info(f"Loading augmentation config from {config_path}")

        # 设置文件路径
        self.train_file = config_dict.get('train_file', 'data/train.conll')
        self.dev_file = config_dict.get('dev_file', 'data/dev.conll')
        self.test_file = config_dict.get('test_file', 'data/final_test.txt')
        self.work_dir = config_dict.get('work_dir', 'result')

        # 设置模型和预测设置
        self.model_name = config_dict.get(
            'model_name', 'hfl/chinese-roberta-wwm-ext')
        self.device = config_dict.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config_dict.get('batch_size', 16)
        self.use_swa = config_dict.get('use_swa', True)

        # 设置数据增强特定参数
        self.augmentation_min_pattern_freq = config_dict.get(
            'augmentation_min_pattern_freq', 2)
        self.augmentation_factor = config_dict.get('augmentation_factor', 2)
        self.augmentation_max_sentences = config_dict.get(
            'augmentation_max_sentences', 10000)
        self.augmentation_confidence_threshold = config_dict.get(
            'augmentation_confidence_threshold', 0.5)

        # 设置伪标签策略
        self.use_test_for_pseudo_labeling = config_dict.get(
            'use_test_for_pseudo_labeling', False)
        self.validate_pseudo_labels = config_dict.get(
            'validate_pseudo_labels', True)

        # 处理标签映射
        label_map_dict = config_dict.get('label_map', {})
        if not label_map_dict:
            raise ValueError(
                "label_map configuration is missing in the YAML file")

        self.label_map = LabelMap(
            labels=label_map_dict.get('labels', []),
            type=label_map_dict.get('type', 'BIOES')
        )

        logger.info(f"Augmentation config loaded successfully:")
        logger.info(f"  - Model: {self.model_name}")
        logger.info(f"  - Device: {self.device}")
        logger.info(
            f"  - Min pattern frequency: {self.augmentation_min_pattern_freq}")
        logger.info(f"  - Augmentation factor: {self.augmentation_factor}")
        logger.info(f"  - Max sentences: {self.augmentation_max_sentences}")
        logger.info(
            f"  - Confidence threshold: {self.augmentation_confidence_threshold}")
        logger.info(
            f"  - Use test for pseudo-labeling: {self.use_test_for_pseudo_labeling}")
        logger.info(
            f"  - Validate pseudo-labels: {self.validate_pseudo_labels}")
        logger.info(f"  - Label scheme: {self.label_map.type}")
        logger.info(f"  - Number of labels: {len(self.label_map.labels)}")
