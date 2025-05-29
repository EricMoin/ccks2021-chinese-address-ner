# Span-based and Biaffine NER Models

本项目在原有的序列标注方法基础上，引入了**span-based**和**biaffine attention**机制，提供了更强大的命名实体识别能力。

## 🚀 新增功能

### 1. Span-based NER

- **直接预测实体边界**：不依赖于序列标注的BIO/BIOES标签，直接预测实体的起始和结束位置
- **避免标签不一致问题**：消除了序列标注中可能出现的标签冲突（如I-PER跟在B-LOC后面）
- **更好的长实体处理**：对于长实体的识别更加准确

### 2. Biaffine Attention

- **丰富的交互建模**：通过biaffine机制建模span起始和结束位置之间的复杂交互
- **参数高效**：相比全连接层，biaffine attention能够用更少的参数捕获更复杂的关系
- **数学表达**：`score(i,j) = x_i^T W x_j + U_1 x_i + U_2 x_j + b`

### 3. Hybrid Model

- **多任务学习**：同时进行序列标注和span预测，互相增强
- **集成预测**：结合两种方法的优势，提高整体性能
- **灵活权重**：可以调整序列标注和span预测的损失权重

## 📁 文件结构

```
├── span_biaffine_model.py    # Span-based和Biaffine模型定义
├── span_converter.py         # BIOES与Span格式转换工具
├── span_trainer.py          # Span模型训练器
├── span_predictor.py        # Span模型预测器
├── main_span.py            # Span模型训练主脚本
├── config_span.yaml       # Span模型配置文件
└── README_SPAN.md         # 本文档
```

## 🔧 模型架构

### 1. SpanBiaffineNER

```
BERT → BiLSTM → SpanClassifier(Biaffine) → Span Predictions
```

### 2. HybridNER

```
BERT → BiLSTM → ┌─ SequenceClassifier → CRF → Sequence Predictions
                └─ SpanClassifier(Biaffine) → Span Predictions
```

## ⚙️ 配置说明

### 核心配置参数

```yaml
# 模型类型选择
model_type: 'hybrid'  # 'sequence', 'span', 'hybrid'

# Span和Biaffine相关配置
use_biaffine: true              # 是否使用biaffine attention
span_threshold: 0.5             # span预测的置信度阈值
sequence_loss_weight: 0.4       # 混合模型中序列标注损失权重
span_loss_weight: 0.6           # 混合模型中span损失权重
biaffine_hidden_dim: 512        # biaffine attention隐藏维度
span_dropout: 0.1               # span分类器dropout率
```

### 模型类型说明

1. **`sequence`**: 传统的序列标注模型（BERT+LSTM+CRF）
2. **`span`**: 纯span-based模型，使用biaffine attention
3. **`hybrid`**: 混合模型，同时使用序列标注和span预测

## 🚀 使用方法

### 1. 训练模型

#### 单模型训练

```bash
python main_span.py --config config_span.yaml --mode single --model_type hybrid
```

#### K折交叉验证

```bash
python main_span.py --config config_span.yaml --mode kfold --model_type hybrid
```

#### 不同模型类型训练

```bash
# 纯span模型
python main_span.py --config config_span.yaml --model_type span

# 混合模型
python main_span.py --config config_span.yaml --model_type hybrid
```

### 2. 模型预测

```python
from config import Config
from span_predictor import SpanPredictor

# 加载配置和模型
config = Config('config_span.yaml')
predictor = SpanPredictor(config, 'result_span/best_hybrid_model.pt')

# 单文本预测
text = "北京市朝阳区建国门外大街1号"
entities = predictor.predict_single(text)
print(entities)

# 批量预测
texts = ["北京市朝阳区建国门外大街1号", "上海市浦东新区陆家嘴金融中心"]
batch_entities = predictor.predict_batch(texts)

# 文件预测
predictor.predict_file('data/test.txt', 'result/predictions.txt')
```

### 3. 集成预测

```python
from span_predictor import EnsembleSpanPredictor

# 多模型集成
model_paths = [
    'result_span/fold_1/best_hybrid_model.pt',
    'result_span/fold_2/best_hybrid_model.pt',
    'result_span/fold_3/best_hybrid_model.pt'
]

ensemble_predictor = EnsembleSpanPredictor(config, model_paths)
ensemble_predictor.predict_file('data/test.txt', 'result/ensemble_predictions.txt')
```

## 📊 性能对比

### 理论优势

| 方法 | 优势 | 劣势 |
|------|------|------|
| 序列标注 | 简单直观，训练稳定 | 标签不一致问题，长实体识别困难 |
| Span-based | 直接预测边界，无标签冲突 | 计算复杂度较高 |
| Biaffine | 丰富的交互建模，参数高效 | 需要更多调参 |
| Hybrid | 结合多种方法优势 | 模型复杂度增加 |

### 实验建议

1. **数据量充足**：推荐使用`hybrid`模型
2. **计算资源有限**：使用`span`模型
3. **追求稳定性**：使用`sequence`模型
4. **长实体较多**：优先考虑`span`或`hybrid`模型

## 🔍 技术细节

### 1. Biaffine Attention机制

```python
def biaffine_attention(x1, x2, W, U1, U2):
    """
    x1, x2: [batch, seq_len, hidden_dim]
    W: [hidden_dim, output_dim, hidden_dim]
    U1: [hidden_dim, output_dim]
    U2: [2*hidden_dim, output_dim]
    """
    # Biaffine term
    biaffine = torch.einsum('bsi,ioj,btj->bsto', x1, W, x2)
    
    # Linear terms
    linear1 = U1(x1).unsqueeze(2).expand(-1, -1, seq_len, -1)
    linear2 = U2(torch.cat([x1.unsqueeze(2).expand(-1, -1, seq_len, -1),
                           x2.unsqueeze(1).expand(-1, seq_len, -1, -1)], dim=-1))
    
    return biaffine + linear1 + linear2
```

### 2. Span标签转换

```python
# BIOES → Spans
def bioes_to_spans(tokens, labels):
    spans = []
    current_span = None
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith('B-'):
            if current_span:
                spans.append(current_span)
            current_span = {'start': i, 'end': i, 'label': label[2:]}
        elif label.startswith('I-') and current_span:
            current_span['end'] = i
        elif label.startswith('E-') and current_span:
            current_span['end'] = i
            spans.append(current_span)
            current_span = None
        elif label.startswith('S-'):
            spans.append({'start': i, 'end': i, 'label': label[2:]})
        else:  # 'O'
            if current_span:
                spans.append(current_span)
                current_span = None
    
    return spans
```

### 3. 损失函数设计

```python
# Hybrid模型损失
def hybrid_loss(sequence_logits, span_scores, sequence_labels, span_labels, mask):
    # 序列标注损失（CRF）
    sequence_loss = -crf(sequence_logits, sequence_labels, mask=mask)
    
    # Span损失（交叉熵）
    span_loss = cross_entropy(span_scores, span_labels, ignore_index=-100)
    
    # 加权组合
    total_loss = sequence_weight * sequence_loss + span_weight * span_loss
    return total_loss
```

## 🛠️ 调参建议

### 1. 学习率设置

```yaml
learning_rate: 3.0e-5  # 基础学习率
# 不同组件使用不同学习率倍数：
# BERT embeddings: 5x
# LSTM: 25x
# Classifiers: 25x
# CRF: 50x
```

### 2. 损失权重调整

```yaml
# 对于实体边界清晰的任务
sequence_loss_weight: 0.3
span_loss_weight: 0.7

# 对于实体边界模糊的任务
sequence_loss_weight: 0.6
span_loss_weight: 0.4
```

### 3. Span阈值优化

```yaml
span_threshold: 0.5  # 默认值
# 高精度要求：0.7-0.8
# 高召回要求：0.3-0.4
```

## 🐛 常见问题

### 1. 内存不足

- 减小`batch_size`
- 减小`biaffine_hidden_dim`
- 使用梯度累积

### 2. 训练不稳定

- 降低学习率
- 增加`span_dropout`
- 使用梯度裁剪

### 3. Span预测为空

- 降低`span_threshold`
- 检查标签转换是否正确
- 增加训练轮数

## 📈 进阶使用

### 1. 自定义Biaffine层

```python
class CustomBiaffine(nn.Module):
    def __init__(self, input_dim, output_dim, rank=None):
        super().__init__()
        # 低秩分解减少参数
        if rank:
            self.W1 = nn.Linear(input_dim, rank)
            self.W2 = nn.Linear(rank, output_dim * input_dim)
        else:
            self.W = nn.Parameter(torch.randn(input_dim, output_dim, input_dim))
```

### 2. 多任务学习扩展

```python
class MultiTaskNER(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence_head = SequenceClassifier()
        self.span_head = SpanClassifier()
        self.relation_head = RelationClassifier()  # 实体关系预测
        self.type_head = EntityTypeClassifier()    # 实体类型细分
```

### 3. 动态权重调整

```python
def dynamic_loss_weight(epoch, total_epochs):
    # 前期注重序列标注，后期注重span预测
    sequence_weight = 0.8 * (1 - epoch / total_epochs) + 0.2
    span_weight = 1 - sequence_weight
    return sequence_weight, span_weight
```

## 📚 参考文献

1. **Biaffine Attention**: "Deep Biaffine Attention for Neural Dependency Parsing" (Dozat & Manning, 2017)
2. **Span-based NER**: "A Unified MRC Framework for Named Entity Recognition" (Li et al., 2020)
3. **Multi-task Learning**: "Multi-Task Learning for Sequence Tagging" (Søgaard & Goldberg, 2016)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

### 开发环境设置

```bash
git clone <repository>
cd tianchi
pip install -r requirements.txt
```

### 代码规范

- 遵循PEP 8
- 添加类型提示
- 编写完整的docstring
- 添加单元测试

---

**Happy Coding! 🎉**
