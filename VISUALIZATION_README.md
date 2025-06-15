# 训练可视化功能说明

## 概述

本项目已集成了训练过程可视化功能，能够自动记录并生成训练过程中的关键指标图表，包括：

- 训练损失和验证损失的变化
- 验证F1分数的变化
- 验证准确率的变化
- 所有指标的综合对比图

## 功能特性

### 1. 自动记录训练指标

- 每个epoch结束后自动记录训练损失、验证损失、验证F1分数和验证准确率
- 将所有指标保存为JSON格式，便于后续分析

### 2. 生成可视化图表

- **损失和F1变化图** (`training_losses_and_f1.png`): 显示训练和验证损失的对比，以及F1分数的变化趋势
- **综合指标图** (`training_metrics_summary.png`): 四个子图展示所有关键指标的变化情况

### 3. 保存最佳模型信息

- 自动识别并记录最佳F1分数对应的epoch和相关指标
- 保存完整的训练统计信息

## 文件结构

训练完成后，会在工作目录下生成以下文件：

```
work_dir/
├── plots/
│   ├── training_losses_and_f1.png      # 损失和F1变化图
│   └── training_metrics_summary.png    # 综合指标图
├── training_metrics.json               # 完整训练历史数据
├── best_metrics.json                   # 最佳模型指标信息
├── best_model.pt                       # 最佳模型权重（原有功能）
└── swa_model.pt                        # SWA模型权重（如启用）
```

## 使用方法

### 1. 正常训练

可视化功能已集成到现有的训练流程中，无需额外配置：

```bash
python main.py
```

### 2. 测试可视化功能

运行测试脚本验证可视化是否正常工作：

```bash
python test_visualization.py
```

## 图表说明

### 1. 训练损失和F1变化图 (`training_losses_and_f1.png`)

- **左侧子图**: 显示训练损失（蓝色）和验证损失（红色）的变化
- **右侧子图**: 显示验证F1分数（绿色）的变化趋势
- 包含数值标注，便于精确读取关键epoch的数值

### 2. 综合指标图 (`training_metrics_summary.png`)

- **左上**: 训练损失vs验证损失对比
- **右上**: 验证F1分数变化
- **左下**: 验证准确率变化
- **右下**: 所有指标的标准化对比（便于观察整体趋势一致性）

## 数据文件

### training_metrics.json

包含完整的训练历史数据：

```json
{
  "epochs": [1, 2, 3, ...],
  "train_losses": [2.1, 1.8, 1.5, ...],
  "val_losses": [2.0, 1.7, 1.4, ...],
  "val_f1_scores": [0.65, 0.72, 0.78, ...],
  "val_accuracies": [0.62, 0.69, 0.75, ...]
}
```

### best_metrics.json

包含最佳模型的详细信息：

```json
{
  "best_epoch": 8,
  "best_f1_score": 0.8456,
  "best_accuracy": 0.8123,
  "total_epochs": 15,
  "final_train_loss": 0.342,
  "final_val_loss": 0.389,
  "final_f1_score": 0.834,
  "final_accuracy": 0.801
}
```

## 技术细节

### 依赖要求

新增的可视化功能需要以下Python库：

- `matplotlib`: 图表绘制
- `numpy`: 数值计算
- `json`: 数据序列化

这些依赖已添加到 `requirements.txt` 文件中。

### 后端设置

可视化工具自动设置matplotlib为非交互式后端（Agg），适用于服务器环境和无显示设备的情况。

### 内存管理

每次生成图表后会自动释放matplotlib图形对象，避免内存泄漏。

## 自定义选项

如需自定义可视化行为，可以修改 `visualization.py` 中的相关参数：

- 图像尺寸：修改 `figsize` 参数
- 图像分辨率：修改 `dpi` 参数
- 标注频率：修改标注条件中的比例参数
- 颜色和样式：修改plot函数中的相关参数

## 故障排除

### 1. 图像生成失败

- 检查matplotlib是否正确安装
- 确保工作目录有写入权限
- 查看日志输出中的错误信息

### 2. 图像显示异常

- 确认使用的matplotlib版本兼容
- 检查系统字体设置（如果有中文显示问题）

### 3. 数据不完整

- 确认训练循环正常执行
- 检查分类报告是否成功生成
- 验证F1分数计算是否正确

## 示例输出

成功运行后，日志中会显示类似信息：

```
INFO - 训练可视化工具已初始化，图表将保存在: result/plots
INFO - 记录第1轮指标: train_loss=1.8542, val_loss=1.7234, val_f1=0.6789, val_acc=0.6543
...
INFO - 损失和F1分数变化图已保存到: result/plots/training_losses_and_f1.png
INFO - 训练指标综合图已保存到: result/plots/training_metrics_summary.png
INFO - 最佳模型: Epoch 8, F1=0.8456, Accuracy=0.8123
INFO - 训练可视化已完成
```

## 注意事项

1. 可视化功能不会影响原有的训练逻辑和模型保存
2. 图表生成在训练完成后进行，不会影响训练速度
3. 如果训练中断，可视化数据会丢失（建议结合checkpoint机制）
4. 大型项目建议定期清理旧的可视化文件以节省磁盘空间
