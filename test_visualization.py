#!/usr/bin/env python3
"""
测试可视化功能的简单脚本
"""

import os
import numpy as np
from visualization import TrainingVisualizer


def test_visualization():
    """测试可视化功能"""

    # 创建测试目录
    test_dir = "test_visualization_output"
    os.makedirs(test_dir, exist_ok=True)

    # 初始化可视化工具
    visualizer = TrainingVisualizer(test_dir)

    # 模拟训练过程数据
    num_epochs = 10

    for epoch in range(num_epochs):
        # 模拟训练指标
        train_loss = 2.0 * np.exp(-epoch * 0.3) + \
            0.1 + np.random.normal(0, 0.05)
        val_loss = 1.8 * np.exp(-epoch * 0.25) + 0.15 + \
            np.random.normal(0, 0.08)
        val_f1 = 0.3 + 0.6 * (1 - np.exp(-epoch * 0.4)) + \
            np.random.normal(0, 0.02)
        val_accuracy = 0.25 + 0.65 * \
            (1 - np.exp(-epoch * 0.35)) + np.random.normal(0, 0.03)

        # 确保指标在合理范围内
        train_loss = max(0.01, train_loss)
        val_loss = max(0.01, val_loss)
        val_f1 = max(0.0, min(1.0, val_f1))
        val_accuracy = max(0.0, min(1.0, val_accuracy))

        # 记录epoch指标
        visualizer.record_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_f1=val_f1,
            val_accuracy=val_accuracy
        )

    # 完成可视化
    visualizer.finalize_training_visualization()

    print(f"测试完成！可视化文件保存在: {test_dir}")
    print("生成的文件包括:")
    print(f"- {test_dir}/plots/training_losses_and_f1.png")
    print(f"- {test_dir}/plots/training_metrics_summary.png")
    print(f"- {test_dir}/training_metrics.json")
    print(f"- {test_dir}/best_metrics.json")


if __name__ == "__main__":
    test_visualization()
