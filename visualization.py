from logger import logger
import json
from typing import List, Dict, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端，适用于服务器环境


class TrainingVisualizer:
    """
    训练过程可视化工具类，用于保存和绘制训练历史图表
    """

    def __init__(self, work_dir: str):
        """
        初始化可视化工具

        Args:
            work_dir: 工作目录路径
        """
        self.work_dir = work_dir
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_accuracies = []
        self.epochs = []

        # 创建plots目录
        self.plots_dir = os.path.join(work_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        logger.info(f"训练可视化工具已初始化，图表将保存在: {self.plots_dir}")

    def record_epoch(self, epoch: int, train_loss: float, val_loss: float,
                     val_f1: float, val_accuracy: float):
        """
        记录单个epoch的训练指标

        Args:
            epoch: 当前epoch数
            train_loss: 训练损失
            val_loss: 验证损失
            val_f1: 验证F1分数
            val_accuracy: 验证准确率
        """
        self.epochs.append(epoch + 1)  # 从1开始计数
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_f1_scores.append(val_f1)
        self.val_accuracies.append(val_accuracy)

        logger.info(f"记录第{epoch+1}轮指标: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_f1={val_f1:.4f}, val_acc={val_accuracy:.4f}")

    def save_metrics_history(self):
        """保存训练历史指标到JSON文件"""
        metrics_data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'val_accuracies': self.val_accuracies
        }

        metrics_file = os.path.join(self.work_dir, "training_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)

        logger.info(f"训练历史指标已保存到: {metrics_file}")

    def plot_losses(self, figsize: tuple = (12, 6)):
        """
        绘制训练和验证损失变化图

        Args:
            figsize: 图像尺寸
        """
        plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.train_losses, 'b-',
                 label='Train Loss', linewidth=2, marker='o')
        plt.plot(self.epochs, self.val_losses, 'r-',
                 label='Validation Loss', linewidth=2, marker='s')
        plt.title('Training and Validation Loss',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 添加数值标注
        for i, (epoch, train_loss, val_loss) in enumerate(zip(self.epochs, self.train_losses, self.val_losses)):
            if i % max(1, len(self.epochs) // 5) == 0:  # 每隔几个epoch标注一次
                plt.annotate(f'{train_loss:.3f}', (epoch, train_loss),
                             textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
                plt.annotate(f'{val_loss:.3f}', (epoch, val_loss),
                             textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8)

        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.val_f1_scores, 'g-',
                 label='Validation F1 Score', linewidth=2, marker='^')
        plt.title('Validation F1 Score', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 添加F1分数标注
        for i, (epoch, f1_score) in enumerate(zip(self.epochs, self.val_f1_scores)):
            if i % max(1, len(self.epochs) // 5) == 0:
                plt.annotate(f'{f1_score:.3f}', (epoch, f1_score),
                             textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        plt.tight_layout()

        # 保存图像
        loss_plot_path = os.path.join(
            self.plots_dir, "training_losses_and_f1.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"损失和F1分数变化图已保存到: {loss_plot_path}")

    def plot_metrics_summary(self, figsize: tuple = (15, 10)):
        """
        绘制训练指标综合图表

        Args:
            figsize: 图像尺寸
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Training Metrics Summary',
                     fontsize=16, fontweight='bold')

        # 1. 训练和验证损失
        axes[0, 0].plot(self.epochs, self.train_losses, 'b-',
                        label='Train Loss', linewidth=2, marker='o')
        axes[0, 0].plot(self.epochs, self.val_losses, 'r-',
                        label='Validation Loss', linewidth=2, marker='s')
        axes[0, 0].set_title('Loss Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 验证F1分数
        axes[0, 1].plot(self.epochs, self.val_f1_scores, 'g-',
                        label='Validation F1', linewidth=2, marker='^')
        axes[0, 1].set_title('Validation F1 Score', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 验证准确率
        axes[1, 0].plot(self.epochs, self.val_accuracies, 'orange',
                        label='Validation Accuracy', linewidth=2, marker='d')
        axes[1, 0].set_title('Validation Accuracy', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 所有指标的标准化比较
        # 标准化所有指标到[0,1]范围进行比较
        norm_train_loss = np.array(self.train_losses)
        norm_val_loss = np.array(self.val_losses)
        norm_f1 = np.array(self.val_f1_scores)
        norm_acc = np.array(self.val_accuracies)

        # 对损失进行反转标准化（损失越低越好）
        norm_train_loss = 1 - (norm_train_loss - np.min(norm_train_loss)) / \
            (np.max(norm_train_loss) - np.min(norm_train_loss) + 1e-8)
        norm_val_loss = 1 - (norm_val_loss - np.min(norm_val_loss)) / \
            (np.max(norm_val_loss) - np.min(norm_val_loss) + 1e-8)

        axes[1, 1].plot(self.epochs, norm_train_loss, 'b--',
                        label='Train Loss (Normalized)', linewidth=2, alpha=0.7)
        axes[1, 1].plot(self.epochs, norm_val_loss, 'r--',
                        label='Val Loss (Normalized)', linewidth=2, alpha=0.7)
        axes[1, 1].plot(self.epochs, norm_f1, 'g-',
                        label='Val F1 Score', linewidth=2)
        axes[1, 1].plot(self.epochs, norm_acc, 'orange',
                        label='Val Accuracy', linewidth=2)
        axes[1, 1].set_title(
            'Normalized Metrics Comparison', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Normalized Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图像
        summary_plot_path = os.path.join(
            self.plots_dir, "training_metrics_summary.png")
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"训练指标综合图已保存到: {summary_plot_path}")

    def save_best_metrics_info(self, best_epoch: int, best_f1: float, best_accuracy: float):
        """
        保存最佳指标信息

        Args:
            best_epoch: 最佳epoch
            best_f1: 最佳F1分数
            best_accuracy: 最佳准确率
        """
        best_metrics = {
            'best_epoch': best_epoch,
            'best_f1_score': best_f1,
            'best_accuracy': best_accuracy,
            'total_epochs': len(self.epochs),
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'final_f1_score': self.val_f1_scores[-1] if self.val_f1_scores else None,
            'final_accuracy': self.val_accuracies[-1] if self.val_accuracies else None
        }

        best_metrics_file = os.path.join(self.work_dir, "best_metrics.json")
        with open(best_metrics_file, 'w', encoding='utf-8') as f:
            json.dump(best_metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"最佳指标信息已保存到: {best_metrics_file}")
        logger.info(
            f"最佳模型: Epoch {best_epoch}, F1={best_f1:.4f}, Accuracy={best_accuracy:.4f}")

    def finalize_training_visualization(self):
        """
        完成训练后的最终可视化处理
        """
        if not self.epochs:
            logger.warning("没有训练历史数据，跳过可视化")
            return

        # 保存指标历史
        self.save_metrics_history()

        # 生成所有图表
        self.plot_losses()
        self.plot_metrics_summary()

        # 找到最佳指标
        if self.val_f1_scores:
            best_f1_idx = np.argmax(self.val_f1_scores)
            best_epoch = self.epochs[best_f1_idx]
            best_f1 = self.val_f1_scores[best_f1_idx]
            best_accuracy = self.val_accuracies[best_f1_idx]

            self.save_best_metrics_info(best_epoch, best_f1, best_accuracy)

        logger.info("训练可视化已完成")
