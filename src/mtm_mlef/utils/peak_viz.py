"""
峰值拟合效果可视化与评估
专门观察日间尖峰的预测质量
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from .peak_extractor import PeakExtractor, print_peak_statistics, save_peak_metrics_to_csv
import os


class PeakVisualizer:
    """
    峰值可视化工具
    """

    def __init__(self, hours_per_day: int = 24, figsize: Tuple[int, int] = (15, 10)):
        """
        Args:
            hours_per_day: 每天小时数
            figsize: 图形大小
        """
        self.hours_per_day = hours_per_day
        self.figsize = figsize
        self.peak_extractor = PeakExtractor(hours_per_day)

        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def plot_daily_curves_with_peaks(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        is_daytime: np.ndarray,
        time_indices: Optional[np.ndarray] = None,
        sample_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制每日曲线及峰值标记

        Args:
            y_true: 真实序列, shape (batch_size, seq_len)
            y_pred: 预测序列, shape (batch_size, seq_len)
            is_daytime: 白天标志, shape (batch_size, seq_len)
            time_indices: 时间索引, shape (batch_size, seq_len)
            sample_indices: 要绘制的样本索引列表，None则绘制前4个
            save_path: 保存路径
            show: 是否显示图形
        """
        # 提取峰值信息
        peak_info = self.peak_extractor.extract_daily_peaks(
            y_true, y_pred, is_daytime, time_indices
        )

        num_samples, seq_len = y_true.shape
        num_days = seq_len // self.hours_per_day

        # 选择要绘制的样本
        if sample_indices is None:
            sample_indices = list(range(min(4, num_samples)))

        # 创建子图
        n_samples = len(sample_indices)
        n_cols = 2
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * n_rows / 2))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

        for plot_idx, sample_idx in enumerate(sample_indices):
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = fig.add_subplot(gs[row, col])

            # 绘制每一天的曲线
            for day in range(num_days):
                start_idx = day * self.hours_per_day
                end_idx = start_idx + self.hours_per_day

                # 时间轴
                hours = np.arange(self.hours_per_day)

                # True curve
                ax.plot(
                    hours + day * self.hours_per_day,
                    y_true[sample_idx, start_idx:end_idx],
                    'b-', alpha=0.6, linewidth=2, label='True' if day == 0 else ''
                )

                # Predicted curve
                ax.plot(
                    hours + day * self.hours_per_day,
                    y_pred[sample_idx, start_idx:end_idx],
                    'r--', alpha=0.6, linewidth=2, label='Predicted' if day == 0 else ''
                )

                # Mark true peak
                true_peak_time = peak_info['true_peak_times'][sample_idx, day]
                true_peak_value = peak_info['true_peak_values'][sample_idx, day]
                ax.scatter(
                    true_peak_time + day * self.hours_per_day,
                    true_peak_value,
                    color='blue', s=150, marker='*', edgecolors='black',
                    linewidths=1.5, zorder=5, label='True Peak' if day == 0 else ''
                )

                # Mark predicted peak
                pred_peak_time = peak_info['pred_peak_times'][sample_idx, day]
                pred_peak_value = peak_info['pred_peak_values'][sample_idx, day]
                ax.scatter(
                    pred_peak_time + day * self.hours_per_day,
                    pred_peak_value,
                    color='red', s=150, marker='X', edgecolors='black',
                    linewidths=1.5, zorder=5, label='Pred Peak' if day == 0 else ''
                )

                # 用虚线连接真实峰值和预测峰值
                ax.plot(
                    [true_peak_time + day * self.hours_per_day,
                     pred_peak_time + day * self.hours_per_day],
                    [true_peak_value, pred_peak_value],
                    'k:', alpha=0.3, linewidth=1
                )

            ax.set_xlabel('Time (hours)', fontsize=10)
            ax.set_ylabel('Power (kW)', fontsize=10)
            ax.set_title(f'Sample {sample_idx} - Daily Curve vs Peak Comparison', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_peak_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        is_daytime: np.ndarray,
        time_indices: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制峰值误差分布图

        Args:
            y_true: 真实序列
            y_pred: 预测序列
            is_daytime: 白天标志
            time_indices: 时间索引
            save_path: 保存路径
            show: 是否显示
        """
        # 提取峰值信息
        peak_info = self.peak_extractor.extract_daily_peaks(
            y_true, y_pred, is_daytime, time_indices
        )

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # 1. 峰值大小误差分布
        ax = axes[0, 0]
        value_errors = peak_info['peak_value_errors'].flatten()
        ax.hist(value_errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Peak Value Error (kW)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Peak Value Error Distribution\n(RMSE={peak_info["peak_value_rmse"]:.2f}, '
                     f'MAE={peak_info["peak_value_mae"]:.2f})',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Peak time error distribution
        ax = axes[0, 1]
        time_errors = peak_info['peak_time_errors'].flatten()
        ax.hist(time_errors, bins=np.arange(-12, 13, 1), alpha=0.7,
                color='coral', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(-1, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='±1 hour')
        ax.axvline(1, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Peak Time Error (hours)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Peak Time Error Distribution\n(MAE={peak_info["peak_time_mae"]:.2f}h, '
                     f'±1h Accuracy={peak_info["peak_time_within_1h"]*100:.1f}%)',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. True Peak vs Predicted Peak scatter plot
        ax = axes[1, 0]
        true_peaks = peak_info['true_peak_values'].flatten()
        pred_peaks = peak_info['pred_peak_values'].flatten()
        ax.scatter(true_peaks, pred_peaks, alpha=0.5, s=30, color='purple')

        # Perfect prediction line
        min_val = min(true_peaks.min(), pred_peaks.min())
        max_val = max(true_peaks.max(), pred_peaks.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                label='Perfect Prediction')

        ax.set_xlabel('True Peak (kW)', fontsize=11)
        ax.set_ylabel('Predicted Peak (kW)', fontsize=11)
        ax.set_title('True Peak vs Predicted Peak', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Peak error boxplot by day
        ax = axes[1, 1]
        num_days = peak_info['peak_value_errors'].shape[1]
        error_by_day = [peak_info['peak_value_errors'][:, day] for day in range(num_days)]
        bp = ax.boxplot(error_by_day, labels=[f'D{i+1}' for i in range(num_days)],
                        patch_artist=True)

        # Set boxplot colors
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Day', fontsize=11)
        ax.set_ylabel('Peak Value Error (kW)', fontsize=11)
        ax.set_title('Peak Error Distribution by Day', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_comparison_across_models(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        is_daytime: np.ndarray,
        time_indices: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        对比多个模型的峰值拟合效果

        Args:
            predictions: 各模型预测, {model_name: predictions}
            y_true: 真实值
            is_daytime: 白天标志
            time_indices: 时间索引
            save_path: 保存路径
            show: 是否显示
        """
        model_names = list(predictions.keys())
        n_models = len(model_names)

        # 计算各模型的峰值指标
        model_metrics = {}
        for name, y_pred in predictions.items():
            peak_info = self.peak_extractor.extract_daily_peaks(
                y_true, y_pred, is_daytime, time_indices
            )
            model_metrics[name] = peak_info

        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # 1. Peak RMSE comparison
        ax = axes[0, 0]
        peak_rmses = [model_metrics[name]['peak_value_rmse'] for name in model_names]
        bars = ax.bar(model_names, peak_rmses, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_ylabel('Peak RMSE (kW)', fontsize=11)
        ax.set_title('Peak Value RMSE Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate values on bars
        for bar, val in zip(bars, peak_rmses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        # 2. Peak time MAE comparison
        ax = axes[0, 1]
        time_maes = [model_metrics[name]['peak_time_mae'] for name in model_names]
        bars = ax.bar(model_names, time_maes, alpha=0.7, color='coral', edgecolor='black')
        ax.set_ylabel('Peak Time MAE (hours)', fontsize=11)
        ax.set_title('Peak Time MAE Comparison', fontsize=12, fontweight='bold')
        ax.axhline(1.0, color='green', linestyle='--', linewidth=1.5,
                   alpha=0.7, label='1 hour threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, time_maes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        # 3. Peak ±1h accuracy comparison
        ax = axes[1, 0]
        within_1h = [model_metrics[name]['peak_time_within_1h'] * 100 for name in model_names]
        bars = ax.bar(model_names, within_1h, alpha=0.7, color='mediumseagreen', edgecolor='black')
        ax.set_ylabel('±1 hour Accuracy (%)', fontsize=11)
        ax.set_title('Peak Time ±1h Accuracy Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, within_1h):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

        # 4. Radar chart
        ax = axes[1, 1]
        categories = ['Peak RMSE\n(norm)', 'Peak MAE\n(norm)', 'Time MAE\n(norm)',
                      '±1h Acc']
        N = len(categories)

        # 归一化指标（越小越好的指标取倒数）
        peak_rmse_norm = np.array([model_metrics[name]['peak_value_rmse'] for name in model_names])
        peak_mae_norm = np.array([model_metrics[name]['peak_value_mae'] for name in model_names])
        time_mae_norm = np.array([model_metrics[name]['peak_time_mae'] for name in model_names])
        within_1h_score = np.array([model_metrics[name]['peak_time_within_1h'] for name in model_names])

        # 归一化到0-1，越大越好
        peak_rmse_norm = 1 - (peak_rmse_norm - peak_rmse_norm.min()) / (peak_rmse_norm.max() - peak_rmse_norm.min() + 1e-6)
        peak_mae_norm = 1 - (peak_mae_norm - peak_mae_norm.min()) / (peak_mae_norm.max() - peak_mae_norm.min() + 1e-6)
        time_mae_norm = 1 - (time_mae_norm - time_mae_norm.min()) / (time_mae_norm.max() - time_mae_norm.min() + 1e-6)

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        ax = plt.subplot(2, 2, 4, projection='polar')
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        for i, name in enumerate(model_names):
            values = [peak_rmse_norm[i], peak_mae_norm[i], time_mae_norm[i], within_1h_score[i]]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title('Model Peak Performance Radar\n(higher is better)', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def create_peak_evaluation_report(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    is_daytime: np.ndarray,
    time_indices: Optional[np.ndarray] = None,
    output_dir: str = './peak_evaluation',
    model_names: Optional[List[str]] = None
):
    """
    创建完整的峰值评估报告

    Args:
        predictions: 各模型预测, {model_name: predictions}
        y_true: 真实值
        is_daytime: 白天标志
        time_indices: 时间索引
        output_dir: 输出目录
        model_names: 模型名称列表（用于排序）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化工具
    visualizer = PeakVisualizer()
    extractor = PeakExtractor()

    print(f"\n{'='*60}")
    print(f"{'Peak Fitting Evaluation Report':^60}")
    print(f"{'='*60}\n")

    # 1. 计算各模型指标
    all_metrics = []
    model_peak_info = {}

    if model_names is None:
        model_names = list(predictions.keys())

    for name in model_names:
        y_pred = predictions[name]
        peak_info = extractor.extract_daily_peaks(y_true, y_pred, is_daytime, time_indices)
        model_peak_info[name] = peak_info

        # 打印统计
        print_peak_statistics(peak_info, prefix=f"[{name}] ")

        # 收集指标
        metrics_row = {
            'model': name,
            'peak_value_rmse': peak_info['peak_value_rmse'],
            'peak_value_mae': peak_info['peak_value_mae'],
            'peak_time_mae': peak_info['peak_time_mae'],
            'peak_time_within_1h': peak_info['peak_time_within_1h'],
            'mean_true_peak': np.mean(peak_info['true_peak_values']),
            'mean_pred_peak': np.mean(peak_info['pred_peak_values']),
            'overall_rmse': np.sqrt(np.mean((y_pred - y_true) ** 2))
        }
        all_metrics.append(metrics_row)

    # 2. 保存指标到CSV
    df_metrics = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(output_dir, 'peak_metrics_summary.csv')
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nMetrics summary saved to: {metrics_path}")

    # 3. Generate visualizations
    print(f"\n{'='*60}")
    print("Generating visualization charts...")
    print(f"{'='*60}\n")

    # 3.1 各模型对比
    visualizer.plot_comparison_across_models(
        predictions, y_true, is_daytime, time_indices,
        save_path=os.path.join(output_dir, 'model_comparison.png'),
        show=False
    )

    # 3.2 每个模型的详细分析
    for name in model_names:
        y_pred = predictions[name]

        # 误差分布图
        visualizer.plot_peak_error_distribution(
            y_true, y_pred, is_daytime, time_indices,
            save_path=os.path.join(output_dir, f'{name}_error_distribution.png'),
            show=False
        )

        # 日曲线对比图
        visualizer.plot_daily_curves_with_peaks(
            y_true, y_pred, is_daytime, time_indices,
            sample_indices=list(range(min(6, y_true.shape[0]))),
            save_path=os.path.join(output_dir, f'{name}_daily_curves.png'),
            show=False
        )

    # 4. Generate detailed daily metrics
    print("\nGenerating daily peak metrics...")
    for name in model_names:
        peak_info = model_peak_info[name]
        num_samples, num_days = peak_info['true_peak_values'].shape

        daily_data = []
        for sample_idx in range(num_samples):
            for day_idx in range(num_days):
                daily_data.append({
                    'model': name,
                    'sample_idx': sample_idx,
                    'day': day_idx + 1,
                    'true_peak_value': peak_info['true_peak_values'][sample_idx, day_idx],
                    'pred_peak_value': peak_info['pred_peak_values'][sample_idx, day_idx],
                    'peak_value_error': peak_info['peak_value_errors'][sample_idx, day_idx],
                    'true_peak_time': peak_info['true_peak_times'][sample_idx, day_idx],
                    'pred_peak_time': peak_info['pred_peak_times'][sample_idx, day_idx],
                    'peak_time_error': peak_info['peak_time_errors'][sample_idx, day_idx]
                })

        df_daily = pd.DataFrame(daily_data)
        daily_path = os.path.join(output_dir, f'{name}_daily_peak_details.csv')
        df_daily.to_csv(daily_path, index=False)

    print(f"\nAll results saved to directory: {output_dir}")
    print(f"\n{'='*60}")
    print("Evaluation report generation complete!")
    print(f"{'='*60}\n")

    return df_metrics, model_peak_info


# ==================== 使用示例 ====================

"""
=== 快速使用 ===

from peak_visualization import create_peak_evaluation_report

# 准备数据
predictions = {
    'seq2seq': seq2seq_predictions,
    'lightgbm': lgb_predictions,
    'xgboost': xgb_predictions,
    'ensemble': ensemble_predictions
}

# 生成完整报告
df_metrics, peak_info = create_peak_evaluation_report(
    predictions=predictions,
    y_true=y_test,
    is_daytime=is_daytime_test,
    time_indices=time_indices_test,
    output_dir='./results/peak_evaluation',
    model_names=['seq2seq', 'lightgbm', 'xgboost', 'ensemble']
)

# 输出内容:
# ./results/peak_evaluation/
# ├── peak_metrics_summary.csv          # 所有模型的峰值指标汇总
# ├── model_comparison.png              # 模型对比图
# ├── seq2seq_error_distribution.png    # Seq2Seq误差分布
# ├── seq2seq_daily_curves.png          # Seq2Seq日曲线对比
# ├── seq2seq_daily_peak_details.csv    # Seq2Seq每日峰值详情
# ├── lightgbm_error_distribution.png
# ├── lightgbm_daily_curves.png
# ├── lightgbm_daily_peak_details.csv
# └── ... (其他模型的图表和数据)


=== 单独使用各可视化功能 ===

from peak_visualization import PeakVisualizer

visualizer = PeakVisualizer(hours_per_day=24, figsize=(15, 10))

# 1. 绘制日曲线与峰值
visualizer.plot_daily_curves_with_peaks(
    y_true=y_test,
    y_pred=predictions['seq2seq'],
    is_daytime=is_daytime_test,
    sample_indices=[0, 1, 2, 3],  # 绘制前4个样本
    save_path='daily_curves.png',
    show=True
)

# 2. 绘制峰值误差分布
visualizer.plot_peak_error_distribution(
    y_true=y_test,
    y_pred=predictions['seq2seq'],
    is_daytime=is_daytime_test,
    save_path='error_distribution.png',
    show=True
)

# 3. 对比多个模型
visualizer.plot_comparison_across_models(
    predictions={
        'model_a': pred_a,
        'model_b': pred_b,
        'model_c': pred_c
    },
    y_true=y_test,
    is_daytime=is_daytime_test,
    save_path='model_comparison.png',
    show=True
)


=== 在训练脚本中集成 ===

# 在训练循环结束后，生成峰值评估报告
def evaluate_peak_performance(model, test_loader, device, output_dir):
    model.eval()
    all_preds = []
    all_targets = []
    all_is_daytime = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, is_daytime = batch
            inputs = inputs.to(device)

            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_is_daytime.append(is_daytime.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    is_daytime = np.concatenate(all_is_daytime, axis=0)

    # 生成报告
    predictions = {'trained_model': y_pred}
    create_peak_evaluation_report(
        predictions=predictions,
        y_true=y_true,
        is_daytime=is_daytime,
        output_dir=output_dir
    )

# 训练完成后调用
evaluate_peak_performance(model, test_loader, device, './results/final_evaluation')
"""


if __name__ == "__main__":
    # Test code
    print("=== Testing Peak Visualization ===")

    # 创建模拟数据
    np.random.seed(42)
    num_samples = 50
    seq_len = 168

    # 真实值
    y_true = np.random.rand(num_samples, seq_len) * 50 + 100
    for i in range(7):
        peak_time = 12 + i * 24
        if peak_time < seq_len:
            y_true[:, peak_time] += 150

    # 模拟预测
    y_pred_a = y_true + np.random.randn(num_samples, seq_len) * 15
    y_pred_b = y_true + np.random.randn(num_samples, seq_len) * 20

    predictions = {
        'model_a': y_pred_a,
        'model_b': y_pred_b
    }

    # 白天标志
    time_indices = np.tile(np.arange(24), (num_samples, 7))[:, :seq_len]
    is_daytime = ((time_indices >= 6) & (time_indices <= 20)).astype(float)

    # 生成完整报告
    df_metrics, peak_info = create_peak_evaluation_report(
        predictions=predictions,
        y_true=y_true,
        is_daytime=is_daytime,
        time_indices=time_indices,
        output_dir='./test_peak_evaluation'
    )

    print("\nTest complete! Check ./test_peak_evaluation directory")
