"""
峰值定义与提取工具模块
用于光伏发电预测中的日间尖峰分析
"""

import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
import pandas as pd


class PeakExtractor:
    """
    峰值提取器：从168小时时间序列中提取每日峰值信息
    """

    def __init__(self, hours_per_day: int = 24):
        """
        Args:
            hours_per_day: 每天的小时数，默认24小时
        """
        self.hours_per_day = hours_per_day

    def extract_daily_peaks(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        is_daytime: Optional[np.ndarray] = None,
        time_indices: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        从真实序列和预测序列中提取每日峰值信息

        Args:
            y_true: 真实序列, shape (batch_size, 168)
            y_pred: 预测序列, shape (batch_size, 168)
            is_daytime: 白天标志, shape (batch_size, 168)，1表示白天，0表示夜晚
                       如果为None，则默认6:00-20:00为白天
            time_indices: 时间索引, shape (batch_size, 168)
                         表示每个时间步在一天中的小时(0-23)

        Returns:
            包含以下键值的字典：
            - 'true_peak_values': 每天真实峰值大小, shape (batch_size, num_days)
            - 'pred_peak_values': 每天预测峰值大小, shape (batch_size, num_days)
            - 'true_peak_times': 每天真实峰值时刻（小时索引）, shape (batch_size, num_days)
            - 'pred_peak_times': 每天预测峰值时刻（小时索引）, shape (batch_size, num_days)
            - 'peak_value_errors': 峰值大小误差, shape (batch_size, num_days)
            - 'peak_time_errors': 峰值时刻误差（小时）, shape (batch_size, num_days)
            - 'peak_value_rmse': 峰值大小的RMSE (标量)
            - 'peak_value_mae': 峰值大小的MAE (标量)
            - 'peak_time_mae': 峰值时刻的MAE (标量)
            - 'peak_time_within_1h': 峰值时刻误差在±1小时内的比例 (标量)
        """
        batch_size, seq_len = y_true.shape
        num_days = seq_len // self.hours_per_day

        # 如果没有提供白天标志，默认6:00-20:00为白天
        if is_daytime is None:
            if time_indices is None:
                # 假设序列从第0天0时开始
                time_indices = np.tile(np.arange(self.hours_per_day), (batch_size, num_days))[:, :seq_len]
            is_daytime = ((time_indices >= 6) & (time_indices <= 20)).astype(float)

        # 初始化结果数组
        true_peak_values = np.zeros((batch_size, num_days))
        pred_peak_values = np.zeros((batch_size, num_days))
        true_peak_times = np.zeros((batch_size, num_days), dtype=int)
        pred_peak_times = np.zeros((batch_size, num_days), dtype=int)

        # 对每一天进行处理
        for day in range(num_days):
            start_idx = day * self.hours_per_day
            end_idx = start_idx + self.hours_per_day

            # 提取当天的数据
            y_true_day = y_true[:, start_idx:end_idx]  # (batch_size, 24)
            y_pred_day = y_pred[:, start_idx:end_idx]
            is_daytime_day = is_daytime[:, start_idx:end_idx]

            # 对每个样本找出白天的峰值
            for i in range(batch_size):
                # 获取白天时段的mask
                daytime_mask = is_daytime_day[i] > 0.5

                if daytime_mask.sum() > 0:
                    # 只在白天时段中寻找峰值
                    true_daytime = y_true_day[i] * daytime_mask
                    pred_daytime = y_pred_day[i] * daytime_mask

                    # 找到真实峰值
                    true_peak_idx = np.argmax(true_daytime)
                    true_peak_values[i, day] = y_true_day[i, true_peak_idx]
                    true_peak_times[i, day] = true_peak_idx

                    # 找到预测峰值
                    pred_peak_idx = np.argmax(pred_daytime)
                    pred_peak_values[i, day] = y_pred_day[i, pred_peak_idx]
                    pred_peak_times[i, day] = pred_peak_idx
                else:
                    # 如果没有白天时段，使用全天最大值
                    true_peak_idx = np.argmax(y_true_day[i])
                    pred_peak_idx = np.argmax(y_pred_day[i])

                    true_peak_values[i, day] = y_true_day[i, true_peak_idx]
                    pred_peak_values[i, day] = y_pred_day[i, pred_peak_idx]
                    true_peak_times[i, day] = true_peak_idx
                    pred_peak_times[i, day] = pred_peak_idx

        # 计算误差统计
        peak_value_errors = pred_peak_values - true_peak_values
        peak_time_errors = pred_peak_times - true_peak_times

        # 全局统计指标
        peak_value_rmse = np.sqrt(np.mean(peak_value_errors ** 2))
        peak_value_mae = np.mean(np.abs(peak_value_errors))
        peak_time_mae = np.mean(np.abs(peak_time_errors))
        peak_time_within_1h = np.mean(np.abs(peak_time_errors) <= 1)

        return {
            'true_peak_values': true_peak_values,
            'pred_peak_values': pred_peak_values,
            'true_peak_times': true_peak_times,
            'pred_peak_times': pred_peak_times,
            'peak_value_errors': peak_value_errors,
            'peak_time_errors': peak_time_errors,
            'peak_value_rmse': peak_value_rmse,
            'peak_value_mae': peak_value_mae,
            'peak_time_mae': peak_time_mae,
            'peak_time_within_1h': peak_time_within_1h
        }

    def extract_daily_peaks_torch(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        is_daytime: Optional[torch.Tensor] = None,
        time_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        PyTorch版本的峰值提取（用于损失函数计算）

        Args:
            y_true: 真实序列, shape (batch_size, 168)
            y_pred: 预测序列, shape (batch_size, 168)
            is_daytime: 白天标志, shape (batch_size, 168)
            time_indices: 时间索引, shape (batch_size, 168)

        Returns:
            包含峰值信息的字典（所有值为torch.Tensor）
        """
        batch_size, seq_len = y_true.shape
        num_days = seq_len // self.hours_per_day
        device = y_true.device

        # 如果没有提供白天标志，默认6:00-20:00为白天
        if is_daytime is None:
            if time_indices is None:
                time_indices = torch.arange(self.hours_per_day, device=device).repeat(batch_size, num_days)[:, :seq_len]
            is_daytime = ((time_indices >= 6) & (time_indices <= 20)).float()

        # 初始化结果
        true_peak_values = torch.zeros(batch_size, num_days, device=device)
        pred_peak_values = torch.zeros(batch_size, num_days, device=device)
        true_peak_times = torch.zeros(batch_size, num_days, dtype=torch.long, device=device)
        pred_peak_times = torch.zeros(batch_size, num_days, dtype=torch.long, device=device)

        # 对每一天进行处理
        for day in range(num_days):
            start_idx = day * self.hours_per_day
            end_idx = start_idx + self.hours_per_day

            # 提取当天的数据
            y_true_day = y_true[:, start_idx:end_idx]  # (batch_size, 24)
            y_pred_day = y_pred[:, start_idx:end_idx]
            is_daytime_day = is_daytime[:, start_idx:end_idx]

            # 创建白天mask（将夜晚设为极小值）
            true_masked = torch.where(is_daytime_day > 0.5, y_true_day, torch.tensor(-1e9, device=device))
            pred_masked = torch.where(is_daytime_day > 0.5, y_pred_day, torch.tensor(-1e9, device=device))

            # 找到峰值索引
            true_peak_idx = torch.argmax(true_masked, dim=1)  # (batch_size,)
            pred_peak_idx = torch.argmax(pred_masked, dim=1)

            # 提取峰值
            true_peak_values[:, day] = torch.gather(y_true_day, 1, true_peak_idx.unsqueeze(1)).squeeze(1)
            pred_peak_values[:, day] = torch.gather(y_pred_day, 1, pred_peak_idx.unsqueeze(1)).squeeze(1)
            true_peak_times[:, day] = true_peak_idx
            pred_peak_times[:, day] = pred_peak_idx

        # 计算误差
        peak_value_errors = pred_peak_values - true_peak_values
        peak_time_errors = (pred_peak_times - true_peak_times).float()

        return {
            'true_peak_values': true_peak_values,
            'pred_peak_values': pred_peak_values,
            'true_peak_times': true_peak_times,
            'pred_peak_times': pred_peak_times,
            'peak_value_errors': peak_value_errors,
            'peak_time_errors': peak_time_errors
        }


def print_peak_statistics(peak_info: Dict[str, np.ndarray], prefix: str = ""):
    """
    打印峰值统计信息

    Args:
        peak_info: extract_daily_peaks返回的字典
        prefix: 打印前缀
    """
    print(f"\n{prefix}峰值拟合统计:")
    print(f"  峰值大小 RMSE: {peak_info['peak_value_rmse']:.4f}")
    print(f"  峰值大小 MAE:  {peak_info['peak_value_mae']:.4f}")
    print(f"  峰值时刻 MAE:  {peak_info['peak_time_mae']:.4f} 小时")
    print(f"  峰值时刻 ±1h准确率: {peak_info['peak_time_within_1h']*100:.2f}%")

    # 额外统计
    mean_true_peak = np.mean(peak_info['true_peak_values'])
    mean_pred_peak = np.mean(peak_info['pred_peak_values'])
    print(f"  平均真实峰值: {mean_true_peak:.4f}")
    print(f"  平均预测峰值: {mean_pred_peak:.4f}")
    print(f"  峰值相对误差: {(mean_pred_peak - mean_true_peak) / mean_true_peak * 100:.2f}%")


def save_peak_metrics_to_csv(peak_info: Dict[str, np.ndarray], save_path: str):
    """
    将峰值指标保存到CSV文件

    Args:
        peak_info: extract_daily_peaks返回的字典
        save_path: 保存路径
    """
    metrics = {
        'metric': [
            'peak_value_rmse',
            'peak_value_mae',
            'peak_time_mae',
            'peak_time_within_1h_ratio',
            'mean_true_peak',
            'mean_pred_peak',
            'peak_relative_error'
        ],
        'value': [
            peak_info['peak_value_rmse'],
            peak_info['peak_value_mae'],
            peak_info['peak_time_mae'],
            peak_info['peak_time_within_1h'],
            np.mean(peak_info['true_peak_values']),
            np.mean(peak_info['pred_peak_values']),
            (np.mean(peak_info['pred_peak_values']) - np.mean(peak_info['true_peak_values'])) /
            np.mean(peak_info['true_peak_values'])
        ]
    }

    df = pd.DataFrame(metrics)
    df.to_csv(save_path, index=False)
    print(f"峰值指标已保存到: {save_path}")


def analyze_peak_error_patterns(
    peak_info: Dict[str, np.ndarray],
    y_true: np.ndarray,
    is_daytime: np.ndarray,
    additional_features: Optional[Dict[str, np.ndarray]] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    分析峰值误差的模式，识别问题类型

    Args:
        peak_info: extract_daily_peaks返回的字典
        y_true: 真实序列, shape (batch_size, seq_len)
        is_daytime: 白天标志
        additional_features: 额外特征（如GHI、温度等）

    Returns:
        分析结果字典，包含误差模式统计
    """
    true_peak_values = peak_info['true_peak_values']  # (batch_size, num_days)
    peak_value_errors = peak_info['peak_value_errors']
    peak_time_errors = peak_info['peak_time_errors']

    batch_size, num_days = true_peak_values.shape

    analysis = {}

    # 1. 按峰值功率分组分析（低/中/高功率日）
    peak_flat = true_peak_values.flatten()
    error_flat = peak_value_errors.flatten()
    time_error_flat = peak_time_errors.flatten()

    # 分位数分组
    low_threshold = np.percentile(peak_flat, 33)
    high_threshold = np.percentile(peak_flat, 67)

    low_mask = peak_flat <= low_threshold
    mid_mask = (peak_flat > low_threshold) & (peak_flat <= high_threshold)
    high_mask = peak_flat > high_threshold

    analysis['by_power_level'] = {
        'low': {
            'count': low_mask.sum(),
            'mean_peak': peak_flat[low_mask].mean(),
            'value_rmse': np.sqrt(np.mean(error_flat[low_mask]**2)),
            'value_mae': np.mean(np.abs(error_flat[low_mask])),
            'time_mae': np.mean(np.abs(time_error_flat[low_mask])),
            'within_1h': np.mean(np.abs(time_error_flat[low_mask]) <= 1)
        },
        'mid': {
            'count': mid_mask.sum(),
            'mean_peak': peak_flat[mid_mask].mean(),
            'value_rmse': np.sqrt(np.mean(error_flat[mid_mask]**2)),
            'value_mae': np.mean(np.abs(error_flat[mid_mask])),
            'time_mae': np.mean(np.abs(time_error_flat[mid_mask])),
            'within_1h': np.mean(np.abs(time_error_flat[mid_mask]) <= 1)
        },
        'high': {
            'count': high_mask.sum(),
            'mean_peak': peak_flat[high_mask].mean(),
            'value_rmse': np.sqrt(np.mean(error_flat[high_mask]**2)),
            'value_mae': np.mean(np.abs(error_flat[high_mask])),
            'time_mae': np.mean(np.abs(time_error_flat[high_mask])),
            'within_1h': np.mean(np.abs(time_error_flat[high_mask]) <= 1)
        }
    }

    # 2. 按天数分析（周内哪一天误差更大）
    analysis['by_day'] = {}
    for day in range(num_days):
        day_errors = peak_value_errors[:, day]
        day_time_errors = peak_time_errors[:, day]
        analysis['by_day'][f'day_{day+1}'] = {
            'value_rmse': np.sqrt(np.mean(day_errors**2)),
            'value_mae': np.mean(np.abs(day_errors)),
            'time_mae': np.mean(np.abs(day_time_errors)),
            'within_1h': np.mean(np.abs(day_time_errors) <= 1)
        }

    # 3. 误差模式分类
    # - 低估模式：预测峰值 < 真实峰值
    # - 高估模式：预测峰值 > 真实峰值
    # - 时刻偏早：预测时刻 < 真实时刻
    # - 时刻偏晚：预测时刻 > 真实时刻
    underestimate_ratio = np.mean(error_flat < 0)
    overestimate_ratio = np.mean(error_flat > 0)
    early_ratio = np.mean(time_error_flat < 0)
    late_ratio = np.mean(time_error_flat > 0)

    analysis['error_patterns'] = {
        'underestimate_ratio': underestimate_ratio,
        'overestimate_ratio': overestimate_ratio,
        'early_prediction_ratio': early_ratio,
        'late_prediction_ratio': late_ratio,
        'mean_value_bias': np.mean(error_flat),  # 负=低估, 正=高估
        'mean_time_bias': np.mean(time_error_flat)  # 负=偏早, 正=偏晚
    }

    # 4. 识别极端误差样本
    value_error_threshold = np.percentile(np.abs(error_flat), 90)
    time_error_threshold = 2  # 超过2小时视为严重

    extreme_value_mask = np.abs(peak_value_errors) > value_error_threshold
    extreme_time_mask = np.abs(peak_time_errors) > time_error_threshold

    analysis['extreme_errors'] = {
        'value_threshold': value_error_threshold,
        'value_extreme_count': extreme_value_mask.sum(),
        'value_extreme_ratio': extreme_value_mask.mean(),
        'time_threshold': time_error_threshold,
        'time_extreme_count': extreme_time_mask.sum(),
        'time_extreme_ratio': extreme_time_mask.mean()
    }

    # 5. 与日均功率的相关性（是否高功率日误差更大）
    daily_avg_power = y_true.reshape(batch_size, num_days, -1).mean(axis=2)  # (batch, days)
    from scipy.stats import pearsonr, spearmanr

    try:
        pearson_val, p_val = pearsonr(daily_avg_power.flatten(), np.abs(error_flat))
        spearman_val, sp_pval = spearmanr(daily_avg_power.flatten(), np.abs(error_flat))

        analysis['power_error_correlation'] = {
            'pearson_r': pearson_val,
            'pearson_pval': p_val,
            'spearman_r': spearman_val,
            'spearman_pval': sp_pval,
            'interpretation': 'positive = higher power days have larger errors'
        }
    except Exception:
        analysis['power_error_correlation'] = {'error': 'correlation calculation failed'}

    if verbose:
        print("\n" + "=" * 60)
        print("峰值误差模式分析报告")
        print("=" * 60)

        print("\n1. 按功率水平分组:")
        for level, stats in analysis['by_power_level'].items():
            print(f"   {level.upper()} (n={stats['count']}, 均值={stats['mean_peak']:.1f}kW):")
            print(f"     - 峰值RMSE: {stats['value_rmse']:.2f} kW")
            print(f"     - 时刻MAE:  {stats['time_mae']:.2f} h")
            print(f"     - ±1h准确率: {stats['within_1h']*100:.1f}%")

        print("\n2. 误差模式:")
        patterns = analysis['error_patterns']
        print(f"   - 低估比例: {patterns['underestimate_ratio']*100:.1f}%")
        print(f"   - 高估比例: {patterns['overestimate_ratio']*100:.1f}%")
        print(f"   - 预测偏早: {patterns['early_prediction_ratio']*100:.1f}%")
        print(f"   - 预测偏晚: {patterns['late_prediction_ratio']*100:.1f}%")
        print(f"   - 平均值偏差: {patterns['mean_value_bias']:.2f} kW")
        print(f"   - 平均时刻偏差: {patterns['mean_time_bias']:.2f} h")

        print("\n3. 极端误差:")
        extreme = analysis['extreme_errors']
        print(f"   - 峰值误差>P90({extreme['value_threshold']:.1f}kW): {extreme['value_extreme_count']} ({extreme['value_extreme_ratio']*100:.1f}%)")
        print(f"   - 时刻误差>2h: {extreme['time_extreme_count']} ({extreme['time_extreme_ratio']*100:.1f}%)")

        if 'pearson_r' in analysis.get('power_error_correlation', {}):
            corr = analysis['power_error_correlation']
            print(f"\n4. 功率-误差相关性:")
            print(f"   - Pearson r: {corr['pearson_r']:.3f} (p={corr['pearson_pval']:.3e})")
            print(f"   - Spearman r: {corr['spearman_r']:.3f} (p={corr['spearman_pval']:.3e})")

        print("=" * 60)

    return analysis


def compare_models_peak_performance(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    is_daytime: np.ndarray,
    time_indices: Optional[np.ndarray] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    对比多个模型的峰值预测性能

    Args:
        predictions: 各模型预测 {model_name: predictions}
        y_true: 真实值
        is_daytime: 白天标志
        time_indices: 时间索引
        output_path: 输出CSV路径

    Returns:
        对比结果DataFrame
    """
    extractor = PeakExtractor()

    results = []
    for model_name, y_pred in predictions.items():
        peak_info = extractor.extract_daily_peaks(y_true, y_pred, is_daytime, time_indices)

        # 整体RMSE/R²
        overall_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        overall_r2 = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

        results.append({
            'model': model_name,
            'overall_rmse': overall_rmse,
            'overall_r2': overall_r2,
            'peak_value_rmse': peak_info['peak_value_rmse'],
            'peak_value_mae': peak_info['peak_value_mae'],
            'peak_time_mae': peak_info['peak_time_mae'],
            'peak_time_within_1h': peak_info['peak_time_within_1h'],
            'mean_true_peak': np.mean(peak_info['true_peak_values']),
            'mean_pred_peak': np.mean(peak_info['pred_peak_values']),
            'peak_bias': np.mean(peak_info['pred_peak_values']) - np.mean(peak_info['true_peak_values'])
        })

    df = pd.DataFrame(results)
    df = df.sort_values('peak_value_rmse')

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"模型对比结果已保存到: {output_path}")

    return df


def get_peak_window_mask(
    y_true: np.ndarray,
    is_daytime: np.ndarray,
    window_size: int = 2,
    hours_per_day: int = 24
) -> np.ndarray:
    """
    获取峰值窗口mask（用于计算峰值窗口内的误差）

    Args:
        y_true: 真实序列
        is_daytime: 白天标志
        window_size: 窗口大小（峰值前后±N小时）
        hours_per_day: 每天小时数

    Returns:
        峰值窗口mask, shape同y_true
    """
    batch_size, seq_len = y_true.shape
    num_days = seq_len // hours_per_day

    extractor = PeakExtractor(hours_per_day)
    peak_info = extractor.extract_daily_peaks(y_true, y_true, is_daytime)
    true_peak_times = peak_info['true_peak_times']  # (batch, num_days)

    mask = np.zeros_like(y_true, dtype=bool)

    for day in range(num_days):
        start_idx = day * hours_per_day
        for i in range(batch_size):
            peak_hour = true_peak_times[i, day]
            window_start = max(0, peak_hour - window_size)
            window_end = min(hours_per_day, peak_hour + window_size + 1)

            for h in range(window_start, window_end):
                global_idx = start_idx + h
                if global_idx < seq_len:
                    mask[i, global_idx] = True

    return mask


# 使用示例
if __name__ == "__main__":
    # 创建模拟数据
    batch_size = 32
    seq_len = 168  # 7天

    # 模拟真实数据（带有白天峰值）
    y_true = np.random.rand(batch_size, seq_len) * 100
    for i in range(7):  # 7天
        # 在每天的12:00左右添加峰值
        peak_time = 12 + i * 24
        if peak_time < seq_len:
            y_true[:, peak_time] += 200

    # 模拟预测数据（峰值略有偏移）
    y_pred = y_true + np.random.randn(batch_size, seq_len) * 10

    # 创建白天标志
    time_indices = np.tile(np.arange(24), (batch_size, 7))[:, :seq_len]
    is_daytime = ((time_indices >= 6) & (time_indices <= 20)).astype(float)

    # 提取峰值
    extractor = PeakExtractor()
    peak_info = extractor.extract_daily_peaks(y_true, y_pred, is_daytime, time_indices)

    # 打印统计信息
    print_peak_statistics(peak_info, prefix="测试")

    # 保存到CSV
    save_peak_metrics_to_csv(peak_info, "test_peak_metrics.csv")

    print("\n每日峰值信息示例（第一个样本）:")
    print("真实峰值:", peak_info['true_peak_values'][0])
    print("预测峰值:", peak_info['pred_peak_values'][0])
    print("真实峰值时刻:", peak_info['true_peak_times'][0])
    print("预测峰值时刻:", peak_info['pred_peak_times'][0])
