"""
峰值感知损失函数
专门针对光伏发电预测中的日间尖峰优化设计

包含:
- PeakAwareLoss: 基础峰值感知损失
- WeightedPeakAwareLoss: 带时间步/样本加权的峰值感知损失
- SoftPeakAwareLoss: 使用软可微峰值提取的损失（梯度更平滑）
- CurriculumPeakLoss: 支持课程学习的峰值损失
- PeakShapeLoss: 峰值曲线形状损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from ..utils.peak_extractor import PeakExtractor


class PeakAwareLoss(nn.Module):
    """
    峰值感知损失函数

    数学形式：
    L_total = α * L_overall + β * L_peak_magnitude + γ * L_peak_timing

    其中：
    - L_overall: 整体曲线的MSE损失
    - L_peak_magnitude: 峰值大小的MSE/MAE损失
    - L_peak_timing: 峰值时刻误差的惩罚项
    - α, β, γ: 可调节的权重超参数
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 1.0,
        peak_magnitude_loss: str = 'mse',
        peak_timing_penalty: str = 'l1',
        timing_penalty_scale: float = 10.0,
        hours_per_day: int = 24
    ):
        """
        Args:
            alpha: 整体曲线损失的权重
            beta: 峰值大小损失的权重
            gamma: 峰值时刻损失的权重
            peak_magnitude_loss: 峰值大小损失类型，'mse' 或 'mae'
            peak_timing_penalty: 峰值时刻惩罚类型，'l1', 'l2', 或 'smooth_l1'
            timing_penalty_scale: 时刻惩罚的缩放因子（将小时差异放大）
            hours_per_day: 每天小时数
        """
        super(PeakAwareLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.peak_magnitude_loss = peak_magnitude_loss
        self.peak_timing_penalty = peak_timing_penalty
        self.timing_penalty_scale = timing_penalty_scale
        self.peak_extractor = PeakExtractor(hours_per_day=hours_per_day)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        is_daytime: torch.Tensor = None,
        time_indices: torch.Tensor = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        计算峰值感知损失

        Args:
            y_pred: 预测值, shape (batch_size, seq_len)
            y_true: 真实值, shape (batch_size, seq_len)
            is_daytime: 白天标志, shape (batch_size, seq_len)
            time_indices: 时间索引, shape (batch_size, seq_len)
            return_components: 是否返回各组成部分的损失

        Returns:
            总损失，如果return_components=True则返回字典
        """
        # 1. 整体曲线MSE损失
        L_overall = F.mse_loss(y_pred, y_true)

        # 2. 提取峰值信息
        peak_info = self.peak_extractor.extract_daily_peaks_torch(
            y_true, y_pred, is_daytime, time_indices
        )

        true_peaks = peak_info['true_peak_values']  # (batch_size, num_days)
        pred_peaks = peak_info['pred_peak_values']
        peak_time_errors = peak_info['peak_time_errors']  # (batch_size, num_days)

        # 3. 峰值大小损失
        if self.peak_magnitude_loss == 'mse':
            L_peak_magnitude = F.mse_loss(pred_peaks, true_peaks)
        elif self.peak_magnitude_loss == 'mae':
            L_peak_magnitude = F.l1_loss(pred_peaks, true_peaks)
        else:
            raise ValueError(f"Unknown peak_magnitude_loss: {self.peak_magnitude_loss}")

        # 4. 峰值时刻损失
        # 将时刻误差放大，使其在损失中更显著
        scaled_time_errors = peak_time_errors * self.timing_penalty_scale

        if self.peak_timing_penalty == 'l1':
            L_peak_timing = torch.mean(torch.abs(scaled_time_errors))
        elif self.peak_timing_penalty == 'l2':
            L_peak_timing = torch.mean(scaled_time_errors ** 2)
        elif self.peak_timing_penalty == 'smooth_l1':
            L_peak_timing = F.smooth_l1_loss(scaled_time_errors, torch.zeros_like(scaled_time_errors))
        else:
            raise ValueError(f"Unknown peak_timing_penalty: {self.peak_timing_penalty}")

        # 5. 总损失
        total_loss = self.alpha * L_overall + self.beta * L_peak_magnitude + self.gamma * L_peak_timing

        if return_components:
            return {
                'total_loss': total_loss,
                'overall_loss': L_overall,
                'peak_magnitude_loss': L_peak_magnitude,
                'peak_timing_loss': L_peak_timing,
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma
            }
        else:
            return total_loss


class WeightedPeakAwareLoss(nn.Module):
    """
    加权峰值感知损失函数

    在PeakAwareLoss基础上，对不同时间步和样本进行加权：
    - 白天时刻权重更高
    - 峰值附近时刻权重更高
    - 高功率样本权重更高
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 1.0,
        daytime_weight: float = 2.0,
        nighttime_weight: float = 0.5,
        peak_window_weight: float = 3.0,
        peak_window_size: int = 2,
        high_power_weight: float = 1.5,
        power_percentile: float = 0.75,
        hours_per_day: int = 24
    ):
        """
        Args:
            alpha, beta, gamma: 与PeakAwareLoss相同
            daytime_weight: 白天时刻的权重
            nighttime_weight: 夜晚时刻的权重
            peak_window_weight: 峰值窗口（±peak_window_size小时）的额外权重
            peak_window_size: 峰值窗口大小（小时）
            high_power_weight: 高功率样本的额外权重
            power_percentile: 高功率的分位数阈值
            hours_per_day: 每天小时数
        """
        super(WeightedPeakAwareLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.daytime_weight = daytime_weight
        self.nighttime_weight = nighttime_weight
        self.peak_window_weight = peak_window_weight
        self.peak_window_size = peak_window_size
        self.high_power_weight = high_power_weight
        self.power_percentile = power_percentile
        self.hours_per_day = hours_per_day
        self.peak_extractor = PeakExtractor(hours_per_day=hours_per_day)

    def compute_sample_weights(
        self,
        y_true: torch.Tensor,
        is_daytime: torch.Tensor,
        peak_times: torch.Tensor
    ) -> torch.Tensor:
        """
        计算每个时间步的权重

        Args:
            y_true: 真实值, shape (batch_size, seq_len)
            is_daytime: 白天标志, shape (batch_size, seq_len)
            peak_times: 每天的峰值时刻, shape (batch_size, num_days)

        Returns:
            权重矩阵, shape (batch_size, seq_len)
        """
        batch_size, seq_len = y_true.shape
        num_days = seq_len // self.hours_per_day
        device = y_true.device

        # 初始化权重：根据是否白天
        weights = torch.where(
            is_daytime > 0.5,
            torch.tensor(self.daytime_weight, device=device),
            torch.tensor(self.nighttime_weight, device=device)
        )

        # 对峰值附近窗口增加权重
        for day in range(num_days):
            start_idx = day * self.hours_per_day
            for i in range(batch_size):
                peak_hour = peak_times[i, day].item()
                # 峰值窗口范围
                window_start = max(0, peak_hour - self.peak_window_size)
                window_end = min(self.hours_per_day, peak_hour + self.peak_window_size + 1)

                # 在峰值窗口内增加权重
                for h in range(window_start, window_end):
                    global_idx = start_idx + h
                    if global_idx < seq_len:
                        weights[i, global_idx] *= self.peak_window_weight

        return weights

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        is_daytime: torch.Tensor = None,
        time_indices: torch.Tensor = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        计算加权峰值感知损失
        """
        batch_size, seq_len = y_true.shape
        device = y_true.device

        # 如果没有白天标志，创建默认值
        if is_daytime is None:
            if time_indices is None:
                num_days = seq_len // self.hours_per_day
                time_indices = torch.arange(self.hours_per_day, device=device).repeat(
                    batch_size, num_days
                )[:, :seq_len]
            is_daytime = ((time_indices >= 6) & (time_indices <= 20)).float()

        # 提取峰值信息（用于计算权重）
        peak_info = self.peak_extractor.extract_daily_peaks_torch(
            y_true, y_pred, is_daytime, time_indices
        )

        # 计算时间步权重
        step_weights = self.compute_sample_weights(
            y_true, is_daytime, peak_info['true_peak_times']
        )

        # 1. 加权整体曲线MSE损失
        squared_errors = (y_pred - y_true) ** 2
        L_overall = torch.sum(squared_errors * step_weights) / torch.sum(step_weights)

        # 2. 峰值大小损失（对高功率样本加权）
        true_peaks = peak_info['true_peak_values']
        pred_peaks = peak_info['pred_peak_values']

        # 计算每个样本的平均峰值
        sample_avg_peaks = torch.mean(true_peaks, dim=1)  # (batch_size,)
        power_threshold = torch.quantile(sample_avg_peaks, self.power_percentile)

        # 高功率样本标记
        high_power_mask = (sample_avg_peaks > power_threshold).float()  # (batch_size,)

        # 扩展到每天
        num_days = true_peaks.shape[1]
        sample_weights = torch.ones_like(true_peaks)  # (batch_size, num_days)
        for i in range(batch_size):
            if high_power_mask[i] > 0.5:
                sample_weights[i, :] *= self.high_power_weight

        # 加权峰值大小损失
        peak_squared_errors = (pred_peaks - true_peaks) ** 2
        L_peak_magnitude = torch.sum(peak_squared_errors * sample_weights) / torch.sum(sample_weights)

        # 3. 峰值时刻损失
        peak_time_errors = peak_info['peak_time_errors']
        L_peak_timing = torch.mean(torch.abs(peak_time_errors) * 10.0)  # 放大10倍

        # 4. 总损失
        total_loss = self.alpha * L_overall + self.beta * L_peak_magnitude + self.gamma * L_peak_timing

        if return_components:
            return {
                'total_loss': total_loss,
                'overall_loss': L_overall,
                'peak_magnitude_loss': L_peak_magnitude,
                'peak_timing_loss': L_peak_timing,
                'step_weights': step_weights,
                'sample_weights': sample_weights,
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma
            }
        else:
            return total_loss


class SoftPeakAwareLoss(nn.Module):
    """
    软可微峰值感知损失函数

    使用温度系数的softmax加权提取峰值位置，而非硬argmax，
    使得峰值时刻的梯度可以平滑回传。

    数学形式:
    - 软峰值时刻: soft_peak_time = Σ(t * softmax(y/τ))
    - 软峰值大小: soft_peak_value = Σ(y * softmax(y/τ))

    其中τ是温度系数，τ越小越接近硬argmax。
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 1.0,
        delta: float = 0.5,
        temperature: float = 0.1,
        hours_per_day: int = 24
    ):
        """
        Args:
            alpha: 整体曲线损失的权重
            beta: 峰值大小损失的权重
            gamma: 峰值时刻损失的权重
            delta: 峰值曲线形状损失的权重
            temperature: softmax温度系数（越小越尖锐）
            hours_per_day: 每天小时数
        """
        super(SoftPeakAwareLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.temperature = temperature
        self.hours_per_day = hours_per_day

    def _soft_peak_extraction(
        self,
        y: torch.Tensor,
        is_daytime: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用softmax进行软峰值提取

        Args:
            y: 功率序列, shape (batch_size, seq_len)
            is_daytime: 白天标志

        Returns:
            soft_peak_values: 软峰值大小, (batch_size, num_days)
            soft_peak_times: 软峰值时刻, (batch_size, num_days)
        """
        batch_size, seq_len = y.shape
        num_days = seq_len // self.hours_per_day
        device = y.device

        soft_peak_values = []
        soft_peak_times = []

        for day in range(num_days):
            start_idx = day * self.hours_per_day
            end_idx = start_idx + self.hours_per_day

            # 当日数据
            y_day = y[:, start_idx:end_idx]  # (batch, 24)
            is_daytime_day = is_daytime[:, start_idx:end_idx]

            # 使用白天mask：将夜间值设为极小值，不参与softmax
            masked_y = torch.where(
                is_daytime_day > 0.5,
                y_day,
                torch.tensor(-1e9, device=device)
            )

            # 温度缩放的softmax
            attention = F.softmax(masked_y / self.temperature, dim=1)  # (batch, 24)

            # 软峰值大小：加权平均
            soft_peak_value = (attention * y_day).sum(dim=1)  # (batch,)
            soft_peak_values.append(soft_peak_value)

            # 软峰值时刻：加权位置
            time_indices = torch.arange(self.hours_per_day, device=device).float()
            soft_peak_time = (attention * time_indices.unsqueeze(0)).sum(dim=1)  # (batch,)
            soft_peak_times.append(soft_peak_time)

        soft_peak_values = torch.stack(soft_peak_values, dim=1)  # (batch, num_days)
        soft_peak_times = torch.stack(soft_peak_times, dim=1)    # (batch, num_days)

        return soft_peak_values, soft_peak_times

    def _peak_shape_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        is_daytime: torch.Tensor,
        window_size: int = 2
    ) -> torch.Tensor:
        """
        计算峰值附近曲线形状的损失

        对峰值前后±window_size小时的曲线形状进行额外惩罚
        """
        batch_size, seq_len = y_true.shape
        num_days = seq_len // self.hours_per_day
        device = y_true.device

        shape_loss = torch.tensor(0.0, device=device)
        count = 0

        for day in range(num_days):
            start_idx = day * self.hours_per_day
            end_idx = start_idx + self.hours_per_day

            y_true_day = y_true[:, start_idx:end_idx]
            y_pred_day = y_pred[:, start_idx:end_idx]
            is_daytime_day = is_daytime[:, start_idx:end_idx]

            # 找到每个样本的峰值位置
            masked_true = torch.where(
                is_daytime_day > 0.5,
                y_true_day,
                torch.tensor(-1e9, device=device)
            )
            peak_indices = torch.argmax(masked_true, dim=1)  # (batch,)

            # 对每个样本提取峰值窗口
            for i in range(batch_size):
                peak_idx = peak_indices[i].item()
                win_start = max(0, peak_idx - window_size)
                win_end = min(self.hours_per_day, peak_idx + window_size + 1)

                true_window = y_true_day[i, win_start:win_end]
                pred_window = y_pred_day[i, win_start:win_end]

                # 形状损失：归一化后的MSE
                true_norm = true_window / (true_window.max() + 1e-6)
                pred_norm = pred_window / (pred_window.max() + 1e-6)

                shape_loss += F.mse_loss(pred_norm, true_norm)
                count += 1

        return shape_loss / (count + 1e-6)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        is_daytime: torch.Tensor = None,
        time_indices: torch.Tensor = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        计算软可微峰值感知损失

        Args:
            y_pred: 预测值, shape (batch_size, seq_len)
            y_true: 真实值, shape (batch_size, seq_len)
            is_daytime: 白天标志
            time_indices: 时间索引
            return_components: 是否返回各组成部分

        Returns:
            总损失
        """
        batch_size, seq_len = y_true.shape
        device = y_true.device

        # 默认白天标志
        if is_daytime is None:
            if time_indices is None:
                num_days = seq_len // self.hours_per_day
                time_indices = torch.arange(self.hours_per_day, device=device).repeat(
                    batch_size, num_days
                )[:, :seq_len]
            is_daytime = ((time_indices >= 6) & (time_indices <= 20)).float()

        # 1. 整体曲线MSE损失
        L_overall = F.mse_loss(y_pred, y_true)

        # 2. 软峰值提取
        true_peak_values, true_peak_times = self._soft_peak_extraction(y_true, is_daytime)
        pred_peak_values, pred_peak_times = self._soft_peak_extraction(y_pred, is_daytime)

        # 3. 峰值大小损失
        L_peak_magnitude = F.mse_loss(pred_peak_values, true_peak_values)

        # 4. 峰值时刻损失（软可微）
        time_diff = pred_peak_times - true_peak_times
        L_peak_timing = torch.mean(time_diff ** 2)  # L2 loss on time difference

        # 5. 峰值曲线形状损失
        L_peak_shape = self._peak_shape_loss(y_pred, y_true, is_daytime)

        # 6. 总损失
        total_loss = (self.alpha * L_overall +
                      self.beta * L_peak_magnitude +
                      self.gamma * L_peak_timing +
                      self.delta * L_peak_shape)

        if return_components:
            return {
                'total_loss': total_loss,
                'overall_loss': L_overall,
                'peak_magnitude_loss': L_peak_magnitude,
                'peak_timing_loss': L_peak_timing,
                'peak_shape_loss': L_peak_shape,
                'true_peak_values': true_peak_values,
                'pred_peak_values': pred_peak_values,
                'true_peak_times': true_peak_times,
                'pred_peak_times': pred_peak_times
            }
        else:
            return total_loss


class CurriculumPeakLoss(nn.Module):
    """
    支持课程学习的峰值损失函数

    训练初期关注整体曲线拟合，后期逐步增加峰值相关损失的权重。
    这有助于模型先学习基本模式，再精调峰值。

    权重调度策略:
    - linear: 线性增长
    - cosine: 余弦退火风格增长
    - step: 阶梯式增长
    """

    def __init__(
        self,
        alpha_start: float = 1.0,
        alpha_end: float = 0.8,
        beta_start: float = 0.5,
        beta_end: float = 3.0,
        gamma_start: float = 0.2,
        gamma_end: float = 1.5,
        schedule: str = 'cosine',
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        hours_per_day: int = 24
    ):
        """
        Args:
            alpha_start/end: 整体损失权重的起始/结束值
            beta_start/end: 峰值大小损失权重的起始/结束值
            gamma_start/end: 峰值时刻损失权重的起始/结束值
            schedule: 权重调度策略 ('linear', 'cosine', 'step')
            warmup_epochs: 预热期（权重保持初始值）
            total_epochs: 总训练轮数
            hours_per_day: 每天小时数
        """
        super(CurriculumPeakLoss, self).__init__()

        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.hours_per_day = hours_per_day

        self.peak_extractor = PeakExtractor(hours_per_day)

        # 当前epoch（需要外部设置）
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch

    def _get_schedule_factor(self) -> float:
        """获取当前的调度因子（0到1之间）"""
        if self.current_epoch < self.warmup_epochs:
            return 0.0

        progress = (self.current_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        progress = min(1.0, max(0.0, progress))

        if self.schedule == 'linear':
            return progress
        elif self.schedule == 'cosine':
            return 0.5 * (1 - torch.cos(torch.tensor(progress * 3.14159)).item())
        elif self.schedule == 'step':
            # 4个阶段
            if progress < 0.25:
                return 0.25
            elif progress < 0.5:
                return 0.5
            elif progress < 0.75:
                return 0.75
            else:
                return 1.0
        else:
            return progress

    def _get_current_weights(self) -> Tuple[float, float, float]:
        """获取当前的权重值"""
        factor = self._get_schedule_factor()

        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * factor
        beta = self.beta_start + (self.beta_end - self.beta_start) * factor
        gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * factor

        return alpha, beta, gamma

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        is_daytime: torch.Tensor = None,
        time_indices: torch.Tensor = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        计算课程学习峰值损失
        """
        batch_size, seq_len = y_true.shape
        device = y_true.device

        # 获取当前权重
        alpha, beta, gamma = self._get_current_weights()

        # 默认白天标志
        if is_daytime is None:
            if time_indices is None:
                num_days = seq_len // self.hours_per_day
                time_indices = torch.arange(self.hours_per_day, device=device).repeat(
                    batch_size, num_days
                )[:, :seq_len]
            is_daytime = ((time_indices >= 6) & (time_indices <= 20)).float()

        # 1. 整体曲线MSE损失
        L_overall = F.mse_loss(y_pred, y_true)

        # 2. 提取峰值信息
        peak_info = self.peak_extractor.extract_daily_peaks_torch(
            y_true, y_pred, is_daytime, time_indices
        )

        true_peaks = peak_info['true_peak_values']
        pred_peaks = peak_info['pred_peak_values']
        peak_time_errors = peak_info['peak_time_errors']

        # 3. 峰值大小损失
        L_peak_magnitude = F.mse_loss(pred_peaks, true_peaks)

        # 4. 峰值时刻损失
        L_peak_timing = torch.mean(torch.abs(peak_time_errors) * 10.0)

        # 5. 总损失
        total_loss = alpha * L_overall + beta * L_peak_magnitude + gamma * L_peak_timing

        if return_components:
            return {
                'total_loss': total_loss,
                'overall_loss': L_overall,
                'peak_magnitude_loss': L_peak_magnitude,
                'peak_timing_loss': L_peak_timing,
                'current_alpha': alpha,
                'current_beta': beta,
                'current_gamma': gamma,
                'schedule_factor': self._get_schedule_factor()
            }
        else:
            return total_loss


class CombinedPeakLoss(nn.Module):
    """
    组合峰值损失函数

    结合多种峰值损失策略，提供统一的接口

    数学形式:
    L_total = α * L_overall + β * L_peak_value + γ * L_peak_time + δ * L_peak_shape + ε * L_daytime

    其中:
    - L_overall: 整体MSE
    - L_peak_value: 峰值大小MSE
    - L_peak_time: 峰值时刻误差
    - L_peak_shape: 峰值附近曲线形状误差
    - L_daytime: 白天时段额外加权MSE
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 1.0,
        delta: float = 0.5,
        epsilon: float = 0.5,
        peak_window_size: int = 2,
        daytime_weight: float = 2.0,
        use_soft_peak: bool = True,
        temperature: float = 0.1,
        hours_per_day: int = 24
    ):
        """
        Args:
            alpha: 整体损失权重
            beta: 峰值大小损失权重
            gamma: 峰值时刻损失权重
            delta: 峰值形状损失权重
            epsilon: 白天加权损失权重
            peak_window_size: 峰值窗口大小
            daytime_weight: 白天时段权重
            use_soft_peak: 是否使用软可微峰值提取
            temperature: softmax温度（仅use_soft_peak=True时有效）
            hours_per_day: 每天小时数
        """
        super(CombinedPeakLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.peak_window_size = peak_window_size
        self.daytime_weight = daytime_weight
        self.use_soft_peak = use_soft_peak
        self.temperature = temperature
        self.hours_per_day = hours_per_day

        self.peak_extractor = PeakExtractor(hours_per_day)

    def _soft_peak_extraction(
        self,
        y: torch.Tensor,
        is_daytime: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """软峰值提取"""
        batch_size, seq_len = y.shape
        num_days = seq_len // self.hours_per_day
        device = y.device

        soft_peak_values = []
        soft_peak_times = []

        for day in range(num_days):
            start_idx = day * self.hours_per_day
            end_idx = start_idx + self.hours_per_day

            y_day = y[:, start_idx:end_idx]
            is_daytime_day = is_daytime[:, start_idx:end_idx]

            masked_y = torch.where(
                is_daytime_day > 0.5,
                y_day,
                torch.tensor(-1e9, device=device)
            )

            attention = F.softmax(masked_y / self.temperature, dim=1)

            soft_peak_value = (attention * y_day).sum(dim=1)
            soft_peak_values.append(soft_peak_value)

            time_indices = torch.arange(self.hours_per_day, device=device).float()
            soft_peak_time = (attention * time_indices.unsqueeze(0)).sum(dim=1)
            soft_peak_times.append(soft_peak_time)

        return torch.stack(soft_peak_values, dim=1), torch.stack(soft_peak_times, dim=1)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        is_daytime: torch.Tensor = None,
        time_indices: torch.Tensor = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """计算组合峰值损失"""
        batch_size, seq_len = y_true.shape
        device = y_true.device
        num_days = seq_len // self.hours_per_day

        # 默认白天标志
        if is_daytime is None:
            if time_indices is None:
                time_indices = torch.arange(self.hours_per_day, device=device).repeat(
                    batch_size, num_days
                )[:, :seq_len]
            is_daytime = ((time_indices >= 6) & (time_indices <= 20)).float()

        # 1. 整体MSE损失
        L_overall = F.mse_loss(y_pred, y_true)

        # 2. 峰值损失
        if self.use_soft_peak:
            true_peak_values, true_peak_times = self._soft_peak_extraction(y_true, is_daytime)
            pred_peak_values, pred_peak_times = self._soft_peak_extraction(y_pred, is_daytime)
        else:
            peak_info = self.peak_extractor.extract_daily_peaks_torch(
                y_true, y_pred, is_daytime, time_indices
            )
            true_peak_values = peak_info['true_peak_values']
            pred_peak_values = peak_info['pred_peak_values']
            true_peak_times = peak_info['true_peak_times'].float()
            pred_peak_times = peak_info['pred_peak_times'].float()

        L_peak_value = F.mse_loss(pred_peak_values, true_peak_values)
        L_peak_time = torch.mean((pred_peak_times - true_peak_times) ** 2)

        # 3. 峰值形状损失
        L_peak_shape = torch.tensor(0.0, device=device)
        count = 0

        for day in range(num_days):
            start_idx = day * self.hours_per_day
            end_idx = start_idx + self.hours_per_day

            y_true_day = y_true[:, start_idx:end_idx]
            y_pred_day = y_pred[:, start_idx:end_idx]
            is_daytime_day = is_daytime[:, start_idx:end_idx]

            masked_true = torch.where(
                is_daytime_day > 0.5,
                y_true_day,
                torch.tensor(-1e9, device=device)
            )
            peak_indices = torch.argmax(masked_true, dim=1)

            for i in range(batch_size):
                peak_idx = peak_indices[i].item()
                win_start = max(0, int(peak_idx - self.peak_window_size))
                win_end = min(self.hours_per_day, int(peak_idx + self.peak_window_size + 1))

                true_window = y_true_day[i, win_start:win_end]
                pred_window = y_pred_day[i, win_start:win_end]

                if true_window.max() > 1e-6:
                    true_norm = true_window / true_window.max()
                    pred_norm = pred_window / (pred_window.max() + 1e-6)
                    L_peak_shape += F.mse_loss(pred_norm, true_norm)
                    count += 1

        L_peak_shape = L_peak_shape / (count + 1e-6)

        # 4. 白天加权MSE损失
        daytime_weights = torch.where(
            is_daytime > 0.5,
            torch.tensor(self.daytime_weight, device=device),
            torch.tensor(1.0, device=device)
        )
        weighted_mse = ((y_pred - y_true) ** 2 * daytime_weights).mean()
        L_daytime = weighted_mse - L_overall  # 额外部分

        # 5. 总损失
        total_loss = (self.alpha * L_overall +
                      self.beta * L_peak_value +
                      self.gamma * L_peak_time +
                      self.delta * L_peak_shape +
                      self.epsilon * L_daytime)

        if return_components:
            return {
                'total_loss': total_loss,
                'overall_loss': L_overall,
                'peak_value_loss': L_peak_value,
                'peak_time_loss': L_peak_time,
                'peak_shape_loss': L_peak_shape,
                'daytime_loss': L_daytime,
                'true_peak_values': true_peak_values,
                'pred_peak_values': pred_peak_values
            }
        else:
            return total_loss


# ==================== 工具函数 ====================

def create_peak_loss(
    loss_type: str = 'combined',
    **kwargs
) -> nn.Module:
    """
    创建峰值感知损失函数的工厂函数

    Args:
        loss_type: 损失类型
            - 'basic': PeakAwareLoss
            - 'weighted': WeightedPeakAwareLoss
            - 'soft': SoftPeakAwareLoss
            - 'curriculum': CurriculumPeakLoss
            - 'combined': CombinedPeakLoss (推荐)
        **kwargs: 传递给具体损失类的参数

    Returns:
        损失函数实例

    Example:
        >>> loss_fn = create_peak_loss('combined', alpha=1.0, beta=2.0)
        >>> loss = loss_fn(y_pred, y_true, is_daytime)
    """
    loss_classes = {
        'basic': PeakAwareLoss,
        'weighted': WeightedPeakAwareLoss,
        'soft': SoftPeakAwareLoss,
        'curriculum': CurriculumPeakLoss,
        'combined': CombinedPeakLoss
    }

    if loss_type not in loss_classes:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_classes.keys())}")

    return loss_classes[loss_type](**kwargs)
