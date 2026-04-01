"""
训练数据重加权与重采样策略
用于提升光伏发电预测中的峰值拟合效果

包含:
- PowerBasedWeightedSampler: 基于功率的加权采样器
- DynamicLossWeighter: 动态损失加权器
- WeightedTrainingStep: 加权训练步骤
- CurriculumWeightScheduler: 课程学习权重调度器
- AdaptivePeakWeighter: 自适应峰值窗口加权器
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
from typing import List, Optional, Tuple, Dict


class PowerBasedWeightedSampler(Sampler):
    """
    基于功率的加权采样器

    对高功率、日照强的样本进行过采样，使模型更关注这些样本
    同时保证夜间样本不被完全忽略
    """

    def __init__(
        self,
        power_values: np.ndarray,
        num_samples: int,
        power_percentile: float = 0.75,
        high_power_weight: float = 3.0,
        low_power_weight: float = 1.0,
        replacement: bool = True
    ):
        """
        Args:
            power_values: 每个样本的平均功率或峰值功率, shape (num_samples,)
            num_samples: 每个epoch采样的样本数
            power_percentile: 高功率的分位数阈值
            high_power_weight: 高功率样本的权重
            low_power_weight: 低功率样本的权重
            replacement: 是否有放回采样
        """
        self.power_values = power_values
        self.num_samples = num_samples
        self.replacement = replacement

        # 计算功率阈值
        power_threshold = np.percentile(power_values, power_percentile * 100)

        # 计算每个样本的权重
        self.weights = np.where(
            power_values >= power_threshold,
            high_power_weight,
            low_power_weight
        )

        # 归一化权重
        self.weights = self.weights / self.weights.sum()

    def __iter__(self):
        # 根据权重进行采样
        indices = np.random.choice(
            len(self.power_values),
            size=self.num_samples,
            replace=self.replacement,
            p=self.weights
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


class DynamicLossWeighter:
    """
    动态损失加权器

    在训练过程中为每个样本和时间步计算动态权重
    """

    def __init__(
        self,
        daytime_weight: float = 2.0,
        nighttime_weight: float = 0.5,
        high_power_weight: float = 1.5,
        power_percentile: float = 0.75,
        peak_window_weight: float = 3.0,
        peak_window_size: int = 2,
        use_temporal_weight: bool = True,
        use_sample_weight: bool = True
    ):
        """
        Args:
            daytime_weight: 白天时刻的基础权重
            nighttime_weight: 夜晚时刻的基础权重
            high_power_weight: 高功率样本的权重倍数
            power_percentile: 高功率的分位数阈值
            peak_window_weight: 峰值窗口的额外权重倍数
            peak_window_size: 峰值前后±N小时
            use_temporal_weight: 是否使用时间步权重（白天/夜晚/峰值窗口）
            use_sample_weight: 是否使用样本权重（高功率/低功率）
        """
        self.daytime_weight = daytime_weight
        self.nighttime_weight = nighttime_weight
        self.high_power_weight = high_power_weight
        self.power_percentile = power_percentile
        self.peak_window_weight = peak_window_weight
        self.peak_window_size = peak_window_size
        self.use_temporal_weight = use_temporal_weight
        self.use_sample_weight = use_sample_weight

    def compute_temporal_weights(
        self,
        is_daytime: torch.Tensor,
        peak_times: Optional[torch.Tensor] = None,
        hours_per_day: int = 24
    ) -> torch.Tensor:
        """
        计算时间步权重

        Args:
            is_daytime: 白天标志, shape (batch_size, seq_len)
            peak_times: 每天的峰值时刻, shape (batch_size, num_days)，可选
            hours_per_day: 每天小时数

        Returns:
            时间步权重, shape (batch_size, seq_len)
        """
        if not self.use_temporal_weight:
            return torch.ones_like(is_daytime)

        batch_size, seq_len = is_daytime.shape
        device = is_daytime.device

        # 基础权重：白天 vs 夜晚
        weights = torch.where(
            is_daytime > 0.5,
            torch.tensor(self.daytime_weight, device=device),
            torch.tensor(self.nighttime_weight, device=device)
        )

        # 如果提供了峰值时刻，对峰值窗口加权
        if peak_times is not None:
            num_days = seq_len // hours_per_day
            for day in range(num_days):
                start_idx = day * hours_per_day
                for i in range(batch_size):
                    peak_hour = peak_times[i, day].item()
                    # 峰值窗口范围
                    window_start = max(0, peak_hour - self.peak_window_size)
                    window_end = min(hours_per_day, peak_hour + self.peak_window_size + 1)

                    # 在峰值窗口内增加权重
                    for h in range(window_start, window_end):
                        global_idx = start_idx + h
                        if global_idx < seq_len:
                            weights[i, global_idx] *= self.peak_window_weight

        return weights

    def compute_sample_weights(
        self,
        power_values: torch.Tensor
    ) -> torch.Tensor:
        """
        计算样本权重（基于功率大小）

        Args:
            power_values: 每个样本的平均或峰值功率, shape (batch_size,)

        Returns:
            样本权重, shape (batch_size,)
        """
        if not self.use_sample_weight:
            return torch.ones_like(power_values)

        # 计算功率阈值
        power_threshold = torch.quantile(power_values, self.power_percentile)

        # 高功率样本权重更高
        weights = torch.where(
            power_values >= power_threshold,
            torch.tensor(self.high_power_weight, device=power_values.device),
            torch.tensor(1.0, device=power_values.device)
        )

        return weights

    def apply_weights_to_loss(
        self,
        loss: torch.Tensor,
        temporal_weights: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        将权重应用到损失上

        Args:
            loss: 原始损失, shape (batch_size, seq_len) 或 (batch_size,)
            temporal_weights: 时间步权重, shape (batch_size, seq_len)
            sample_weights: 样本权重, shape (batch_size,)，可选

        Returns:
            加权后的损失 (标量)
        """
        if loss.dim() == 2:  # (batch_size, seq_len)
            # 应用时间步权重
            weighted_loss = loss * temporal_weights

            # 应用样本权重
            if sample_weights is not None:
                weighted_loss = weighted_loss * sample_weights.unsqueeze(1)

            # 归一化
            total_weight = temporal_weights.sum()
            if sample_weights is not None:
                total_weight = total_weight * sample_weights.mean()

            return weighted_loss.sum() / total_weight

        elif loss.dim() == 1:  # (batch_size,)
            # 只应用样本权重
            if sample_weights is not None:
                weighted_loss = loss * sample_weights
                return weighted_loss.sum() / sample_weights.sum()
            else:
                return loss.mean()

        else:
            raise ValueError(f"Unsupported loss shape: {loss.shape}")


class WeightedTrainingStep:
    """
    加权训练步骤的完整实现

    封装了在训练循环中如何使用动态加权
    """

    def __init__(
        self,
        loss_weighter: DynamicLossWeighter,
        criterion: torch.nn.Module,
        use_peak_windows: bool = True
    ):
        """
        Args:
            loss_weighter: 损失加权器
            criterion: 基础损失函数（如MSE）
            use_peak_windows: 是否使用峰值窗口加权
        """
        self.loss_weighter = loss_weighter
        self.criterion = criterion
        self.use_peak_windows = use_peak_windows

    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        is_daytime: torch.Tensor,
        peak_times: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> torch.Tensor:
        """
        计算加权损失

        Args:
            y_pred: 预测值, shape (batch_size, seq_len)
            y_true: 真实值, shape (batch_size, seq_len)
            is_daytime: 白天标志, shape (batch_size, seq_len)
            peak_times: 峰值时刻, shape (batch_size, num_days)，可选
            return_details: 是否返回详细信息

        Returns:
            加权损失，如果return_details=True则返回字典
        """
        # 计算基础损失（每个时间步）
        pointwise_loss = (y_pred - y_true) ** 2  # (batch_size, seq_len)

        # 计算时间步权重
        temporal_weights = self.loss_weighter.compute_temporal_weights(
            is_daytime,
            peak_times if self.use_peak_windows else None
        )

        # 计算样本权重（基于平均功率）
        avg_power = y_true.mean(dim=1)  # (batch_size,)
        sample_weights = self.loss_weighter.compute_sample_weights(avg_power)

        # 应用权重
        weighted_loss = self.loss_weighter.apply_weights_to_loss(
            pointwise_loss,
            temporal_weights,
            sample_weights
        )

        if return_details:
            return {
                'loss': weighted_loss,
                'pointwise_loss': pointwise_loss,
                'temporal_weights': temporal_weights,
                'sample_weights': sample_weights,
                'avg_power': avg_power
            }
        else:
            return weighted_loss


# ==================== 使用示例 ====================

"""
=== 方案1: 使用加权采样器（DataLoader级别）===

# 准备数据
train_dataset = YourDataset(...)
power_values = np.array([sample['target'].mean() for sample in train_dataset])

# 创建加权采样器
weighted_sampler = PowerBasedWeightedSampler(
    power_values=power_values,
    num_samples=len(train_dataset),  # 每个epoch的样本数
    power_percentile=0.75,  # 前25%视为高功率
    high_power_weight=3.0,  # 高功率样本权重3倍
    low_power_weight=1.0,   # 低功率样本权重1倍
    replacement=True        # 有放回采样
)

# 创建DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=weighted_sampler,  # 使用加权采样器
    num_workers=4
)

# 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets, is_daytime = batch
        outputs = model(inputs)

        # 使用普通损失函数即可，因为采样已经偏向高功率
        loss = F.mse_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


=== 方案2: 使用动态损失加权（训练步骤级别）===

# 创建损失加权器
loss_weighter = DynamicLossWeighter(
    daytime_weight=2.0,       # 白天权重2倍
    nighttime_weight=0.5,     # 夜晚权重0.5倍
    high_power_weight=1.5,    # 高功率样本权重1.5倍
    power_percentile=0.75,    # 前25%视为高功率
    peak_window_weight=3.0,   # 峰值窗口权重3倍
    peak_window_size=2,       # 峰值前后±2小时
    use_temporal_weight=True,
    use_sample_weight=True
)

# 创建加权训练步骤
weighted_step = WeightedTrainingStep(
    loss_weighter=loss_weighter,
    criterion=nn.MSELoss(reduction='none'),
    use_peak_windows=True
)

# 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets, is_daytime, peak_times = batch
        outputs = model(inputs)

        # 计算加权损失
        loss = weighted_step.compute_loss(
            y_pred=outputs,
            y_true=targets,
            is_daytime=is_daytime,
            peak_times=peak_times  # 可选，如果需要峰值窗口加权
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


=== 方案3: 结合峰值感知损失和动态加权 ===

from peak_aware_loss import WeightedPeakAwareLoss

# 使用已经内置加权机制的峰值感知损失
criterion = WeightedPeakAwareLoss(
    alpha=1.0,
    beta=2.0,
    gamma=1.0,
    daytime_weight=2.0,
    nighttime_weight=0.5,
    peak_window_weight=3.0,
    peak_window_size=2,
    high_power_weight=1.5,
    power_percentile=0.75
)

# 训练循环（最简单）
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets, is_daytime, time_indices = batch
        outputs = model(inputs)

        # 直接计算损失，已包含所有加权逻辑
        loss = criterion(
            y_pred=outputs,
            y_true=targets,
            is_daytime=is_daytime,
            time_indices=time_indices
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


=== 推荐策略 ===

1. 如果使用标准MSE损失，推荐方案1（加权采样）+ 方案2（动态加权）
2. 如果使用峰值感知损失，推荐方案3（已内置加权）
3. 如果峰值问题非常严重，可以方案1+方案3组合使用

=== 超参数调节建议 ===

daytime_weight / nighttime_weight:
- 保守: 1.5 / 0.7 （白天稍微重要）
- 平衡: 2.0 / 0.5 （推荐起点）
- 激进: 3.0 / 0.3 （强烈关注白天，但可能夜间失真）

high_power_weight:
- 保守: 1.2 （轻微偏向高功率）
- 平衡: 1.5 （推荐）
- 激进: 2.0-3.0 （强烈偏向高功率）

peak_window_weight:
- 保守: 1.5 （轻微强调峰值窗口）
- 平衡: 2.0-3.0 （推荐）
- 激进: 5.0+ （极度关注峰值，可能导致其他区域拟合差）

power_percentile:
- 0.5: 上半部分都是高功率
- 0.75: 前25%是高功率（推荐）
- 0.9: 只有前10%是高功率（更极端）

=== 注意事项 ===

1. 加权不是越大越好
   - 过度加权会导致模型过拟合到高权重样本
   - 夜间预测可能完全失真
   - 建议从保守参数开始，逐步调整

2. 监控整体指标
   - 在validation set上监控整体RMSE、R²
   - 如果整体指标下降>10%，说明加权过度
   - 适当降低权重对比度

3. 结合early stopping
   - 使用validation set的综合指标（如 RMSE + peak_RMSE）
   - 防止在训练集上过拟合高权重样本

4. 权重warm-up
   - 训练初期使用较小的权重对比
   - 随着训练进行逐步增大权重对比
   - 有助于稳定训练

示例: 权重warm-up
def get_weight_schedule(epoch, max_epochs, start_ratio=0.5):
    # epoch 0: 权重对比度为start_ratio
    # 最后: 权重对比度为1.0
    ratio = start_ratio + (1.0 - start_ratio) * (epoch / max_epochs)
    return ratio

for epoch in range(num_epochs):
    schedule = get_weight_schedule(epoch, num_epochs)
    loss_weighter.daytime_weight = 1.0 + (2.0 - 1.0) * schedule
    loss_weighter.nighttime_weight = 1.0 - (1.0 - 0.5) * schedule
    # ... 训练 ...
"""


if __name__ == "__main__":
    # 测试代码
    print("=== 测试加权采样器 ===")
    power_values = np.random.rand(1000) * 100
    sampler = PowerBasedWeightedSampler(
        power_values=power_values,
        num_samples=1000,
        power_percentile=0.75,
        high_power_weight=3.0
    )
    print(f"权重分布: min={sampler.weights.min():.4f}, max={sampler.weights.max():.4f}")
    print(f"高功率样本数: {(power_values >= np.percentile(power_values, 75)).sum()}")

    print("\n=== 测试动态损失加权 ===")
    batch_size = 16
    seq_len = 168
    y_true = torch.randn(batch_size, seq_len) * 100 + 200
    y_pred = y_true + torch.randn(batch_size, seq_len) * 20
    is_daytime = torch.randint(0, 2, (batch_size, seq_len)).float()
    peak_times = torch.randint(10, 14, (batch_size, 7))  # 7天，峰值在10-14点

    weighter = DynamicLossWeighter()
    temporal_weights = weighter.compute_temporal_weights(is_daytime, peak_times)
    print(f"时间步权重: min={temporal_weights.min():.4f}, max={temporal_weights.max():.4f}")

    avg_power = y_true.mean(dim=1)
    sample_weights = weighter.compute_sample_weights(avg_power)
    print(f"样本权重: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}")

    print("\n=== 测试加权训练步骤 ===")
    loss_weighter = DynamicLossWeighter()
    weighted_step = WeightedTrainingStep(loss_weighter, nn.MSELoss(reduction='none'))
    loss_details = weighted_step.compute_loss(
        y_pred, y_true, is_daytime, peak_times, return_details=True
    )
    print(f"加权损失: {loss_details['loss']:.4f}")
    print(f"原始损失: {loss_details['pointwise_loss'].mean():.4f}")

    print("\n测试通过！")


class CurriculumWeightScheduler:
    """
    课程学习权重调度器

    随着训练进行，动态调整各类权重：
    - 训练初期：关注整体拟合，低峰值权重
    - 训练中期：逐步增加峰值权重
    - 训练后期：高峰值权重，精调峰值

    支持多种调度策略：linear, cosine, step, exponential
    """

    def __init__(
        self,
        total_epochs: int,
        warmup_epochs: int = 5,
        schedule: str = 'cosine',
        # 白天/夜晚权重
        daytime_weight_start: float = 1.5,
        daytime_weight_end: float = 2.5,
        nighttime_weight_start: float = 0.8,
        nighttime_weight_end: float = 0.3,
        # 峰值窗口权重
        peak_window_weight_start: float = 1.5,
        peak_window_weight_end: float = 4.0,
        # 高功率样本权重
        high_power_weight_start: float = 1.2,
        high_power_weight_end: float = 2.0
    ):
        """
        Args:
            total_epochs: 总训练轮数
            warmup_epochs: 预热期（权重保持初始值）
            schedule: 调度策略 ('linear', 'cosine', 'step', 'exponential')
            *_start/*_end: 各权重的起始/结束值
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule

        self.daytime_weight_start = daytime_weight_start
        self.daytime_weight_end = daytime_weight_end
        self.nighttime_weight_start = nighttime_weight_start
        self.nighttime_weight_end = nighttime_weight_end
        self.peak_window_weight_start = peak_window_weight_start
        self.peak_window_weight_end = peak_window_weight_end
        self.high_power_weight_start = high_power_weight_start
        self.high_power_weight_end = high_power_weight_end

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch

    def _get_progress(self) -> float:
        """获取训练进度 (0到1)"""
        if self.current_epoch < self.warmup_epochs:
            return 0.0

        progress = (self.current_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        return min(1.0, max(0.0, progress))

    def _apply_schedule(self, progress: float) -> float:
        """根据调度策略转换进度"""
        if self.schedule == 'linear':
            return progress
        elif self.schedule == 'cosine':
            return 0.5 * (1 - np.cos(progress * np.pi))
        elif self.schedule == 'step':
            if progress < 0.25:
                return 0.25
            elif progress < 0.5:
                return 0.5
            elif progress < 0.75:
                return 0.75
            else:
                return 1.0
        elif self.schedule == 'exponential':
            return 1 - np.exp(-3 * progress)
        else:
            return progress

    def get_current_weights(self) -> Dict[str, float]:
        """获取当前的所有权重"""
        progress = self._get_progress()
        factor = self._apply_schedule(progress)

        return {
            'daytime_weight': self.daytime_weight_start + (self.daytime_weight_end - self.daytime_weight_start) * factor,
            'nighttime_weight': self.nighttime_weight_start + (self.nighttime_weight_end - self.nighttime_weight_start) * factor,
            'peak_window_weight': self.peak_window_weight_start + (self.peak_window_weight_end - self.peak_window_weight_start) * factor,
            'high_power_weight': self.high_power_weight_start + (self.high_power_weight_end - self.high_power_weight_start) * factor,
            'progress': progress,
            'factor': factor,
            'epoch': self.current_epoch
        }

    def update_weighter(self, weighter: 'DynamicLossWeighter'):
        """更新DynamicLossWeighter的权重"""
        weights = self.get_current_weights()
        weighter.daytime_weight = weights['daytime_weight']
        weighter.nighttime_weight = weights['nighttime_weight']
        weighter.peak_window_weight = weights['peak_window_weight']
        weighter.high_power_weight = weights['high_power_weight']

    def __str__(self):
        weights = self.get_current_weights()
        return (f"CurriculumWeightScheduler(epoch={self.current_epoch}/{self.total_epochs}, "
                f"progress={weights['progress']:.2f}, daytime={weights['daytime_weight']:.2f}, "
                f"peak_window={weights['peak_window_weight']:.2f})")


class AdaptivePeakWeighter:
    """
    自适应峰值窗口加权器

    根据真实峰值位置动态创建权重矩阵，在峰值附近应用更高的权重。
    支持根据功率强度自适应调整权重强度。
    """

    def __init__(
        self,
        base_weight: float = 1.0,
        peak_weight_multiplier: float = 3.0,
        window_size: int = 2,
        power_adaptive: bool = True,
        power_scale: float = 0.5,
        hours_per_day: int = 24
    ):
        """
        Args:
            base_weight: 基础权重（非峰值区域）
            peak_weight_multiplier: 峰值区域的权重倍数
            window_size: 峰值窗口大小（±N小时）
            power_adaptive: 是否根据功率强度自适应调整
            power_scale: 功率自适应的缩放因子
            hours_per_day: 每天小时数
        """
        self.base_weight = base_weight
        self.peak_weight_multiplier = peak_weight_multiplier
        self.window_size = window_size
        self.power_adaptive = power_adaptive
        self.power_scale = power_scale
        self.hours_per_day = hours_per_day

    def compute_weights(
        self,
        y_true: torch.Tensor,
        is_daytime: torch.Tensor
    ) -> torch.Tensor:
        """
        计算每个时间步的权重

        Args:
            y_true: 真实值序列, shape (batch_size, seq_len)
            is_daytime: 白天标志, shape (batch_size, seq_len)

        Returns:
            权重矩阵, shape (batch_size, seq_len)
        """
        batch_size, seq_len = y_true.shape
        num_days = seq_len // self.hours_per_day
        device = y_true.device

        # 初始化权重
        weights = torch.full((batch_size, seq_len), self.base_weight, device=device)

        for day in range(num_days):
            start_idx = day * self.hours_per_day
            end_idx = start_idx + self.hours_per_day

            y_day = y_true[:, start_idx:end_idx]
            is_daytime_day = is_daytime[:, start_idx:end_idx]

            # 找到白天的峰值位置
            masked_y = torch.where(
                is_daytime_day > 0.5,
                y_day,
                torch.tensor(-1e9, device=device)
            )
            peak_indices = torch.argmax(masked_y, dim=1)  # (batch,)

            # 对每个样本设置峰值窗口权重
            for i in range(batch_size):
                peak_idx = peak_indices[i].item()
                peak_value = y_day[i, int(peak_idx)].item()

                # 计算权重倍数
                if self.power_adaptive and peak_value > 0:
                    # 功率越高，权重越大
                    adaptive_factor = 1.0 + self.power_scale * (peak_value / (y_true[i].max().item() + 1e-6))
                else:
                    adaptive_factor = 1.0

                # 设置峰值窗口权重
                win_start = max(0, int(peak_idx - self.window_size))
                win_end = min(self.hours_per_day, int(peak_idx + self.window_size + 1))

                for h in range(win_start, win_end):
                    global_idx = start_idx + h
                    # 距离峰值越近，权重越高（高斯型）
                    distance = abs(h - peak_idx)
                    gaussian_weight = np.exp(-0.5 * (distance / max(1, self.window_size)) ** 2)
                    weights[i, global_idx] = (
                        self.base_weight +
                        (self.peak_weight_multiplier - self.base_weight) * gaussian_weight * adaptive_factor
                    )

        return weights

    def apply_to_loss(
        self,
        loss: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        将权重应用到点对点损失

        Args:
            loss: 点对点损失, shape (batch_size, seq_len)
            weights: 权重, shape (batch_size, seq_len)

        Returns:
            加权损失 (标量)
        """
        weighted_loss = loss * weights
        return weighted_loss.sum() / weights.sum()


class CompositeLossWeighter:
    """
    综合损失加权器

    组合多种加权策略，提供统一的接口
    """

    def __init__(
        self,
        use_temporal_weight: bool = True,
        use_sample_weight: bool = True,
        use_peak_adaptive: bool = True,
        use_curriculum: bool = True,
        # 时间权重参数
        daytime_weight: float = 2.0,
        nighttime_weight: float = 0.5,
        # 峰值权重参数
        peak_window_weight: float = 3.0,
        peak_window_size: int = 2,
        # 样本权重参数
        high_power_weight: float = 1.5,
        power_percentile: float = 0.75,
        # 课程学习参数
        total_epochs: int = 50,
        warmup_epochs: int = 5,
        schedule: str = 'cosine'
    ):
        """
        Args:
            use_*: 是否使用各类加权
            其他参数: 各加权策略的参数
        """
        self.use_temporal_weight = use_temporal_weight
        self.use_sample_weight = use_sample_weight
        self.use_peak_adaptive = use_peak_adaptive
        self.use_curriculum = use_curriculum

        # 创建各加权器
        self.dynamic_weighter = DynamicLossWeighter(
            daytime_weight=daytime_weight,
            nighttime_weight=nighttime_weight,
            high_power_weight=high_power_weight,
            power_percentile=power_percentile,
            peak_window_weight=peak_window_weight,
            peak_window_size=peak_window_size,
            use_temporal_weight=use_temporal_weight,
            use_sample_weight=use_sample_weight
        )

        if use_peak_adaptive:
            self.peak_weighter = AdaptivePeakWeighter(
                peak_weight_multiplier=peak_window_weight,
                window_size=peak_window_size
            )
        else:
            self.peak_weighter = None

        if use_curriculum:
            self.curriculum_scheduler = CurriculumWeightScheduler(
                total_epochs=total_epochs,
                warmup_epochs=warmup_epochs,
                schedule=schedule,
                daytime_weight_start=1.5,
                daytime_weight_end=daytime_weight,
                peak_window_weight_start=1.5,
                peak_window_weight_end=peak_window_weight
            )
        else:
            self.curriculum_scheduler = None

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch
        if self.curriculum_scheduler:
            self.curriculum_scheduler.set_epoch(epoch)
            self.curriculum_scheduler.update_weighter(self.dynamic_weighter)

    def compute_weighted_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        is_daytime: torch.Tensor,
        peak_times: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算综合加权损失

        Args:
            y_pred: 预测值, shape (batch_size, seq_len)
            y_true: 真实值, shape (batch_size, seq_len)
            is_daytime: 白天标志
            peak_times: 峰值时刻（可选）

        Returns:
            加权损失 (标量)
        """
        # 点对点损失
        pointwise_loss = (y_pred - y_true) ** 2

        # 计算时间步权重
        temporal_weights = self.dynamic_weighter.compute_temporal_weights(
            is_daytime, peak_times
        )

        # 计算样本权重
        avg_power = y_true.mean(dim=1)
        sample_weights = self.dynamic_weighter.compute_sample_weights(avg_power)

        # 如果使用峰值自适应加权，则叠加
        if self.peak_weighter:
            peak_weights = self.peak_weighter.compute_weights(y_true, is_daytime)
            temporal_weights = temporal_weights * peak_weights

        # 应用权重
        weighted_loss = self.dynamic_weighter.apply_weights_to_loss(
            pointwise_loss,
            temporal_weights,
            sample_weights
        )

        return weighted_loss

    def get_current_config(self) -> Dict:
        """获取当前配置"""
        config = {
            'epoch': self.current_epoch,
            'daytime_weight': self.dynamic_weighter.daytime_weight,
            'nighttime_weight': self.dynamic_weighter.nighttime_weight,
            'peak_window_weight': self.dynamic_weighter.peak_window_weight,
            'high_power_weight': self.dynamic_weighter.high_power_weight
        }
        if self.curriculum_scheduler:
            config['curriculum_progress'] = self.curriculum_scheduler._get_progress()
        return config
