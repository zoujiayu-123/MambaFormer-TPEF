# 峰值感知损失函数设计文档

## 概述

本文档介绍专门针对光伏发电预测中**日间尖峰优化**设计的损失函数系列。这些损失函数旨在解决标准MSE损失导致的峰值预测被"平滑化"的问题。

**文件位置**: `src/mtm_mlef/losses/peak_loss.py`

---

## 问题背景

### 标准MSE损失的局限性

在光伏发电功率预测任务中，使用标准MSE损失函数会导致：

1. **峰值低估**: 模型倾向于预测保守的均值，导致峰值被严重低估（实测Mamba模型低估24.7%~33.9%）
2. **峰值时刻偏移**: 预测的峰值时刻与真实峰值时刻存在偏差
3. **曲线形状失真**: 峰值附近的曲线形状被过度平滑

### 设计目标

- **峰值大小RMSE** 显著下降
- **峰值时刻误差** 控制在 ±1 小时内
- 在不明显恶化整体R²的前提下，优先提升峰值拟合度

---

## 损失函数架构

```
┌─────────────────────────────────────────────────────────────┐
│                    create_peak_loss()                        │
│                      工厂函数入口                             │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ PeakAwareLoss   │ │WeightedPeakLoss │ │ SoftPeakLoss    │
│   (基础版)       │ │   (加权版)       │ │  (软可微版)      │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    ┌─────────────────┐
                    │CurriculumLoss   │
                    │  (课程学习版)    │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ CombinedPeakLoss│
                    │   (组合版)       │
                    └─────────────────┘
```

---

## 损失函数详解

### 1. PeakAwareLoss（基础峰值感知损失）

**数学形式**:
```
L_total = α × L_overall + β × L_peak_magnitude + γ × L_peak_timing
```

**组件说明**:
| 组件 | 公式 | 作用 |
|------|------|------|
| L_overall | MSE(y_pred, y_true) | 整体曲线拟合 |
| L_peak_magnitude | MSE/MAE(pred_peak, true_peak) | 峰值大小准确性 |
| L_peak_timing | \|t_pred - t_true\| × scale | 峰值时刻惩罚 |

**参数**:
```python
PeakAwareLoss(
    alpha=1.0,              # 整体损失权重
    beta=2.0,               # 峰值大小权重
    gamma=1.0,              # 峰值时刻权重
    peak_magnitude_loss='mse',  # 'mse' 或 'mae'
    peak_timing_penalty='l1',   # 'l1', 'l2', 'smooth_l1'
    timing_penalty_scale=10.0,  # 时刻误差放大系数
    hours_per_day=24
)
```

**使用示例**:
```python
from src.mtm_mlef.losses import create_peak_loss

loss_fn = create_peak_loss('basic', alpha=1.0, beta=2.0, gamma=1.0)
loss = loss_fn(y_pred, y_true, is_daytime, time_indices)
```

---

### 2. WeightedPeakAwareLoss（加权峰值感知损失）

**特点**: 在基础版上增加时间步和样本加权

**加权策略**:
- **白天/夜间权重**: 白天时刻权重更高（默认2.0 vs 0.5）
- **峰值窗口权重**: 峰值前后±N小时权重额外提升（默认3.0倍）
- **高功率样本权重**: 高功率日的样本权重更高

**参数**:
```python
WeightedPeakAwareLoss(
    alpha=1.0, beta=2.0, gamma=1.0,
    daytime_weight=2.0,      # 白天权重
    nighttime_weight=0.5,    # 夜间权重
    peak_window_weight=3.0,  # 峰值窗口额外权重
    peak_window_size=2,      # 峰值窗口大小（±小时）
    high_power_weight=1.5,   # 高功率样本权重
    power_percentile=0.75    # 高功率阈值分位数
)
```

---

### 3. SoftPeakAwareLoss（软可微峰值感知损失）

**核心创新**: 使用温度缩放的softmax替代硬argmax，使峰值时刻的梯度可平滑回传

**数学形式**:
```
软峰值时刻: soft_peak_time = Σ(t × softmax(y/τ))
软峰值大小: soft_peak_value = Σ(y × softmax(y/τ))
```

其中τ是温度系数，τ越小越接近硬argmax。

**损失组件**:
```
L_total = α × L_overall + β × L_peak_value + γ × L_peak_time + δ × L_peak_shape
```

| 组件 | 说明 |
|------|------|
| L_overall | 整体MSE |
| L_peak_value | 软峰值大小MSE |
| L_peak_time | 软峰值时刻L2误差 |
| L_peak_shape | 峰值窗口内归一化形状MSE |

**参数**:
```python
SoftPeakAwareLoss(
    alpha=1.0,          # 整体损失权重
    beta=2.0,           # 峰值大小权重
    gamma=1.0,          # 峰值时刻权重
    delta=0.5,          # 形状损失权重
    temperature=0.1,    # softmax温度（越小越尖锐）
    hours_per_day=24
)
```

**注意事项**:
- temperature过低（<0.5）可能导致梯度不稳定
- 建议temperature范围: 0.5~2.0
- 形状损失（delta）可能导致数值不稳定，建议设为0或较小值

---

### 4. CurriculumPeakLoss（课程学习峰值损失）

**设计理念**: 训练初期关注整体拟合，后期逐步增加峰值权重

**权重调度策略**:
| 策略 | 说明 |
|------|------|
| linear | 线性增长 |
| cosine | 余弦退火风格增长（推荐） |
| step | 阶梯式增长（4阶段） |

**参数**:
```python
CurriculumPeakLoss(
    alpha_start=1.0, alpha_end=0.8,   # 整体损失权重范围
    beta_start=0.5, beta_end=3.0,     # 峰值大小权重范围
    gamma_start=0.2, gamma_end=1.5,   # 峰值时刻权重范围
    schedule='cosine',                 # 调度策略
    warmup_epochs=5,                   # 预热期
    total_epochs=50
)
```

**使用方式**:
```python
loss_fn = create_peak_loss('curriculum', schedule='cosine')

for epoch in range(total_epochs):
    loss_fn.set_epoch(epoch)  # 更新当前epoch
    for batch in dataloader:
        loss = loss_fn(y_pred, y_true, is_daytime)
```

---

### 5. CombinedPeakLoss（组合峰值损失）⭐推荐

**特点**: 结合所有策略的综合损失函数

**数学形式**:
```
L_total = α × L_overall + β × L_peak_value + γ × L_peak_time + δ × L_peak_shape + ε × L_daytime
```

| 组件 | 公式 | 作用 |
|------|------|------|
| L_overall | MSE(y_pred, y_true) | 整体曲线拟合 |
| L_peak_value | MSE(pred_peak, true_peak) | 峰值大小准确性 |
| L_peak_time | (t_pred - t_true)² | 峰值时刻准确性 |
| L_peak_shape | 归一化窗口MSE | 峰值曲线形状 |
| L_daytime | 白天加权MSE - 整体MSE | 白天时段强化 |

**参数**:
```python
CombinedPeakLoss(
    alpha=1.0,           # 整体损失权重
    beta=2.0,            # 峰值大小权重
    gamma=1.0,           # 峰值时刻权重
    delta=0.5,           # 形状损失权重
    epsilon=0.5,         # 白天加权权重
    peak_window_size=2,  # 峰值窗口大小
    daytime_weight=2.0,  # 白天时段权重
    use_soft_peak=True,  # 使用软可微峰值提取
    temperature=0.1      # softmax温度
)
```

---

## 工厂函数

```python
from src.mtm_mlef.losses import create_peak_loss

# 创建不同类型的损失函数
loss_basic = create_peak_loss('basic', alpha=1.0, beta=2.0)
loss_weighted = create_peak_loss('weighted', daytime_weight=2.0)
loss_soft = create_peak_loss('soft', temperature=1.0)
loss_curriculum = create_peak_loss('curriculum', schedule='cosine')
loss_combined = create_peak_loss('combined', alpha=1.0, beta=0.5)
```

---

## 实验结果与建议

### 实验结果

| 损失函数 | Mamba R² | 峰值偏差 | 训练稳定性 |
|----------|----------|----------|------------|
| MSE (基线) | 0.479 | -24.7% | ✓ 稳定 |
| CombinedPeakLoss (β=2.5) | 不稳定 | - | ✗ 梯度爆炸 |
| SoftPeakAwareLoss (β=0.3) | 0.492 | -33.9% | ✓ 稳定 |

### 关键发现

1. **峰值损失权重不宜过高**: β>1.0 容易导致训练不稳定
2. **温度系数需要调优**: temperature过低导致梯度爆炸，建议≥1.0
3. **形状损失有风险**: L_peak_shape中的除法可能产生数值不稳定

### 推荐配置

**保守配置（稳定优先）**:
```python
peak_loss_config = {
    'alpha': 1.0,       # 整体权重主导
    'beta': 0.3,        # 峰值大小权重（小）
    'gamma': 0.1,       # 峰值时刻权重（很小）
    'delta': 0.0,       # 禁用形状损失
    'temperature': 2.0, # 高温度，平滑softmax
}
```

**激进配置（峰值优先）**:
```python
peak_loss_config = {
    'alpha': 0.5,       # 降低整体权重
    'beta': 1.5,        # 提高峰值大小权重
    'gamma': 0.5,       # 提高时刻权重
    'delta': 0.0,       # 禁用形状损失
    'temperature': 0.5, # 较低温度
}
```

---

## 与trainers.py集成

`train_rnn_model()` 函数已支持峰值损失：

```python
from src.mtm_mlef.trainers import train_rnn_model

# 使用MSE损失（默认）
model = train_rnn_model(
    MambaSeq2Seq, "Mamba", n_features, input_len, output_len,
    config, train_loader, val_loader, device, save_dir,
    loss_type='mse'
)

# 使用峰值损失
model = train_rnn_model(
    MambaSeq2Seq, "Mamba", n_features, input_len, output_len,
    config, train_loader, val_loader, device, save_dir,
    loss_type='soft',  # 'peak' 或 'soft'
    loss_config={
        'alpha': 1.0,
        'beta': 0.3,
        'gamma': 0.1,
        'temperature': 2.0,
    }
)
```

**支持的loss_type**:
- `'mse'`: 标准MSE损失
- `'peak'`: CombinedPeakLoss
- `'soft'`: SoftPeakAwareLoss

---

## 总结

| 损失函数 | 适用场景 | 复杂度 | 推荐指数 |
|----------|----------|--------|----------|
| PeakAwareLoss | 简单场景，快速实验 | 低 | ⭐⭐ |
| WeightedPeakAwareLoss | 需要样本/时间步加权 | 中 | ⭐⭐⭐ |
| SoftPeakAwareLoss | 需要平滑梯度 | 中 | ⭐⭐⭐ |
| CurriculumPeakLoss | 长期训练，渐进优化 | 中 | ⭐⭐⭐ |
| CombinedPeakLoss | 综合优化，功能全面 | 高 | ⭐⭐⭐⭐ |

**最终建议**:
- 对于Mamba等序列模型，峰值损失函数训练不稳定风险较高
- **推荐策略**: 使用标准MSE训练基础模型，通过MLEF集成和树模型修正峰值
