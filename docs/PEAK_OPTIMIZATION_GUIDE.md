# 峰值优化模块 - 快速开始指南

## 概述

针对光伏发电预测中**日间尖峰拟合**的优化模块，包含：
- 峰值感知损失函数
- 动态加权策略
- 峰值集成逻辑
- 专项评估工具

## 3步快速集成

### 步骤1: 使用峰值感知损失

```python
from mtm_mlef import WeightedPeakAwareLoss

# 创建损失函数
criterion = WeightedPeakAwareLoss(
    alpha=1.0,    # 整体曲线权重
    beta=2.5,     # 峰值大小权重（关键参数）
    gamma=1.5,    # 峰值时刻权重
    daytime_weight=2.0,
    nighttime_weight=0.5,
    peak_window_weight=3.0
)

# 训练循环
for batch in dataloader:
    inputs, targets, is_daytime, time_indices = batch
    outputs = model(inputs)

    loss = criterion(
        y_pred=outputs,
        y_true=targets,
        is_daytime=is_daytime,
        time_indices=time_indices
    )

    loss.backward()
    optimizer.step()
```

### 步骤2: 评估峰值性能

```python
from mtm_mlef import create_peak_evaluation_report

# 获取预测结果
predictions = {'my_model': y_pred}

# 生成完整评估报告
create_peak_evaluation_report(
    predictions=predictions,
    y_true=y_test,
    is_daytime=is_daytime_test,
    time_indices=time_indices_test,
    output_dir='./results/peak_eval'
)
```

### 步骤3: (可选) 使用集成提升

```python
from mtm_mlef import PeakAwareEnsemble

# 收集多个模型预测
predictions = {
    'seq2seq': seq2seq_pred,
    'lightgbm': lgb_pred,
    'xgboost': xgb_pred
}

# 创建集成器
ensemble = PeakAwareEnsemble(peak_window_size=2)

# 在验证集校准权重
ensemble.calibrate_on_validation(
    predictions_val, y_val, is_daytime_val
)

# 预测
y_pred_final = ensemble.predict(
    predictions_test, is_daytime_test
)
```

## 核心API

### 损失函数

#### `PeakAwareLoss`
基础峰值感知损失，包含三个部分：
- 整体MSE (α)
- 峰值大小MSE (β)
- 峰值时刻惩罚 (γ)

#### `WeightedPeakAwareLoss`
增强版，额外支持：
- 昼夜时段加权
- 峰值窗口加权
- 高功率样本加权

### 工具类

#### `PeakExtractor`
```python
from mtm_mlef import PeakExtractor

extractor = PeakExtractor(hours_per_day=24)
peak_info = extractor.extract_daily_peaks(
    y_true, y_pred, is_daytime, time_indices
)

# 返回：峰值高度、时刻、RMSE、MAE等统计
```

#### `PeakVisualizer`
```python
from mtm_mlef import PeakVisualizer

viz = PeakVisualizer()

# 日曲线对比图
viz.plot_daily_curves_with_peaks(
    y_true, y_pred, is_daytime
)

# 误差分布图
viz.plot_peak_error_distribution(
    y_true, y_pred, is_daytime
)

# 多模型对比
viz.plot_comparison_across_models(
    predictions, y_true, is_daytime
)
```

### 数据加权

#### `PowerBasedWeightedSampler`
```python
from mtm_mlef import PowerBasedWeightedSampler
from torch.utils.data import DataLoader

# 创建加权采样器
sampler = PowerBasedWeightedSampler(
    power_values=avg_power_per_sample,
    num_samples=len(dataset),
    high_power_weight=3.0
)

# 在DataLoader中使用
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

#### `DynamicLossWeighter`
```python
from mtm_mlef import DynamicLossWeighter, WeightedTrainingStep

weighter = DynamicLossWeighter(
    daytime_weight=2.0,
    nighttime_weight=0.5
)

step = WeightedTrainingStep(weighter, nn.MSELoss(reduction='none'))

# 训练中使用
loss = step.compute_loss(
    y_pred, y_true, is_daytime, peak_times
)
```

### 集成模型

#### `PeakAwareEnsemble`
时段依赖的动态集成：
- 峰值窗口：优先使用在该时段表现好的模型
- 其他时段：使用整体拟合好的模型

#### `MLEFPeakEnhancer`
```python
from mtm_mlef import MLEFPeakEnhancer

enhancer = MLEFPeakEnhancer(
    seq2seq_model=seq2seq,
    tree_models={'lgb': lgb_model, 'xgb': xgb_model}
)

enhancer.calibrate(val_data, is_daytime_val)
y_pred = enhancer.predict(X_test, is_daytime_test)
```

## 推荐配置

### 平衡型（推荐起点）
```python
criterion = WeightedPeakAwareLoss(
    alpha=1.0, beta=2.5, gamma=1.5,
    daytime_weight=2.0,
    nighttime_weight=0.5,
    peak_window_weight=2.5
)
```
**预期效果：** 峰值RMSE↓25-35%，整体R²↓3-5%

### 峰值优先型
```python
criterion = WeightedPeakAwareLoss(
    alpha=1.0, beta=4.0, gamma=2.0,
    daytime_weight=3.0,
    nighttime_weight=0.3,
    peak_window_weight=4.0
)
```
**预期效果：** 峰值RMSE↓40-50%，整体R²↓8-12%

### 保守型
```python
criterion = WeightedPeakAwareLoss(
    alpha=1.0, beta=1.5, gamma=0.8,
    daytime_weight=1.5,
    nighttime_weight=0.7,
    peak_window_weight=2.0
)
```
**预期效果：** 峰值RMSE↓15-20%，整体R²↓1-2%

## 超参数调优

### 核心参数

| 参数 | 默认值 | 作用 | 调优建议 |
|-----|--------|------|---------|
| **β (beta)** | 2.0 | 峰值大小权重 | 峰值偏低→增大；抖动→减小 |
| **γ (gamma)** | 1.0 | 峰值时刻权重 | 时刻误差>2h→增大；形状扭曲→减小 |
| **daytime_weight** | 2.0 | 白天权重 | 保持在1.5-3.0之间 |
| **nighttime_weight** | 0.5 | 夜晚权重 | 不低于0.3，避免夜间失真 |
| **peak_window_weight** | 3.0 | 峰值窗口额外权重 | 2.0-4.0合适，过大会过拟合 |

### 权重平衡策略

**如果整体R²下降>10%：**
- 降低β和γ
- 提高nighttime_weight
- 使用权重warm-up

**如果峰值改善不明显：**
- 增大β
- 检查is_daytime标记是否正确
- 确认峰值在白天时段

**如果峰值时刻误差大：**
- 增大γ
- 使用smooth_l1惩罚
- 增大timing_penalty_scale

## 评估指标

### 峰值专项指标
- **Peak Value RMSE** - 峰值高度RMSE
- **Peak Time MAE** - 峰值时刻MAE（小时）
- **±1h Accuracy** - 峰值时刻±1小时准确率

### 整体指标
- **Overall RMSE** - 整体RMSE
- **Overall R²** - 拟合优度
- **Daytime/Nighttime RMSE** - 分时段RMSE

### 报告输出
运行评估后会生成：
```
results/peak_eval/
├── peak_metrics_summary.csv          # 指标汇总表
├── model_comparison.png              # 模型对比图
├── my_model_error_distribution.png   # 误差分布
├── my_model_daily_curves.png         # 日曲线对比
└── my_model_daily_peak_details.csv   # 每日峰值详情
```

## 数据准备

### is_daytime标志
```python
# 方法1: 基于时间索引
time_hour = timestamps.hour
is_daytime = ((time_hour >= 6) & (time_hour <= 20)).astype(float)

# 方法2: 基于辐射数据
is_daytime = (solar_radiation > threshold).astype(float)
```

### time_indices索引
```python
# 创建小时索引（0-23循环）
time_indices = np.tile(np.arange(24), (num_samples, 7))[:, :168]
```

## 常见问题

**Q: 使用后整体R²下降太多？**
A: 降低β和γ，或提高nighttime_weight到0.7

**Q: 峰值改善不明显？**
A: 检查is_daytime是否正确，尝试增大β到3.0-4.0

**Q: 训练不稳定？**
A: 使用权重warm-up，或改用smooth_l1惩罚

**Q: 峰值窗口内很好，其他地方变差？**
A: 降低peak_window_weight，或增大peak_window_size

## 完整示例

```python
from mtm_mlef import (
    WeightedPeakAwareLoss,
    create_peak_evaluation_report,
    PeakAwareEnsemble
)
import torch
import torch.nn as nn

# 1. 准备数据
# 假设已有 train_loader, val_loader, test_loader
# 每个batch包含: (inputs, targets, is_daytime, time_indices)

# 2. 创建模型和损失
model = YourSeq2SeqModel()
criterion = WeightedPeakAwareLoss(
    alpha=1.0, beta=2.5, gamma=1.5
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. 训练
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets, is_daytime, time_indices = batch
        outputs = model(inputs)

        loss = criterion(
            outputs, targets, is_daytime, time_indices
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. 评估
model.eval()
with torch.no_grad():
    # 获取测试集预测
    all_preds, all_targets, all_is_daytime, all_time_indices = [], [], [], []
    for batch in test_loader:
        inputs, targets, is_daytime, time_indices = batch
        outputs = model(inputs)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy())
        all_is_daytime.append(is_daytime.numpy())
        all_time_indices.append(time_indices.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    is_daytime_test = np.concatenate(all_is_daytime)
    time_indices_test = np.concatenate(all_time_indices)

# 5. 生成评估报告
predictions = {'my_model': y_pred}
create_peak_evaluation_report(
    predictions=predictions,
    y_true=y_true,
    is_daytime=is_daytime_test,
    time_indices=time_indices_test,
    output_dir='./results/peak_evaluation'
)

print("评估完成！请查看 ./results/peak_evaluation")
```

## 详细说明

更多设计原理、数学推导和详细用例，请参考各模块代码中的docstring：
- `losses/peak_loss.py` - 损失函数详解
- `utils/peak_extractor.py` - 峰值提取算法
- `utils/peak_viz.py` - 可视化方法
- `data/peak_sampler.py` - 加权采样策略
- `models/peak_ensemble.py` - 集成逻辑

---

**祝训练顺利！** 如有问题请查看代码注释或提issue。
