# Task4 光伏发电预测 - 任务与模型配置总结

## 1. 项目概述

### 1.1 项目目标
- **核心任务**: 光伏发电功率预测
- **预测模式**: 720小时历史数据 → 168小时（7天）逐小时预测
- **技术路线**: mamba底部拟合 + 树模型峰值增强 + MLEF集成

### 1.2 任务定义
| 任务 | 名称 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| task4_hourly | 720h_to_168h | 720小时 | 168小时 | Seq2Seq逐小时预测 |

---

## 2. 数据配置

### 2.1 数据来源
- **数据文件**: `data/combined_pv_data{2016,2017,2018}.csv`
- **时间粒度**: 小时级
- **目标变量**: `InvPAC_kW_Avg`（逆变器交流功率）

### 2.2 特征列表

**电气特征（16个）**:
| 特征名称 | 描述 |
|----------|------|
| InvVb_Avg | 逆变器B相电压 |
| InvIa_Avg | 逆变器A相电流 |
| InvIb_Avg | 逆变器B相电流 |
| InvIc_Avg | 逆变器C相电流 |
| InvFreq_Avg | 逆变器频率 |
| InvPAC_kW_Avg | 逆变器交流功率（目标） |
| InvPDC_kW_Avg | 逆变器直流功率 |
| InvOpStatus_Avg | 逆变器运行状态 |
| InvVoltageFault_Max | 逆变器电压故障 |
| PwrMtrIa_Avg | 电表A相电流 |
| PwrMtrIb_Avg | 电表B相电流 |
| PwrMtrFreq_Avg | 电表频率 |
| PwrMtrPhaseRev_Avg | 电表相序反转 |
| PwrMtrVa_Avg | 电表A相电压 |
| Battery_A_Avg | 电池电流 |
| Qloss_Ah_Max | 电荷损失 |

**时间特征（9个）**:
| 特征名称 | 描述 |
|----------|------|
| hour | 小时 (0-23) |
| day_of_week | 星期几 (0-6) |
| month | 月份 (1-12) |
| day_of_year | 一年中第几天 |
| hour_sin/hour_cos | 小时循环编码 |
| day_sin/day_cos | 日期循环编码 |
| is_daytime | 白天标记 (6:00-18:00) |

**总计**: 25个特征

### 2.3 数据划分
```yaml
训练月份: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
测试月份: [11, 12]
训练/验证比例: 90% / 10%
```

### 2.4 数据预处理
- 缺失值处理: 前向填充 → 后向填充 → 填0
- 归一化: MinMaxScaler [0, 1]
- 序列生成: 滑动窗口

---

## 3. 模型架构与参数

### 3.1 MambaSeq2Seq（主模型）

**架构**:
- 编码器: Input → Linear → MambaBlock → TransformerBlock → LayerNorm
- 解码器: Cross-Attention → MambaBlock → TransformerBlock → Output

**配置参数**:
```yaml
d_model: 128           # 模型维度
n_heads: 8             # 注意力头数
d_state: 64            # Mamba状态维度
dropout: 0.2           # Dropout比例
learning_rate: 0.0001  # 学习率
weight_decay: 0.0001   # 权重衰减
epochs: 150            # 训练轮次
batch_size: 128        # 批次大小
patience: 20           # 早停耐心
scheduler: cosine      # 学习率调度
warmup_epochs: 10      # 预热轮次
gradient_clip: 1.0     # 梯度裁剪
```

**输入输出维度**:
- 输入: (batch, 720, 25)
- 输出: (batch, 168)

### 3.2 LSTMSeq2Seq

**配置参数**:
```yaml
hidden_size: 128       # 隐藏层大小
num_layers: 3          # LSTM层数
dropout: 0.3           # Dropout比例
learning_rate: 0.0005  # 学习率
weight_decay: 0.0002   # 权重衰减
epochs: 200            # 训练轮次
batch_size: 32         # 批次大小
patience: 35           # 早停耐心
gradient_clip: 3.0     # 梯度裁剪
```

### 3.3 GRUSeq2Seq

**配置参数**:
```yaml
hidden_size: 128       # 隐藏层大小
num_layers: 3          # GRU层数
dropout: 0.3           # Dropout比例
learning_rate: 0.0005  # 学习率
weight_decay: 0.0002   # 权重衰减
epochs: 200            # 训练轮次
batch_size: 32         # 批次大小
patience: 35           # 早停耐心
gradient_clip: 3.0     # 梯度裁剪
```

### 3.4 LightGBM峰值预测器

**峰值功率预测**:
```yaml
objective: regression
n_estimators: 200      # 树数量
learning_rate: 0.1     # 学习率
num_leaves: 63         # 叶子节点数
max_depth: 8           # 最大深度
min_child_samples: 10  # 最小子节点样本
subsample: 0.9         # 子采样比例
colsample_bytree: 0.9  # 特征采样比例
reg_alpha: 0.05        # L1正则
reg_lambda: 0.5        # L2正则
```

**峰值时刻分类器（最优参数）**:
```yaml
objective: multiclass
num_class: 24          # 24小时分类
n_estimators: 50       # 树数量
max_depth: 3           # 最大深度
learning_rate: 0.1     # 学习率
num_leaves: 31         # 叶子节点数
min_child_samples: 20  # 最小子节点样本
n_jobs: 4              # 并行数
```

### 3.5 XGBoost峰值预测器

**配置参数**:
```yaml
objective: reg:squarederror
n_estimators: 200      # 树数量
learning_rate: 0.1     # 学习率
max_depth: 6           # 最大深度
min_child_weight: 1    # 最小子节点权重
subsample: 0.9         # 子采样比例
colsample_bytree: 0.9  # 特征采样比例
reg_alpha: 0.05        # L1正则
reg_lambda: 0.5        # L2正则
```

---

## 4. MLEF集成框架

### 4.1 集成策略
采用**动态置信度加权融合**策略，根据各模型在验证集上的逐时刻不确定性动态调整权重。

### 4.2 权重计算算法

```python
# 1. 模型过滤：保留 R² > 0 的模型
# 2. 计算逐时刻不确定性
for model in models:
    errors = pred_val - y_val
    uncertainty[model] = np.std(errors, axis=0)  # (168,)

# 3. 逆方差加权
for t in range(168):
    weights = 1.0 / (uncertainties + 1e-6)
    weights = weights / weights.sum()
    mlef_pred[:, t] = sum(w * pred[:, t] for w, pred in zip(weights, preds))
```

### 4.3 MLEF配置
```yaml
quality_threshold: 0.0   # 模型质量阈值
weighting_method: "r2"   # 权重方法
min_models: 2            # 最少保留模型数
```

---

## 5. 训练配置

### 5.1 优化器与损失函数
- **优化器**: AdamW
- **损失函数**: MSELoss
- **学习率调度**: CosineAnnealingLR（预热+余弦退火）

### 5.2 训练流程
1. 加载数据并预处理
2. 训练RNN模型（Mamba/LSTM/GRU）
3. 训练树模型峰值预测器
4. MLEF动态置信度加权集成
5. 评估与可视化

---

## 6. 评估指标

### 6.1 回归指标
| 指标 | 说明 |
|------|------|
| R² | 决定系数 |
| MSE | 均方误差 |
| RMSE | 均方根误差 |
| MAE | 平均绝对误差 |
| MAPE | 平均绝对百分比误差 |
| sMAPE | 对称MAPE |

### 6.2 分类指标（峰值时刻）
| 指标 | 说明 |
|------|------|
| Accuracy | 准确率 |
| F1-Score | F1分数（Macro） |
| PR-AUC | 精确率-召回率AUC |

### 6.3 分时域指标
- Horizon_1h: 第1小时预测性能
- Horizon_24h: 第24小时预测性能
- Horizon_72h: 第72小时预测性能
- Horizon_168h: 第168小时预测性能

---

## 7. 输出文件

### 7.1 指标文件
| 文件名 | 内容 |
|--------|------|
| `metrics.csv` | 所有模型性能指标 |
| `regression_metrics.csv` | 回归指标详情 |
| `classification_metrics.csv` | 分类指标详情 |
| `metrics_summary.json` | 指标汇总 |

### 7.2 可视化文件
| 文件名 | 内容 |
|--------|------|
| `r2_comparison.png` | R²对比柱状图 |
| `prediction_comparison.png` | 预测vs真实值散点图 |
| `peak_prediction_comparison.png` | 峰值预测对比 |
| `regression_metrics_comparison.png` | 回归指标对比图 |
| `classification_metrics_comparison.png` | 分类指标对比图 |

### 7.3 模型与数据文件
| 文件名 | 内容 |
|--------|------|
| `mlef_weights_by_hour.npy` | MLEF逐小时权重矩阵 |
| `mlef_model_names.json` | 参与MLEF的模型列表 |
| `all_predictions.pkl` | 所有模型预测结果 |

---

