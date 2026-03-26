"""
集成学习模块 (MLEF - Multi-Layer Ensemble Framework)

包含多种集成方法:
- 基础集成: 简单平均、加权平均、投票、Stacking、Blending、最优组合
- MLEF框架: 两层/三层元学习集成
- 峰值感知集成: 时段依赖的动态权重集成
"""

import numpy as np
import pickle
from typing import Dict, Optional, List, Tuple
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from scipy.optimize import minimize

from ..utils.peak_extractor import PeakExtractor


def simple_average_ensemble(predictions_dict):
    """
    简单平均集成

    Args:
        predictions_dict: 字典,键为模型名,值为预测数组

    Returns:
        平均预测值
    """
    predictions = np.column_stack(list(predictions_dict.values()))
    return np.mean(predictions, axis=1)


def weighted_average_ensemble(predictions_val_dict, predictions_test_dict, y_val):
    """
    基于验证集性能的加权平均集成

    Args:
        predictions_val_dict: 验证集预测字典
        predictions_test_dict: 测试集预测字典
        y_val: 验证集真实值

    Returns:
        weights: 权重字典
        pred_test: 加权平均预测值
    """
    # 计算每个模型在验证集上的R²
    val_scores = []
    model_names = list(predictions_val_dict.keys())

    for name in model_names:
        pred_val = predictions_val_dict[name]
        score = r2_score(y_val, pred_val)
        val_scores.append(max(0, score))  # 避免负权重

    # 归一化权重
    weights_array = np.array(val_scores)
    weights_array = weights_array / (weights_array.sum() + 1e-8)

    # 创建权重字典
    weights = {name: w for name, w in zip(model_names, weights_array)}

    # 测试集加权预测
    predictions_test = np.column_stack([predictions_test_dict[name] for name in model_names])
    pred_test = np.average(predictions_test, weights=weights_array, axis=1)

    return weights, pred_test


def soft_voting_ensemble(predictions_val_dict, predictions_test_dict, y_val):
    """
    软投票集成 (实际上等同于加权平均)

    Args:
        predictions_val_dict: 验证集预测字典
        predictions_test_dict: 测试集预测字典
        y_val: 验证集真实值

    Returns:
        pred_test: 投票预测值
    """
    # 使用加权平均实现软投票
    _, pred_test = weighted_average_ensemble(predictions_val_dict, predictions_test_dict, y_val)
    return pred_test


def stacking_ensemble(predictions_val_dict, predictions_test_dict, y_val, config=None):
    """
    Stacking集成 (使用Ridge回归作为元学习器)

    Args:
        predictions_val_dict: 验证集预测字典
        predictions_test_dict: 测试集预测字典
        y_val: 验证集真实值
        config: 配置字典

    Returns:
        meta_model: 训练好的元模型
        pred_test: 测试集预测值
    """
    if config is None:
        config = {'alpha': 1.0, 'random_state': 42}

    # 准备元特征
    X_meta_val = np.column_stack(list(predictions_val_dict.values()))
    X_meta_test = np.column_stack(list(predictions_test_dict.values()))

    # 训练Ridge元学习器
    meta_model = Ridge(**config)
    meta_model.fit(X_meta_val, y_val)

    # 预测
    pred_test = meta_model.predict(X_meta_test)

    return meta_model, pred_test


def blending_ensemble(predictions_val_dict, predictions_test_dict, y_val, config=None):
    """
    Blending集成 (使用XGBoost作为元学习器)

    Args:
        predictions_val_dict: 验证集预测字典
        predictions_test_dict: 测试集预测字典
        y_val: 验证集真实值
        config: 配置字典

    Returns:
        meta_model: 训练好的元模型
        pred_test: 测试集预测值
    """
    if config is None:
        config = {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 3,
            'random_state': 42
        }

    # 准备元特征
    X_meta_val = np.column_stack(list(predictions_val_dict.values()))
    X_meta_test = np.column_stack(list(predictions_test_dict.values()))

    # 训练XGBoost元学习器
    meta_model = xgb.XGBRegressor(**config)
    meta_model.fit(X_meta_val, y_val, verbose=False)

    # 预测
    pred_test = meta_model.predict(X_meta_test)

    return meta_model, pred_test


def optimal_combination_ensemble(predictions_val_dict, predictions_test_dict, y_val):
    """
    最优组合集成 (使用优化算法寻找最佳权重)

    Args:
        predictions_val_dict: 验证集预测字典
        predictions_test_dict: 测试集预测字典
        y_val: 验证集真实值

    Returns:
        optimal_weights: 最优权重字典
        pred_test: 测试集预测值
    """
    # 准备元特征
    X_meta_val = np.column_stack(list(predictions_val_dict.values()))
    X_meta_test = np.column_stack(list(predictions_test_dict.values()))

    n_models = X_meta_val.shape[1]
    model_names = list(predictions_val_dict.keys())

    def objective(weights):
        """目标函数:最小化验证集MSE"""
        pred = np.average(X_meta_val, weights=weights, axis=1)
        return mean_squared_error(y_val, pred)

    # 约束条件:权重和为1,每个权重非负
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n_models)]
    initial_weights = np.ones(n_models) / n_models

    # 优化
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights_array = result.x

    # 创建权重字典
    optimal_weights = {name: w for name, w in zip(model_names, optimal_weights_array)}

    # 测试集预测
    pred_test = np.average(X_meta_test, weights=optimal_weights_array, axis=1)

    return optimal_weights, pred_test


class MLEFThreeLayerEnsemble:
    """
    MLEF三层元学习集成框架 (针对Seq2Seq模型)

    架构:
    ┌─────────────────────────────────────────────────────────┐
    │ Layer 1: Weighted Voting (加权投票层)                     │
    │   - 基于验证集R²/RMSE计算权重                              │
    │   - 输出: 加权平均预测                                     │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │ Layer 2: Meta-Learning with Features (元学习层)          │
    │   - 元特征: 基模型预测 + 预测方差 + 模型一致性              │
    │   - 元学习器: Ridge/XGBoost/LightGBM                      │
    │   - 输出: 元学习预测                                       │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │ Layer 3: Final Meta-Learner (最终融合层)                 │
    │   - 输入: Layer1输出 + Layer2输出 + 原始特征             │
    │   - 自适应权重: 根据验证集动态调整                         │
    │   - 输出: 最终预测                                        │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, meta_learner='ridge', verbose=True):
        """
        参数:
            meta_learner: 元学习器类型 ('ridge', 'xgboost', 'lightgbm')
            verbose: 是否打印详细信息
        """
        self.meta_learner_type = meta_learner
        self.verbose = verbose

        # 存储各层结果
        self.layer1_weights = None
        self.layer2_model = None
        self.layer3_model = None

        self.layer1_pred = None
        self.layer2_pred = None
        self.final_pred = None

    def _create_meta_features(self, predictions_dict):
        """
        创建增强的元特征

        元特征包括:
        1. 基模型预测值
        2. 预测方差 (模型不确定性)
        3. 模型一致性 (预测间的标准差)
        """
        # 基础预测特征
        base_predictions = np.column_stack(list(predictions_dict.values()))

        # 预测方差 (每个样本的方差)
        pred_variance = np.var(base_predictions, axis=1, keepdims=True)

        # 模型一致性 (每个样本的标准差)
        pred_std = np.std(base_predictions, axis=1, keepdims=True)

        # 拼接所有元特征
        meta_features = np.hstack([
            base_predictions,    # n_models列
            pred_variance,       # 1列
            pred_std            # 1列
        ])

        return meta_features

    def fit(self, predictions_val_dict, predictions_test_dict, y_val):
        """
        训练MLEF三层集成

        参数:
            predictions_val_dict: 验证集预测字典 {model_name: predictions}
            predictions_test_dict: 测试集预测字典
            y_val: 验证集真实值
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"MLEF Three-Layer Ensemble Training")
            print(f"{'='*70}")
            print(f"Number of base models: {len(predictions_val_dict)}")
            print(f"Meta-learner type: {self.meta_learner_type}")

        model_names = list(predictions_val_dict.keys())

        # ===== Layer 1: Weighted Voting =====
        if self.verbose:
            print(f"\n[Layer 1] Weighted Voting...")

        # 计算R²权重
        val_scores = []
        for name in model_names:
            pred_val = predictions_val_dict[name]
            score = r2_score(y_val, pred_val)
            val_scores.append(max(0, score))

        # 归一化权重
        weights_array = np.array(val_scores)
        weights_array = weights_array / (weights_array.sum() + 1e-8)
        self.layer1_weights = {name: w for name, w in zip(model_names, weights_array)}

        # Layer1预测
        predictions_val = np.column_stack([predictions_val_dict[name] for name in model_names])
        predictions_test = np.column_stack([predictions_test_dict[name] for name in model_names])

        layer1_pred_val = np.average(predictions_val, weights=weights_array, axis=1)
        self.layer1_pred = np.average(predictions_test, weights=weights_array, axis=1)

        if self.verbose:
            layer1_r2 = r2_score(y_val, layer1_pred_val)
            print(f"  Validation R²: {layer1_r2:.6f}")

        # ===== Layer 2: Meta-Learning with Features =====
        if self.verbose:
            print(f"\n[Layer 2] Meta-Learning with Enhanced Features...")

        # 创建元特征
        X_meta_val = self._create_meta_features(predictions_val_dict)
        X_meta_test = self._create_meta_features(predictions_test_dict)

        if self.verbose:
            print(f"  Meta-features shape: {X_meta_val.shape}")

        # 训练元学习器
        if self.meta_learner_type == 'ridge':
            self.layer2_model = Ridge(alpha=1.0, random_state=42)
        elif self.meta_learner_type == 'xgboost':
            self.layer2_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
                verbosity=0
            )
        elif self.meta_learner_type == 'lightgbm':
            import lightgbm as lgb
            self.layer2_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown meta-learner: {self.meta_learner_type}")

        self.layer2_model.fit(X_meta_val, y_val)

        # Layer2预测
        layer2_pred_val = self.layer2_model.predict(X_meta_val)
        self.layer2_pred = self.layer2_model.predict(X_meta_test)

        if self.verbose:
            layer2_r2 = r2_score(y_val, layer2_pred_val)
            print(f"  Validation R²: {layer2_r2:.6f}")

        # ===== Layer 3: Final Meta-Learner =====
        if self.verbose:
            print(f"\n[Layer 3] Final Fusion...")

        # 组合Layer1和Layer2的预测作为最终元特征
        X_final_val = np.column_stack([
            layer1_pred_val,     # Layer1输出
            layer2_pred_val,     # Layer2输出
            X_meta_val          # 原始元特征
        ])

        X_final_test = np.column_stack([
            self.layer1_pred,
            self.layer2_pred,
            X_meta_test
        ])

        # 最终元学习器 (简单Ridge)
        self.layer3_model = Ridge(alpha=0.5, random_state=42)
        self.layer3_model.fit(X_final_val, y_val)

        # 最终预测
        layer3_pred_val = self.layer3_model.predict(X_final_val)
        self.final_pred = self.layer3_model.predict(X_final_test)

        if self.verbose:
            layer3_r2 = r2_score(y_val, layer3_pred_val)
            print(f"  Validation R²: {layer3_r2:.6f}")

            print(f"\n{'='*70}")
            print(f"MLEF Training Complete!")
            print(f"{'='*70}\n")

        return self

    def predict(self):
        """返回最终预测"""
        return self.final_pred

    def get_layer_predictions(self):
        """返回各层预测结果(用于分析)"""
        return {
            'Layer1_WeightedVoting': self.layer1_pred,
            'Layer2_MetaLearning': self.layer2_pred,
            'Layer3_FinalFusion': self.final_pred
        }

    def get_layer1_weights(self):
        """返回Layer1的权重"""
        return self.layer1_weights


class EnsembleFramework:
    """集成学习框架类"""

    def __init__(self, config=None):
        """
        初始化集成框架

        Args:
            config: 集成配置
        """
        self.config = config or {}
        self.results = {}
        self.weights = {}

    def fit_all(self, predictions_val_dict, predictions_test_dict, y_val):
        """
        训练所有集成方法

        Args:
            predictions_val_dict: 验证集预测字典
            predictions_test_dict: 测试集预测字典
            y_val: 验证集真实值

        Returns:
            self
        """
        # 1. 简单平均
        self.results['Simple Average'] = simple_average_ensemble(predictions_test_dict)

        # 2. 加权平均
        weights, pred = weighted_average_ensemble(
            predictions_val_dict, predictions_test_dict, y_val
        )
        self.results['Weighted Average'] = pred
        self.weights['Weighted Average'] = weights

        # 3. 软投票
        self.results['Soft Voting'] = soft_voting_ensemble(
            predictions_val_dict, predictions_test_dict, y_val
        )

        # 4. Stacking
        stacking_config = self.config.get('stacking', {'alpha': 1.0, 'random_state': 42})
        meta_model, pred = stacking_ensemble(
            predictions_val_dict, predictions_test_dict, y_val, stacking_config
        )
        self.results['Stacking (Ridge)'] = pred

        # 5. Blending
        blending_config = self.config.get('blending', {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 3,
            'random_state': 42
        })
        meta_model, pred = blending_ensemble(
            predictions_val_dict, predictions_test_dict, y_val, blending_config
        )
        self.results['Blending (XGBoost)'] = pred

        # 6. 最优组合
        weights, pred = optimal_combination_ensemble(
            predictions_val_dict, predictions_test_dict, y_val
        )
        self.results['Optimal Combination'] = pred
        self.weights['Optimal Combination'] = weights

        return self

    def get_predictions(self, method_name):
        """获取指定方法的预测结果"""
        return self.results.get(method_name)

    def get_weights(self, method_name):
        """获取指定方法的权重"""
        return self.weights.get(method_name)

    def get_all_results(self):
        """获取所有预测结果"""
        return self.results

    def get_all_weights(self):
        """获取所有权重"""
        return self.weights


class MLEFTwoLayerEnsemble:
    """
    MLEF简化两层元学习集成 (避免过拟合的优化版本)

    架构:
    ┌──────────────────────────────────────────────────────────┐
    │ Layer 1: Weighted Voting (加权投票层)                      │
    │   - 基于验证集R²计算权重                                   │
    │   - 过滤负R²模型(权重设为0)                                │
    │   - 输出: 加权平均预测                                     │
    └──────────────────────────────────────────────────────────┘
                            ↓
    ┌──────────────────────────────────────────────────────────┐
    │ Layer 2: Ridge Meta-Learning (简化元学习层)               │
    │   - 元特征: 高质量基模型预测 + 预测方差                     │
    │   - 元学习器: Ridge回归 (L2正则化避免过拟合)               │
    │   - 输出: 最终预测                                         │
    └──────────────────────────────────────────────────────────┘

    相比三层版本的改进:
    1. 去掉Layer 3,减少过拟合风险
    2. 固定使用Ridge元学习器,添加L2正则化
    3. 自动过滤低质量模型(R²<0)
    4. 简化元特征(只用预测值+方差,去掉std避免冗余)
    """

    def __init__(self, alpha=5.0, verbose=True):
        """
        参数:
            alpha: Ridge正则化系数 (越大越保守)
            verbose: 是否打印详细信息
        """
        self.alpha = alpha
        self.verbose = verbose

        self.layer1_weights = None
        self.layer2_model = None
        self.final_pred = None
        self.filtered_models = None  # 过滤后的高质量模型列表

    def fit(self, predictions_val_dict, predictions_test_dict, y_val):
        """
        训练MLEF两层集成

        参数:
            predictions_val_dict: 验证集预测字典 {model_name: predictions}
            predictions_test_dict: 测试集预测字典
            y_val: 验证集真实值
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"MLEF Two-Layer Ensemble Training (Simplified)")
            print(f"{'='*70}")
            print(f"Number of base models: {len(predictions_val_dict)}")
            print(f"Ridge alpha: {self.alpha}")

        model_names = list(predictions_val_dict.keys())

        # ===== Layer 1: Weighted Voting with Filtering =====
        if self.verbose:
            print(f"\n[Layer 1] Weighted Voting (Filtering low-quality models)...")

        # 计算R²权重并过滤负R²模型
        val_scores = []
        valid_models = []
        for name in model_names:
            pred_val = predictions_val_dict[name]
            score = r2_score(y_val, pred_val)
            if score > 0:  # 只保留正R²的模型
                val_scores.append(score)
                valid_models.append(name)
            elif self.verbose:
                print(f"  ⚠ 过滤低质量模型: {name} (R²={score:.4f})")

        self.filtered_models = valid_models

        if len(valid_models) == 0:
            raise ValueError("所有模型R²都为负,无法集成")

        # 归一化权重
        weights_array = np.array(val_scores)
        weights_array = weights_array / (weights_array.sum() + 1e-8)
        self.layer1_weights = {name: w for name, w in zip(valid_models, weights_array)}

        # Layer1预测(只使用高质量模型)
        predictions_val = np.column_stack([predictions_val_dict[name] for name in valid_models])
        predictions_test = np.column_stack([predictions_test_dict[name] for name in valid_models])

        layer1_pred_val = np.average(predictions_val, weights=weights_array, axis=1)
        layer1_pred_test = np.average(predictions_test, weights=weights_array, axis=1)

        if self.verbose:
            layer1_r2 = r2_score(y_val, layer1_pred_val)
            print(f"  Filtered models: {len(valid_models)}/{len(model_names)}")
            print(f"  Validation R²: {layer1_r2:.6f}")

        # ===== Layer 2: Ridge Meta-Learning =====
        if self.verbose:
            print(f"\n[Layer 2] Ridge Meta-Learning...")

        # 创建简化元特征
        pred_variance = np.var(predictions_val, axis=1, keepdims=True)
        X_meta_val = np.hstack([predictions_val, pred_variance])

        pred_variance_test = np.var(predictions_test, axis=1, keepdims=True)
        X_meta_test = np.hstack([predictions_test, pred_variance_test])

        if self.verbose:
            print(f"  Meta-features shape: {X_meta_val.shape}")

        # 训练Ridge元学习器
        self.layer2_model = Ridge(alpha=self.alpha, random_state=42)
        self.layer2_model.fit(X_meta_val, y_val)

        # 最终预测
        layer2_pred_val = self.layer2_model.predict(X_meta_val)
        self.final_pred = self.layer2_model.predict(X_meta_test)

        if self.verbose:
            layer2_r2 = r2_score(y_val, layer2_pred_val)
            print(f"  Validation R²: {layer2_r2:.6f}")

            print(f"\n{'='*70}")
            print(f"MLEF Two-Layer Training Complete!")
            print(f"{'='*70}\n")

        return self

    def predict(self):
        """返回最终预测"""
        return self.final_pred

    def get_layer1_weights(self):
        """返回Layer1的权重"""
        return self.layer1_weights

    def get_filtered_models(self):
        """返回过滤后的高质量模型列表"""
        return self.filtered_models


# ==================== 峰值集成函数 ====================

def peak_to_hourly_prediction(peak_values, rnn_baseline, predicted_peak_hours, sigma=4.0):
    """
    将日峰值预测转换为逐小时预测

    使用峰值信息对RNN基线进行增强。在预测的峰值时刻附近应用提升因子，
    使用高斯窗口平滑避免突变。

    Args:
        peak_values: 树模型预测的日峰值，形状 (n_samples, 7)
        rnn_baseline: RNN模型的逐小时预测，形状 (n_samples, 168)
        predicted_peak_hours: 分类器预测的峰值时刻 (0-23)，形状 (n_samples, 7)
        sigma: 高斯平滑参数，控制提升范围

    Returns:
        hourly_pred: 峰值增强的逐小时预测，形状 (n_samples, 168)

    Example:
        >>> hourly_pred = peak_to_hourly_prediction(
        ...     peak_values, rnn_pred, peak_hours, sigma=4.0
        ... )
    """
    n_samples = peak_values.shape[0]
    hourly_pred = rnn_baseline.copy()

    for i in range(n_samples):
        for day in range(7):
            day_start = day * 24
            day_end = (day + 1) * 24

            # 当日RNN预测
            day_rnn = rnn_baseline[i, day_start:day_end]

            # 树模型预测的峰值
            tree_peak = peak_values[i, day]

            # RNN当日峰值
            rnn_peak = day_rnn.max()

            # 计算提升因子
            if rnn_peak > 1e-6:  # 避免除零
                boost_factor = tree_peak / rnn_peak

                # 限制提升范围（避免过度放大）
                boost_factor = np.clip(boost_factor, 0.3, 3.0)

                # 应用提升（高斯平滑，避免突变）
                boost_profile = np.ones(24)

                # 使用分类器预测的峰值时刻
                peak_hour = predicted_peak_hours[i, day]

                # 以峰值时刻为中心的高斯权重
                for h in range(24):
                    distance = abs(h - peak_hour)
                    weight = np.exp(-distance**2 / (2 * sigma**2))
                    boost_profile[h] = 1.0 + (boost_factor - 1.0) * weight

                # 应用提升
                hourly_pred[i, day_start:day_end] = day_rnn * boost_profile

    return hourly_pred


def mlef_confidence_weighted(predictions_val, y_val, predictions_test):
    """
    动态置信度加权融合（MLEF最佳策略）

    根据每个模型在验证集上的逐时刻不确定性动态调整权重。
    不确定性越小的模型在该时刻权重越高（逆方差加权）。

    相比基线通常提升约10%。

    Args:
        predictions_val: 验证集预测字典 {model_name: (n_samples, 168)}
        y_val: 验证集真实值，形状 (n_samples, 168)
        predictions_test: 测试集预测字典 {model_name: (n_samples, 168)}

    Returns:
        mlef_pred: 融合预测，形状 (n_samples, 168)
        weights_by_hour: 每个时刻的权重，形状 (168, n_models)
        model_names: 模型名称列表

    Example:
        >>> mlef_pred, weights, names = mlef_confidence_weighted(
        ...     predictions_val, y_val, predictions_test
        ... )
    """
    print("\n" + "=" * 80)
    print("[MLEF] 动态置信度加权集成")
    print("=" * 80)

    # 1. 过滤低质量模型
    model_names = []
    preds_val = []
    preds_test = []

    print("\n  模型过滤（保留R² > 0的模型）:")
    for model_name in predictions_val.keys():
        r2 = r2_score(y_val.flatten(), predictions_val[model_name].flatten())
        if r2 > 0:
            model_names.append(model_name)
            preds_val.append(predictions_val[model_name])
            preds_test.append(predictions_test[model_name])
            print(f"    ✓ {model_name}: R² = {r2:.4f}")
        else:
            print(f"    ✗ {model_name}: R² = {r2:.4f} (过滤)")

    n_models = len(model_names)
    print(f"\n  保留模型数: {n_models}")

    # 2. 计算每个模型在验证集上的逐时刻不确定性（标准差）
    print(f"\n  计算模型不确定性...")
    model_uncertainty = {}
    for i, name in enumerate(model_names):
        errors = preds_val[i] - y_val
        uncertainty = np.std(errors, axis=0)  # (168,) 每个时刻的标准差
        model_uncertainty[name] = uncertainty
        print(f"    {name}: 平均不确定性 = {uncertainty.mean():.3f} kW")

    # 3. 动态加权预测（逆方差加权）
    print(f"\n  执行动态加权融合...")
    mlef_pred = np.zeros_like(preds_test[0])
    weights_by_hour = np.zeros((168, n_models))

    for t in range(168):
        # 该时刻各模型的不确定性
        uncertainties = np.array([model_uncertainty[name][t] for name in model_names])

        # 不确定性越小，权重越大（逆方差加权）
        weights = 1.0 / (uncertainties + 1e-6)
        weights = weights / weights.sum()  # 归一化
        weights_by_hour[t] = weights

        # 加权融合
        for i, pred in enumerate(preds_test):
            mlef_pred[:, t] += weights[i] * pred[:, t]

    # 4. 统计平均权重
    avg_weights = weights_by_hour.mean(axis=0)
    print(f"\n  平均权重:")
    for name, weight in zip(model_names, avg_weights):
        print(f"    {name}: {weight:.6f}")

    # 5. 分析权重分布
    print(f"\n  权重统计:")
    for i, name in enumerate(model_names):
        weights_i = weights_by_hour[:, i]
        print(f"    {name}: min={weights_i.min():.4f}, max={weights_i.max():.4f}, std={weights_i.std():.4f}")

    return mlef_pred, weights_by_hour, model_names


def mlef_ridge_ensemble(predictions_val, y_val, predictions_test, alpha=10.0):
    """
    MLEF Ridge元学习集成（带自动低质模型过滤）

    Args:
        predictions_val: 验证集预测字典 {model_name: (n_samples, 168)}
        y_val: 验证集真实值，形状 (n_samples, 168)
        predictions_test: 测试集预测字典 {model_name: (n_samples, 168)}
        alpha: Ridge正则化参数

    Returns:
        ridge: 训练好的Ridge模型（如果成功）
        coefficients: 模型系数字典
        y_mlef_pred: 融合预测，形状 (n_samples, 168)

    Example:
        >>> ridge, coefs, pred = mlef_ridge_ensemble(
        ...     predictions_val, y_val, predictions_test, alpha=10.0
        ... )
    """
    print("\n" + "=" * 80)
    print("[MLEF] Ridge 元学习集成")
    print("=" * 80)

    # 过滤掉性能太差的模型（R² < 0）
    model_names = []
    valid_preds_val = []
    valid_preds_test = []

    print("\n  模型过滤（保留R² > 0的模型）:")
    for name, pred_val in predictions_val.items():
        r2 = r2_score(y_val.flatten(), pred_val.flatten())
        if r2 > 0:  # 只保留R² > 0的模型
            model_names.append(name)
            valid_preds_val.append(pred_val)
            valid_preds_test.append(predictions_test[name])
            print(f"    ✓ {name}: R² = {r2:.4f}")
        else:
            print(f"    ✗ {name}: R² = {r2:.4f} (过滤)")

    if len(model_names) < 2:
        print("\n  警告：有效模型少于2个，返回最佳单模型预测")
        if len(model_names) == 1:
            return None, {}, valid_preds_test[0]
        else:
            # 没有有效模型，返回零预测
            return None, {}, np.zeros_like(y_val)

    # 准备元特征
    X_meta_train = np.stack(valid_preds_val, axis=-1)
    X_meta_train = X_meta_train.reshape(-1, len(model_names))
    y_meta_train = y_val.reshape(-1, 1)

    print(f"\n  元训练数据: {X_meta_train.shape}")
    print(f"  元标签: {y_meta_train.shape}")

    # 训练Ridge
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_meta_train, y_meta_train)

    # 提取系数
    coef_array = ridge.coef_.flatten()
    coefficients = {name: float(coef_array[i]) for i, name in enumerate(model_names)}

    print(f"\n  Ridge 系数:")
    for name, coef in coefficients.items():
        print(f"    {name}: {coef:.6f}")

    # 截距
    intercept_val = ridge.intercept_[0] if hasattr(ridge.intercept_, '__len__') else ridge.intercept_
    print(f"  截距: {intercept_val:.6f}")

    # 测试集预测
    X_meta_test = np.stack(valid_preds_test, axis=-1)
    n_test, output_len, _ = X_meta_test.shape
    X_meta_test = X_meta_test.reshape(-1, len(model_names))

    y_mlef_pred = ridge.predict(X_meta_test)
    y_mlef_pred = y_mlef_pred.reshape(n_test, output_len)

    return ridge, coefficients, y_mlef_pred


# ==================== 峰值感知集成类 ====================

class PeakAwareEnsemble:
    """
    峰值感知的集成预测器

    核心思想：
    1. 在验证集上统计各模型在不同时段（峰值窗口 vs 其他时段）的表现
    2. 预测时，根据时段动态调整各模型权重
    3. 峰值窗口内优先使用在该时段表现好的模型

    Example:
        >>> ensemble = PeakAwareEnsemble(peak_window_size=2)
        >>> ensemble.calibrate_on_validation(predictions, y_true, is_daytime)
        >>> y_pred = ensemble.predict(predictions_test, is_daytime_test)
    """

    def __init__(
        self,
        hours_per_day: int = 24,
        peak_window_size: int = 2,
        use_dynamic_weights: bool = True
    ):
        """
        Args:
            hours_per_day: 每天小时数
            peak_window_size: 峰值前后±N小时视为峰值窗口
            use_dynamic_weights: 是否使用动态权重（否则使用固定权重）
        """
        self.hours_per_day = hours_per_day
        self.peak_window_size = peak_window_size
        self.use_dynamic_weights = use_dynamic_weights
        self.peak_extractor = PeakExtractor(hours_per_day)

        # 模型在不同时段的性能统计
        self.model_performance = {}  # {model_name: {'peak': rmse, 'non_peak': rmse}}
        self.base_weights = {}       # {model_name: weight} 基础权重
        self.peak_weights = {}       # {model_name: weight} 峰值窗口权重
        self.non_peak_weights = {}   # {model_name: weight} 非峰值窗口权重

    def calibrate_on_validation(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        is_daytime: np.ndarray,
        time_indices: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        在验证集上校准各模型的权重

        Args:
            predictions: 各模型的预测结果, {model_name: predictions}
                        predictions shape: (num_samples, seq_len)
            y_true: 真实值, shape (num_samples, seq_len)
            is_daytime: 白天标志, shape (num_samples, seq_len)
            time_indices: 时间索引, shape (num_samples, seq_len)
            verbose: 是否打印统计信息
        """
        num_samples, seq_len = y_true.shape
        num_days = seq_len // self.hours_per_day

        # 1. 提取真实峰值位置
        peak_info = self.peak_extractor.extract_daily_peaks(
            y_true, y_true,  # 使用真实值两次，只为获取峰值位置
            is_daytime, time_indices
        )
        true_peak_times = peak_info['true_peak_times']  # (num_samples, num_days)

        # 2. 创建峰值窗口mask
        peak_window_mask = np.zeros_like(y_true, dtype=bool)
        for day in range(num_days):
            start_idx = day * self.hours_per_day
            for i in range(num_samples):
                peak_hour = true_peak_times[i, day]
                window_start = max(0, peak_hour - self.peak_window_size)
                window_end = min(self.hours_per_day, peak_hour + self.peak_window_size + 1)

                for h in range(window_start, window_end):
                    global_idx = start_idx + h
                    if global_idx < seq_len:
                        peak_window_mask[i, global_idx] = True

        # 3. 计算每个模型在峰值窗口和非峰值窗口的RMSE
        for model_name, y_pred in predictions.items():
            # 峰值窗口误差
            peak_errors = y_pred[peak_window_mask] - y_true[peak_window_mask]
            peak_rmse = np.sqrt(np.mean(peak_errors ** 2))

            # 非峰值窗口误差
            non_peak_mask = ~peak_window_mask
            non_peak_errors = y_pred[non_peak_mask] - y_true[non_peak_mask]
            non_peak_rmse = np.sqrt(np.mean(non_peak_errors ** 2))

            # 整体误差
            overall_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

            # 存储性能
            self.model_performance[model_name] = {
                'peak_rmse': peak_rmse,
                'non_peak_rmse': non_peak_rmse,
                'overall_rmse': overall_rmse
            }

            if verbose:
                print(f"\n{model_name} 在验证集上的表现:")
                print(f"  峰值窗口 RMSE: {peak_rmse:.4f}")
                print(f"  非峰值窗口 RMSE: {non_peak_rmse:.4f}")
                print(f"  整体 RMSE: {overall_rmse:.4f}")

        # 4. 计算动态权重
        self._compute_dynamic_weights(verbose)

    def _compute_dynamic_weights(self, verbose: bool = True):
        """
        根据各模型在不同时段的表现计算权重

        使用反比例权重：RMSE越小，权重越大
        """
        model_names = list(self.model_performance.keys())

        # 基础权重（基于整体RMSE）
        overall_rmses = np.array([
            self.model_performance[name]['overall_rmse']
            for name in model_names
        ])
        # 使用反平方作为权重（更aggressive的差异化）
        base_weights_raw = 1.0 / (overall_rmses ** 2 + 1e-6)
        base_weights_norm = base_weights_raw / base_weights_raw.sum()

        for i, name in enumerate(model_names):
            self.base_weights[name] = base_weights_norm[i]

        # 峰值窗口权重
        peak_rmses = np.array([
            self.model_performance[name]['peak_rmse']
            for name in model_names
        ])
        peak_weights_raw = 1.0 / (peak_rmses ** 2 + 1e-6)
        peak_weights_norm = peak_weights_raw / peak_weights_raw.sum()

        for i, name in enumerate(model_names):
            self.peak_weights[name] = peak_weights_norm[i]

        # 非峰值窗口权重
        non_peak_rmses = np.array([
            self.model_performance[name]['non_peak_rmse']
            for name in model_names
        ])
        non_peak_weights_raw = 1.0 / (non_peak_rmses ** 2 + 1e-6)
        non_peak_weights_norm = non_peak_weights_raw / non_peak_weights_raw.sum()

        for i, name in enumerate(model_names):
            self.non_peak_weights[name] = non_peak_weights_norm[i]

        if verbose:
            print("\n=== 计算的集成权重 ===")
            print(f"{'模型':<20} {'基础权重':<12} {'峰值窗口':<12} {'非峰值窗口':<12}")
            print("-" * 60)
            for name in model_names:
                print(f"{name:<20} {self.base_weights[name]:<12.4f} "
                      f"{self.peak_weights[name]:<12.4f} {self.non_peak_weights[name]:<12.4f}")

    def predict(
        self,
        predictions: Dict[str, np.ndarray],
        is_daytime: np.ndarray,
        time_indices: Optional[np.ndarray] = None,
        use_peak_detection: bool = True
    ) -> np.ndarray:
        """
        使用动态权重进行集成预测

        Args:
            predictions: 各模型的预测结果, {model_name: predictions}
            is_daytime: 白天标志, shape (num_samples, seq_len)
            time_indices: 时间索引, shape (num_samples, seq_len)
            use_peak_detection: 是否使用峰值检测来确定峰值窗口

        Returns:
            集成预测结果, shape (num_samples, seq_len)
        """
        model_names = list(predictions.keys())
        num_samples, seq_len = next(iter(predictions.values())).shape

        # 初始化集成结果
        ensemble_pred = np.zeros((num_samples, seq_len))

        if not self.use_dynamic_weights:
            # 使用固定基础权重
            for name in model_names:
                ensemble_pred += predictions[name] * self.base_weights[name]
            return ensemble_pred

        # 使用动态权重
        if use_peak_detection:
            # 先用基础权重得到初步预测，用于检测峰值位置
            temp_pred = np.zeros((num_samples, seq_len))
            for name in model_names:
                temp_pred += predictions[name] * self.base_weights[name]

            # 检测峰值位置
            peak_info = self.peak_extractor.extract_daily_peaks(
                temp_pred, temp_pred,
                is_daytime, time_indices
            )
            pred_peak_times = peak_info['true_peak_times']  # (num_samples, num_days)
        else:
            # 使用固定峰值时刻（例如正午12点）
            num_days = seq_len // self.hours_per_day
            pred_peak_times = np.full((num_samples, num_days), 12, dtype=int)

        # 创建峰值窗口mask
        num_days = seq_len // self.hours_per_day
        peak_window_mask = np.zeros((num_samples, seq_len), dtype=bool)

        for day in range(num_days):
            start_idx = day * self.hours_per_day
            for i in range(num_samples):
                peak_hour = pred_peak_times[i, day]
                window_start = max(0, peak_hour - self.peak_window_size)
                window_end = min(self.hours_per_day, peak_hour + self.peak_window_size + 1)

                for h in range(window_start, window_end):
                    global_idx = start_idx + h
                    if global_idx < seq_len:
                        peak_window_mask[i, global_idx] = True

        # 对每个时间步使用相应的权重
        for name in model_names:
            # 峰值窗口使用峰值权重，其他地方使用非峰值权重
            weight_map = np.where(
                peak_window_mask,
                self.peak_weights[name],
                self.non_peak_weights[name]
            )
            ensemble_pred += predictions[name] * weight_map

        return ensemble_pred

    def save_weights(self, path: str):
        """保存权重到文件"""
        weights = {
            'base_weights': self.base_weights,
            'peak_weights': self.peak_weights,
            'non_peak_weights': self.non_peak_weights,
            'model_performance': self.model_performance,
            'config': {
                'hours_per_day': self.hours_per_day,
                'peak_window_size': self.peak_window_size,
                'use_dynamic_weights': self.use_dynamic_weights
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(weights, f)
        print(f"权重已保存到: {path}")

    def load_weights(self, path: str):
        """从文件加载权重"""
        with open(path, 'rb') as f:
            weights = pickle.load(f)

        self.base_weights = weights['base_weights']
        self.peak_weights = weights['peak_weights']
        self.non_peak_weights = weights['non_peak_weights']
        self.model_performance = weights['model_performance']

        config = weights['config']
        self.hours_per_day = config['hours_per_day']
        self.peak_window_size = config['peak_window_size']
        self.use_dynamic_weights = config['use_dynamic_weights']

        print(f"权重已从 {path} 加载")


class MLEFPeakEnhancer:
    """
    MLEF (Multi-Level Ensemble Framework) 的峰值增强版本

    在原有MLEF基础上，专门针对峰值时段优化集成策略

    Example:
        >>> enhancer = MLEFPeakEnhancer(seq2seq_model, tree_models)
        >>> enhancer.calibrate(val_data, is_daytime_val)
        >>> y_pred = enhancer.predict(X_test, is_daytime_test)
    """

    def __init__(
        self,
        seq2seq_model,  # 主模型（Seq2Seq）
        tree_models: Dict[str, any],  # 树模型 {'lgb': model, 'xgb': model}
        peak_window_size: int = 2,
        hours_per_day: int = 24
    ):
        """
        Args:
            seq2seq_model: Seq2Seq主模型
            tree_models: 树模型字典
            peak_window_size: 峰值窗口大小
            hours_per_day: 每天小时数
        """
        self.seq2seq_model = seq2seq_model
        self.tree_models = tree_models
        self.peak_ensemble = PeakAwareEnsemble(hours_per_day, peak_window_size)

    def calibrate(
        self,
        val_data,
        is_daytime: np.ndarray,
        time_indices: Optional[np.ndarray] = None
    ):
        """
        在验证集上校准集成权重

        Args:
            val_data: 验证集数据，包含 (X_val, y_val)
            is_daytime: 白天标志
            time_indices: 时间索引
        """
        X_val, y_val = val_data

        # 获取各模型预测
        predictions = {}

        # Seq2Seq预测
        with torch.no_grad():
            seq2seq_pred = self.seq2seq_model(torch.tensor(X_val).float())
            predictions['seq2seq'] = seq2seq_pred.cpu().numpy()

        # 树模型预测
        for name, model in self.tree_models.items():
            # 假设树模型需要reshape输入
            X_reshaped = X_val.reshape(X_val.shape[0], -1)
            tree_pred = model.predict(X_reshaped)
            predictions[name] = tree_pred.reshape(y_val.shape)

        # 校准权重
        self.peak_ensemble.calibrate_on_validation(
            predictions, y_val, is_daytime, time_indices, verbose=True
        )

    def predict(
        self,
        X: np.ndarray,
        is_daytime: np.ndarray,
        time_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        使用峰值感知集成进行预测

        Args:
            X: 输入特征
            is_daytime: 白天标志
            time_indices: 时间索引

        Returns:
            集成预测结果
        """
        predictions = {}

        # Seq2Seq预测
        with torch.no_grad():
            seq2seq_pred = self.seq2seq_model(torch.tensor(X).float())
            predictions['seq2seq'] = seq2seq_pred.cpu().numpy()

        # 树模型预测
        for name, model in self.tree_models.items():
            X_reshaped = X.reshape(X.shape[0], -1)
            tree_pred = model.predict(X_reshaped)
            predictions[name] = tree_pred.reshape(predictions['seq2seq'].shape)

        # 峰值感知集成
        ensemble_pred = self.peak_ensemble.predict(
            predictions, is_daytime, time_indices, use_peak_detection=True
        )

        return ensemble_pred


class EnhancedPeakAwareEnsemble:
    """
    增强版峰值感知集成（简化版）

    核心改进：
    1. 使用高斯软过渡，峰值窗口边缘权重平滑变化
    2. 根据验证集统计自动选择"峰值窗口更可靠"的模型

    Example:
        >>> ensemble = EnhancedPeakAwareEnsemble()
        >>> ensemble.calibrate(predictions, y_true, is_daytime)
        >>> y_pred = ensemble.predict(predictions_test, is_daytime_test)
    """

    def __init__(
        self,
        hours_per_day: int = 24,
        peak_window_size: int = 2,
        soft_transition_sigma: float = 1.0
    ):
        """
        Args:
            hours_per_day: 每天小时数
            peak_window_size: 峰值窗口大小（±N小时）
            soft_transition_sigma: 高斯过渡的标准差
        """
        self.hours_per_day = hours_per_day
        self.peak_window_size = peak_window_size
        self.soft_transition_sigma = soft_transition_sigma

        self.peak_extractor = PeakExtractor(hours_per_day)

        self.model_performance = {}
        self.peak_weights = {}
        self.non_peak_weights = {}

    def calibrate(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        is_daytime: np.ndarray,
        verbose: bool = True
    ):
        """在验证集上校准权重"""
        num_samples, seq_len = y_true.shape
        num_days = seq_len // self.hours_per_day
        model_names = list(predictions.keys())

        # 1. 提取峰值位置，创建峰值窗口mask
        peak_info = self.peak_extractor.extract_daily_peaks(
            y_true, y_true, is_daytime
        )
        true_peak_times = peak_info['true_peak_times']

        peak_mask = np.zeros((num_samples, seq_len), dtype=bool)
        for day in range(num_days):
            start_idx = day * self.hours_per_day
            for i in range(num_samples):
                peak_hour = true_peak_times[i, day]
                for h in range(max(0, peak_hour - self.peak_window_size),
                               min(self.hours_per_day, peak_hour + self.peak_window_size + 1)):
                    if start_idx + h < seq_len:
                        peak_mask[i, start_idx + h] = True

        # 2. 计算各模型在峰值/非峰值区域的RMSE
        for name, y_pred in predictions.items():
            peak_rmse = np.sqrt(np.mean((y_pred[peak_mask] - y_true[peak_mask]) ** 2))
            non_peak_rmse = np.sqrt(np.mean((y_pred[~peak_mask] - y_true[~peak_mask]) ** 2))
            overall_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

            self.model_performance[name] = {
                'peak_rmse': peak_rmse,
                'non_peak_rmse': non_peak_rmse,
                'overall_rmse': overall_rmse
            }

        # 3. 计算权重（RMSE越小权重越大）
        peak_rmses = np.array([self.model_performance[n]['peak_rmse'] for n in model_names])
        non_peak_rmses = np.array([self.model_performance[n]['non_peak_rmse'] for n in model_names])

        peak_weights_raw = 1.0 / (peak_rmses ** 2 + 1e-8)
        peak_weights_norm = peak_weights_raw / peak_weights_raw.sum()

        non_peak_weights_raw = 1.0 / (non_peak_rmses ** 2 + 1e-8)
        non_peak_weights_norm = non_peak_weights_raw / non_peak_weights_raw.sum()

        for i, name in enumerate(model_names):
            self.peak_weights[name] = peak_weights_norm[i]
            self.non_peak_weights[name] = non_peak_weights_norm[i]

        if verbose:
            print("\n=== 增强峰值集成校准 ===")
            print(f"{'模型':<15} {'峰值RMSE':<12} {'非峰值RMSE':<12} {'峰值权重':<12} {'非峰值权重':<12}")
            for name in model_names:
                perf = self.model_performance[name]
                print(f"{name:<15} {perf['peak_rmse']:<12.4f} {perf['non_peak_rmse']:<12.4f} "
                      f"{self.peak_weights[name]:<12.4f} {self.non_peak_weights[name]:<12.4f}")

    def predict(
        self,
        predictions: Dict[str, np.ndarray],
        is_daytime: np.ndarray
    ) -> np.ndarray:
        """使用软过渡权重进行集成预测"""
        model_names = list(predictions.keys())
        num_samples, seq_len = next(iter(predictions.values())).shape
        num_days = seq_len // self.hours_per_day

        # 1. 用基础权重初步预测，检测峰值位置
        temp_pred = sum(predictions[n] * self.peak_weights[n] for n in model_names)

        peak_info = self.peak_extractor.extract_daily_peaks(temp_pred, temp_pred, is_daytime)
        pred_peak_times = peak_info['true_peak_times']

        # 2. 创建高斯软过渡权重
        peak_weight_map = np.zeros((num_samples, seq_len))
        for day in range(num_days):
            start_idx = day * self.hours_per_day
            for i in range(num_samples):
                peak_hour = pred_peak_times[i, day]
                for h in range(self.hours_per_day):
                    if start_idx + h < seq_len:
                        distance = abs(h - peak_hour)
                        # 高斯软过渡
                        peak_weight_map[i, start_idx + h] = np.exp(
                            -0.5 * (distance / self.soft_transition_sigma) ** 2
                        )

        non_peak_weight_map = 1.0 - peak_weight_map

        # 3. 加权集成
        ensemble_pred = np.zeros((num_samples, seq_len))
        for name in model_names:
            weight = (peak_weight_map * self.peak_weights[name] +
                      non_peak_weight_map * self.non_peak_weights[name])
            ensemble_pred += predictions[name] * weight

        # 归一化（因为权重和不一定为1）
        weight_sum = sum(
            peak_weight_map * self.peak_weights[n] + non_peak_weight_map * self.non_peak_weights[n]
            for n in model_names
        )
        ensemble_pred = ensemble_pred / (weight_sum + 1e-8)

        return ensemble_pred

    def save(self, path: str):
        """保存权重"""
        data = {
            'peak_weights': self.peak_weights,
            'non_peak_weights': self.non_peak_weights,
            'model_performance': self.model_performance,
            'config': {
                'hours_per_day': self.hours_per_day,
                'peak_window_size': self.peak_window_size,
                'soft_transition_sigma': self.soft_transition_sigma
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'EnhancedPeakAwareEnsemble':
        """加载权重"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        instance = cls(**data['config'])
        instance.peak_weights = data['peak_weights']
        instance.non_peak_weights = data['non_peak_weights']
        instance.model_performance = data['model_performance']
        return instance


# ==================== 高级峰值集成优化 ====================

class AdaptiveGaussianEnsemble:
    """
    自适应非对称高斯权重集成

    核心改进：
    1. 使用非对称高斯（峰值前后不同sigma）避免权重突变
    2. 可选：根据峰值尖锐度自适应调整sigma

    Example:
        >>> ensemble = AdaptiveGaussianEnsemble(sigma_before=1.5, sigma_after=2.0)
        >>> ensemble.calibrate(predictions_val, y_val, is_daytime_val)
        >>> y_pred = ensemble.predict(predictions_test, is_daytime_test)
    """

    def __init__(
        self,
        sigma_before: float = 1.5,
        sigma_after: float = 2.0,
        adaptive_sigma: bool = True,
        hours_per_day: int = 24
    ):
        """
        Args:
            sigma_before: 峰值前的高斯标准差（较小=较陡）
            sigma_after: 峰值后的高斯标准差（较大=较缓）
            adaptive_sigma: 是否根据峰值尖锐度自适应调整sigma
            hours_per_day: 每天小时数
        """
        self.sigma_before = sigma_before
        self.sigma_after = sigma_after
        self.adaptive_sigma = adaptive_sigma
        self.hours_per_day = hours_per_day
        self.peak_extractor = PeakExtractor(hours_per_day)

        self.peak_weights = {}
        self.non_peak_weights = {}
        self.model_performance = {}

    def _compute_peak_sharpness(
        self,
        y: np.ndarray,
        peak_times: np.ndarray
    ) -> np.ndarray:
        """
        计算峰值尖锐度（使用负二阶导数）

        Args:
            y: 预测/真实值, shape (batch_size, seq_len)
            peak_times: 峰值时刻, shape (batch_size, num_days)

        Returns:
            sharpness: 峰值尖锐度, shape (batch_size, num_days)
        """
        batch_size, seq_len = y.shape
        num_days = seq_len // self.hours_per_day
        sharpness = np.zeros((batch_size, num_days))

        for day in range(num_days):
            day_start = day * self.hours_per_day
            for i in range(batch_size):
                ph = int(peak_times[i, day])
                # 确保峰值不在边界
                if 1 <= ph <= self.hours_per_day - 2:
                    idx = day_start + ph
                    # 二阶导数: f''(x) ≈ f(x-1) - 2f(x) + f(x+1)
                    curv = y[i, idx - 1] + y[i, idx + 1] - 2 * y[i, idx]
                    sharpness[i, day] = -curv  # 负号使峰值处为正

        return sharpness

    def _compute_gaussian_weights(
        self,
        peak_positions: np.ndarray,
        seq_len: int,
        peak_sharpness: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算非对称高斯过渡权重

        Args:
            peak_positions: 峰值位置, shape (batch_size, num_days)
            seq_len: 序列长度
            peak_sharpness: 峰值尖锐度（可选）

        Returns:
            weight_map: 权重映射, shape (batch_size, seq_len)
        """
        batch_size = peak_positions.shape[0]
        num_days = peak_positions.shape[1]

        weight_map = np.zeros((batch_size, seq_len))

        for day in range(num_days):
            day_start = day * self.hours_per_day
            for i in range(batch_size):
                peak_hour = int(peak_positions[i, day])

                # 自适应sigma（尖峰更窄，钝峰更宽）
                sigma_scale = 1.0
                if self.adaptive_sigma and peak_sharpness is not None:
                    sharp = peak_sharpness[i, day]
                    # 尖锐度越高，sigma越小（权重更集中）
                    sigma_scale = 1.0 / (1.0 + max(0, sharp) * 0.01)

                for h in range(self.hours_per_day):
                    global_idx = day_start + h
                    if global_idx >= seq_len:
                        continue

                    distance = h - peak_hour

                    # 非对称高斯
                    if distance < 0:
                        sigma = self.sigma_before * sigma_scale
                    else:
                        sigma = self.sigma_after * sigma_scale

                    weight_map[i, global_idx] = np.exp(-0.5 * (distance / sigma) ** 2)

        return weight_map

    def calibrate(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        is_daytime: np.ndarray,
        verbose: bool = True
    ):
        """
        在验证集上校准权重

        Args:
            predictions: 各模型预测, {model_name: (batch_size, seq_len)}
            y_true: 真实值, shape (batch_size, seq_len)
            is_daytime: 白天标志
            verbose: 是否打印信息
        """
        num_samples, seq_len = y_true.shape
        num_days = seq_len // self.hours_per_day
        model_names = list(predictions.keys())

        # 1. 提取真实峰值位置
        peak_info = self.peak_extractor.extract_daily_peaks(y_true, y_true, is_daytime)
        true_peak_times = peak_info['true_peak_times']

        # 2. 计算峰值窗口mask（用于统计性能）
        # 使用±2小时作为峰值窗口
        peak_window = 2
        peak_mask = np.zeros((num_samples, seq_len), dtype=bool)
        for day in range(num_days):
            start_idx = day * self.hours_per_day
            for i in range(num_samples):
                ph = int(true_peak_times[i, day])
                for h in range(max(0, ph - peak_window), min(self.hours_per_day, ph + peak_window + 1)):
                    if start_idx + h < seq_len:
                        peak_mask[i, start_idx + h] = True

        # 3. 计算各模型在峰值/非峰值区域的RMSE
        for name, y_pred in predictions.items():
            peak_rmse = np.sqrt(np.mean((y_pred[peak_mask] - y_true[peak_mask]) ** 2))
            non_peak_rmse = np.sqrt(np.mean((y_pred[~peak_mask] - y_true[~peak_mask]) ** 2))
            overall_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

            self.model_performance[name] = {
                'peak_rmse': peak_rmse,
                'non_peak_rmse': non_peak_rmse,
                'overall_rmse': overall_rmse
            }

        # 4. 计算逆方差权重
        peak_rmses = np.array([self.model_performance[n]['peak_rmse'] for n in model_names])
        non_peak_rmses = np.array([self.model_performance[n]['non_peak_rmse'] for n in model_names])

        peak_weights_raw = 1.0 / (peak_rmses ** 2 + 1e-8)
        non_peak_weights_raw = 1.0 / (non_peak_rmses ** 2 + 1e-8)

        peak_weights_norm = peak_weights_raw / peak_weights_raw.sum()
        non_peak_weights_norm = non_peak_weights_raw / non_peak_weights_raw.sum()

        for i, name in enumerate(model_names):
            self.peak_weights[name] = peak_weights_norm[i]
            self.non_peak_weights[name] = non_peak_weights_norm[i]

        if verbose:
            print("\n" + "=" * 70)
            print("[AdaptiveGaussianEnsemble] 校准完成")
            print("=" * 70)
            print(f"Sigma设置: before={self.sigma_before}, after={self.sigma_after}")
            print(f"自适应Sigma: {self.adaptive_sigma}")
            print(f"\n{'模型':<20} {'峰值RMSE':<12} {'非峰值RMSE':<12} {'峰值权重':<12} {'非峰值权重':<12}")
            print("-" * 70)
            for name in model_names:
                perf = self.model_performance[name]
                print(f"{name:<20} {perf['peak_rmse']:<12.4f} {perf['non_peak_rmse']:<12.4f} "
                      f"{self.peak_weights[name]:<12.4f} {self.non_peak_weights[name]:<12.4f}")

    def predict(
        self,
        predictions: Dict[str, np.ndarray],
        is_daytime: np.ndarray
    ) -> np.ndarray:
        """
        使用非对称高斯权重进行集成预测

        Args:
            predictions: 各模型预测
            is_daytime: 白天标志

        Returns:
            集成预测结果
        """
        model_names = list(predictions.keys())
        num_samples, seq_len = next(iter(predictions.values())).shape
        num_days = seq_len // self.hours_per_day

        # 1. 初步预测以检测峰值位置
        temp_pred = np.zeros((num_samples, seq_len))
        for name in model_names:
            w = (self.peak_weights.get(name, 0) + self.non_peak_weights.get(name, 0)) / 2
            temp_pred += predictions[name] * w

        peak_info = self.peak_extractor.extract_daily_peaks(temp_pred, temp_pred, is_daytime)
        pred_peak_times = peak_info['true_peak_times']

        # 2. 计算峰值尖锐度
        sharpness = self._compute_peak_sharpness(temp_pred, pred_peak_times) if self.adaptive_sigma else None

        # 3. 计算高斯权重图
        peak_weight_map = self._compute_gaussian_weights(pred_peak_times, seq_len, sharpness)
        non_peak_weight_map = 1.0 - peak_weight_map

        # 4. 加权集成
        ensemble_pred = np.zeros((num_samples, seq_len))
        weight_sum = np.zeros((num_samples, seq_len))

        for name in model_names:
            w_peak = self.peak_weights.get(name, 0)
            w_non_peak = self.non_peak_weights.get(name, 0)

            weight = peak_weight_map * w_peak + non_peak_weight_map * w_non_peak
            ensemble_pred += predictions[name] * weight
            weight_sum += weight

        return ensemble_pred / (weight_sum + 1e-8)


class TimeDependentEnsemble:
    """
    时段依赖的动态权重集成

    核心思想：
    对于峰值附近的每个偏移位置(-6h到+6h)，统计各模型的历史表现，
    并使用逆方差加权，使得在该位置表现好的模型获得更高权重。

    Example:
        >>> ensemble = TimeDependentEnsemble(max_offset=6)
        >>> ensemble.calibrate_time_weights(predictions_val, y_val, is_daytime_val)
        >>> y_pred = ensemble.predict(predictions_test, is_daytime_test)
    """

    def __init__(
        self,
        hours_per_day: int = 24,
        max_offset: int = 6
    ):
        """
        Args:
            hours_per_day: 每天小时数
            max_offset: 相对于峰值的最大偏移量（±max_offset小时）
        """
        self.hours_per_day = hours_per_day
        self.max_offset = max_offset
        self.peak_extractor = PeakExtractor(hours_per_day)

        # 时段依赖权重: {model_name: {offset: weight}}
        self.time_weights = {}
        # 基础权重（用于峰值窗口外）
        self.base_weights = {}

    def calibrate_time_weights(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        is_daytime: np.ndarray,
        verbose: bool = True
    ):
        """
        在验证集上校准时段依赖权重

        Args:
            predictions: 各模型预测
            y_true: 真实值
            is_daytime: 白天标志
            verbose: 是否打印信息
        """
        model_names = list(predictions.keys())
        num_samples, seq_len = y_true.shape
        num_days = seq_len // self.hours_per_day

        # 1. 提取真实峰值位置
        peak_info = self.peak_extractor.extract_daily_peaks(y_true, y_true, is_daytime)
        true_peak_times = peak_info['true_peak_times']

        # 2. 收集各偏移位置的误差
        offset_errors = {
            m: {dt: [] for dt in range(-self.max_offset, self.max_offset + 1)}
            for m in model_names
        }

        for i in range(num_samples):
            for day in range(num_days):
                peak_hour = int(true_peak_times[i, day])
                day_start = day * self.hours_per_day

                for dt in range(-self.max_offset, self.max_offset + 1):
                    target_hour = peak_hour + dt
                    if 0 <= target_hour < self.hours_per_day:
                        global_idx = day_start + target_hour
                        if global_idx < seq_len:
                            true_val = y_true[i, global_idx]
                            for m in model_names:
                                pred_val = predictions[m][i, global_idx]
                                offset_errors[m][dt].append((pred_val - true_val) ** 2)

        # 3. 计算RMSE和逆方差权重
        for m in model_names:
            self.time_weights[m] = {}
            for dt in range(-self.max_offset, self.max_offset + 1):
                if offset_errors[m][dt]:
                    rmse = np.sqrt(np.mean(offset_errors[m][dt]))
                    self.time_weights[m][dt] = 1.0 / (rmse ** 2 + 1e-6)
                else:
                    self.time_weights[m][dt] = 1e-6

        # 4. 归一化（每个偏移位置的权重和为1）
        for dt in range(-self.max_offset, self.max_offset + 1):
            total = sum(self.time_weights[m][dt] for m in model_names)
            for m in model_names:
                self.time_weights[m][dt] /= (total + 1e-8)

        # 5. 计算基础权重（用于峰值窗口外）
        overall_rmses = {}
        for m in model_names:
            overall_rmses[m] = np.sqrt(np.mean((predictions[m] - y_true) ** 2))

        total_inv = sum(1.0 / (r ** 2 + 1e-6) for r in overall_rmses.values())
        for m in model_names:
            self.base_weights[m] = (1.0 / (overall_rmses[m] ** 2 + 1e-6)) / total_inv

        if verbose:
            print("\n" + "=" * 70)
            print("[TimeDependentEnsemble] 时段依赖权重校准完成")
            print("=" * 70)
            print(f"偏移范围: ±{self.max_offset}小时")
            print(f"\n时段权重表:")
            print(f"{'Offset':<8}", end='')
            for m in model_names:
                print(f"{m:<15}", end='')
            print()
            print("-" * (8 + 15 * len(model_names)))
            for dt in range(-self.max_offset, self.max_offset + 1):
                print(f"{dt:>+3}h     ", end='')
                for m in model_names:
                    print(f"{self.time_weights[m][dt]:<15.4f}", end='')
                print()
            print(f"\n基础权重（峰值窗口外）:")
            for m in model_names:
                print(f"  {m}: {self.base_weights[m]:.4f}")

    def predict(
        self,
        predictions: Dict[str, np.ndarray],
        is_daytime: np.ndarray
    ) -> np.ndarray:
        """
        使用时段依赖权重进行预测
        """
        model_names = list(predictions.keys())
        num_samples, seq_len = next(iter(predictions.values())).shape
        num_days = seq_len // self.hours_per_day

        # 1. 用基础权重初步预测，检测峰值位置
        temp_pred = sum(predictions[m] * self.base_weights.get(m, 1/len(model_names))
                       for m in model_names)

        peak_info = self.peak_extractor.extract_daily_peaks(temp_pred, temp_pred, is_daytime)
        pred_peak_times = peak_info['true_peak_times']

        # 2. 应用时段依赖权重
        ensemble_pred = np.zeros((num_samples, seq_len))

        for i in range(num_samples):
            for day in range(num_days):
                peak_hour = int(pred_peak_times[i, day])
                day_start = day * self.hours_per_day

                for h in range(self.hours_per_day):
                    global_idx = day_start + h
                    if global_idx >= seq_len:
                        continue

                    dt = h - peak_hour

                    # 在峰值窗口内使用时段权重，否则使用基础权重
                    if -self.max_offset <= dt <= self.max_offset:
                        for m in model_names:
                            weight = self.time_weights[m].get(dt, self.base_weights.get(m, 0))
                            ensemble_pred[i, global_idx] += weight * predictions[m][i, global_idx]
                    else:
                        for m in model_names:
                            ensemble_pred[i, global_idx] += self.base_weights.get(m, 0) * predictions[m][i, global_idx]

        return ensemble_pred


def find_optimal_window_size(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    is_daytime: np.ndarray,
    candidate_windows: List[int] = None,
    metric_weights: Dict[str, float] = None,
    verbose: bool = True
) -> Tuple[int, float]:
    """
    搜索最优峰值窗口大小

    Args:
        predictions: 各模型预测
        y_true: 真实值
        is_daytime: 白天标志
        candidate_windows: 候选窗口大小列表
        metric_weights: 指标权重 {'peak_rmse': w1, 'timing_accuracy': w2}
        verbose: 是否打印信息

    Returns:
        best_window: 最优窗口大小
        best_score: 最优得分
    """
    if candidate_windows is None:
        candidate_windows = [1, 2, 3, 4, 5]
    if metric_weights is None:
        metric_weights = {'peak_rmse': 0.6, 'timing_accuracy': 0.4}

    peak_extractor = PeakExtractor()
    best_window = 2
    best_score = float('inf')
    results = []

    if verbose:
        print("\n" + "=" * 70)
        print("[find_optimal_window_size] 搜索最优峰值窗口")
        print("=" * 70)

    for window in candidate_windows:
        # 创建临时集成器
        temp_ensemble = EnhancedPeakAwareEnsemble(
            peak_window_size=window,
            soft_transition_sigma=window * 0.5  # sigma与窗口成比例
        )
        temp_ensemble.calibrate(predictions, y_true, is_daytime, verbose=False)

        # 预测
        ensemble_pred = temp_ensemble.predict(predictions, is_daytime)

        # 评估峰值指标
        peak_info = peak_extractor.extract_daily_peaks(y_true, ensemble_pred, is_daytime)

        # 综合得分（越低越好）
        score = (
            metric_weights['peak_rmse'] * peak_info['peak_value_rmse'] +
            metric_weights['timing_accuracy'] * (1 - peak_info['peak_time_within_1h']) * 100
        )

        results.append({
            'window': window,
            'peak_rmse': peak_info['peak_value_rmse'],
            'peak_mae': peak_info['peak_value_mae'],
            'time_mae': peak_info['peak_time_mae'],
            'within_1h': peak_info['peak_time_within_1h'],
            'score': score
        })

        if score < best_score:
            best_score = score
            best_window = window

    if verbose:
        print(f"\n{'Window':<10} {'Peak RMSE':<12} {'Time MAE':<10} {'±1h率':<10} {'得分':<10}")
        print("-" * 55)
        for r in results:
            marker = " ✓" if r['window'] == best_window else ""
            print(f"{r['window']:<10} {r['peak_rmse']:<12.4f} {r['time_mae']:<10.2f} "
                  f"{r['within_1h']*100:<10.1f}% {r['score']:<10.4f}{marker}")
        print(f"\n最优窗口: {best_window}h (得分: {best_score:.4f})")

    return best_window, best_score


class KFoldCalibratedEnsemble:
    """
    K折交叉验证权重校准集成

    使用K折交叉验证获得更稳健的集成权重，避免过拟合单一验证集。

    Example:
        >>> ensemble = KFoldCalibratedEnsemble(n_folds=5)
        >>> ensemble.calibrate_with_kfold(predictions_val, y_val, is_daytime_val)
        >>> y_pred = ensemble.predict(predictions_test, is_daytime_test)
    """

    def __init__(
        self,
        n_folds: int = 5,
        base_ensemble_class=None,
        **ensemble_kwargs
    ):
        """
        Args:
            n_folds: 折数
            base_ensemble_class: 基础集成类（默认EnhancedPeakAwareEnsemble）
            **ensemble_kwargs: 传递给基础集成类的参数
        """
        self.n_folds = n_folds
        self.base_ensemble_class = base_ensemble_class or EnhancedPeakAwareEnsemble
        self.ensemble_kwargs = ensemble_kwargs

        self.final_peak_weights = {}
        self.final_non_peak_weights = {}
        self.base_ensemble = None

    def calibrate_with_kfold(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        is_daytime: np.ndarray,
        verbose: bool = True
    ):
        """
        使用K折交叉验证校准权重
        """
        from sklearn.model_selection import KFold

        n_samples = y_true.shape[0]
        model_names = list(predictions.keys())
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        all_peak_weights = []
        all_non_peak_weights = []
        fold_performances = []

        if verbose:
            print("\n" + "=" * 70)
            print(f"[KFoldCalibratedEnsemble] {self.n_folds}折交叉验证校准")
            print("=" * 70)

        for fold_idx, (_, val_idx) in enumerate(kf.split(range(n_samples))):
            # 提取该折数据
            fold_preds = {m: pred[val_idx] for m, pred in predictions.items()}
            fold_y = y_true[val_idx]
            fold_daytime = is_daytime[val_idx]

            # 创建并校准该折的集成器
            fold_ensemble = self.base_ensemble_class(**self.ensemble_kwargs)
            fold_ensemble.calibrate(fold_preds, fold_y, fold_daytime, verbose=False)

            # 评估该折性能
            fold_pred = fold_ensemble.predict(fold_preds, fold_daytime)
            fold_rmse = np.sqrt(np.mean((fold_pred - fold_y) ** 2))

            # 存储权重
            all_peak_weights.append(fold_ensemble.peak_weights.copy())
            all_non_peak_weights.append(fold_ensemble.non_peak_weights.copy())
            fold_performances.append(1.0 / (fold_rmse + 1e-6))

            if verbose:
                print(f"  Fold {fold_idx + 1}: RMSE = {fold_rmse:.4f}")

        # 性能加权平均
        total_perf = sum(fold_performances)
        fold_weights_normalized = [p / total_perf for p in fold_performances]

        # 计算最终权重
        for m in model_names:
            self.final_peak_weights[m] = sum(
                w * fold_w[m]
                for w, fold_w in zip(fold_weights_normalized, all_peak_weights)
            )
            self.final_non_peak_weights[m] = sum(
                w * fold_w[m]
                for w, fold_w in zip(fold_weights_normalized, all_non_peak_weights)
            )

        # 创建最终集成器
        self.base_ensemble = self.base_ensemble_class(**self.ensemble_kwargs)
        self.base_ensemble.peak_weights = self.final_peak_weights
        self.base_ensemble.non_peak_weights = self.final_non_peak_weights

        if verbose:
            print(f"\n最终校准权重（{self.n_folds}折平均）:")
            print(f"{'模型':<20} {'峰值权重':<12} {'非峰值权重':<12}")
            print("-" * 45)
            for m in model_names:
                print(f"{m:<20} {self.final_peak_weights[m]:<12.4f} "
                      f"{self.final_non_peak_weights[m]:<12.4f}")

    def predict(
        self,
        predictions: Dict[str, np.ndarray],
        is_daytime: np.ndarray
    ) -> np.ndarray:
        """使用校准后的权重预测"""
        if self.base_ensemble is None:
            raise ValueError("请先调用 calibrate_with_kfold()")
        return self.base_ensemble.predict(predictions, is_daytime)


class PeriodBasedEnsemble:
    """
    分时段权重集成

    根据时间段（夜间/白天/峰值时刻）使用不同的模型权重。
    这种方法假设不同模型在不同时段有不同的优势。

    时段定义:
    - 夜间 (night): 0-5h, 19-23h
    - 白天 (daytime): 6-18h (排除峰值时刻)
    - 峰值时刻 (peak): 10-14h (典型光伏峰值时段)

    Example:
        >>> ensemble = PeriodBasedEnsemble()
        >>> ensemble.calibrate(predictions_val, y_val, is_daytime_val)
        >>> y_pred = ensemble.predict(predictions_test, is_daytime_test)
    """

    def __init__(
        self,
        hours_per_day: int = 24,
        night_hours: tuple = (0, 5, 19, 23),  # 0-5h 和 19-23h
        peak_hours: tuple = (10, 14),  # 峰值时刻 10-14h
        temperature: float = 0.1  # softmax 温度（越小权重差异越大）
    ):
        """
        Args:
            hours_per_day: 每天小时数
            night_hours: 夜间时段 (start1, end1, start2, end2)
            peak_hours: 峰值时段 (start, end)
            temperature: softmax 温度
        """
        self.hours_per_day = hours_per_day
        self.night_hours = night_hours
        self.peak_hours = peak_hours
        self.temperature = temperature

        # 各时段的模型权重
        self.period_weights = {
            'night': {},
            'daytime': {},
            'peak': {}
        }

    def _get_period(self, hour: int) -> str:
        """判断时刻属于哪个时段"""
        # 夜间
        if hour <= self.night_hours[1] or hour >= self.night_hours[2]:
            return 'night'
        # 峰值时刻
        elif self.peak_hours[0] <= hour <= self.peak_hours[1]:
            return 'peak'
        # 白天（非峰值）
        else:
            return 'daytime'

    def calibrate(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        is_daytime: np.ndarray,
        verbose: bool = True
    ):
        """
        在验证集上校准分时段权重

        Args:
            predictions: 各模型预测
            y_true: 真实值
            is_daytime: 白天标志
            verbose: 是否打印信息
        """
        model_names = list(predictions.keys())
        num_samples, seq_len = y_true.shape
        num_days = seq_len // self.hours_per_day

        # 收集各时段的误差
        period_errors = {
            period: {m: [] for m in model_names}
            for period in ['night', 'daytime', 'peak']
        }

        for i in range(num_samples):
            for day in range(num_days):
                day_start = day * self.hours_per_day
                for h in range(self.hours_per_day):
                    global_idx = day_start + h
                    if global_idx >= seq_len:
                        continue

                    period = self._get_period(h)
                    true_val = y_true[i, global_idx]

                    for m in model_names:
                        pred_val = predictions[m][i, global_idx]
                        period_errors[period][m].append((pred_val - true_val) ** 2)

        # 计算各时段的 RMSE 和权重
        for period in ['night', 'daytime', 'peak']:
            rmses = {}
            for m in model_names:
                if period_errors[period][m]:
                    rmses[m] = np.sqrt(np.mean(period_errors[period][m]))
                else:
                    rmses[m] = 1e6  # 无数据时给很大的 RMSE

            # 使用 softmax 计算权重（RMSE 越小权重越大）
            inv_rmses = np.array([1.0 / (rmses[m] + 1e-6) for m in model_names])
            weights = np.exp(inv_rmses / self.temperature)
            weights = weights / weights.sum()

            for idx, m in enumerate(model_names):
                self.period_weights[period][m] = weights[idx]

        if verbose:
            print("\n" + "=" * 70)
            print("[PeriodBasedEnsemble] 分时段权重校准完成")
            print("=" * 70)
            print(f"时段定义:")
            print(f"  夜间: {self.night_hours[0]}-{self.night_hours[1]}h, {self.night_hours[2]}-{self.night_hours[3]}h")
            print(f"  峰值: {self.peak_hours[0]}-{self.peak_hours[1]}h")
            print(f"  白天: 其他时段")
            print(f"\n各时段权重:")
            print(f"{'时段':<12}", end='')
            for m in model_names:
                print(f"{m:<15}", end='')
            print()
            print("-" * (12 + 15 * len(model_names)))
            for period in ['night', 'daytime', 'peak']:
                print(f"{period:<12}", end='')
                for m in model_names:
                    print(f"{self.period_weights[period][m]:<15.4f}", end='')
                print()

    def predict(
        self,
        predictions: Dict[str, np.ndarray],
        is_daytime: np.ndarray
    ) -> np.ndarray:
        """
        使用分时段权重进行预测
        """
        model_names = list(predictions.keys())
        num_samples, seq_len = next(iter(predictions.values())).shape
        num_days = seq_len // self.hours_per_day

        ensemble_pred = np.zeros((num_samples, seq_len))

        for i in range(num_samples):
            for day in range(num_days):
                day_start = day * self.hours_per_day
                for h in range(self.hours_per_day):
                    global_idx = day_start + h
                    if global_idx >= seq_len:
                        continue

                    period = self._get_period(h)

                    for m in model_names:
                        weight = self.period_weights[period].get(m, 1.0 / len(model_names))
                        ensemble_pred[i, global_idx] += weight * predictions[m][i, global_idx]

        return ensemble_pred


class CombinedPeriodTimeEnsemble:
    """
    组合集成：分时段权重 + 时段依赖权重

    结合 PeriodBasedEnsemble 和 TimeDependentEnsemble 的优势：
    - 在峰值窗口外：使用分时段权重（夜间/白天/峰值）
    - 在峰值窗口内：使用时段依赖权重（基于偏移位置的精细权重）
    """

    def __init__(
        self,
        hours_per_day: int = 24,
        max_offset: int = 6,
        peak_hours: tuple = (10, 14),
        blend_alpha: float = 0.5  # 混合系数（0=纯分时段，1=纯时段依赖）
    ):
        self.hours_per_day = hours_per_day
        self.max_offset = max_offset
        self.peak_hours = peak_hours
        self.blend_alpha = blend_alpha

        self.period_ensemble = PeriodBasedEnsemble(
            hours_per_day=hours_per_day,
            peak_hours=peak_hours
        )
        self.time_ensemble = TimeDependentEnsemble(
            hours_per_day=hours_per_day,
            max_offset=max_offset
        )
        self.peak_extractor = PeakExtractor(hours_per_day)

    def calibrate(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        is_daytime: np.ndarray,
        verbose: bool = True
    ):
        """校准两种权重"""
        if verbose:
            print("\n" + "=" * 70)
            print("[CombinedPeriodTimeEnsemble] 组合集成校准")
            print("=" * 70)

        # 校准分时段权重
        self.period_ensemble.calibrate(predictions, y_true, is_daytime, verbose=verbose)

        # 校准时段依赖权重
        self.time_ensemble.calibrate_time_weights(predictions, y_true, is_daytime, verbose=verbose)

    def predict(
        self,
        predictions: Dict[str, np.ndarray],
        is_daytime: np.ndarray
    ) -> np.ndarray:
        """
        组合预测

        在峰值窗口内使用 blend_alpha 混合两种权重
        在峰值窗口外使用分时段权重
        """
        model_names = list(predictions.keys())
        num_samples, seq_len = next(iter(predictions.values())).shape
        num_days = seq_len // self.hours_per_day

        # 获取分时段预测
        period_pred = self.period_ensemble.predict(predictions, is_daytime)

        # 获取时段依赖预测
        time_pred = self.time_ensemble.predict(predictions, is_daytime)

        # 提取预测峰值位置
        peak_info = self.peak_extractor.extract_daily_peaks(period_pred, period_pred, is_daytime)
        pred_peak_times = peak_info['true_peak_times']

        # 组合预测
        ensemble_pred = np.zeros((num_samples, seq_len))

        for i in range(num_samples):
            for day in range(num_days):
                peak_hour = int(pred_peak_times[i, day])
                day_start = day * self.hours_per_day

                for h in range(self.hours_per_day):
                    global_idx = day_start + h
                    if global_idx >= seq_len:
                        continue

                    dt = abs(h - peak_hour)

                    # 在峰值窗口内混合两种预测
                    if dt <= self.max_offset:
                        # 距离峰值越近，时段依赖权重占比越高
                        alpha = self.blend_alpha * (1 - dt / self.max_offset)
                        ensemble_pred[i, global_idx] = (
                            alpha * time_pred[i, global_idx] +
                            (1 - alpha) * period_pred[i, global_idx]
                        )
                    else:
                        # 峰值窗口外使用分时段预测
                        ensemble_pred[i, global_idx] = period_pred[i, global_idx]

        return ensemble_pred


class PeakResidualCorrector:
    """
    峰值残差校正器

    训练专门的后处理模型校正峰值预测残差。
    使用LightGBM对每天的峰值误差进行学习和校正。

    特征：
    - 预测峰值大小
    - 预测峰值时刻
    - 模型间分歧度（标准差）
    - 峰值尖锐度

    Example:
        >>> corrector = PeakResidualCorrector()
        >>> corrector.fit(predictions_val, ensemble_pred_val, y_val, is_daytime_val)
        >>> corrected = corrector.correct(predictions_test, ensemble_pred_test, is_daytime_test)
    """

    def __init__(
        self,
        hours_per_day: int = 24,
        correction_sigma: float = 2.0,
        n_estimators: int = 50,
        max_depth: int = 4
    ):
        """
        Args:
            hours_per_day: 每天小时数
            correction_sigma: 校正量的高斯平滑sigma
            n_estimators: LightGBM树数量
            max_depth: 树最大深度
        """
        self.hours_per_day = hours_per_day
        self.correction_sigma = correction_sigma
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.peak_extractor = PeakExtractor(hours_per_day)
        self.corrector_models = []  # 每天一个模型

    def _extract_features(
        self,
        predictions: Dict[str, np.ndarray],
        ensemble_pred: np.ndarray,
        is_daytime: np.ndarray
    ) -> List[np.ndarray]:
        """
        提取峰值校正特征

        Returns:
            features_by_day: 每天的特征数组列表
        """
        batch_size, seq_len = ensemble_pred.shape
        num_days = seq_len // self.hours_per_day
        model_names = list(predictions.keys())

        # 提取峰值信息
        peak_info = self.peak_extractor.extract_daily_peaks(
            ensemble_pred, ensemble_pred, is_daytime
        )

        features_by_day = []
        for day in range(num_days):
            day_start = day * self.hours_per_day
            features = []

            # 特征1: 预测峰值大小
            features.append(peak_info['true_peak_values'][:, day:day+1])

            # 特征2: 预测峰值时刻
            features.append(peak_info['true_peak_times'][:, day:day+1].astype(float))

            # 特征3: 模型间分歧度
            model_peaks = []
            for m in model_names:
                day_pred = predictions[m][:, day_start:day_start+self.hours_per_day]
                model_peaks.append(day_pred.max(axis=1))
            disagreement = np.std(model_peaks, axis=0)
            features.append(disagreement[:, np.newaxis])

            # 特征4: 峰值尖锐度
            day_ensemble = ensemble_pred[:, day_start:day_start+self.hours_per_day]
            sharpness = []
            for i in range(batch_size):
                ph = int(peak_info['true_peak_times'][i, day])
                if 1 <= ph <= self.hours_per_day - 2:
                    curv = day_ensemble[i, ph-1] + day_ensemble[i, ph+1] - 2 * day_ensemble[i, ph]
                    sharpness.append(-curv)
                else:
                    sharpness.append(0)
            features.append(np.array(sharpness)[:, np.newaxis])

            features_by_day.append(np.hstack(features))

        return features_by_day

    def fit(
        self,
        predictions: Dict[str, np.ndarray],
        ensemble_pred: np.ndarray,
        y_true: np.ndarray,
        is_daytime: np.ndarray,
        verbose: bool = True
    ):
        """
        训练峰值残差校正模型
        """
        import lightgbm as lgb

        seq_len = y_true.shape[1]
        num_days = seq_len // self.hours_per_day

        # 提取特征
        features_by_day = self._extract_features(predictions, ensemble_pred, is_daytime)

        # 提取真实和预测峰值
        true_peak_info = self.peak_extractor.extract_daily_peaks(y_true, y_true, is_daytime)
        pred_peak_info = self.peak_extractor.extract_daily_peaks(y_true, ensemble_pred, is_daytime)

        if verbose:
            print("\n" + "=" * 70)
            print("[PeakResidualCorrector] 训练峰值残差校正器")
            print("=" * 70)

        self.corrector_models = []

        for day in range(num_days):
            X = features_by_day[day]
            residuals = (
                true_peak_info['true_peak_values'][:, day] -
                pred_peak_info['pred_peak_values'][:, day]
            )

            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.1,
                verbose=-1,
                n_jobs=-1
            )
            model.fit(X, residuals)
            self.corrector_models.append(model)

            if verbose:
                # 评估校正效果
                pred_correction = model.predict(X)
                mae_before = np.mean(np.abs(residuals))
                mae_after = np.mean(np.abs(residuals - pred_correction))
                print(f"  Day {day+1}: MAE {mae_before:.2f} -> {mae_after:.2f} kW")

    def correct(
        self,
        predictions: Dict[str, np.ndarray],
        ensemble_pred: np.ndarray,
        is_daytime: np.ndarray
    ) -> np.ndarray:
        """
        应用残差校正
        """
        batch_size, seq_len = ensemble_pred.shape
        num_days = seq_len // self.hours_per_day

        # 提取特征
        features_by_day = self._extract_features(predictions, ensemble_pred, is_daytime)

        # 提取预测峰值位置
        peak_info = self.peak_extractor.extract_daily_peaks(
            ensemble_pred, ensemble_pred, is_daytime
        )

        corrected_pred = ensemble_pred.copy()

        for day in range(num_days):
            if day >= len(self.corrector_models):
                continue

            X = features_by_day[day]
            corrections = self.corrector_models[day].predict(X)

            day_start = day * self.hours_per_day

            for i in range(batch_size):
                peak_hour = int(peak_info['true_peak_times'][i, day])
                correction = corrections[i]

                # 使用高斯窗口平滑校正量
                for h in range(self.hours_per_day):
                    global_idx = day_start + h
                    if global_idx >= seq_len:
                        continue

                    distance = abs(h - peak_hour)
                    weight = np.exp(-0.5 * (distance / self.correction_sigma) ** 2)
                    corrected_pred[i, global_idx] += correction * weight

        return corrected_pred
