#!/usr/bin/env python
"""
任务4峰值预测版: 720小时 → 168小时 + 树模型峰值预测 + MLEF集成

核心改进:
1. RNN模型（Mamba/LSTM/GRU）: 逐小时预测，擅长底部拟合
2. 树模型（LightGBM/XGBoost）: 峰值预测，弥补RNN峰值不足
3. MLEF集成: 自动学习不同时段的最优模型组合
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from datetime import datetime
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from src.mtm_mlef.config import load_config
from src.mtm_mlef.data_processing import (
    load_data,
    filter_data_by_months,
    create_hourly_sequences_for_weekly_prediction,
    extract_statistical_features,
    extract_peak_hour_features,
    extract_improved_peak_features,  # 改进的峰值特征
    create_peak_targets
)
from src.mtm_mlef.metrics import (
    calculate_seq2seq_metrics,
    calculate_regression_metrics,
    calculate_multiclass_metrics
)
from src.mtm_mlef.models.seq2seq_models import MambaSeq2Seq, LSTMSeq2Seq, GRUSeq2Seq, TransformerSeq2Seq
from src.mtm_mlef.models.ensemble import (
    peak_to_hourly_prediction,
    mlef_confidence_weighted,
    mlef_ridge_ensemble
)
from src.mtm_mlef.training import predict_seq2seq
from src.mtm_mlef.trainers import (
    train_rnn_model,
    train_peak_predictor,
    train_peak_hour_predictor,
    train_peak_hour_regressor,  # 峰值时刻回归
    train_peak_hour_coarse_classifier  # 粗粒度分类
)
from src.mtm_mlef.checkpoint_utils import CheckpointManager
from src.mtm_mlef.visualization import (
    set_non_interactive_backend,
    plot_seq2seq_predictions,
    plot_regression_metrics,
    plot_classification_metrics,
    plot_prediction_comparison
)
from src.mtm_mlef.utils import set_random_seed, GPUMemoryManager, print_gpu_info
from src.mtm_mlef.utils.peak_extractor import (
    PeakExtractor,
    analyze_peak_error_patterns,
    compare_models_peak_performance
)
from src.mtm_mlef.utils.peak_viz import (
    PeakVisualizer,
    create_peak_evaluation_report
)
from src.mtm_mlef.models.ensemble import (
    EnhancedPeakAwareEnsemble,
    AdaptiveGaussianEnsemble,
    TimeDependentEnsemble,
    KFoldCalibratedEnsemble,
    PeakResidualCorrector,
    PeriodBasedEnsemble,
    CombinedPeriodTimeEnsemble,
    find_optimal_window_size
)

# 设置matplotlib非交互模式
set_non_interactive_backend()


def main():
    parser = argparse.ArgumentParser(description='任务4峰值预测+MLEF集成')
    parser.add_argument('--epochs', type=int, default=50, help='RNN训练轮次')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--seed', type=int, default=16, help='随机种子(默认42)')
    parser.add_argument('--skip-advanced-ensemble', action='store_true',
                        help='跳过高级集成方法(阶段5-6)以加快训练速度')
    parser.add_argument('--input-len', type=int, default=720,
                        help='输入序列长度(默认720h=30天，可选360h/1080h)')
    parser.add_argument('--save-history', action='store_true',
                        help='保存训练历史(用于绘制训练曲线)')
    args = parser.parse_args()

    # 设置随机种子确保结果可复现
    set_random_seed(args.seed)

    print("=" * 80)
    print("任务4峰值预测版: RNN底部拟合 + 树模型峰值预测 + MLEF集成")
    print("=" * 80)
    print(f"随机种子: {args.seed}")

    # 加载配置
    config = load_config('config.yaml')
    task_config = config.tasks['task4_hourly']

    # 支持命令行参数覆盖输入长度
    input_len = args.input_len if args.input_len else task_config['input_len']
    output_len = task_config['output_len']
    test_months = task_config['test_months']
    train_months = task_config['train_months']

    print(f"输入序列长度: {input_len}h ({input_len//24}天)")
    print(f"输出序列长度: {output_len}h ({output_len//24}天)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[设备] {device}")

    # 初始化GPU显存管理器
    gpu_manager = GPUMemoryManager(
        target_utilization=0.90,  # 目标使用90%显存
        safety_margin_gb=1.0,     # 预留1GB安全余量
        verbose=True
    )
    print_gpu_info()

    # 加载数据
    print("\n[数据加载]")
    train_2016 = load_data('data/combined_pv_data2016.csv')
    train_2017 = load_data('data/combined_pv_data2017.csv')
    train_2016 = filter_data_by_months(train_2016, include_months=train_months)
    train_2017 = filter_data_by_months(train_2017, include_months=train_months)
    train_data = pd.concat([train_2016, train_2017], ignore_index=True)

    test_2016 = filter_data_by_months(load_data('data/combined_pv_data2016.csv'), include_months=test_months)
    test_2017 = filter_data_by_months(load_data('data/combined_pv_data2017.csv'), include_months=test_months)
    test_2018 = filter_data_by_months(load_data('data/combined_pv_data2018.csv'), include_months=test_months)
    test_data = pd.concat([test_2016, test_2017, test_2018], ignore_index=True)

    # 创建序列
    print("\n[创建序列]")
    target_col = config.data['target_col']
    X_train_seq, y_train_seq, _ = create_hourly_sequences_for_weekly_prediction(
        train_data, target_col=target_col, input_hours=input_len, output_hours=output_len,
        add_time_features=True, check_continuity=False
    )
    X_test, y_test, _ = create_hourly_sequences_for_weekly_prediction(
        test_data, target_col=target_col, input_hours=input_len, output_hours=output_len,
        add_time_features=True, check_continuity=False
    )

    # 划分训练/验证
    n_train = int(len(X_train_seq) * 0.9)
    X_train, y_train = X_train_seq[:n_train], y_train_seq[:n_train]
    X_val, y_val = X_train_seq[n_train:], y_train_seq[n_train:]

    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")

    # 归一化
    n_features = X_train.shape[2]
    scaler_X = MinMaxScaler()
    scaler_X.fit(X_train.reshape(-1, n_features))

    X_train_norm = scaler_X.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_norm = scaler_X.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_norm = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    scaler_y = MinMaxScaler()
    y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val_norm = scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    # 创建DataLoader（后续会根据模型动态调整batch_size）
    # 这里先用默认值创建，实际训练时会重新创建
    default_batch_size = args.batch_size
    train_dataset = TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train_norm))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_norm), torch.FloatTensor(y_val_norm))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_norm), torch.FloatTensor(y_test_norm))

    def create_loaders(batch_size):
        """创建DataLoader，支持动态batch_size"""
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        return train_loader, val_loader, test_loader

    # 初始化默认loader（用于后续树模型等）
    train_loader, val_loader, test_loader = create_loaders(default_batch_size)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/task4_peak_mlef_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    predictions_val = {}
    predictions_test = {}
    results = {}
    training_histories = {}  # 保存训练历史
    training_times = {}  # 保存训练时间
    inference_times = {}  # 保存推理时间

    import time

    # ========== 阶段1: 训练RNN模型（底部拟合） ==========
    print("\n" + "=" * 80)
    print("[阶段1] 训练RNN模型（逐小时预测）")
    print("=" * 80)

    mamba_config = config.models['mamba_seq2seq_long']

    # Mamba 使用标准 MSE 损失（峰值损失效果不佳，靠 MLEF 集成修正）

    # 统一使用命令行指定的batch_size
    gpu_manager.clear_memory()
    unified_batch_size = args.batch_size
    print(f"  [统一] 所有模型使用batch_size: {unified_batch_size}")
    train_loader, val_loader, test_loader = create_loaders(unified_batch_size)

    mamba_start_time = time.time()
    if args.save_history:
        mamba, mamba_history = train_rnn_model(
            MambaSeq2Seq, "Mamba", n_features, input_len, output_len,
            mamba_config, train_loader, val_loader, device, output_dir,
            loss_type='mse', return_history=True
        )
        training_histories['Mamba'] = mamba_history
    else:
        mamba = train_rnn_model(
            MambaSeq2Seq, "Mamba", n_features, input_len, output_len,
            mamba_config, train_loader, val_loader, device, output_dir,
            loss_type='mse'
        )
    mamba_train_time = time.time() - mamba_start_time
    training_times['Mamba'] = mamba_train_time
    print(f"  Mamba 训练时间: {mamba_train_time:.1f}s ({mamba_train_time/60:.1f}min)")

    infer_start = time.time()
    y_val_pred_norm = predict_seq2seq(mamba, val_loader, device)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_norm.reshape(-1, 1)).reshape(-1, output_len)
    predictions_val['Mamba'] = y_val_pred

    y_test_pred_norm = predict_seq2seq(mamba, test_loader, device)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_norm.reshape(-1, 1)).reshape(-1, output_len)
    predictions_test['Mamba'] = y_test_pred
    inference_times['Mamba'] = time.time() - infer_start

    metrics = calculate_seq2seq_metrics(y_test, y_test_pred, 'Mamba', horizons=[1, 24, 72, 168])
    results['Mamba'] = metrics
    print(f"  Mamba R²: {metrics['Overall_R²']:.4f}")

    del mamba
    gpu_manager.clear_memory()

    # LSTM
    lstm_config = config.models['lstm_seq2seq_long']

    lstm_start_time = time.time()
    if args.save_history:
        lstm, lstm_history = train_rnn_model(
            LSTMSeq2Seq, "LSTM", n_features, input_len, output_len,
            lstm_config, train_loader, val_loader, device, output_dir,
            return_history=True
        )
        training_histories['LSTM'] = lstm_history
    else:
        lstm = train_rnn_model(
            LSTMSeq2Seq, "LSTM", n_features, input_len, output_len,
            lstm_config, train_loader, val_loader, device, output_dir
        )
    lstm_train_time = time.time() - lstm_start_time
    training_times['LSTM'] = lstm_train_time
    print(f"  LSTM 训练时间: {lstm_train_time:.1f}s ({lstm_train_time/60:.1f}min)")

    infer_start = time.time()
    y_val_pred_norm = predict_seq2seq(lstm, val_loader, device)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_norm.reshape(-1, 1)).reshape(-1, output_len)
    predictions_val['LSTM'] = y_val_pred

    y_test_pred_norm = predict_seq2seq(lstm, test_loader, device)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_norm.reshape(-1, 1)).reshape(-1, output_len)
    predictions_test['LSTM'] = y_test_pred
    inference_times['LSTM'] = time.time() - infer_start

    metrics = calculate_seq2seq_metrics(y_test, y_test_pred, 'LSTM', horizons=[1, 24, 72, 168])
    results['LSTM'] = metrics
    print(f"  LSTM R²: {metrics['Overall_R²']:.4f}")

    del lstm
    gpu_manager.clear_memory()

    # GRU
    gru_config = config.models['gru_seq2seq_long']

    gru_start_time = time.time()
    if args.save_history:
        gru, gru_history = train_rnn_model(
            GRUSeq2Seq, "GRU", n_features, input_len, output_len,
            gru_config, train_loader, val_loader, device, output_dir,
            return_history=True
        )
        training_histories['GRU'] = gru_history
    else:
        gru = train_rnn_model(
            GRUSeq2Seq, "GRU", n_features, input_len, output_len,
            gru_config, train_loader, val_loader, device, output_dir
        )
    gru_train_time = time.time() - gru_start_time
    training_times['GRU'] = gru_train_time
    print(f"  GRU 训练时间: {gru_train_time:.1f}s ({gru_train_time/60:.1f}min)")

    infer_start = time.time()
    y_val_pred_norm = predict_seq2seq(gru, val_loader, device)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_norm.reshape(-1, 1)).reshape(-1, output_len)
    predictions_val['GRU'] = y_val_pred

    y_test_pred_norm = predict_seq2seq(gru, test_loader, device)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_norm.reshape(-1, 1)).reshape(-1, output_len)
    predictions_test['GRU'] = y_test_pred
    inference_times['GRU'] = time.time() - infer_start

    metrics = calculate_seq2seq_metrics(y_test, y_test_pred, 'GRU', horizons=[1, 24, 72, 168])
    results['GRU'] = metrics
    print(f"  GRU R²: {metrics['Overall_R²']:.4f}")

    del gru
    gpu_manager.clear_memory()

    # Transformer
    transformer_config = config.models['transformer_seq2seq_long']

    transformer_start_time = time.time()
    if args.save_history:
        transformer, transformer_history = train_rnn_model(
            TransformerSeq2Seq, "Transformer", n_features, input_len, output_len,
            transformer_config, train_loader, val_loader, device, output_dir,
            return_history=True
        )
        training_histories['Transformer'] = transformer_history
    else:
        transformer = train_rnn_model(
            TransformerSeq2Seq, "Transformer", n_features, input_len, output_len,
            transformer_config, train_loader, val_loader, device, output_dir
        )
    transformer_train_time = time.time() - transformer_start_time
    training_times['Transformer'] = transformer_train_time
    print(f"  Transformer 训练时间: {transformer_train_time:.1f}s ({transformer_train_time/60:.1f}min)")

    infer_start = time.time()
    y_val_pred_norm = predict_seq2seq(transformer, val_loader, device)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_norm.reshape(-1, 1)).reshape(-1, output_len)
    predictions_val['Transformer'] = y_val_pred

    y_test_pred_norm = predict_seq2seq(transformer, test_loader, device)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_norm.reshape(-1, 1)).reshape(-1, output_len)
    predictions_test['Transformer'] = y_test_pred
    inference_times['Transformer'] = time.time() - infer_start

    metrics = calculate_seq2seq_metrics(y_test, y_test_pred, 'Transformer', horizons=[1, 24, 72, 168])
    results['Transformer'] = metrics
    print(f"  Transformer R²: {metrics['Overall_R²']:.4f}")

    del transformer
    gpu_manager.clear_memory()

    # ========== 阶段2: 树模型峰值预测 ==========
    print("\n" + "=" * 80)
    print("[阶段2] 树模型峰值预测")
    print("=" * 80)

    # 提取统计特征（原始3375维）
    print("\n提取统计特征...")
    X_train_stat = extract_statistical_features(X_train_norm)
    X_val_stat = extract_statistical_features(X_val_norm)
    X_test_stat = extract_statistical_features(X_test_norm)
    print(f"  统计特征维度: {X_train_stat.shape[1]}")

    # 创建峰值目标
    print("\n创建峰值目标...")
    y_train_peak, y_train_peak_hour = create_peak_targets(y_train)
    y_val_peak, y_val_peak_hour = create_peak_targets(y_val)
    y_test_peak, y_test_peak_hour = create_peak_targets(y_test)
    print(f"  峰值目标维度: {y_train_peak.shape}")

    # LightGBM峰值预测
    lgbm_config = config.models['lightgbm_peak']
    lgbm_models = train_peak_predictor(
        'lightgbm', X_train_stat, y_train_peak, X_val_stat, y_val_peak,
        lgbm_config, 'LightGBM_Peak', output_dir
    )

    # 预测峰值
    lgbm_val_peak = np.stack([m.predict(X_val_stat) for m in lgbm_models], axis=1)
    lgbm_test_peak = np.stack([m.predict(X_test_stat) for m in lgbm_models], axis=1)

    # 训练峰值时刻分类器（使用轻量级特征）
    print("\n提取峰值时刻预测特征（轻量级）...")
    X_train_hour_feat = extract_peak_hour_features(X_train_norm)
    X_val_hour_feat = extract_peak_hour_features(X_val_norm)
    X_test_hour_feat = extract_peak_hour_features(X_test_norm)
    print(f"  峰值时刻特征维度: {X_train_hour_feat.shape[1]}")

    print("\n训练LightGBM峰值时刻分类器...")
    lgbm_hour_classifiers = train_peak_hour_predictor(
        'lightgbm', X_train_hour_feat, y_train_peak_hour, X_val_hour_feat, y_val_peak_hour,
        lgbm_config, 'LightGBM_PeakHour', output_dir
    )

    # 预测峰值时刻（分类结果，直接使用）
    lgbm_val_peak_hour = np.stack([clf.predict(X_val_hour_feat) for clf in lgbm_hour_classifiers], axis=1)
    lgbm_test_peak_hour = np.stack([clf.predict(X_test_hour_feat) for clf in lgbm_hour_classifiers], axis=1)

    # 转换为逐小时预测（使用Mamba作为基线）
    lgbm_val_hourly = peak_to_hourly_prediction(lgbm_val_peak, predictions_val['Mamba'], lgbm_val_peak_hour)
    lgbm_test_hourly = peak_to_hourly_prediction(lgbm_test_peak, predictions_test['Mamba'], lgbm_test_peak_hour)

    predictions_val['LightGBM_Peak'] = lgbm_val_hourly
    predictions_test['LightGBM_Peak'] = lgbm_test_hourly

    metrics = calculate_seq2seq_metrics(y_test, lgbm_test_hourly, 'LightGBM_Peak', horizons=[1, 24, 72, 168])
    results['LightGBM_Peak'] = metrics
    print(f"\n  LightGBM_Peak R²: {metrics['Overall_R²']:.4f}")

    # XGBoost峰值预测
    xgb_config = config.models['xgboost_peak']
    xgb_models = train_peak_predictor(
        'xgboost', X_train_stat, y_train_peak, X_val_stat, y_val_peak,
        xgb_config, 'XGBoost_Peak', output_dir
    )

    xgb_val_peak = np.stack([m.predict(X_val_stat) for m in xgb_models], axis=1)
    xgb_test_peak = np.stack([m.predict(X_test_stat) for m in xgb_models], axis=1)

    # 训练峰值时刻分类器（使用轻量级特征）
    print("\n训练XGBoost峰值时刻分类器...")
    xgb_hour_classifiers = train_peak_hour_predictor(
        'xgboost', X_train_hour_feat, y_train_peak_hour, X_val_hour_feat, y_val_peak_hour,
        xgb_config, 'XGBoost_PeakHour', output_dir
    )

    # 预测峰值时刻（分类结果，直接使用）
    xgb_val_peak_hour = np.stack([clf.predict(X_val_hour_feat) for clf in xgb_hour_classifiers], axis=1)
    xgb_test_peak_hour = np.stack([clf.predict(X_test_hour_feat) for clf in xgb_hour_classifiers], axis=1)

    xgb_val_hourly = peak_to_hourly_prediction(xgb_val_peak, predictions_val['Mamba'], xgb_val_peak_hour)
    xgb_test_hourly = peak_to_hourly_prediction(xgb_test_peak, predictions_test['Mamba'], xgb_test_peak_hour)

    predictions_val['XGBoost_Peak'] = xgb_val_hourly
    predictions_test['XGBoost_Peak'] = xgb_test_hourly

    metrics = calculate_seq2seq_metrics(y_test, xgb_test_hourly, 'XGBoost_Peak', horizons=[1, 24, 72, 168])
    results['XGBoost_Peak'] = metrics
    print(f"  XGBoost_Peak R²: {metrics['Overall_R²']:.4f}")

    # ========== 阶段3: MLEF集成 ==========
    # 使用动态置信度加权融合（最佳策略）
    mlef_pred, weights_by_hour, mlef_model_names = mlef_confidence_weighted(
        predictions_val, y_val, predictions_test
    )

    predictions_test['MLEF'] = mlef_pred
    metrics = calculate_seq2seq_metrics(y_test, mlef_pred, 'MLEF', horizons=[1, 24, 72, 168])
    results['MLEF'] = metrics
    print(f"\n  MLEF R²: {metrics['Overall_R²']:.4f}")

    # 保存权重矩阵
    np.save(f'{output_dir}/mlef_weights_by_hour.npy', weights_by_hour)
    with open(f'{output_dir}/mlef_model_names.json', 'w') as f:
        json.dump(mlef_model_names, f, indent=2)
    print(f"  权重矩阵已保存: {output_dir}/mlef_weights_by_hour.npy")

    # 保存结果
    print("\n" + "=" * 80)
    print("[保存结果]")
    print("=" * 80)

    df_metrics = pd.DataFrame(results).T
    df_metrics = df_metrics.sort_values('Overall_R²', ascending=False)
    df_metrics.to_csv(f"{output_dir}/metrics.csv")

    # 分类展示结果
    rnn_models = ['Mamba', 'LSTM', 'GRU', 'Transformer']
    peak_fusion_models = ['LightGBM_Peak', 'XGBoost_Peak']
    ensemble_models = ['MLEF']

    print(f"\n" + "=" * 80)
    print("性能对比")
    print("=" * 80)

    print(f"\n1. 时序预测模型 (RNN/Transformer):")
    df_rnn = df_metrics.loc[[m for m in rnn_models if m in df_metrics.index]]
    print(df_rnn[['Overall_R²', 'Overall_RMSE', 'Overall_MAE']].to_string())

    print(f"\n2. 峰值融合策略 (树模型峰值 + Mamba基线):")
    df_peak = df_metrics.loc[[m for m in peak_fusion_models if m in df_metrics.index]]
    print(df_peak[['Overall_R²', 'Overall_RMSE', 'Overall_MAE']].to_string())

    print(f"\n3. MLEF集成 (动态置信度加权融合):")
    df_mlef = df_metrics.loc[[m for m in ensemble_models if m in df_metrics.index]]
    print(df_mlef[['Overall_R²', 'Overall_RMSE', 'Overall_MAE']].to_string())

    print(f"\n性能排名 (全部):")
    print(df_metrics[['Overall_R²', 'Overall_RMSE', 'Overall_MAE']].to_string())

    # ========== 可视化 ==========
    print("\n" + "=" * 80)
    print("[生成可视化]")
    print("=" * 80)

    # ========== 计算额外指标 ==========
    print("\n计算额外评估指标...")

    # 回归指标（对每个模型的整体预测）
    all_regression_metrics = []
    for model_name, pred in predictions_test.items():
        y_test_flat = y_test.flatten()
        pred_flat = pred.flatten()
        reg_metrics = calculate_regression_metrics(y_test_flat, pred_flat, model_name)
        all_regression_metrics.append(reg_metrics)

    # 保存回归指标到CSV
    df_reg_metrics = pd.DataFrame(all_regression_metrics)
    df_reg_metrics = df_reg_metrics.sort_values('R2', ascending=False)
    df_reg_metrics.to_csv(f"{output_dir}/regression_metrics.csv", index=False)
    print("  ✓ 回归指标已保存: regression_metrics.csv")

    # 绘制回归指标对比图
    plot_regression_metrics(all_regression_metrics, f"{output_dir}/regression_metrics_comparison.png")
    print("  ✓ 回归指标对比图已保存")

    # 分类指标（峰值时刻预测）
    all_classification_metrics = []

    # LightGBM峰值时刻分类器
    if lgbm_hour_classifiers:
        lgbm_hour_pred = np.stack([clf.predict(X_test_hour_feat) for clf in lgbm_hour_classifiers], axis=1)
        # 计算每天的分类指标
        for i in range(lgbm_hour_pred.shape[1]):
            cls_metrics = calculate_multiclass_metrics(
                y_test_peak_hour[:, i], lgbm_hour_pred[:, i],
                f'LightGBM_Hour_Day{i+1}', num_classes=24
            )
            all_classification_metrics.append(cls_metrics)

        # 整体分类指标（所有天合并）
        lgbm_all_true = y_test_peak_hour.flatten()
        lgbm_all_pred = lgbm_hour_pred.flatten()
        cls_metrics = calculate_multiclass_metrics(
            lgbm_all_true, lgbm_all_pred, 'LightGBM_PeakHour_Overall', num_classes=24
        )
        all_classification_metrics.append(cls_metrics)

    # XGBoost峰值时刻分类器
    if xgb_hour_classifiers:
        xgb_hour_pred = np.stack([clf.predict(X_test_hour_feat) for clf in xgb_hour_classifiers], axis=1)
        # 计算每天的分类指标
        for i in range(xgb_hour_pred.shape[1]):
            cls_metrics = calculate_multiclass_metrics(
                y_test_peak_hour[:, i], xgb_hour_pred[:, i],
                f'XGBoost_Hour_Day{i+1}', num_classes=24
            )
            all_classification_metrics.append(cls_metrics)

        # 整体分类指标（所有天合并）
        xgb_all_true = y_test_peak_hour.flatten()
        xgb_all_pred = xgb_hour_pred.flatten()
        cls_metrics = calculate_multiclass_metrics(
            xgb_all_true, xgb_all_pred, 'XGBoost_PeakHour_Overall', num_classes=24
        )
        all_classification_metrics.append(cls_metrics)

    if all_classification_metrics:
        # 保存分类指标到CSV
        df_cls_metrics = pd.DataFrame(all_classification_metrics)
        df_cls_metrics.to_csv(f"{output_dir}/classification_metrics.csv", index=False)
        print("  ✓ 分类指标已保存: classification_metrics.csv")

        # 只绘制Overall指标对比图
        overall_cls_metrics = [m for m in all_classification_metrics if 'Overall' in m['Model']]
        if overall_cls_metrics:
            plot_classification_metrics(overall_cls_metrics, f"{output_dir}/classification_metrics_comparison.png")
            print("  ✓ 分类指标对比图已保存")

    # 绘制预测对比图（真实值 vs 预测值散点图）
    plot_prediction_comparison(y_test, predictions_test, f"{output_dir}/prediction_comparison.png")
    print("  ✓ 预测对比图已保存")

    # 保存所有指标摘要到JSON
    metrics_summary = {
        'regression_metrics': all_regression_metrics,
        'classification_metrics': all_classification_metrics
    }
    with open(f'{output_dir}/metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print("  ✓ 指标摘要已保存: metrics_summary.json")

    # 1. 所有模型预测对比（添加标注）
    # 为峰值融合模型重命名
    predictions_test_renamed = {}
    for model_name, pred in predictions_test.items():
        if 'Peak' in model_name and model_name != 'MLEF':
            predictions_test_renamed[f"{model_name} (Mamba_base)"] = pred
        else:
            predictions_test_renamed[model_name] = pred

    plot_seq2seq_predictions(
        y_test, predictions_test_renamed, output_dir,
        n_samples=3, dpi=300
    )
    print("  ✓ 预测对比图已保存")

    # 2. R²对比柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    models = df_metrics.index.tolist()
    r2_scores = df_metrics['Overall_R²'].values

    # 为峰值融合模型添加(Mamba_base)标注
    display_names = []
    for model in models:
        if 'Peak' in model and model != 'MLEF':
            display_names.append(f"{model} (Mamba_base)")
        else:
            display_names.append(model)

    colors = ['green' if r2 >= 0.5 else 'orange' if r2 >= 0 else 'red' for r2 in r2_scores]
    bars = ax.barh(display_names, r2_scores, color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('R² Score', fontsize=12)
    ax.set_title('Peak MLEF - Model R² Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    # 添加数值标签
    for i, (model, r2) in enumerate(zip(display_names, r2_scores)):
        ax.text(r2 + 0.01, i, f'{r2:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/r2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ R²对比图已保存")

    # 3. 峰值预测对比（展示前3个样本的日峰值）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i in range(min(3, len(y_test))):
        # 提取每日峰值
        sample_true = y_test[i].reshape(7, 24).max(axis=1)

        axes[i].plot(range(7), sample_true, 'o-', label='True Peak',
                    linewidth=2.5, markersize=8, color='black', alpha=0.8)

        # 绘制主要模型的峰值预测
        for model_name in ['Mamba', 'LightGBM_Peak', 'XGBoost_Peak', 'MLEF']:
            if model_name in predictions_test:
                sample_pred = predictions_test[model_name][i].reshape(7, 24).max(axis=1)
                # 为峰值融合模型添加(Mamba_base)标注
                display_label = f"{model_name} (Mamba_base)" if 'Peak' in model_name else model_name
                axes[i].plot(range(7), sample_pred, 'o-', label=display_label,
                           linewidth=1.5, markersize=6, alpha=0.7)

        axes[i].set_title(f'Sample {i+1} - Daily Peak Power', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Day', fontsize=10)
        axes[i].set_ylabel('Peak Power (kW)', fontsize=10)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(range(7))
        axes[i].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/peak_prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 峰值对比图已保存")

    # 保存预测结果供后续MLEF策略测试使用
    print("\n保存预测结果...")
    predictions_data = {
        'predictions_val': predictions_val,
        'predictions_test': predictions_test,
        'y_val': y_val,
        'y_test': y_test
    }
    with open(f'{output_dir}/all_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_data, f)
    print(f"  ✓ 预测结果已保存: {output_dir}/all_predictions.pkl")

    # ========== 阶段4: 峰值专项评估 ==========
    print("\n" + "=" * 80)
    print("[阶段4] 峰值拟合效果专项评估")
    print("=" * 80)

    # 创建白天标志（6:00-20:00视为白天）
    seq_len = output_len
    time_indices = np.tile(np.arange(24), (len(y_test), 7))[:, :seq_len]
    is_daytime_test = ((time_indices >= 6) & (time_indices <= 20)).astype(float)

    # 生成峰值评估报告
    peak_eval_dir = os.path.join(output_dir, 'peak_evaluation')
    os.makedirs(peak_eval_dir, exist_ok=True)

    try:
        df_peak_metrics, model_peak_info = create_peak_evaluation_report(
            predictions=predictions_test,
            y_true=y_test,
            is_daytime=is_daytime_test,
            time_indices=time_indices,
            output_dir=peak_eval_dir,
            model_names=['Mamba', 'LSTM', 'GRU', 'LightGBM_Peak', 'XGBoost_Peak', 'MLEF']
        )

        # 打印峰值指标摘要
        print("\n峰值拟合指标摘要:")
        print(df_peak_metrics[['model', 'peak_value_rmse', 'peak_time_mae', 'peak_time_within_1h']].to_string(index=False))

        # 保存峰值指标
        df_peak_metrics.to_csv(f'{output_dir}/peak_metrics.csv', index=False)
        print(f"\n  ✓ 峰值指标已保存: {output_dir}/peak_metrics.csv")

        # 分析MLEF模型的峰值误差模式
        if 'MLEF' in predictions_test:
            extractor = PeakExtractor()
            mlef_peak_info = extractor.extract_daily_peaks(
                y_test, predictions_test['MLEF'], is_daytime_test, time_indices
            )
            analysis = analyze_peak_error_patterns(
                mlef_peak_info, y_test, is_daytime_test, verbose=True
            )

            # 保存分析结果
            with open(f'{peak_eval_dir}/mlef_error_analysis.json', 'w') as f:
                # 转换numpy类型为Python原生类型
                def convert_to_native(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.int64, np.int32)):
                        return int(obj)
                    elif isinstance(obj, (np.float64, np.float32)):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_to_native(v) for k, v in obj.items()}
                    return obj
                json.dump(convert_to_native(analysis), f, indent=2)
            print(f"  ✓ MLEF误差分析已保存: {peak_eval_dir}/mlef_error_analysis.json")

    except Exception as e:
        print(f"  警告: 峰值评估失败 - {e}")

    # ========== 阶段5: 增强峰值集成测试 ==========
    # 基础模型列表（排除集成结果）
    base_model_names = ['Mamba', 'LSTM', 'GRU', 'Transformer', 'LightGBM_Peak', 'XGBoost_Peak']
    base_predictions_val = {k: v for k, v in predictions_val.items() if k in base_model_names}
    base_predictions_test = {k: v for k, v in predictions_test.items() if k in base_model_names}

    if args.skip_advanced_ensemble:
        print("\n" + "=" * 80)
        print("[阶段5-6] 跳过高级集成方法 (--skip-advanced-ensemble)")
        print("=" * 80)
        print("  提示: 如需运行高级集成方法，请去掉 --skip-advanced-ensemble 参数")
    else:
        print("\n" + "=" * 80)
        print("[阶段5] 增强峰值集成测试")
        print("=" * 80)

        # 生成验证集的 is_daytime
        is_daytime_val = ((np.tile(np.arange(24), (len(y_val), 7))[:, :output_len] >= 6) &
                          (np.tile(np.arange(24), (len(y_val), 7))[:, :output_len] <= 20)).astype(float)

        try:
            # 使用增强版峰值感知集成
            enhanced_ensemble = EnhancedPeakAwareEnsemble(
                peak_window_size=2,
                soft_transition_sigma=1.5
            )

            enhanced_ensemble.calibrate(base_predictions_val, y_val, is_daytime_val, verbose=True)

            # 预测
            enhanced_pred = enhanced_ensemble.predict(base_predictions_test, is_daytime_test)
            predictions_test['Enhanced_MLEF'] = enhanced_pred

            # 计算指标
            enhanced_metrics = calculate_seq2seq_metrics(
                y_test, enhanced_pred, 'Enhanced_MLEF', horizons=[1, 24, 72, 168]
            )
            results['Enhanced_MLEF'] = enhanced_metrics
            print(f"\n  Enhanced_MLEF R²: {enhanced_metrics['Overall_R²']:.4f}")

            # 保存增强集成权重
            enhanced_ensemble.save(f'{output_dir}/enhanced_ensemble_weights.pkl')
            print(f"  ✓ 增强集成权重已保存")

            # 对比原始MLEF和增强版
            if 'MLEF' in results:
                print(f"\n  对比:")
                print(f"    原始MLEF R²:   {results['MLEF']['Overall_R²']:.4f}")
                print(f"    增强MLEF R²:   {enhanced_metrics['Overall_R²']:.4f}")
                improvement = enhanced_metrics['Overall_R²'] - results['MLEF']['Overall_R²']
                print(f"    R²提升:        {improvement:+.4f}")

        except Exception as e:
            print(f"  警告: 增强峰值集成测试失败 - {e}")

        # ========== 阶段6: 高级峰值集成优化 ==========
        print("\n" + "=" * 80)
        print("[阶段6] 高级峰值集成优化")
        print("=" * 80)

        try:
            # 6.1 搜索最优峰值窗口
            print("\n6.1 搜索最优峰值窗口...")
            optimal_window, best_score = find_optimal_window_size(
                base_predictions_val, y_val, is_daytime_val,
                candidate_windows=[1, 2, 3, 4, 5],
                verbose=True
            )

            # 6.2 自适应高斯集成
            print("\n6.2 自适应高斯权重集成...")
            adaptive_ensemble = AdaptiveGaussianEnsemble(
                sigma_before=1.5,
                sigma_after=2.0,
                adaptive_sigma=True
            )
            adaptive_ensemble.calibrate(base_predictions_val, y_val, is_daytime_val, verbose=True)
            adaptive_pred = adaptive_ensemble.predict(base_predictions_test, is_daytime_test)
            predictions_test['Adaptive_Gaussian'] = adaptive_pred

            adaptive_metrics = calculate_seq2seq_metrics(
                y_test, adaptive_pred, 'Adaptive_Gaussian', horizons=[1, 24, 72, 168]
            )
            results['Adaptive_Gaussian'] = adaptive_metrics
            print(f"\n  Adaptive_Gaussian R²: {adaptive_metrics['Overall_R²']:.4f}")

            # 6.3 时段依赖权重集成
            print("\n6.3 时段依赖权重集成...")
            time_ensemble = TimeDependentEnsemble(max_offset=6)
            time_ensemble.calibrate_time_weights(base_predictions_val, y_val, is_daytime_val, verbose=True)
            time_pred = time_ensemble.predict(base_predictions_test, is_daytime_test)
            predictions_test['Time_Dependent'] = time_pred

            time_metrics = calculate_seq2seq_metrics(
                y_test, time_pred, 'Time_Dependent', horizons=[1, 24, 72, 168]
            )
            results['Time_Dependent'] = time_metrics
            print(f"\n  Time_Dependent R²: {time_metrics['Overall_R²']:.4f}")

            # 6.4 K折交叉验证校准
            print("\n6.4 K折交叉验证权重校准...")
            kfold_ensemble = KFoldCalibratedEnsemble(
                n_folds=5,
                peak_window_size=optimal_window,
                soft_transition_sigma=optimal_window * 0.5
            )
            kfold_ensemble.calibrate_with_kfold(base_predictions_val, y_val, is_daytime_val, verbose=True)
            kfold_pred = kfold_ensemble.predict(base_predictions_test, is_daytime_test)
            predictions_test['KFold_Calibrated'] = kfold_pred

            kfold_metrics = calculate_seq2seq_metrics(
                y_test, kfold_pred, 'KFold_Calibrated', horizons=[1, 24, 72, 168]
            )
            results['KFold_Calibrated'] = kfold_metrics
            print(f"\n  KFold_Calibrated R²: {kfold_metrics['Overall_R²']:.4f}")

            # 6.5 残差校正器
            print("\n6.5 峰值残差校正...")
            corrector = PeakResidualCorrector(correction_sigma=2.0)

            # 使用K折集成的预测作为基础
            kfold_val_pred = kfold_ensemble.predict(base_predictions_val, is_daytime_val)
            corrector.fit(base_predictions_val, kfold_val_pred, y_val, is_daytime_val, verbose=True)
            corrected_pred = corrector.correct(base_predictions_test, kfold_pred, is_daytime_test)
            predictions_test['Residual_Corrected'] = corrected_pred

            corrected_metrics = calculate_seq2seq_metrics(
                y_test, corrected_pred, 'Residual_Corrected', horizons=[1, 24, 72, 168]
            )
            results['Residual_Corrected'] = corrected_metrics
            print(f"\n  Residual_Corrected R²: {corrected_metrics['Overall_R²']:.4f}")

            # 6.6 分时段权重集成
            print("\n6.6 分时段权重集成...")
            period_ensemble = PeriodBasedEnsemble(
                peak_hours=(10, 14),  # 典型光伏峰值时段
                temperature=0.1  # 更尖锐的权重分布
            )
            period_ensemble.calibrate(base_predictions_val, y_val, is_daytime_val, verbose=True)
            period_pred = period_ensemble.predict(base_predictions_test, is_daytime_test)
            predictions_test['Period_Based'] = period_pred

            period_metrics = calculate_seq2seq_metrics(
                y_test, period_pred, 'Period_Based', horizons=[1, 24, 72, 168]
            )
            results['Period_Based'] = period_metrics
            print(f"\n  Period_Based R²: {period_metrics['Overall_R²']:.4f}")

            # 6.7 组合集成（分时段 + 时段依赖）
            print("\n6.7 组合集成（分时段 + 时段依赖）...")
            combined_ensemble = CombinedPeriodTimeEnsemble(
                max_offset=6,
                peak_hours=(10, 14),
                blend_alpha=0.6  # 60% 时段依赖 + 40% 分时段
            )
            combined_ensemble.calibrate(base_predictions_val, y_val, is_daytime_val, verbose=True)
            combined_pred = combined_ensemble.predict(base_predictions_test, is_daytime_test)
            predictions_test['Combined_Period_Time'] = combined_pred

            combined_metrics = calculate_seq2seq_metrics(
                y_test, combined_pred, 'Combined_Period_Time', horizons=[1, 24, 72, 168]
            )
            results['Combined_Period_Time'] = combined_metrics
            print(f"\n  Combined_Period_Time R²: {combined_metrics['Overall_R²']:.4f}")

            # 6.8 综合对比
            print("\n" + "=" * 70)
            print("高级集成方法对比")
            print("=" * 70)

            advanced_methods = [
                ('MLEF (原始)', results.get('MLEF', {})),
                ('Enhanced_MLEF', results.get('Enhanced_MLEF', {})),
                ('Adaptive_Gaussian', results.get('Adaptive_Gaussian', {})),
                ('Time_Dependent', results.get('Time_Dependent', {})),
                ('KFold_Calibrated', results.get('KFold_Calibrated', {})),
                ('Period_Based', results.get('Period_Based', {})),
                ('Combined_Period_Time', results.get('Combined_Period_Time', {})),
                ('Residual_Corrected', results.get('Residual_Corrected', {}))
            ]

            print(f"\n{'方法':<25} {'R²':<12} {'RMSE':<12} {'MAE':<12}")
            print("-" * 65)
            for name, m in advanced_methods:
                if m:
                    print(f"{name:<25} {m.get('Overall_R²', 0):<12.4f} "
                          f"{m.get('Overall_RMSE', 0):<12.2f} {m.get('Overall_MAE', 0):<12.2f}")

            # 6.9 峰值专项指标对比
            print("\n峰值专项指标对比:")
            extractor = PeakExtractor()

            print(f"\n{'方法':<25} {'峰值RMSE':<12} {'时刻MAE':<12} {'±1h准确率':<12}")
            print("-" * 65)
            for name in ['MLEF', 'Adaptive_Gaussian', 'Time_Dependent', 'KFold_Calibrated', 'Period_Based', 'Combined_Period_Time', 'Residual_Corrected']:
                if name in predictions_test:
                    peak_info = extractor.extract_daily_peaks(
                        y_test, predictions_test[name], is_daytime_test
                    )
                    print(f"{name:<25} {peak_info['peak_value_rmse']:<12.2f} "
                          f"{peak_info['peak_time_mae']:<12.2f} {peak_info['peak_time_within_1h']*100:<12.1f}%")

            # 保存最佳模型预测
            best_method = max(
                ['KFold_Calibrated', 'Residual_Corrected', 'Adaptive_Gaussian', 'Time_Dependent', 'Period_Based', 'Combined_Period_Time'],
                key=lambda m: results.get(m, {}).get('Overall_R²', 0) if m in results else 0
            )
            print(f"\n最佳高级集成方法: {best_method}")
            print(f"  R²: {results[best_method]['Overall_R²']:.4f}")

            # 更新结果CSV
            df_metrics_updated = pd.DataFrame(results).T
            df_metrics_updated = df_metrics_updated.sort_values('Overall_R²', ascending=False)
            df_metrics_updated.to_csv(f"{output_dir}/metrics_all.csv")
            print(f"\n  ✓ 完整指标已保存: {output_dir}/metrics_all.csv")

        except Exception as e:
            import traceback
            print(f"  警告: 高级峰值集成优化失败 - {e}")
            traceback.print_exc()

    # ========== 保存训练历史和效率数据 ==========
    if args.save_history and training_histories:
        print("\n" + "=" * 80)
        print("[保存训练历史和效率数据]")
        print("=" * 80)

        # 保存训练历史
        with open(f'{output_dir}/training_histories.pkl', 'wb') as f:
            pickle.dump(training_histories, f)
        print(f"  ✓ 训练历史已保存: {output_dir}/training_histories.pkl")

        # 保存效率数据
        efficiency_data = {
            'training_times': training_times,
            'inference_times': inference_times,
            'input_len': input_len,
            'output_len': output_len,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        }
        with open(f'{output_dir}/efficiency_data.json', 'w') as f:
            json.dump(efficiency_data, f, indent=2)
        print(f"  ✓ 效率数据已保存: {output_dir}/efficiency_data.json")

        # 打印效率摘要
        print("\n训练效率摘要:")
        print(f"  {'模型':<15} {'训练时间(s)':<15} {'推理时间(s)':<15}")
        print("-" * 45)
        for model_name in ['Mamba', 'LSTM', 'GRU', 'Transformer']:
            train_t = training_times.get(model_name, 0)
            infer_t = inference_times.get(model_name, 0)
            print(f"  {model_name:<15} {train_t:<15.1f} {infer_t:<15.3f}")

    print(f"\n结果已保存至: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
