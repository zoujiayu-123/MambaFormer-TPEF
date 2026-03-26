#!/usr/bin/env python
"""
生成论文所需的所有图表
包括：训练曲线、效率对比、输入长度消融实验
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 实验目录
EXPERIMENTS = {
    '360h': 'Mamba-pv/output/task4_peak_mlef_20251130_190135',
    '720h': 'Mamba-pv/output/task4_peak_mlef_20251130_004123',
    '1080h': 'Mamba-pv/output/task4_peak_mlef_20251130_195255'
}

OUTPUT_DIR = 'Mamba-pv/output/paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_experiment_data(exp_dir):
    """加载实验数据"""
    data = {}

    # 加载效率数据
    efficiency_file = os.path.join(exp_dir, 'efficiency_data.json')
    if os.path.exists(efficiency_file):
        with open(efficiency_file, 'r') as f:
            data['efficiency'] = json.load(f)

    # 加载训练历史
    history_file = os.path.join(exp_dir, 'training_histories.pkl')
    if os.path.exists(history_file):
        with open(history_file, 'rb') as f:
            data['history'] = pickle.load(f)

    # 加载指标
    metrics_file = os.path.join(exp_dir, 'metrics.csv')
    if os.path.exists(metrics_file):
        data['metrics'] = pd.read_csv(metrics_file, index_col=0)

    return data


def plot_training_curves(data_720h, output_path):
    """绘制训练曲线（使用720h实验）"""
    if 'history' not in data_720h:
        print("警告: 720h实验没有训练历史数据")
        return

    history = data_720h['history']
    models = ['Mamba', 'LSTM', 'GRU', 'Transformer']
    colors = {'Mamba': '#2ecc71', 'LSTM': '#3498db', 'GRU': '#9b59b6', 'Transformer': '#e74c3c'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        if model in history:
            h = history[model]
            epochs = h.get('epochs', range(1, len(h['train_loss']) + 1))

            ax.plot(epochs, h['train_loss'], label='Train Loss', color=colors[model], linewidth=2)
            ax.plot(epochs, h['val_loss'], label='Val Loss', color=colors[model], linewidth=2, linestyle='--')

            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss (MSE)', fontsize=11)
            ax.set_title(f'{model} Training Curve', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1, max(epochs))
        else:
            ax.text(0.5, 0.5, f'{model}\nNo Data', ha='center', va='center', fontsize=14)
            ax.set_title(f'{model}', fontsize=12, fontweight='bold')

    plt.suptitle('Training Curves (720h Input, 168h Output)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 训练曲线图已保存: {output_path}")


def plot_efficiency_comparison(all_data, output_path):
    """绘制计算效率对比图"""
    models = ['Mamba', 'LSTM', 'GRU', 'Transformer']
    input_lens = ['360h', '720h', '1080h']

    # 提取训练时间
    train_times = {model: [] for model in models}
    for input_len in input_lens:
        if input_len in all_data and 'efficiency' in all_data[input_len]:
            eff = all_data[input_len]['efficiency']
            for model in models:
                train_times[model].append(eff['training_times'].get(model, 0) / 60)  # 转换为分钟
        else:
            for model in models:
                train_times[model].append(0)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: 训练时间柱状图
    ax1 = axes[0]
    x = np.arange(len(input_lens))
    width = 0.2
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    for i, model in enumerate(models):
        ax1.bar(x + i * width, train_times[model], width, label=model, color=colors[i], alpha=0.8)

    ax1.set_xlabel('Input Length', fontsize=11)
    ax1.set_ylabel('Training Time (minutes)', fontsize=11)
    ax1.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(['360h\n(15 days)', '720h\n(30 days)', '1080h\n(45 days)'])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 子图2: 推理时间对比（使用720h数据）
    ax2 = axes[1]
    if '720h' in all_data and 'efficiency' in all_data['720h']:
        eff = all_data['720h']['efficiency']
        infer_times = [eff['inference_times'].get(m, 0) for m in models]

        bars = ax2.bar(models, infer_times, color=colors, alpha=0.8)
        ax2.set_xlabel('Model', fontsize=11)
        ax2.set_ylabel('Inference Time (seconds)', fontsize=11)
        ax2.set_title('Inference Time (720h Input)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, val in zip(bars, infer_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}s', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 效率对比图已保存: {output_path}")


def plot_input_length_ablation(all_data, output_path):
    """绘制输入长度消融实验图"""
    models = ['Mamba', 'LSTM', 'GRU', 'Transformer', 'MLEF']
    input_lens = ['360h', '720h', '1080h']
    input_days = [15, 30, 45]

    # 提取R²分数
    r2_scores = {model: [] for model in models}
    for input_len in input_lens:
        if input_len in all_data and 'metrics' in all_data[input_len]:
            metrics = all_data[input_len]['metrics']
            for model in models:
                if model in metrics.index:
                    r2_scores[model].append(metrics.loc[model, 'Overall_R²'])
                else:
                    r2_scores[model].append(np.nan)
        else:
            for model in models:
                r2_scores[model].append(np.nan)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Mamba': '#2ecc71', 'LSTM': '#3498db', 'GRU': '#9b59b6',
              'Transformer': '#e74c3c', 'MLEF': '#f39c12'}
    markers = {'Mamba': 'o', 'LSTM': 's', 'GRU': '^', 'Transformer': 'D', 'MLEF': '*'}

    for model in models:
        scores = r2_scores[model]
        valid_idx = [i for i, s in enumerate(scores) if not np.isnan(s)]
        if valid_idx:
            valid_days = [input_days[i] for i in valid_idx]
            valid_scores = [scores[i] for i in valid_idx]
            ax.plot(valid_days, valid_scores, marker=markers[model], label=model,
                   color=colors[model], linewidth=2, markersize=10, alpha=0.8)

    ax.set_xlabel('Input Length (days)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Input Length Ablation Study', fontsize=14, fontweight='bold')
    ax.set_xticks(input_days)
    ax.set_xticklabels(['15 days\n(360h)', '30 days\n(720h)', '45 days\n(1080h)'])
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 输入长度消融图已保存: {output_path}")


def plot_model_comparison_bar(all_data, output_path):
    """绘制模型性能对比柱状图（使用720h数据）"""
    if '720h' not in all_data or 'metrics' not in all_data['720h']:
        print("警告: 没有720h指标数据")
        return

    metrics = all_data['720h']['metrics']

    # 选择要展示的模型
    display_models = ['Mamba', 'LSTM', 'GRU', 'Transformer', 'LightGBM_Peak', 'XGBoost_Peak', 'MLEF']
    display_models = [m for m in display_models if m in metrics.index]

    r2_scores = [metrics.loc[m, 'Overall_R²'] for m in display_models]
    rmse_scores = [metrics.loc[m, 'Overall_RMSE'] for m in display_models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 颜色
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#1abc9c', '#f39c12', '#e67e22']

    # R² 对比
    ax1 = axes[0]
    bars1 = ax1.barh(display_models, r2_scores, color=colors[:len(display_models)], alpha=0.8)
    ax1.set_xlabel('R² Score', fontsize=11)
    ax1.set_title('R² Score Comparison (720h → 168h)', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars1, r2_scores):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=10)

    # RMSE 对比
    ax2 = axes[1]
    bars2 = ax2.barh(display_models, rmse_scores, color=colors[:len(display_models)], alpha=0.8)
    ax2.set_xlabel('RMSE (kW)', fontsize=11)
    ax2.set_title('RMSE Comparison (720h → 168h)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars2, rmse_scores):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 模型对比图已保存: {output_path}")


def create_ablation_table(all_data, output_path):
    """创建消融实验表格"""
    models = ['Mamba', 'LSTM', 'GRU', 'Transformer', 'MLEF']
    input_lens = ['360h', '720h', '1080h']

    # 构建表格数据
    table_data = []
    for model in models:
        row = {'Model': model}
        for input_len in input_lens:
            if input_len in all_data and 'metrics' in all_data[input_len]:
                metrics = all_data[input_len]['metrics']
                if model in metrics.index:
                    row[f'{input_len}_R2'] = f"{metrics.loc[model, 'Overall_R²']:.4f}"
                    row[f'{input_len}_RMSE'] = f"{metrics.loc[model, 'Overall_RMSE']:.2f}"
                else:
                    row[f'{input_len}_R2'] = '-'
                    row[f'{input_len}_RMSE'] = '-'
            else:
                row[f'{input_len}_R2'] = '-'
                row[f'{input_len}_RMSE'] = '-'
        table_data.append(row)

    df = pd.DataFrame(table_data)
    df.to_csv(output_path, index=False)
    print(f"  ✓ 消融实验表格已保存: {output_path}")
    return df


def main():
    print("=" * 60)
    print("生成论文图表")
    print("=" * 60)

    # 加载所有实验数据
    all_data = {}
    for name, exp_dir in EXPERIMENTS.items():
        print(f"\n加载 {name} 实验数据...")
        all_data[name] = load_experiment_data(exp_dir)
        if all_data[name]:
            print(f"  ✓ 已加载: {list(all_data[name].keys())}")
        else:
            print(f"  ✗ 加载失败")

    print("\n" + "=" * 60)
    print("生成图表")
    print("=" * 60)

    # 1. 训练曲线图
    print("\n[1] 生成训练曲线图...")
    plot_training_curves(all_data.get('720h', {}),
                        os.path.join(OUTPUT_DIR, 'training_curves.png'))

    # 2. 效率对比图
    print("\n[2] 生成效率对比图...")
    plot_efficiency_comparison(all_data,
                              os.path.join(OUTPUT_DIR, 'efficiency_comparison.png'))

    # 3. 输入长度消融图
    print("\n[3] 生成输入长度消融图...")
    plot_input_length_ablation(all_data,
                              os.path.join(OUTPUT_DIR, 'input_length_ablation.png'))

    # 4. 模型对比图
    print("\n[4] 生成模型对比图...")
    plot_model_comparison_bar(all_data,
                             os.path.join(OUTPUT_DIR, 'model_comparison.png'))

    # 5. 消融实验表格
    print("\n[5] 生成消融实验表格...")
    df = create_ablation_table(all_data,
                              os.path.join(OUTPUT_DIR, 'ablation_results.csv'))
    print("\n消融实验结果:")
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"所有图表已保存至: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
