#!/usr/bin/env python
"""
Task4光伏功率预测 - SHAP可解释性分析

对LightGBM和XGBoost峰值预测模型进行SHAP分析：
1. 特征重要性排序（全局）
2. 特征影响方向（Summary Plot）
3. 关键特征依赖关系（Dependence Plot）
4. 单样本预测分解（Waterfall Plot）
5. 白天/夜间特征对比

输出：
- shap_summary_*.png: 蜂群图
- shap_bar_*.png: 特征重要性柱状图
- shap_dependence_*.png: 依赖图
- shap_waterfall_*.png: 瀑布图
- feature_importance.csv: 特征重要性表格
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置非交互模式
plt.switch_backend('Agg')

# 添加项目路径
sys.path.insert(0, 'Mamba-pv')
sys.path.insert(0, 'Mamba-pv/src')

try:
    import shap
    print(f"SHAP version: {shap.__version__}")
except ImportError:
    print("请先安装shap库: pip install shap")
    sys.exit(1)


# ==================== 配置 ====================
EXPERIMENT_DIR = 'Mamba-pv/output/task4_peak_mlef_20251130_004123'
OUTPUT_DIR = 'Mamba-pv/output/shap_analysis'
DATA_FILE = 'Mamba-pv/data/combined_pv_data2016.csv'

# 输入长度配置（与训练时一致）
INPUT_LEN = 720
OUTPUT_LEN = 168

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 特征名称生成 ====================
def get_statistical_feature_names(n_days=30, n_weeks=4, n_features=25):
    """
    生成统计特征的名称列表

    Args:
        n_days: 天数（360h=15天, 720h=30天, 1080h=45天）
        n_weeks: 周数
        n_features: 原始特征数

    Returns:
        feature_names: 特征名称列表
    """
    # 原始电气特征名
    raw_features = [
        'InvVb_Avg', 'InvIa_Avg', 'InvIb_Avg', 'InvIc_Avg', 'InvFreq_Avg',
        'InvPAC_kW_Avg', 'InvPDC_kW_Avg', 'InvOpStatus_Avg', 'InvVoltageFault_Max',
        'PwrMtrIa_Avg', 'PwrMtrIb_Avg', 'PwrMtrFreq_Avg', 'PwrMtrPhaseRev_Avg',
        'PwrMtrVa_Avg', 'Battery_A_Avg', 'Qloss_Ah_Max'
    ]
    # 时间特征名
    time_features = [
        'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
        'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos', 'is_daytime'
    ]
    base_features = raw_features + time_features

    feature_names = []

    # 1. 每日统计特征 (n_days × 4种统计 × n_features)
    for day in range(n_days):
        for stat in ['mean', 'max', 'std', 'min']:
            for feat in base_features:
                feature_names.append(f'Day{day+1}_{stat}_{feat}')

    # 2. 每周统计特征 (n_weeks × 2种统计 × n_features)
    for week in range(n_weeks):
        for stat in ['mean', 'max']:
            for feat in base_features:
                feature_names.append(f'Week{week+1}_{stat}_{feat}')

    # 3. 全局统计特征 (3种统计 × n_features)
    for stat in ['global_mean', 'global_max', 'global_std']:
        for feat in base_features:
            feature_names.append(f'{stat}_{feat}')

    # 4. 最近时段统计 (3种 × n_features)
    for stat in ['last24h_mean', 'last24h_max', 'last72h_mean']:
        for feat in base_features:
            feature_names.append(f'{stat}_{feat}')

    # 5. 趋势特征 (1种 × n_features)
    for feat in base_features:
        feature_names.append(f'trend_{feat}')

    return feature_names


def get_simplified_feature_names(n_stat_features):
    """
    生成简化的特征名称（当特征数量与预期不符时使用）

    Args:
        n_stat_features: 实际统计特征数量

    Returns:
        feature_names: 简化的特征名称列表
    """
    return [f'stat_feat_{i}' for i in range(n_stat_features)]


# ==================== 数据加载和预处理 ====================
def load_data():
    """加载并预处理数据（与训练脚本一致）"""
    from mtm_mlef.data_processing import (
        load_data as load_csv_data,
        filter_data_by_months,
        create_hourly_sequences_for_weekly_prediction,
        extract_statistical_features,
        create_peak_targets
    )
    from sklearn.preprocessing import MinMaxScaler

    print("=" * 60)
    print("加载数据...")
    print("=" * 60)

    # 加载数据（与训练脚本一致）
    train_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    test_months = [11, 12]
    target_col = 'InvPAC_kW_Avg'

    # 加载训练数据（2016-2017年1-10月）
    train_2016 = load_csv_data(DATA_FILE)
    train_2016 = filter_data_by_months(train_2016, include_months=train_months)

    # 加载测试数据（2016年11-12月）
    test_2016 = filter_data_by_months(load_csv_data(DATA_FILE), include_months=test_months)

    print(f"训练数据: {len(train_2016)} 条记录")
    print(f"测试数据: {len(test_2016)} 条记录")

    # 创建序列
    print("\n创建序列...")
    X_train_seq, y_train_seq, _ = create_hourly_sequences_for_weekly_prediction(
        train_2016, target_col=target_col, input_hours=INPUT_LEN, output_hours=OUTPUT_LEN,
        add_time_features=True, check_continuity=False
    )
    X_test, y_test, timestamps = create_hourly_sequences_for_weekly_prediction(
        test_2016, target_col=target_col, input_hours=INPUT_LEN, output_hours=OUTPUT_LEN,
        add_time_features=True, check_continuity=False
    )

    # 划分训练/验证
    n_train = int(len(X_train_seq) * 0.9)
    X_train = X_train_seq[:n_train]
    y_train = y_train_seq[:n_train]

    print(f"序列数据形状: X_train={X_train.shape}, X_test={X_test.shape}")

    # 归一化
    n_features = X_train.shape[2]
    scaler_X = MinMaxScaler()
    scaler_X.fit(X_train.reshape(-1, n_features))

    X_train_norm = scaler_X.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_test_norm = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    scaler_y = MinMaxScaler()
    y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    print(f"训练集: {X_train_norm.shape[0]} 样本")
    print(f"测试集: {X_test_norm.shape[0]} 样本")

    # 提取统计特征（用于树模型）
    print("\n提取统计特征...")
    X_train_stat = extract_statistical_features(X_train_norm)
    X_test_stat = extract_statistical_features(X_test_norm)
    print(f"统计特征维度: {X_train_stat.shape[1]}")

    # 创建峰值预测目标
    y_train_peak, y_train_peak_hour = create_peak_targets(y_train_norm)
    y_test_peak, y_test_peak_hour = create_peak_targets(y_test_norm)

    return {
        'X_train': X_train_norm,
        'X_test': X_test_norm,
        'y_train': y_train_norm,
        'y_test': y_test_norm,
        'X_train_stat': X_train_stat,
        'X_test_stat': X_test_stat,
        'y_train_peak': y_train_peak,
        'y_test_peak': y_test_peak,
        'timestamps': timestamps,
        'scaler_y': scaler_y
    }


def load_models():
    """加载训练好的模型"""
    print("\n" + "=" * 60)
    print("加载模型...")
    print("=" * 60)

    models = {}

    # 加载LightGBM峰值模型
    lgb_path = os.path.join(EXPERIMENT_DIR, 'LightGBM_Peak_peak_models.pkl')
    if os.path.exists(lgb_path):
        with open(lgb_path, 'rb') as f:
            models['LightGBM_Peak'] = pickle.load(f)
        print(f"✓ 加载 LightGBM_Peak: {len(models['LightGBM_Peak'])} 个模型（7天）")

    # 加载XGBoost峰值模型
    xgb_path = os.path.join(EXPERIMENT_DIR, 'XGBoost_Peak_peak_models.pkl')
    if os.path.exists(xgb_path):
        with open(xgb_path, 'rb') as f:
            models['XGBoost_Peak'] = pickle.load(f)
        print(f"✓ 加载 XGBoost_Peak: {len(models['XGBoost_Peak'])} 个模型（7天）")

    return models


# ==================== SHAP分析 ====================
def analyze_tree_model_shap(model, X_data, feature_names, model_name, day_idx=0):
    """
    对单个树模型进行SHAP分析

    Args:
        model: 训练好的树模型
        X_data: 特征数据
        feature_names: 特征名称列表
        model_name: 模型名称
        day_idx: 预测的天数索引（0-6）

    Returns:
        shap_values: SHAP值
        explainer: SHAP解释器
    """
    print(f"\n分析 {model_name} Day{day_idx+1}...")

    # 采样（加速计算）
    max_samples = min(500, len(X_data))
    indices = np.random.choice(len(X_data), max_samples, replace=False)
    X_sample = X_data[indices]

    # 创建TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print(f"  SHAP值形状: {shap_values.shape}")
    print(f"  Expected value: {explainer.expected_value:.4f}")

    return shap_values, explainer, X_sample


def compute_feature_importance(shap_values, feature_names):
    """
    计算特征重要性

    Args:
        shap_values: SHAP值
        feature_names: 特征名称

    Returns:
        DataFrame: 特征重要性排序
    """
    # 计算平均绝对SHAP值
    importance = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'mean_shap': shap_values.mean(axis=0),
        'std_shap': shap_values.std(axis=0)
    })

    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    return df


def aggregate_importance_across_days(all_shap_values, feature_names):
    """
    聚合7天模型的SHAP值，计算整体特征重要性

    Args:
        all_shap_values: 7天的SHAP值列表
        feature_names: 特征名称

    Returns:
        DataFrame: 聚合后的特征重要性
    """
    # 合并所有天的SHAP值
    combined_shap = np.concatenate(all_shap_values, axis=0)

    return compute_feature_importance(combined_shap, feature_names)


# ==================== 可视化 ====================
def plot_shap_summary(shap_values, X_data, feature_names, title, save_path, max_display=20):
    """绘制SHAP Summary Plot（蜂群图）"""
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_data,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=(12, 10)
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


def plot_shap_bar(importance_df, title, save_path, top_n=20):
    """绘制SHAP特征重要性柱状图"""
    df_top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_top)))
    bars = ax.barh(range(len(df_top)), df_top['importance'], color=colors)

    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['feature'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars, df_top['importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


def plot_shap_dependence(shap_values, X_data, feature_idx, feature_names,
                         title, save_path, interaction_idx='auto'):
    """绘制SHAP依赖图"""
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X_data,
        feature_names=feature_names,
        interaction_index=interaction_idx,
        show=False
    )
    plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


def plot_shap_waterfall(explainer, shap_values, X_sample, sample_idx,
                        feature_names, title, save_path):
    """绘制SHAP瀑布图"""
    # 创建Explanation对象
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_sample[sample_idx],
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=15, show=False)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


def plot_comparison_bar(lgb_importance, xgb_importance, save_path, top_n=15):
    """绘制LightGBM vs XGBoost特征重要性对比图"""
    # 合并两个模型的重要性
    merged = lgb_importance[['feature', 'importance']].rename(
        columns={'importance': 'LightGBM'}
    ).merge(
        xgb_importance[['feature', 'importance']].rename(
            columns={'importance': 'XGBoost'}
        ),
        on='feature',
        how='outer'
    ).fillna(0)

    # 计算平均重要性并排序
    merged['avg_importance'] = (merged['LightGBM'] + merged['XGBoost']) / 2
    merged = merged.sort_values('avg_importance', ascending=False).head(top_n)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(merged))
    width = 0.35

    bars1 = ax.barh(x - width/2, merged['LightGBM'], width,
                    label='LightGBM', color='#2ecc71', alpha=0.8)
    bars2 = ax.barh(x + width/2, merged['XGBoost'], width,
                    label='XGBoost', color='#3498db', alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(merged['feature'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title('Feature Importance: LightGBM vs XGBoost', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")

    return merged


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("Task4 光伏功率预测 - SHAP可解释性分析")
    print("=" * 60)

    # 1. 加载数据
    data = load_data()
    X_test_stat = data['X_test_stat']

    # 2. 加载模型
    models = load_models()

    if not models:
        print("错误：没有找到可分析的模型")
        return

    # 3. 生成特征名称
    n_stat_features = X_test_stat.shape[1]
    print(f"\n统计特征数量: {n_stat_features}")

    # 尝试生成详细特征名，如果数量不匹配则使用简化名称
    try:
        feature_names = get_statistical_feature_names(n_days=30, n_weeks=4, n_features=25)
        if len(feature_names) != n_stat_features:
            print(f"特征名数量({len(feature_names)})与实际特征数({n_stat_features})不匹配，使用简化名称")
            feature_names = get_simplified_feature_names(n_stat_features)
    except:
        feature_names = get_simplified_feature_names(n_stat_features)

    print(f"特征名称数量: {len(feature_names)}")

    # 4. 对每个模型进行SHAP分析
    all_results = {}

    for model_name, model_list in models.items():
        print(f"\n{'='*60}")
        print(f"分析模型: {model_name}")
        print(f"{'='*60}")

        all_shap_values = []
        all_X_samples = []

        # 对7天的每个模型进行分析
        for day_idx, model in enumerate(model_list):
            shap_values, explainer, X_sample = analyze_tree_model_shap(
                model, X_test_stat, feature_names, model_name, day_idx
            )
            all_shap_values.append(shap_values)
            all_X_samples.append(X_sample)

        # 聚合7天的SHAP值
        combined_shap = np.concatenate(all_shap_values, axis=0)
        combined_X = np.concatenate(all_X_samples, axis=0)

        # 计算聚合后的特征重要性
        importance_df = compute_feature_importance(combined_shap, feature_names)
        all_results[model_name] = importance_df

        # 保存特征重要性表格
        importance_path = os.path.join(OUTPUT_DIR, f'{model_name}_feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"\n✓ 特征重要性表格已保存: {importance_path}")

        # 打印Top 10特征
        print(f"\n{model_name} Top 10 特征:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['rank']:2d}. {row['feature']}: {row['importance']:.4f}")

        # 生成可视化
        print(f"\n生成 {model_name} 可视化图表...")

        # Summary Plot
        plot_shap_summary(
            combined_shap, combined_X, feature_names,
            f'{model_name} SHAP Summary (All 7 Days)',
            os.path.join(OUTPUT_DIR, f'{model_name}_shap_summary.png'),
            max_display=20
        )

        # Bar Plot
        plot_shap_bar(
            importance_df,
            f'{model_name} Feature Importance (Mean |SHAP|)',
            os.path.join(OUTPUT_DIR, f'{model_name}_shap_bar.png'),
            top_n=20
        )

        # Dependence Plot for top 3 features
        for rank in range(min(3, len(importance_df))):
            feat_name = importance_df.iloc[rank]['feature']
            feat_idx = feature_names.index(feat_name)
            plot_shap_dependence(
                combined_shap, combined_X, feat_idx, feature_names,
                f'{model_name} Dependence: {feat_name}',
                os.path.join(OUTPUT_DIR, f'{model_name}_shap_dependence_top{rank+1}.png')
            )

        # Waterfall Plot for a sample
        # 使用Day1的模型和数据
        plot_shap_waterfall(
            explainer, all_shap_values[0], all_X_samples[0], 0,
            feature_names,
            f'{model_name} Day1 Sample Prediction Breakdown',
            os.path.join(OUTPUT_DIR, f'{model_name}_shap_waterfall.png')
        )

    # 5. 对比分析
    if 'LightGBM_Peak' in all_results and 'XGBoost_Peak' in all_results:
        print(f"\n{'='*60}")
        print("LightGBM vs XGBoost 特征重要性对比")
        print(f"{'='*60}")

        comparison_df = plot_comparison_bar(
            all_results['LightGBM_Peak'],
            all_results['XGBoost_Peak'],
            os.path.join(OUTPUT_DIR, 'feature_importance_comparison.png'),
            top_n=15
        )

        # 保存对比表格
        comparison_path = os.path.join(OUTPUT_DIR, 'feature_importance_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"✓ 对比表格已保存: {comparison_path}")

    # 6. 生成分析报告摘要
    print(f"\n{'='*60}")
    print("SHAP分析完成")
    print(f"{'='*60}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"\n生成的文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")


if __name__ == '__main__':
    main()
