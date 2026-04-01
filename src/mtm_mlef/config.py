"""
配置管理模块

负责加载和管理YAML配置文件。
"""

import yaml
from pathlib import Path


class Config:
    """配置类,封装所有配置参数"""

    def __init__(self, config_dict):
        """
        初始化配置对象

        Args:
            config_dict: 配置字典
        """
        self._config = config_dict

    def __getitem__(self, key):
        """支持字典式访问"""
        return self._config[key]

    def get(self, key, default=None):
        """获取配置值,支持默认值"""
        return self._config.get(key, default)

    @property
    def data(self):
        """数据配置"""
        return self._config.get('data', {})

    @property
    def random_seed(self):
        """随机种子"""
        return self._config.get('random_seed', 42)

    @property
    def sequences(self):
        """序列配置"""
        return self._config.get('sequences', {})

    @property
    def tasks(self):
        """任务配置"""
        return self._config.get('tasks', {})

    @property
    def models(self):
        """模型配置"""
        return self._config.get('models', {})

    @property
    def ensemble(self):
        """集成学习配置"""
        return self._config.get('ensemble', {})

    @property
    def mlef(self):
        """MLEF元学习集成框架配置"""
        return self._config.get('mlef', {})

    @property
    def metrics(self):
        """评估指标配置"""
        return self._config.get('metrics', {})

    @property
    def interpretability(self):
        """可解释性配置"""
        return self._config.get('interpretability', {})

    @property
    def visualization(self):
        """可视化配置"""
        return self._config.get('visualization', {})

    @property
    def output(self):
        """输出配置"""
        return self._config.get('output', {})


def load_config(config_path='config.yaml'):
    """
    从YAML文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        Config对象
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def get_default_config():
    """
    获取默认配置 (当配置文件不存在时使用)

    Returns:
        Config对象
    """
    default_config = {
        'data': {
            'file_path': 'combined_pv_data2016.csv',
            'timestamp_col': 'TIMESTAMP',
            'target_col': 'InvPAC_kW_Avg',
            'train_ratio': 0.72,
            'val_ratio': 0.08,
            'test_ratio': 0.20
        },
        'random_seed': 42,
        'sequences': {
            'seq_len': 24,
            'cnn_time_steps': 5
        },
        'models': {
            'xgboost': {
                'objective': 'reg:squarederror',
                'colsample_bytree': 0.3,
                'learning_rate': 0.1,
                'max_depth': 5,
                'alpha': 10,
                'n_estimators': 100
            },
            'random_forest': {
                'n_estimators': 100,
                'n_jobs': -1
            },
            'elastic_net': {
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'max_iter': 2000
            },
            'svr': {
                'kernel': 'rbf',
                'C': 100,
                'epsilon': 0.05
            },
            'lightgbm': {
                'objective': 'regression',
                'n_estimators': 500,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'early_stopping_rounds': 50
            },
            'cnn': {
                'filters': 64,
                'kernel_size': 2,
                'pool_size': 2,
                'dropout': 0.2,
                'dense_units': 100,
                'epochs': 50,
                'batch_size': 32
            },
            'lstm': {
                'units': 50,
                'dropout': 0.2,
                'dense_units': 100,
                'epochs': 50,
                'batch_size': 32
            },
            'mamba_transformer': {
                'd_model': 80,
                'n_heads': 4,
                'd_state': 16,
                'dropout': 0.2,
                'learning_rate': 0.0015,
                'weight_decay': 0.0001,
                'epochs': 60,
                'batch_size_gpu': 256,
                'batch_size_cpu': 128,
                'patience': 10,
                'num_workers_gpu': 4,
                'num_workers_cpu': 0
            }
        },
        'ensemble': {
            'stacking': {
                'alpha': 1.0
            },
            'blending': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 3
            }
        },
        'metrics': {
            'mape_threshold': 0.5,
            'smape_eps': 1e-8
        },
        'visualization': {
            'plot_samples': 500,
            'scatter_samples': 2000,
            'dpi': 300,
            'figsize': {
                'width': 18,
                'height': 12
            }
        },
        'output': {
            'save_models': False,
            'save_predictions': True,
            'save_plots': True,
            'results_file': 'mtm_mlef_complete_results.xlsx',
            'predictions_file': 'mtm_mlef_predictions.xlsx',
            'plot_file': 'mtm_mlef_complete_analysis.png'
        }
    }

    return Config(default_config)
