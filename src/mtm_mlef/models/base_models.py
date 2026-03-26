"""
传统机器学习模型

包含XGBoost, RandomForest, ElasticNet, SVR, LightGBM等基础模型的训练函数。
"""

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from lightgbm import LGBMRegressor, early_stopping


def train_xgboost(X_train, y_train, X_val=None, y_val=None, config=None):
    """
    训练XGBoost模型

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据(可选)
        config: 模型配置

    Returns:
        训练好的模型
    """
    if config is None:
        config = {
            'objective': 'reg:squarederror',
            'colsample_bytree': 0.3,
            'learning_rate': 0.1,
            'max_depth': 5,
            'alpha': 10,
            'n_estimators': 100,
            'random_state': 42
        }

    model = xgb.XGBRegressor(**config)
    model.fit(X_train, y_train, verbose=False)

    return model


def train_random_forest(X_train, y_train, X_val=None, y_val=None, config=None):
    """
    训练Random Forest模型

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据(可选)
        config: 模型配置

    Returns:
        训练好的模型
    """
    if config is None:
        config = {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }

    model = RandomForestRegressor(**config)
    model.fit(X_train, y_train)

    return model


def train_elastic_net(X_train, y_train, X_val=None, y_val=None, config=None):
    """
    训练ElasticNet模型

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据(可选)
        config: 模型配置

    Returns:
        训练好的模型
    """
    if config is None:
        config = {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'random_state': 42,
            'max_iter': 2000
        }

    model = ElasticNet(**config)
    model.fit(X_train, y_train)

    return model


def train_svr(X_train, y_train, X_val=None, y_val=None, config=None):
    """
    训练SVR模型

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据(可选)
        config: 模型配置

    Returns:
        训练好的模型
    """
    if config is None:
        config = {
            'kernel': 'rbf',
            'C': 100,
            'epsilon': 0.05
        }

    model = SVR(**config)
    model.fit(X_train, y_train)

    return model


def train_lightgbm(X_train, y_train, X_val=None, y_val=None, config=None):
    """
    训练LightGBM模型

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据(可选)
        config: 模型配置

    Returns:
        训练好的模型
    """
    if config is None:
        config = {
            'objective': 'regression',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }

    early_stopping_rounds = config.pop('early_stopping_rounds', 50)

    model = LGBMRegressor(**config)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(early_stopping_rounds, verbose=False)]
        )
    else:
        model.fit(X_train, y_train)

    return model
