"""
训练器模块

包含各种模型的训练函数：RNN模型训练、树模型峰值回归训练、峰值时刻分类训练。
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error

# 延迟导入避免循环依赖
lgb = None
xgb = None
_peak_loss_module = None


def _lazy_import_peak_loss():
    """延迟导入峰值损失函数"""
    global _peak_loss_module
    if _peak_loss_module is None:
        from src.mtm_mlef.losses import peak_loss as pl
        _peak_loss_module = pl
    return _peak_loss_module


def _lazy_import_tree_models():
    """延迟导入LightGBM和XGBoost"""
    global lgb, xgb
    if lgb is None:
        import lightgbm as lgb_module
        lgb = lgb_module
    if xgb is None:
        import xgboost as xgb_module
        xgb = xgb_module


def train_rnn_model(model_class, model_name, n_features, input_len, output_len, config,
                    train_loader, val_loader, device, save_dir=None,
                    checkpoint_manager_class=None,
                    loss_type='mse', loss_config=None,
                    return_history=False):
    """
    训练RNN模型（Mamba/LSTM/GRU）

    Args:
        model_class: 模型类（MambaSeq2Seq, LSTMSeq2Seq, GRUSeq2Seq）
        model_name: 模型名称，用于日志和保存
        n_features: 输入特征数
        input_len: 输入序列长度
        output_len: 输出序列长度
        config: 模型配置字典，包含：
            - d_model, n_heads, d_state (Mamba)
            - hidden_size, num_layers (LSTM/GRU)
            - dropout, learning_rate, weight_decay
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 训练设备
        save_dir: 模型保存目录
        checkpoint_manager_class: 检查点管理器类（可选）
        loss_type: 损失函数类型，'mse' 或 'peak'
        loss_config: 峰值损失函数配置字典（当loss_type='peak'时使用）
        return_history: 是否返回训练历史（默认False）

    Returns:
        model: 训练好的模型
        history: 训练历史字典（当return_history=True时）

    Example:
        >>> # 使用标准MSE损失
        >>> model = train_rnn_model(
        ...     MambaSeq2Seq, "Mamba", n_features=25,
        ...     input_len=720, output_len=168,
        ...     config=mamba_config,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     device=device
        ... )

        >>> # 使用峰值感知损失
        >>> peak_config = {'alpha': 1.0, 'beta': 2.5, 'gamma': 1.5}
        >>> model = train_rnn_model(
        ...     MambaSeq2Seq, "Mamba", ...,
        ...     loss_type='peak', loss_config=peak_config
        ... )
    """
    print(f"\n[模型] {model_name}")

    # 根据模型类型创建模型
    if model_name == "Mamba":
        model = model_class(
            n_features=n_features,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_state=config['d_state'],
            dropout=config['dropout'],
            input_len=input_len,
            output_len=output_len
        ).to(device)
    elif model_name == "Transformer":
        model = model_class(
            n_features=n_features,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            input_len=input_len,
            output_len=output_len
        ).to(device)
    else:  # LSTM/GRU
        model = model_class(
            n_features=n_features,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            output_len=output_len
        ).to(device)

    # 优化器配置
    lr = config.get('learning_rate', 0.0003)
    wd = config.get('weight_decay', 0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 损失函数配置
    # 支持: 'mse', 'peak'(combined), 'soft'(SoftPeakAwareLoss)
    use_peak_loss = (loss_type in ['peak', 'soft'])
    if use_peak_loss:
        pl = _lazy_import_peak_loss()
        loss_cfg = loss_config or {}
        # 根据 loss_type 选择损失函数类型
        if loss_type == 'soft':
            criterion = pl.create_peak_loss('soft', **loss_cfg)
            loss_name = 'SoftPeakAwareLoss'
        else:
            criterion = pl.create_peak_loss('combined', **loss_cfg)
            loss_name = 'CombinedPeakLoss'
        print(f"  使用峰值感知损失 ({loss_name})")
        print(f"    配置: alpha={loss_cfg.get('alpha', 1.0)}, beta={loss_cfg.get('beta', 2.0)}, "
              f"gamma={loss_cfg.get('gamma', 1.0)}, temp={loss_cfg.get('temperature', 0.1)}")
    else:
        criterion = nn.MSELoss()

    # 创建检查点管理器
    checkpoint_manager = None
    if save_dir and checkpoint_manager_class:
        checkpoint_manager = checkpoint_manager_class(save_dir, model_name)

    # 训练参数
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 20)
    max_epochs = config.get('max_epochs', 50)

    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }

    for epoch in range(max_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)

            if use_peak_loss:
                # 生成时间索引和白天标志
                batch_size = X_batch.size(0)
                # 168小时 = 7天 × 24小时，生成对应的小时索引 (0-23循环)
                time_indices = torch.arange(output_len, device=device).unsqueeze(0).expand(batch_size, -1) % 24
                is_daytime = ((time_indices >= 6) & (time_indices <= 20)).float()
                loss = criterion(output, y_batch, is_daytime, time_indices)
            else:
                loss = criterion(output, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)

                if use_peak_loss:
                    batch_size = X_batch.size(0)
                    time_indices = torch.arange(output_len, device=device).unsqueeze(0).expand(batch_size, -1) % 24
                    is_daytime = ((time_indices >= 6) & (time_indices <= 20)).float()
                    loss = criterion(output, y_batch, is_daytime, time_indices)
                else:
                    loss = criterion(output, y_batch)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)

        print(f"  Epoch [{epoch+1}/{max_epochs}] Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if checkpoint_manager:
                checkpoint_manager.save_best(
                    model, optimizer, epoch=epoch+1,
                    train_loss=train_loss, val_loss=val_loss
                )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # 加载最佳模型
    if checkpoint_manager:
        checkpoint_manager.load_best(model, device=device)

    if return_history:
        return model, history
    return model


def train_peak_predictor(model_type, X_train, y_peak_train, X_val, y_peak_val,
                         config, model_name, save_dir=None):
    """
    训练树模型进行日峰值回归预测

    为每天训练单独的回归模型，共7个模型。

    Args:
        model_type: 'lightgbm' 或 'xgboost'
        X_train: 训练特征，形状 (n_samples, n_features)
        y_peak_train: 训练标签（日峰值），形状 (n_samples, 7)
        X_val: 验证特征
        y_peak_val: 验证标签
        config: 模型配置字典，包含：
            - objective, n_estimators, learning_rate
            - max_depth, num_leaves, min_child_samples
            - subsample, colsample_bytree, reg_alpha, reg_lambda
        model_name: 模型名称
        save_dir: 保存目录

    Returns:
        models: 7个回归模型的列表

    Example:
        >>> models = train_peak_predictor(
        ...     'lightgbm', X_stat_train, y_peak_train,
        ...     X_stat_val, y_peak_val, lgb_config, 'LightGBM_Peak'
        ... )
    """
    _lazy_import_tree_models()

    print(f"\n[训练] {model_name} (峰值预测)...")
    print(f"  输入特征维度: {X_train.shape[1]}")
    print(f"  输出维度: {y_peak_train.shape[1]} (7天峰值)")

    models = []

    if model_type == 'lightgbm':
        for day in range(7):
            model = lgb.LGBMRegressor(
                objective=config['objective'],
                n_estimators=config['n_estimators'],
                learning_rate=config['learning_rate'],
                num_leaves=config['num_leaves'],
                max_depth=config['max_depth'],
                min_child_samples=config['min_child_samples'],
                subsample=config['subsample'],
                colsample_bytree=config['colsample_bytree'],
                reg_alpha=config['reg_alpha'],
                reg_lambda=config['reg_lambda'],
                n_jobs=config.get('n_jobs', -1),
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_peak_train[:, day])
            models.append(model)

    elif model_type == 'xgboost':
        for day in range(7):
            model = xgb.XGBRegressor(
                objective=config['objective'],
                n_estimators=config['n_estimators'],
                learning_rate=config['learning_rate'],
                max_depth=config['max_depth'],
                min_child_weight=config.get('min_child_weight', 1),
                subsample=config['subsample'],
                colsample_bytree=config['colsample_bytree'],
                reg_alpha=config['reg_alpha'],
                reg_lambda=config['reg_lambda'],
                n_jobs=config.get('n_jobs', -1),
                random_state=42,
                verbosity=0
            )
            model.fit(X_train, y_peak_train[:, day])
            models.append(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 验证集评估
    y_val_pred = np.stack([m.predict(X_val) for m in models], axis=1)
    val_r2 = r2_score(y_peak_val.flatten(), y_val_pred.flatten())
    val_mae = mean_absolute_error(y_peak_val.flatten(), y_val_pred.flatten())

    print(f"  峰值预测 R²: {val_r2:.4f}")
    print(f"  峰值预测 MAE: {val_mae:.2f} kW")

    # 保存模型
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{model_name}_peak_models.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(models, f)
        print(f"  模型已保存: {model_path}")

    return models


def train_peak_hour_predictor(model_type, X_train, y_peak_hour, X_val, y_peak_hour_val,
                              config, model_name, save_dir=None):
    """
    训练树模型分类器预测峰值时刻（0-23小时分类）

    使用最优超参数为每天训练单独的24类分类器。

    最优参数（来自专项训练的网格搜索）:
        - n_estimators=50
        - max_depth=3
        - learning_rate=0.1
        - num_leaves=31
        - min_child_samples=20

    Args:
        model_type: 'lightgbm' 或 'xgboost'
        X_train: 训练特征，形状 (n_samples, n_features)
        y_peak_hour: 峰值时刻标签，形状 (n_samples, 7)，值为0-23
        X_val: 验证特征
        y_peak_hour_val: 验证标签
        config: 配置字典（主要用于n_jobs等参数，核心参数使用最优值）
        model_name: 模型名称
        save_dir: 保存目录

    Returns:
        classifiers: 7个分类器的列表

    Example:
        >>> classifiers = train_peak_hour_predictor(
        ...     'lightgbm', X_feat_train, y_peak_hour_train,
        ...     X_feat_val, y_peak_hour_val, config, 'LightGBM_Hour'
        ... )
    """
    _lazy_import_tree_models()

    print(f"\n[训练] {model_name} (峰值时刻预测 - 分类任务)...")
    print(f"  输入特征维度: {X_train.shape[1]}")
    print(f"  输出: 7天×24小时分类")
    print(f"  使用最优参数: n_estimators=50, max_depth=3")

    classifiers = []
    n_jobs = config.get('n_jobs', 4)

    if model_type == 'lightgbm':
        for day in range(7):
            print(f"  训练第{day+1}天分类器...", end='', flush=True)
            clf = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=24,
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20,
                random_state=42,
                verbose=-1,
                n_jobs=n_jobs
            )
            clf.fit(X_train, y_peak_hour[:, day])

            # 计算验证集准确率
            val_pred = clf.predict(X_val)
            accuracy = (val_pred == y_peak_hour_val[:, day]).mean()
            print(f" 完成 (准确率: {accuracy:.3f})")

            classifiers.append(clf)

    elif model_type == 'xgboost':
        for day in range(7):
            print(f"  训练第{day+1}天分类器...", end='', flush=True)
            clf = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=24,
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                min_child_weight=1,
                random_state=42,
                verbosity=0,
                n_jobs=n_jobs
            )
            clf.fit(X_train, y_peak_hour[:, day])

            # 计算验证集准确率
            val_pred = clf.predict(X_val)
            accuracy = (val_pred == y_peak_hour_val[:, day]).mean()
            print(f" 完成 (准确率: {accuracy:.3f})")

            classifiers.append(clf)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 验证集评估
    val_pred_hours = np.stack([clf.predict(X_val) for clf in classifiers], axis=1)
    accuracy = (val_pred_hours == y_peak_hour_val).mean()
    print(f"  峰值时刻预测准确率: {accuracy:.2%}")

    # 计算平均误差（小时）
    hour_errors = np.abs(val_pred_hours - y_peak_hour_val)
    # 处理跨日误差（如预测23点，真实0点，误差应为1而非23）
    hour_errors = np.minimum(hour_errors, 24 - hour_errors)
    mean_hour_error = hour_errors.mean()
    print(f"  平均时刻误差: {mean_hour_error:.1f} 小时")

    # 保存模型
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{model_name}_hour_classifiers.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(classifiers, f)
        print(f"  时刻分类器已保存: {model_path}")

    return classifiers


def train_peak_hour_regressor(model_type, X_train, y_peak_hour, X_val, y_peak_hour_val,
                               config, model_name, save_dir=None):
    """
    训练树模型回归器预测峰值时刻（回归任务，比24类分类更稳定）

    将峰值时刻预测从24类分类改为回归任务，直接预测6-20之间的连续值。

    Args:
        model_type: 'lightgbm' 或 'xgboost'
        X_train: 训练特征
        y_peak_hour: 峰值时刻标签（0-23）
        X_val: 验证特征
        y_peak_hour_val: 验证标签
        config: 配置字典
        model_name: 模型名称
        save_dir: 保存目录

    Returns:
        regressors: 7个回归器的列表
    """
    _lazy_import_tree_models()

    print(f"\n[训练] {model_name} (峰值时刻预测 - 回归任务)...")
    print(f"  输入特征维度: {X_train.shape[1]}")
    print(f"  输出: 7天峰值时刻（连续值）")

    regressors = []
    n_jobs = config.get('n_jobs', 4)

    if model_type == 'lightgbm':
        for day in range(7):
            print(f"  训练第{day+1}天回归器...", end='', flush=True)
            reg = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=n_jobs
            )
            reg.fit(X_train, y_peak_hour[:, day])

            # 验证集评估
            val_pred = reg.predict(X_val)
            mae = np.abs(val_pred - y_peak_hour_val[:, day]).mean()
            # ±1小时准确率
            within_1h = (np.abs(val_pred - y_peak_hour_val[:, day]) <= 1).mean()
            print(f" MAE={mae:.2f}h, ±1h={within_1h:.1%}")

            regressors.append(reg)

    elif model_type == 'xgboost':
        for day in range(7):
            print(f"  训练第{day+1}天回归器...", end='', flush=True)
            reg = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=0,
                n_jobs=n_jobs
            )
            reg.fit(X_train, y_peak_hour[:, day])

            val_pred = reg.predict(X_val)
            mae = np.abs(val_pred - y_peak_hour_val[:, day]).mean()
            within_1h = (np.abs(val_pred - y_peak_hour_val[:, day]) <= 1).mean()
            print(f" MAE={mae:.2f}h, ±1h={within_1h:.1%}")

            regressors.append(reg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 整体评估
    val_pred_hours = np.stack([reg.predict(X_val) for reg in regressors], axis=1)
    overall_mae = np.abs(val_pred_hours - y_peak_hour_val).mean()
    overall_within_1h = (np.abs(val_pred_hours - y_peak_hour_val) <= 1).mean()
    print(f"  整体峰值时刻 MAE: {overall_mae:.2f} 小时")
    print(f"  整体 ±1h 准确率: {overall_within_1h:.1%}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{model_name}_hour_regressors.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(regressors, f)
        print(f"  时刻回归器已保存: {model_path}")

    return regressors


def train_peak_hour_coarse_classifier(model_type, X_train, y_peak_hour, X_val, y_peak_hour_val,
                                       config, model_name, save_dir=None):
    """
    训练粗粒度峰值时刻分类器（6类代替24类）

    时段划分：
    - 0: 6-8点 (早晨)
    - 1: 9-10点 (上午)
    - 2: 11-12点 (午前)
    - 3: 13-14点 (午后)
    - 4: 15-16点 (下午)
    - 5: 17-20点 (傍晚)

    Args:
        model_type: 'lightgbm' 或 'xgboost'
        X_train, y_peak_hour, X_val, y_peak_hour_val: 训练和验证数据
        config: 配置字典
        model_name: 模型名称
        save_dir: 保存目录

    Returns:
        classifiers: 7个分类器列表
        bin_centers: 分箱中心（用于还原小时）
    """
    _lazy_import_tree_models()

    # 转换为粗粒度分类
    bin_edges = [0, 6, 9, 11, 13, 15, 17, 21, 24]  # 包含夜间
    bin_centers = np.array([3, 7, 10, 12, 14, 16, 18, 22])  # 8个分箱中心

    def to_bins(y):
        y_bins = np.digitize(y, bin_edges[1:-1])  # 返回0-7
        return y_bins

    y_train_bins = to_bins(y_peak_hour)
    y_val_bins = to_bins(y_peak_hour_val)

    n_classes = len(bin_centers)

    print(f"\n[训练] {model_name} (峰值时刻预测 - 粗粒度{n_classes}类分类)...")
    print(f"  输入特征维度: {X_train.shape[1]}")
    print(f"  时段划分: 夜间, 6-8, 9-10, 11-12, 13-14, 15-16, 17-20, 夜间")

    classifiers = []
    n_jobs = config.get('n_jobs', 4)

    if model_type == 'lightgbm':
        for day in range(7):
            print(f"  训练第{day+1}天分类器...", end='', flush=True)
            clf = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=n_classes,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                random_state=42,
                verbose=-1,
                n_jobs=n_jobs
            )
            clf.fit(X_train, y_train_bins[:, day])

            val_pred_bins = clf.predict(X_val)
            val_pred_hours = bin_centers[val_pred_bins]

            # 评估
            mae = np.abs(val_pred_hours - y_peak_hour_val[:, day]).mean()
            within_1h = (np.abs(val_pred_hours - y_peak_hour_val[:, day]) <= 1).mean()
            bin_acc = (val_pred_bins == y_val_bins[:, day]).mean()
            print(f" 分箱准确率={bin_acc:.1%}, MAE={mae:.2f}h, ±1h={within_1h:.1%}")

            classifiers.append(clf)

    elif model_type == 'xgboost':
        for day in range(7):
            print(f"  训练第{day+1}天分类器...", end='', flush=True)
            clf = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=n_classes,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                min_child_weight=5,
                random_state=42,
                verbosity=0,
                n_jobs=n_jobs
            )
            clf.fit(X_train, y_train_bins[:, day])

            val_pred_bins = clf.predict(X_val)
            val_pred_hours = bin_centers[val_pred_bins]

            mae = np.abs(val_pred_hours - y_peak_hour_val[:, day]).mean()
            within_1h = (np.abs(val_pred_hours - y_peak_hour_val[:, day]) <= 1).mean()
            bin_acc = (val_pred_bins == y_val_bins[:, day]).mean()
            print(f" 分箱准确率={bin_acc:.1%}, MAE={mae:.2f}h, ±1h={within_1h:.1%}")

            classifiers.append(clf)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 整体评估
    val_pred_bins_all = np.stack([clf.predict(X_val) for clf in classifiers], axis=1)
    val_pred_hours_all = bin_centers[val_pred_bins_all]
    overall_mae = np.abs(val_pred_hours_all - y_peak_hour_val).mean()
    overall_within_1h = (np.abs(val_pred_hours_all - y_peak_hour_val) <= 1).mean()
    overall_bin_acc = (val_pred_bins_all == y_val_bins).mean()

    print(f"  整体分箱准确率: {overall_bin_acc:.1%}")
    print(f"  整体时刻 MAE: {overall_mae:.2f} 小时")
    print(f"  整体 ±1h 准确率: {overall_within_1h:.1%}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{model_name}_coarse_classifiers.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'classifiers': classifiers, 'bin_centers': bin_centers}, f)
        print(f"  粗粒度分类器已保存: {model_path}")

    return classifiers, bin_centers
