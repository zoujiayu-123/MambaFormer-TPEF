"""
训练工具模块

包含PyTorch模型训练、DataLoader创建等工具函数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class PVPowerLoss(nn.Module):
    """
    光伏功率专用损失函数

    特点:
    1. 基础MSE损失
    2. 低功率区域加权（提升夜间和阴天预测精度）
    3. 负值惩罚（虽然有ReLU，但作为双重保险）
    """

    def __init__(self, low_power_weight=2.0, neg_penalty=10.0, low_power_threshold=1.0):
        super().__init__()
        self.low_power_weight = low_power_weight
        self.neg_penalty = neg_penalty
        self.low_power_threshold = low_power_threshold

    def forward(self, pred, target):
        # 基础MSE损失
        mse_loss = F.mse_loss(pred, target)

        # 负值惩罚（理论上ReLU后不会有，但作为保险）
        neg_mask = pred < 0
        if neg_mask.any():
            neg_penalty_loss = self.neg_penalty * torch.mean(torch.abs(pred[neg_mask]))
        else:
            neg_penalty_loss = torch.tensor(0.0, device=pred.device)

        # 低功率区域加权（提升低功率预测精度）
        low_power_mask = target < self.low_power_threshold
        if low_power_mask.any():
            low_power_loss = self.low_power_weight * F.mse_loss(
                pred[low_power_mask], target[low_power_mask]
            )
        else:
            low_power_loss = torch.tensor(0.0, device=pred.device)

        # 总损失
        total_loss = mse_loss + neg_penalty_loss + low_power_loss

        return total_loss


def create_dataloaders(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                      X_test_seq, y_test_seq, device_type='cpu', config=None):
    """
    创建PyTorch DataLoader

    Args:
        X_train_seq, y_train_seq: 训练序列
        X_val_seq, y_val_seq: 验证序列
        X_test_seq, y_test_seq: 测试序列
        device_type: 设备类型 ('cpu' 或 'cuda')
        config: 配置字典

    Returns:
        train_loader, val_loader, test_loader
    """
    if config is None:
        if device_type == 'cuda':
            batch_size = 256
            num_workers = 4
        else:
            batch_size = 128
            num_workers = 0
    else:
        batch_size = config.get('batch_size', 128)
        num_workers = config.get('num_workers', 0)

    # 创建TensorDataset
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_seq).float(),
        torch.from_numpy(y_train_seq).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_seq).float(),
        torch.from_numpy(y_val_seq).float()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test_seq).float(),
        torch.from_numpy(y_test_seq).float()
    )

    # 创建DataLoader
    def make_loader(dataset, shuffle):
        kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "pin_memory": (device_type == 'cuda'),
            "num_workers": num_workers,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return DataLoader(dataset, **kwargs)

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)
    test_loader = make_loader(test_dataset, shuffle=False)

    return train_loader, val_loader, test_loader


def train_pytorch_model(model, train_loader, val_loader, device='cpu',
                       epochs=60, patience=10, learning_rate=0.0015,
                       weight_decay=1e-4):
    """
    训练PyTorch模型 (通用训练函数)

    Args:
        model: PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        epochs: 训练轮数
        patience: 早停耐心值
        learning_rate: 学习率
        weight_decay: 权重衰减

    Returns:
        训练好的模型
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=False, min_lr=1e-6
    )

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 放宽梯度裁剪: 1.0 → 5.0
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] - Val Loss: {avg_val_loss:.6f}")

        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def train_mamba_transformer_fast(model, train_loader, val_loader, device='cpu',
                                 epochs=60, patience=10, learning_rate=0.0015,
                                 weight_decay=1e-4):
    """
    快速训练Mamba-Transformer模型 (带GPU加速支持)

    Args:
        model: Mamba-Transformer模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        epochs: 训练轮数
        patience: 早停耐心值
        learning_rate: 学习率
        weight_decay: 权重衰减

    Returns:
        训练好的模型
    """
    # 使用定制的光伏功率损失函数 (阶段3修正: 适中权重)
    criterion = PVPowerLoss(low_power_weight=2.5, neg_penalty=15.0, low_power_threshold=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_loss = float('inf')
    best_state = None
    bad = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5
    )

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    out = model(xb).squeeze()
                    loss = criterion(out, yb)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(xb).squeeze()
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total += loss.item()

        # 每3个epoch才完整验证一次,减少评估开销
        do_val = (ep % 3 == 0) or (ep <= 3)
        if do_val:
            model.eval()
            vloss, vcnt = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    if device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            out = model(xb).squeeze()
                            loss = criterion(out, yb)
                    else:
                        out = model(xb).squeeze()
                        loss = criterion(out, yb)
                    vloss += loss.item()
                    vcnt += 1
            vloss /= max(1, vcnt)
            scheduler.step(vloss)

            if vloss < best_loss:
                best_loss = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"  Early stopping @ epoch {ep} | best val loss: {best_loss:.6f}")
                    break

            if (ep % 9 == 0) or (ep <= 3):
                print(f"  Epoch {ep:02d}/{epochs} | train loss {total/len(train_loader):.6f} | val loss {vloss:.6f}")
        else:
            if (ep % 9 == 0) or (ep <= 3):
                print(f"  Epoch {ep:02d}/{epochs} | train loss {total/len(train_loader):.6f}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model


def predict_with_model(model, data_loader, device='cpu'):
    """
    使用训练好的模型进行预测

    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备

    Returns:
        预测结果数组
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            out = model(xb).squeeze()
            predictions.extend(out.detach().cpu().numpy().reshape(-1).tolist())

    return np.array(predictions)


# ==================== Seq2Seq 训练函数 ====================

class Seq2SeqLoss(nn.Module):
    """Seq2Seq专用损失函数，支持时间步加权"""

    def __init__(self, horizon_weights=None, low_power_weight=1.5, neg_penalty=10.0):
        super().__init__()
        self.horizon_weights = horizon_weights
        self.low_power_weight = low_power_weight
        self.neg_penalty = neg_penalty
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, output_len)
            target: (batch, output_len)

        Returns:
            loss: 标量
        """
        # 基础MSE损失
        loss = self.mse(pred, target)  # (batch, output_len)

        # 应用时间步权重（如果提供）
        if self.horizon_weights is not None:
            weights = torch.tensor(self.horizon_weights, device=pred.device)
            # 扩展权重到所有时间步
            output_len = pred.size(1)
            if len(weights) < output_len:
                # 线性插值权重
                horizon_weights_expanded = torch.linspace(
                    weights[0], weights[-1], output_len, device=pred.device
                )
            else:
                horizon_weights_expanded = weights[:output_len]
            loss = loss * horizon_weights_expanded

        # 负值惩罚
        neg_mask = pred < 0
        if neg_mask.any():
            neg_penalty_loss = self.neg_penalty * torch.abs(pred[neg_mask]).mean()
            loss = loss.mean() + neg_penalty_loss
        else:
            loss = loss.mean()

        # 低功率区域加权（可选）
        low_power_mask = target < 1.0
        if low_power_mask.any() and self.low_power_weight > 1.0:
            low_power_loss = self.low_power_weight * self.mse(
                pred[low_power_mask], target[low_power_mask]
            ).mean()
            loss = loss + low_power_loss

        return loss


class ExtremeValueWeightedLoss(nn.Module):
    """
    极值加权损失函数 - 强调极端值预测

    对于峰值预测任务，我们希望模型更准确预测极端值（高峰值和低峰值）
    而不是仅仅预测平均值。此损失函数根据目标值与均值的偏离程度动态加权。
    """

    def __init__(self, extreme_weight=2.0, neg_penalty=10.0):
        """
        Args:
            extreme_weight: 极值权重因子（越大越强调极值）
            neg_penalty: 负值惩罚系数
        """
        super().__init__()
        self.extreme_weight = extreme_weight
        self.neg_penalty = neg_penalty
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, output_len) 预测值
            target: (batch, output_len) 真实值

        Returns:
            loss: 加权后的标量损失
        """
        # 基础MSE损失
        mse_loss = self.mse(pred, target)  # (batch, output_len)

        # 计算动态权重：根据目标值与均值的偏离程度
        target_mean = target.mean()
        target_std = target.std() + 1e-8  # 避免除零

        # 标准化偏离度: |target - mean| / std
        deviation = torch.abs(target - target_mean) / target_std

        # 权重: 1 + extreme_weight * deviation
        # 偏离越大（极值），权重越高
        weights = 1.0 + self.extreme_weight * deviation

        # 应用权重
        weighted_loss = (mse_loss * weights).mean()

        # 负值惩罚
        neg_mask = pred < 0
        if neg_mask.any():
            neg_penalty_loss = self.neg_penalty * torch.abs(pred[neg_mask]).mean()
            weighted_loss = weighted_loss + neg_penalty_loss

        return weighted_loss


def train_seq2seq_model(model, train_loader, val_loader, device='cpu',
                        epochs=150, patience=25, learning_rate=0.0003,
                        weight_decay=1e-4, teacher_forcing_schedule=None,
                        horizon_weights=None, use_extreme_loss=False, extreme_weight=2.0):
    """
    训练Seq2Seq模型

    Args:
        model: Seq2Seq模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        epochs: 训练轮次
        patience: 早停容忍度
        learning_rate: 学习率
        weight_decay: 权重衰减
        teacher_forcing_schedule: 教师强制衰减函数 (epoch -> ratio)
        horizon_weights: 时间步权重
        use_extreme_loss: 是否使用极值加权损失（适合峰值预测）
        extreme_weight: 极值权重因子（仅当use_extreme_loss=True时有效）

    Returns:
        训练好的模型
    """
    if use_extreme_loss:
        criterion = ExtremeValueWeightedLoss(extreme_weight=extreme_weight)
        print(f"  使用极值加权损失 (extreme_weight={extreme_weight})")
    else:
        criterion = Seq2SeqLoss(horizon_weights=horizon_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"  开始训练Seq2Seq模型...")
    print(f"  设备: {device}, Epochs: {epochs}, Patience: {patience}")

    for epoch in range(epochs):
        # 计算当前teacher forcing比例
        if teacher_forcing_schedule:
            tf_ratio = teacher_forcing_schedule(epoch)
        else:
            tf_ratio = max(0.5 * (0.95 ** epoch), 0.1)  # 默认衰减策略

        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            # 根据模型类型决定是否使用teacher forcing
            if hasattr(model, 'forward') and 'teacher_forcing_ratio' in \
               model.forward.__code__.co_varnames:
                outputs = model(batch_x, target=batch_y, teacher_forcing_ratio=tf_ratio)
            else:
                outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 放宽梯度裁剪: 1.0 → 5.0
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        # 打印进度（每10轮）
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:03d}/{epochs}] - "
                  f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, "
                  f"TF: {tf_ratio:.3f}")

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} | best val loss: {best_val_loss:.6f}")
                break

    # 恢复最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def predict_seq2seq(model, data_loader, device='cpu'):
    """
    使用Seq2Seq模型进行预测

    Args:
        model: Seq2Seq模型
        data_loader: 数据加载器
        device: 设备

    Returns:
        predictions: (n_samples, output_len) - 功率值已裁剪至非负
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            out = model(xb)  # (batch, output_len)
            predictions.append(out.detach().cpu().numpy())

    predictions = np.vstack(predictions)  # (n_samples, output_len)

    # 对于Mamba模型，应用非负约束（训练时未使用ReLU以避免梯度消失）
    predictions = np.clip(predictions, 0, None)

    return predictions
