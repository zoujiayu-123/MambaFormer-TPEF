"""
深度学习模型 (PyTorch实现)

包含CNN和LSTM模型的定义和训练函数。
"""

import torch
import torch.nn as nn


class CNN1DModel(nn.Module):
    """1D卷积神经网络"""

    def __init__(self, time_steps, n_features, filters=64, kernel_size=2,
                 pool_size=2, dropout=0.2, dense_units=100):
        super(CNN1DModel, self).__init__()

        self.conv1 = nn.Conv1d(n_features, filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout1 = nn.Dropout(dropout)

        # 计算卷积和池化后的长度
        conv_out_len = time_steps - kernel_size + 1
        pool_out_len = conv_out_len // pool_size
        flatten_size = filters * pool_out_len

        self.fc1 = nn.Linear(flatten_size, dense_units)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, 1)

    def forward(self, x):
        # x shape: (batch, time_steps, n_features)
        # Conv1d expects: (batch, n_features, time_steps)
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class LSTMModel(nn.Module):
    """LSTM神经网络"""

    def __init__(self, n_features, hidden_size=50, dropout=0.2, dense_units=100):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_units)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, 1)

    def forward(self, x):
        # x shape: (batch, time_steps, n_features)
        lstm_out, _ = self.lstm(x)

        # 使用最后一个时间步的输出
        x = lstm_out[:, -1, :]
        x = self.dropout1(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


def train_cnn(X_train, y_train, X_val=None, y_val=None, config=None, device='cpu'):
    """
    训练CNN模型 (PyTorch实现)

    Args:
        X_train, y_train: 训练数据 (X_train shape: (n_samples, time_steps, n_features))
        X_val, y_val: 验证数据(可选)
        config: 模型配置
        device: 设备 ('cpu' or 'cuda')

    Returns:
        训练好的模型
    """
    if config is None:
        config = {
            'filters': 64,
            'kernel_size': 2,
            'pool_size': 2,
            'dropout': 0.2,
            'dense_units': 100,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        }

    time_steps = X_train.shape[1]
    n_features = X_train.shape[2]

    # 创建模型
    model = CNN1DModel(
        time_steps=time_steps,
        n_features=n_features,
        filters=config.get('filters', 64),
        kernel_size=config.get('kernel_size', 2),
        pool_size=config.get('pool_size', 2),
        dropout=config.get('dropout', 0.2),
        dense_units=config.get('dense_units', 100)
    ).to(device)

    # 准备数据
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)

    if X_val is not None and y_val is not None:
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        has_val = True
    else:
        has_val = False

    # 训练设置
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))

    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 32)

    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # Mini-batch训练
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # 验证
        if has_val and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_tensor)
            model.train()

            if (epoch + 1) % 10 == 0:
                avg_train_loss = total_loss / num_batches
                # print(f"  Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

    model.eval()
    return model


def train_lstm(X_train, y_train, X_val=None, y_val=None, config=None, device='cpu'):
    """
    训练LSTM模型 (PyTorch实现)

    Args:
        X_train, y_train: 训练数据 (X_train shape: (n_samples, time_steps, n_features))
        X_val, y_val: 验证数据(可选)
        config: 模型配置
        device: 设备 ('cpu' or 'cuda')

    Returns:
        训练好的模型
    """
    if config is None:
        config = {
            'units': 50,
            'dropout': 0.2,
            'dense_units': 100,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        }

    n_features = X_train.shape[2]

    # 创建模型
    model = LSTMModel(
        n_features=n_features,
        hidden_size=config.get('units', 50),
        dropout=config.get('dropout', 0.2),
        dense_units=config.get('dense_units', 100)
    ).to(device)

    # 准备数据
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)

    if X_val is not None and y_val is not None:
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        has_val = True
    else:
        has_val = False

    # 训练设置
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))

    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 32)

    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # Mini-batch训练
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # 验证
        if has_val and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_tensor)
            model.train()

            if (epoch + 1) % 10 == 0:
                avg_train_loss = total_loss / num_batches
                # print(f"  Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

    model.eval()
    return model


def predict_with_pytorch_model(model, X, device='cpu'):
    """
    使用PyTorch模型进行预测

    Args:
        model: 训练好的PyTorch模型
        X: 输入数据
        device: 设备

    Returns:
        预测结果
    """
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).squeeze()

    return predictions.cpu().numpy()
