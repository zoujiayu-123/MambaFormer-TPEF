"""
Seq2Seq模型实现

包含4个Seq2Seq模型用于任务2（24h→168h）和任务4（30峰值→7峰值）：
1. MambaSeq2Seq: Mamba Encoder-Decoder架构
2. LSTMSeq2Seq: 经典LSTM Seq2Seq
3. GRUSeq2Seq: 轻量级GRU Seq2Seq
4. TransformerSeq2Seq: 标准Transformer Seq2Seq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_transformer import MambaBlock, TransformerBlock


class MambaSeq2Seq(nn.Module):
    """Mamba Seq2Seq模型 - 增强版Encoder-Decoder架构（增加解码器深度）"""

    def __init__(self, n_features, d_model=128, n_heads=8, d_state=32,
                 dropout=0.15, input_len=24, output_len=168):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model

        # 编码器（2层：1 Mamba + 1 Transformer）
        self.encoder_proj = nn.Linear(n_features, d_model)
        self.encoder_norm0 = nn.LayerNorm(d_model)

        self.encoder_mamba = MambaBlock(d_model, d_state, dropout)
        self.encoder_norm1 = nn.LayerNorm(d_model)

        self.encoder_transformer = TransformerBlock(d_model, n_heads, dropout)
        self.encoder_norm2 = nn.LayerNorm(d_model)

        # Cross-Attention: 解码器关注编码器输出
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # 解码器（平衡版：1层Mamba + 1层Transformer）
        self.decoder_mamba = MambaBlock(d_model, d_state, dropout)
        self.decoder_norm1 = nn.LayerNorm(d_model)

        self.decoder_transformer = TransformerBlock(d_model, n_heads, dropout)
        self.decoder_norm2 = nn.LayerNorm(d_model)

        # 输出投影（移除ReLU避免输出坍缩为0）
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
            # 注意：移除ReLU，因为会导致输出坍缩为全0，杀死梯度
            # 功率非负约束在训练后预测时手动应用：predictions = np.clip(predictions, 0, None)
        )

        # 位置编码
        self.encoder_pos = nn.Parameter(torch.randn(1, input_len, d_model) * 0.02)
        self.decoder_pos = nn.Parameter(torch.randn(1, output_len, d_model) * 0.02)

        self._init_parameters()

    def _init_parameters(self):
        """初始化参数（使用更小的初始值避免梯度爆炸）"""
        nn.init.xavier_uniform_(self.encoder_proj.weight, gain=0.5)
        nn.init.constant_(self.encoder_proj.bias, 0)

    def encode(self, x):
        """
        编码输入序列（简化版，减少层数）

        Args:
            x: (batch, input_len, n_features)

        Returns:
            encoder_out: (batch, input_len, d_model)
        """
        # 投影 + 位置编码
        x = self.encoder_proj(x)
        x = self.encoder_norm0(x + self.encoder_pos[:, :x.size(1), :])

        # Mamba层 + 残差
        residual = x
        x = self.encoder_mamba(x)
        x = self.encoder_norm1(x + residual)

        # Transformer层 + 残差
        residual = x
        x = self.encoder_transformer(x)
        x = self.encoder_norm2(x + residual)

        return x

    def decode(self, encoder_out):
        """
        解码生成输出序列（平衡版：1层Mamba + 1层Transformer）

        Args:
            encoder_out: (batch, input_len, d_model)

        Returns:
            output: (batch, output_len)
        """
        batch_size = encoder_out.size(0)

        # 初始解码器输入：位置编码
        decoder_input = self.decoder_pos.repeat(batch_size, 1, 1)  # (batch, output_len, d_model)

        # Cross-Attention: 解码器关注编码器
        attn_out, _ = self.cross_attention(
            decoder_input, encoder_out, encoder_out
        )
        x = self.cross_attn_norm(decoder_input + attn_out)  # 残差连接

        # Mamba解码层 + 残差
        residual = x
        x = self.decoder_mamba(x)
        x = self.decoder_norm1(x + residual)

        # Transformer解码层 + 残差
        residual = x
        x = self.decoder_transformer(x)
        x = self.decoder_norm2(x + residual)

        # 输出投影
        output = self.output_proj(x).squeeze(-1)  # (batch, output_len)
        return output

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, input_len, n_features)

        Returns:
            output: (batch, output_len)
        """
        encoder_out = self.encode(x)
        output = self.decode(encoder_out)
        return output


class LSTMSeq2Seq(nn.Module):
    """LSTM Seq2Seq模型 - 经典Encoder-Decoder with Teacher Forcing"""

    def __init__(self, n_features, hidden_size=128, num_layers=2,
                 dropout=0.2, output_len=168):
        super().__init__()
        self.output_len = output_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_features = n_features

        # 编码器
        self.encoder = nn.LSTM(
            n_features, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # 解码器
        self.decoder = nn.LSTM(
            1, hidden_size, num_layers,  # 输入是上一步预测值（标量）
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.ReLU()  # 功率非负
        )

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        """
        前向传播

        Args:
            x: (batch, input_len, n_features)
            target: (batch, output_len) - 训练时的真实值
            teacher_forcing_ratio: 使用真实值的概率

        Returns:
            output: (batch, output_len)
        """
        batch_size = x.size(0)

        # 编码
        _, (hidden, cell) = self.encoder(x)

        # 解码
        outputs = []
        decoder_input = torch.zeros(batch_size, 1, 1, device=x.device)  # 起始输入

        for t in range(self.output_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            prediction = self.fc(decoder_output.squeeze(1))  # (batch, 1)
            outputs.append(prediction)

            # Teacher forcing决策
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1].unsqueeze(-1)
            else:
                decoder_input = prediction.unsqueeze(1)

        outputs = torch.cat(outputs, dim=1)  # (batch, output_len)
        return outputs


class GRUSeq2Seq(nn.Module):
    """GRU Seq2Seq模型 - 轻量级Encoder-Decoder"""

    def __init__(self, n_features, hidden_size=128, num_layers=2,
                 dropout=0.2, output_len=168):
        super().__init__()
        self.output_len = output_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 编码器
        self.encoder = nn.GRU(
            n_features, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # 解码器
        self.decoder = nn.GRU(
            1, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.ReLU()
        )

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        """
        前向传播

        Args:
            x: (batch, input_len, n_features)
            target: (batch, output_len)
            teacher_forcing_ratio: 使用真实值的概率

        Returns:
            output: (batch, output_len)
        """
        batch_size = x.size(0)

        # 编码
        _, hidden = self.encoder(x)

        # 解码
        outputs = []
        decoder_input = torch.zeros(batch_size, 1, 1, device=x.device)

        for t in range(self.output_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            prediction = self.fc(decoder_output.squeeze(1))
            outputs.append(prediction)

            # Teacher forcing
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1].unsqueeze(-1)
            else:
                decoder_input = prediction.unsqueeze(1)

        outputs = torch.cat(outputs, dim=1)
        return outputs


class TransformerSeq2Seq(nn.Module):
    """Transformer Seq2Seq模型 - 标准Encoder-Decoder"""

    def __init__(self, n_features, d_model=128, n_heads=8, num_layers=4,
                 dropout=0.15, input_len=24, output_len=168):
        super().__init__()
        self.d_model = d_model
        self.output_len = output_len

        # 输入投影
        self.input_proj = nn.Linear(n_features, d_model)

        # Transformer编码器-解码器
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        # 位置编码
        self.encoder_pos = nn.Parameter(torch.randn(1, input_len, d_model))
        self.decoder_pos = nn.Parameter(torch.randn(1, output_len, d_model))

        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()
        )

        self._init_parameters()

    def _init_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.normal_(self.encoder_pos, mean=0, std=0.02)
        nn.init.normal_(self.decoder_pos, mean=0, std=0.02)

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, input_len, n_features)

        Returns:
            output: (batch, output_len)
        """
        batch_size = x.size(0)

        # 编码器输入
        src = self.input_proj(x) + self.encoder_pos[:, :x.size(1), :]

        # 解码器输入（可学习的位置编码）
        tgt = self.decoder_pos.repeat(batch_size, 1, 1)

        # Transformer
        output = self.transformer(src, tgt)

        # 输出投影
        output = self.fc_out(output).squeeze(-1)
        return output  # (batch, output_len)
