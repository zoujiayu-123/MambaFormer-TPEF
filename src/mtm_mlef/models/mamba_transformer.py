"""
Mamba-Transformer混合模型

结合官方Mamba状态空间模型和Transformer注意力机制的混合架构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba
    OFFICIAL_MAMBA_AVAILABLE = True
except ImportError:
    OFFICIAL_MAMBA_AVAILABLE = False
    print("Warning: Official mamba_ssm not available, using fallback implementation")


class MambaBlock(nn.Module):
    """
    Mamba状态空间模块包装器

    优先使用官方mamba_ssm实现，提供10-100倍性能提升
    """

    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        if OFFICIAL_MAMBA_AVAILABLE:
            # 使用官方Mamba实现（CUDA优化，Selective SSM）
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,          # 卷积核大小
                expand=2,          # 扩展因子
                use_fast_path=True # 启用CUDA加速
            )
            self.dropout = nn.Dropout(dropout)
        else:
            # 降级方案：简单的1D卷积层
            self.mamba = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.dropout = None

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            输出张量 (batch, seq_len, d_model)
        """
        if OFFICIAL_MAMBA_AVAILABLE:
            # 官方Mamba：直接处理(batch, seq_len, d_model)
            output = self.mamba(x)
            output = self.dropout(output)
        else:
            # 降级方案：Conv1D需要(batch, d_model, seq_len)
            x_t = x.transpose(1, 2)  # (batch, d_model, seq_len)
            output = self.mamba(x_t)
            output = output.transpose(1, 2)  # (batch, seq_len, d_model)

        return output


class TransformerBlock(nn.Module):
    """Transformer注意力模块"""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            输出张量 (batch, seq_len, d_model)
        """
        # 多头注意力 + 残差
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # 前馈网络 + 残差
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class MambaTransformerHybrid(nn.Module):
    """Mamba-Transformer混合模型"""

    def __init__(self, n_features, d_model=80, n_heads=4, d_state=16,
                 dropout=0.2, seq_len=24):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.seq_len = seq_len

        # 输入投影
        self.input_proj = nn.Linear(n_features, d_model)

        # 第1层: Mamba块和Transformer块
        self.mamba_block1 = MambaBlock(d_model, d_state, dropout)
        self.transformer_block1 = TransformerBlock(d_model, n_heads, dropout)

        # 第2层: Mamba块和Transformer块 (增强模型深度)
        self.mamba_block2 = MambaBlock(d_model, d_state, dropout)
        self.transformer_block2 = TransformerBlock(d_model, n_heads, dropout)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # 输出层 (添加ReLU确保非负输出)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()  # 确保光伏功率预测为非负值
        )

        self._init_parameters()

    def _init_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

    def forward(self, x, return_features=False):
        """
        前向传播

        Args:
            x: (batch, seq_len, n_features)
            return_features: 是否返回隐层特征 (用于混合模型)

        Returns:
            如果return_features=False: 输出张量 (batch,) 或 (batch, 1)
            如果return_features=True: (输出, 隐层特征)
        """
        # 输入投影
        x = self.input_proj(x)

        # 添加位置编码
        x = x + self.pos_encoding[:, :x.size(1), :]

        # 第1层: Mamba + Transformer
        x = self.mamba_block1(x)
        x = self.transformer_block1(x)

        # 第2层: Mamba + Transformer (深度增强)
        x = self.mamba_block2(x)
        x = self.transformer_block2(x)

        # 全局平均池化
        x_pooled = torch.mean(x, dim=1)

        # 如果需要特征，返回池化后的特征（用于混合模型）
        if return_features:
            features = x_pooled.detach()  # 分离梯度
            output = self.output_proj(x_pooled)
            return output, features

        # 正常预测
        output = self.output_proj(x_pooled)
        return output

    def extract_features(self, x):
        """
        提取深层特征（专用于混合模型）

        Args:
            x: (batch, seq_len, n_features)

        Returns:
            features: (batch, d_model) 深层特征向量
        """
        with torch.no_grad():
            _, features = self.forward(x, return_features=True)
        return features
