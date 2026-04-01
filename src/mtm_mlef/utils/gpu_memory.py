"""
GPU显存管理工具

自动检测可用显存并动态调整batch_size，尽量用满显存。
"""

import torch
import gc
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelMemoryProfile:
    """模型显存占用配置"""
    name: str
    base_batch_size: int  # 基准batch_size
    base_memory_gb: float  # 基准显存占用(GB)
    memory_per_sample_mb: float  # 每个样本的显存占用(MB)
    min_batch_size: int = 8  # 最小batch_size
    max_batch_size: int = 512  # 最大batch_size


# 预设的模型显存配置（基于720h->168h任务，RTX 5090实测）
# 实测：LSTM batch_size=48 只用 6GB，需大幅调高
MODEL_MEMORY_PROFILES = {
    'Mamba': ModelMemoryProfile(
        name='Mamba',
        base_batch_size=256,
        base_memory_gb=4.0,
        memory_per_sample_mb=70,  # 实测: batch=336合适
        min_batch_size=512,  # 固定为336
        max_batch_size=512  # 固定为336
    ),
    'LSTM': ModelMemoryProfile(
        name='LSTM',
        base_batch_size=256,
        base_memory_gb=3.0,
        memory_per_sample_mb=25,  # 实测: batch=48用6GB -> 每样本约125MB，调低估算
        min_batch_size=512,
        max_batch_size=512
    ),
    'GRU': ModelMemoryProfile(
        name='GRU',
        base_batch_size=256,
        base_memory_gb=2.5,
        memory_per_sample_mb=20,  # GRU比LSTM更轻量
        min_batch_size=512,
        max_batch_size=512
    ),
    'Transformer': ModelMemoryProfile(
        name='Transformer',
        base_batch_size=128,
        base_memory_gb=4.0,
        memory_per_sample_mb=50,  # Transformer自注意力显存大
        min_batch_size=512,
        max_batch_size=512
    ),
}


class GPUMemoryManager:
    """
    GPU显存管理器

    自动检测可用显存并为不同模型计算最优batch_size
    """

    def __init__(
        self,
        target_utilization: float = 0.90,  # 目标显存利用率
        safety_margin_gb: float = 1.0,     # 安全余量(GB)
        verbose: bool = True
    ):
        """
        Args:
            target_utilization: 目标显存利用率 (0.0-1.0)
            safety_margin_gb: 安全余量，预留给系统和其他进程
            verbose: 是否打印详细信息
        """
        self.target_utilization = target_utilization
        self.safety_margin_gb = safety_margin_gb
        self.verbose = verbose

        # 检测GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_gpu = torch.cuda.is_available()

        if self.has_gpu:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            self.gpu_name = "CPU"
            self.total_memory_gb = 0

        if self.verbose and self.has_gpu:
            print(f"\n[GPU显存管理器初始化]")
            print(f"  设备: {self.gpu_name}")
            print(f"  总显存: {self.total_memory_gb:.1f} GB")
            print(f"  目标利用率: {self.target_utilization*100:.0f}%")
            print(f"  安全余量: {self.safety_margin_gb:.1f} GB")

    def get_available_memory(self) -> Tuple[float, float]:
        """
        获取当前可用显存

        Returns:
            (可用显存GB, 已用显存GB)
        """
        if not self.has_gpu:
            return 0.0, 0.0

        # 先清理缓存
        torch.cuda.empty_cache()
        gc.collect()

        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)

        # 可用 = 总显存 - 已分配 - 安全余量
        available = self.total_memory_gb - allocated - self.safety_margin_gb

        return max(0, available), allocated

    def calculate_optimal_batch_size(
        self,
        model_name: str,
        input_len: int = 720,
        output_len: int = 168,
        n_features: int = 25,
        custom_profile: Optional[ModelMemoryProfile] = None
    ) -> int:
        """
        计算最优batch_size

        Args:
            model_name: 模型名称 ('Mamba', 'LSTM', 'GRU', 'Transformer')
            input_len: 输入序列长度
            output_len: 输出序列长度
            n_features: 特征数量
            custom_profile: 自定义显存配置

        Returns:
            最优batch_size
        """
        if not self.has_gpu:
            # CPU模式，使用较小的batch_size
            return 16

        # 获取模型配置
        if custom_profile:
            profile = custom_profile
        elif model_name in MODEL_MEMORY_PROFILES:
            profile = MODEL_MEMORY_PROFILES[model_name]
        else:
            # 未知模型，使用保守配置
            profile = ModelMemoryProfile(
                name=model_name,
                base_batch_size=32,
                base_memory_gb=8.0,
                memory_per_sample_mb=150,
                min_batch_size=8,
                max_batch_size=256
            )

        # 获取可用显存
        available_gb, used_gb = self.get_available_memory()
        target_memory_gb = available_gb * self.target_utilization

        if self.verbose:
            print(f"\n[{model_name}] 显存分析:")
            print(f"  可用显存: {available_gb:.2f} GB")
            print(f"  目标使用: {target_memory_gb:.2f} GB")

        # 根据序列长度调整显存占用估算
        # 720h -> 168h 比标准配置更大
        seq_factor = (input_len * output_len) / (720 * 168)
        feature_factor = n_features / 25
        adjusted_memory_per_sample = profile.memory_per_sample_mb * seq_factor * feature_factor

        # 模型基础开销 (参数、梯度等)
        base_overhead_gb = profile.base_memory_gb * 0.3  # 约30%是基础开销

        # 可用于数据的显存
        data_memory_gb = target_memory_gb - base_overhead_gb

        if data_memory_gb <= 0:
            if self.verbose:
                print(f"  警告: 可用显存不足，使用最小batch_size")
            return profile.min_batch_size

        # 计算batch_size
        # 训练时需要: 输入 + 输出 + 梯度(约2倍) + 优化器状态(约1倍) ≈ 4倍
        training_multiplier = 4.0
        memory_per_sample_gb = (adjusted_memory_per_sample * training_multiplier) / 1024

        optimal_batch_size = int(data_memory_gb / memory_per_sample_gb)

        # 限制在合理范围内
        optimal_batch_size = max(profile.min_batch_size, optimal_batch_size)
        optimal_batch_size = min(profile.max_batch_size, optimal_batch_size)

        # 调整为8的倍数，优化GPU效率
        optimal_batch_size = (optimal_batch_size // 8) * 8
        optimal_batch_size = max(profile.min_batch_size, optimal_batch_size)

        if self.verbose:
            estimated_usage = base_overhead_gb + (optimal_batch_size * memory_per_sample_gb)
            print(f"  估算每样本显存: {adjusted_memory_per_sample:.1f} MB")
            print(f"  最优batch_size: {optimal_batch_size}")
            print(f"  预计显存使用: {estimated_usage:.2f} GB ({estimated_usage/self.total_memory_gb*100:.1f}%)")

        return optimal_batch_size

    def get_all_batch_sizes(
        self,
        model_names: list = None,
        input_len: int = 720,
        output_len: int = 168,
        n_features: int = 25
    ) -> Dict[str, int]:
        """
        获取所有模型的最优batch_size

        Args:
            model_names: 模型名称列表，默认为所有预设模型
            input_len: 输入序列长度
            output_len: 输出序列长度
            n_features: 特征数量

        Returns:
            {模型名: batch_size} 字典
        """
        if model_names is None:
            model_names = list(MODEL_MEMORY_PROFILES.keys())

        batch_sizes = {}
        for name in model_names:
            batch_sizes[name] = self.calculate_optimal_batch_size(
                name, input_len, output_len, n_features
            )

        return batch_sizes

    def clear_memory(self):
        """清理GPU显存"""
        if self.has_gpu:
            torch.cuda.empty_cache()
            gc.collect()

            if self.verbose:
                available, used = self.get_available_memory()
                print(f"\n[显存清理] 已用: {used:.2f} GB, 可用: {available:.2f} GB")

    def memory_summary(self) -> str:
        """返回当前显存使用摘要"""
        if not self.has_gpu:
            return "CPU模式，无GPU显存"

        available, used = self.get_available_memory()

        return (
            f"GPU: {self.gpu_name}\n"
            f"总显存: {self.total_memory_gb:.1f} GB\n"
            f"已使用: {used:.2f} GB ({used/self.total_memory_gb*100:.1f}%)\n"
            f"可用: {available:.2f} GB ({available/self.total_memory_gb*100:.1f}%)"
        )


def get_optimal_batch_size(
    model_name: str,
    input_len: int = 720,
    output_len: int = 168,
    n_features: int = 25,
    target_utilization: float = 0.90,
    verbose: bool = True
) -> int:
    """
    便捷函数：获取模型的最优batch_size

    Args:
        model_name: 模型名称
        input_len: 输入序列长度
        output_len: 输出序列长度
        n_features: 特征数量
        target_utilization: 目标显存利用率
        verbose: 是否打印信息

    Returns:
        最优batch_size
    """
    manager = GPUMemoryManager(
        target_utilization=target_utilization,
        verbose=verbose
    )
    return manager.calculate_optimal_batch_size(
        model_name, input_len, output_len, n_features
    )


def auto_batch_size_decorator(model_name: str):
    """
    装饰器：自动调整训练函数的batch_size

    用法:
        @auto_batch_size_decorator('Mamba')
        def train_mamba(model, train_loader, ...):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = GPUMemoryManager(verbose=True)
            optimal_bs = manager.calculate_optimal_batch_size(model_name)

            # 如果kwargs中有batch_size，替换它
            if 'batch_size' in kwargs:
                print(f"[自动调整] batch_size: {kwargs['batch_size']} -> {optimal_bs}")
                kwargs['batch_size'] = optimal_bs

            return func(*args, **kwargs)
        return wrapper
    return decorator


# 便捷接口
def print_gpu_info():
    """打印GPU信息"""
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"总显存: {total:.1f} GB")
        print(f"已分配: {allocated:.2f} GB")
        print(f"已保留: {reserved:.2f} GB")
        print(f"可用: {total - reserved:.2f} GB")
    else:
        print("未检测到GPU，使用CPU模式")
