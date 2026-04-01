"""
随机种子设置工具

确保实验可复现性。
"""

import os
import random
import numpy as np


def set_random_seed(seed=42):
    """
    设置所有随机种子以确保结果可复现

    设置的种子包括：
    - Python random
    - NumPy random
    - PyTorch random
    - CUDA random
    - CUDA deterministic
    - LightGBM/XGBoost (通过PYTHONHASHSEED)

    Args:
        seed: 随机种子值，默认42

    Example:
        >>> from mtm_mlef.utils import set_random_seed
        >>> set_random_seed(42)
        >>> # 之后的所有随机操作都是可复现的
    """
    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # 尝试设置 PyTorch random（如果可用）
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # 设置环境变量（影响LightGBM、XGBoost等）
    os.environ['PYTHONHASHSEED'] = str(seed)
