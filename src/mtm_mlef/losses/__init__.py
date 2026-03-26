"""
损失函数模块
"""

from .peak_loss import (
    PeakAwareLoss,
    WeightedPeakAwareLoss,
    SoftPeakAwareLoss,
    CurriculumPeakLoss,
    CombinedPeakLoss,
    create_peak_loss
)

__all__ = [
    'PeakAwareLoss',
    'WeightedPeakAwareLoss',
    'SoftPeakAwareLoss',
    'CurriculumPeakLoss',
    'CombinedPeakLoss',
    'create_peak_loss',
]
