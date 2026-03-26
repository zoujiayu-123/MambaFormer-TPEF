"""
模型模块

包含所有机器学习和深度学习模型的实现。
"""

from .base_models import (
    train_xgboost,
    train_random_forest,
    train_elastic_net,
    train_svr,
    train_lightgbm
)

from .ensemble import (
    PeakAwareEnsemble,
    MLEFPeakEnhancer,
    EnhancedPeakAwareEnsemble,
    AdaptiveGaussianEnsemble,
    TimeDependentEnsemble,
    KFoldCalibratedEnsemble,
    PeakResidualCorrector,
    PeriodBasedEnsemble,
    CombinedPeriodTimeEnsemble,
    find_optimal_window_size
)

__all__ = [
    'train_xgboost',
    'train_random_forest',
    'train_elastic_net',
    'train_svr',
    'train_lightgbm',
    'PeakAwareEnsemble',
    'MLEFPeakEnhancer',
    'EnhancedPeakAwareEnsemble',
    'AdaptiveGaussianEnsemble',
    'TimeDependentEnsemble',
    'KFoldCalibratedEnsemble',
    'PeakResidualCorrector',
    'PeriodBasedEnsemble',
    'CombinedPeriodTimeEnsemble',
    'find_optimal_window_size',
]
