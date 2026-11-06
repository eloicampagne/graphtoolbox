"""Model training, early stopping and forecasting metrics."""

from graphtoolbox._lazy import install_lazy_exports

_SYMBOL_MODULES = {
    "Trainer": ".trainer",
    "RollingTrainer": ".trainer",
    "EarlyStopping": ".trainer",
    "set_device": ".trainer",
    "set_seed": ".trainer",
    "MAE": ".metrics",
    "NMAE": ".metrics",
    "MAPE": ".metrics",
    "RMSE": ".metrics",
    "BIAS": ".metrics",
    "reconcile": ".reconciliation",
    "error_covariance": ".reconciliation",
    "shrinkage_intensity": ".reconciliation",
}

__getattr__, __dir__, __all__ = install_lazy_exports(__name__, globals(), _SYMBOL_MODULES)
