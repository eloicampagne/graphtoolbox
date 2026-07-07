"""Model training, early stopping and forecasting metrics."""

from graphtoolbox._lazy import install_lazy_exports

_SYMBOL_MODULES = {
    "Trainer": ".trainer",
    "RollingTrainer": ".trainer",
    "EarlyStopping": ".trainer",
    "set_device": ".trainer",
    "MAE": ".metrics",
    "NMAE": ".metrics",
    "MAPE": ".metrics",
    "RMSE": ".metrics",
    "BIAS": ".metrics",
}

__getattr__, __dir__, __all__ = install_lazy_exports(__name__, globals(), _SYMBOL_MODULES)
