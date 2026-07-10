"""Statistical evaluation of cached forecasts (significance, no retraining)."""

from graphtoolbox._lazy import install_lazy_exports

_SYMBOL_MODULES = {
    "diebold_mariano": ".significance",
    "pairwise_dm": ".significance",
    "bootstrap_metric": ".significance",
    "bootstrap_table": ".significance",
    "model_confidence_set": ".significance",
    "DMResult": ".significance",
    "BootstrapResult": ".significance",
    "MCSResult": ".significance",
}

__getattr__, __dir__, __all__ = install_lazy_exports(__name__, globals(), _SYMBOL_MODULES)
