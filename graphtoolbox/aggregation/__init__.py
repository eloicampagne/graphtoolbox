"""Online and batch expert aggregation (Opera-style robust prediction)."""

from graphtoolbox._lazy import install_lazy_exports

_SYMBOL_MODULES = {
    "Aggregation": ".aggregation",
}

__getattr__, __dir__, __all__ = install_lazy_exports(__name__, globals(), _SYMBOL_MODULES)
