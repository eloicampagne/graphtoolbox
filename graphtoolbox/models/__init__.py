"""Graph neural network architectures."""

from graphtoolbox._lazy import install_lazy_exports

_SYMBOL_MODULES = {
    "myGNN": ".gnn",
    "ConvAdapter": ".gnn",
    "GatedMultiConvGNN": ".gnn",
    "AdditiveGraphModel": ".gnn",
    "GCNEncoder": ".gnn",
    "VariationalGNNEncoder": ".gnn",
}

__getattr__, __dir__, __all__ = install_lazy_exports(__name__, globals(), _SYMBOL_MODULES)
