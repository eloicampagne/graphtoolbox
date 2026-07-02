"""Dataset construction, graph building and preprocessing."""

from graphtoolbox._lazy import install_lazy_exports

_SYMBOL_MODULES = {
    "DataClass": ".dataset",
    "GraphDataset": ".dataset",
    "GraphBuilder": ".builder",
    "extract_dataframe": ".preprocessing",
    "create_variable": ".preprocessing",
    "sub_df": ".preprocessing",
    "extract_dummies": ".preprocessing",
}

__getattr__, __dir__, __all__ = install_lazy_exports(__name__, globals(), _SYMBOL_MODULES)
