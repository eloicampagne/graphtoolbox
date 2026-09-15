"""Graph neural network architectures."""

from graphtoolbox._lazy import install_lazy_exports

_SYMBOL_MODULES = {
    "myGNN": ".gnn",
    "ConvAdapter": ".gnn",
    "GatedMultiConvGNN": ".gnn",
    "AdditiveGraphModel": ".gnn",
    "GCNEncoder": ".gnn",
    "VariationalGNNEncoder": ".gnn",
    "TemporalGNN": ".temporal",
    "ConvAdapterTemporal": ".temporal",
    "lag_sequence_indices": ".temporal",
    "CovariateSeq2Seq": ".baseline",
    "arimax_forecast": ".baseline",
    "lstm_forecast": ".baseline",
    "chronos2_forecast": ".baseline",
    "xgboost_forecast": ".baseline",
    "gam_forecast_by_slot": ".baseline",
    "chronos_bolt_forecast": ".baseline",
    "fit_tabular_baseline": ".baseline",
}

__getattr__, __dir__, __all__ = install_lazy_exports(__name__, globals(), _SYMBOL_MODULES)
