"""Compatibility imports for the paper's historical example scripts.

Baseline implementations now live in :mod:`graphtoolbox.models.baseline`.
New code should import them from there directly.
"""

from graphtoolbox.models.baseline import (  # noqa: F401
    CovariateSeq2Seq,
    arimax_forecast,
    chronos2_forecast,
    chronos_bolt_forecast,
    gam_forecast_by_slot,
    lstm_forecast,
    mape,
    rmse,
    smape,
    xgboost_forecast,
)

__all__ = [
    "CovariateSeq2Seq", "arimax_forecast", "chronos2_forecast",
    "chronos_bolt_forecast", "gam_forecast_by_slot", "lstm_forecast",
    "mape", "rmse", "smape", "xgboost_forecast",
]
