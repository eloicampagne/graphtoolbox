"""Smoke tests: the package and its public API import without heavy data."""


def test_package_version():
    import graphtoolbox
    assert isinstance(graphtoolbox.__version__, str) and graphtoolbox.__version__


def test_public_symbols_import():
    from graphtoolbox.models import myGNN, ConvAdapter, TemporalGNN, AdditiveGraphModel
    from graphtoolbox.training import Trainer, MAPE, RMSE, set_device
    from graphtoolbox.evaluation import (
        diebold_mariano, bootstrap_metric, model_confidence_set, pairwise_dm,
    )
    # reference them so linters do not flag unused imports
    assert all(obj is not None for obj in (
        myGNN, ConvAdapter, TemporalGNN, AdditiveGraphModel,
        Trainer, MAPE, RMSE, set_device,
        diebold_mariano, bootstrap_metric, model_confidence_set, pairwise_dm,
    ))
