"""GraphToolbox: graph machine learning utilities for time-series forecasting.

The package is organised into focused subpackages that can be imported on
demand, so that lightweight use does not pull in optional heavy dependencies
(mapping, dimensionality reduction, hyper-parameter search):

- ``graphtoolbox.data``           : dataset construction and preprocessing.
- ``graphtoolbox.models``         : graph neural network architectures.
- ``graphtoolbox.training``       : trainers and forecasting metrics.
- ``graphtoolbox.optim``          : hyper-parameter optimisation.
- ``graphtoolbox.aggregation``    : online expert aggregation.
- ``graphtoolbox.interpretability``: post-hoc explanation and ALE analysis.
- ``graphtoolbox.utils``          : graph learning, attention and plotting helpers.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("GraphToolbox")
except PackageNotFoundError:  # package not installed (e.g. running from a source checkout)
    __version__ = "0.1.0"

__all__ = ["__version__"]
