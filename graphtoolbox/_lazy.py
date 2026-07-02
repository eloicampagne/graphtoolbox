"""Lazy public-API loading for subpackages (PEP 562).

Each subpackage exposes its public symbols through a single mapping from symbol
name to the submodule that defines it. Resolution is deferred until the symbol
is first accessed, which keeps ``from graphtoolbox.sub import Symbol`` ergonomic
without importing optional heavy dependencies (mapping, dimensionality
reduction, hyper-parameter search) as soon as a lightweight submodule is
imported, and without eagerly triggering the circular imports that exist
between the data, models and training subpackages.
"""

from importlib import import_module
from typing import Callable, Dict, List, Tuple


def install_lazy_exports(
    package: str,
    package_globals: dict,
    symbol_modules: Dict[str, str],
) -> Tuple[Callable[[str], object], Callable[[], List[str]], List[str]]:
    """Build the ``__getattr__``, ``__dir__`` and ``__all__`` of a subpackage.

    Parameters
    ----------
    package : str
        The importing subpackage's ``__name__``.
    package_globals : dict
        The importing subpackage's ``globals()``, used to cache resolved symbols.
    symbol_modules : dict
        Mapping from public symbol name to the relative submodule (e.g.
        ``".dataset"``) that defines it.

    Returns
    -------
    tuple
        The ``__getattr__`` and ``__dir__`` callables and the sorted ``__all__``.
    """

    public_names = sorted(symbol_modules)

    def __getattr__(name: str) -> object:
        try:
            submodule = symbol_modules[name]
        except KeyError:
            raise AttributeError(
                f"module {package!r} has no attribute {name!r}"
            ) from None
        obj = getattr(import_module(submodule, package), name)
        package_globals[name] = obj
        return obj

    def __dir__() -> List[str]:
        return sorted(set(package_globals) | set(public_names))

    return __getattr__, __dir__, public_names
