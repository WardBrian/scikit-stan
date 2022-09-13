from ._version import __version__  # noqa
from .utils.stan import init_local_cmdstan

__all__ = ["GLM"]

init_local_cmdstan()

from .generalized_linear_regression import GLM
