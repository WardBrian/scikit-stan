import numbers
from inspect import isclass

import numpy as np

from typing import Any, Callable, List, Optional
from numpy.typing import NDArray
from numpy import float64

import warnings

from sk_stan_regression.modelcore import CoreEstimator
from ..exceptions import NotFittedError

# probably unnecessary... do the cast somewhere else?
# def _ensure_no_complex_data(array):
#    if (
#        hasattr(array, "dtype")
#        and array.dtype is not None
#        and hasattr(array.dtype, "kind")
#        and array.dtype.kind == "c"
#    ):
#        raise ValueError(f"Complex data not supported\n{array!r}\n")
#

# TODO: write docstrings for everything
def check_array(
    X: NDArray[float64],
    ensure_2d: bool = True,
    allow_nd: bool = False,
    dtype: str = "numeric",
) -> NDArray[float64]:
    """
    Input validation on an array, list, sparse matrix or similar.
    By default, the input is checked to be a non-empty 2D array containing
    only finite values.
    """
    # TODO PANDAS -> np support?

    array_res = np.asanyarray(X)

    if ensure_2d:
        # input cannot be scalar
        if X.ndim == 0:
            raise ValueError(
                f"""Expected 2D array, got scalar array instead:\narray={X!r}.\n
                    Reshape your data either using array.reshape(-1, 1) if 
                    your data has a single feature or array.reshape(1, -1) 
                    if it contains a single sample."""
            )
        if X.ndim == 1:
            warnings.warn(
                """Passed data is one-dimensional, while estimator expects
                it to be at at least two-dimensional."""
            )
            array_res = np.asanyarray(X)[:, None]

    if not allow_nd and X.ndim > 2:
        raise ValueError(
            f"""
            Passed array with {X.ndim!r} dimensions. Estimator expected <= 2. 
            """
        )

    # TODO: enforce that all values are finite & real
    # _ensure_no_complex_data(array_res)

    # TODO: enforce number of features & samples

    return array_res


# TODO: add additional arguments
def _check_y(y:NDArray[Any], y_numeric:bool=True)->NDArray[float64]:
    y = check_array(y, ensure_2d=False)

    if y_numeric:
        y = y.astype(np.float64)

    return np.asanyarray(y)


# adapted from sklearn's check_X_y validation
def check_X_y(
    X:NDArray[float64],
    y:NDArray[float64],
    ensure_X_2d: bool = True,
    allow_nd: bool = False,
    y_numeric: bool =True,
) -> tuple[NDArray[float64], NDArray[float64]]:
    X_checked = check_array(X, ensure_2d=ensure_X_2d, allow_nd=allow_nd)
    y_checked = _check_y(y, y_numeric=y_numeric)

    return X_checked, y_checked


# taken from official sklearn repo;
# TODO: simplify
def check_is_fitted(estimator:CoreEstimator, attributes:Optional[List[str]]=None, *, msg:Optional[str]=None, all_or_any:Callable=all) -> None:
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.
    If an estimator does not set any attributes with a trailing underscore, it
    can define a ``__sklearn_is_fitted__`` method returning a boolean to specify
    if the estimator is fitted or not.
    Parameters
    ----------
    estimator : estimator instance
        estimator instance for which the check is performed.
    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``
        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.
    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.
    Returns
    -------
    None
    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        fitted = all_or_any([hasattr(estimator, attr) for attr in attributes])
    else:
        fitted = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]

    if not fitted:
        raise NotFittedError(msg % {"name": type(estimator).__name__})


def check_consistent_length(*arrays: List[Any]) -> None:
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(lgth) for lgth in lengths]
        )


def _num_samples(x) -> int:
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error
