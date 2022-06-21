import numbers
import warnings
from inspect import isclass
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp  # type: ignore
from numpy import float64
from numpy.typing import NDArray

from ..exceptions import NotFittedError


# TODO: write docstrings for everything
# adapted from sklearn's data validation scheme
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
    if sp.issparse(X):
        raise ValueError(
            """Estimator does not currently support sparse data
             entry; consider extracting with .data."""
        )

    if np.any(np.iscomplex(X)):
        raise ValueError("""Complex data not supported.""")

    array_res = np.asanyarray(X, dtype=np.float64)

    if np.isnan(array_res).any():
        raise ValueError("Input contains NaN.")

    if not np.isfinite(array_res).all():
        raise ValueError(
            "Input contains infinity or a value too large for this estimator."
        )

    if ensure_2d:
        # input cannot be scalar
        shape = array_res.shape
        if shape is None or len(shape) == 0 or shape[0] == 0:
            raise ValueError(
                "Singleton array or empty array cannot be considered a valid collection."
            )

        if array_res.size == 0:
            raise ValueError(
                f"0 feature(s) (shape=({shape[0]}, 0)) while a minimum of 1 "
                "is required."
            )

        if X.ndim == 0:
            raise ValueError(
                f"""Expected 2D array, got scalar array instead:\narray={X!r}.\n
                    Reshape your data either using array.reshape(-1, 1) if 
                    your data has a single feature or array.reshape(1, -1) 
                    if it contains a single sample."""
            )
        if X.ndim == 1:
            warnings.warn(
                """Passed data is one-dimensional, while estimator expects"""
                + """ it to be at at least two-dimensional."""
            )
            array_res = np.asanyarray(X)[:, None]

    if not allow_nd and X.ndim > 2:
        raise ValueError(
            f"""
            Passed array with {X.ndim!r} dimensions. Estimator expected <= 2. 
            """
        )

    return array_res


# TODO: add additional arguments
def _check_y(y: NDArray[Any], y_numeric: bool = True) -> NDArray[float64]:
    y = check_array(y, ensure_2d=False)

    if y_numeric:
        y = y.astype(np.float64)

    return np.asanyarray(y)


# adapted from sklearn's check_X_y validation
def check_X_y(
    X: NDArray[float64],
    y: NDArray[float64],
    ensure_X_2d: bool = True,
    allow_nd: bool = False,
    y_numeric: bool = True,
) -> Tuple[NDArray[float64], NDArray[float64]]:
    X_checked = check_array(X, ensure_2d=ensure_X_2d, allow_nd=allow_nd)
    y_checked = _check_y(y, y_numeric=y_numeric)

    return X_checked, y_checked


# taken from official sklearn repo;
# TODO: simplify
def check_is_fitted(
    estimator: Any,
    attributes: Optional[List[str]] = None,
    *,
    msg: Optional[str] = None,
    all_or_any: Callable[[Iterable[object]], bool] = all,
) -> None:
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
        if hasattr(estimator, "is_fitted_"):
            fitted = estimator.is_fitted_
        else:
            raise TypeError(
                "{} has no field is_fitted_, does not conform to required API".format(
                    estimator
                )
            )

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
    uniques: NDArray[np.int_] = np.unique(lengths)  # type: ignore
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(lgth) for lgth in lengths]
        )


def _num_samples(x: Any) -> Union[int, numbers.Integral]:
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
