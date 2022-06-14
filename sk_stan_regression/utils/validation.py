import numbers
from inspect import isclass

import numpy as np

from typing import Optional
from numpy.typing import ArrayLike

from ..exceptions import NotFittedError


# TODO: write docstrings for everything 
def check_array(
    X: ArrayLike, 
    ensure_2d: Optional[bool] = True, 
    allow_nd: Optional[bool] = False, 
    ensure_min_features: Optional[int] = 1,
):
    """
    Input validation on an array, list, sparse matrix or similar.
    By default, the input is checked to be a non-empty 2D array containing
    only finite values. 
    """
    # TODO PANDAS -> np support? 

    if ensure_2d: 
        # input cannot be scalar
        if X.ndim == 0: 
            raise ValueError(
                    f"""Expected 2D array, got scalar array instead:\narray={X!r}.\n
                    Reshape your data either using array.reshape(-1, 1) if 
                    your data has a single feature or array.reshape(1, -1) 
                    if it contains a single sample."""
            )

    if not allow_nd and X.ndim > 2: 
        raise ValueError(
            f"""
            Passed array with {X.ndim!r} dimensions. Estimator expected <= 2. 
            """
        )

    
    # TODO: enforce that all values are finite 

    # TODO: enforce number of features & samples
    pass 

# custom function adapted from sklearn's validations
def _validate_data(
    X="no-validation", 
    y="no-validation", 
    ensure_X_2d: Optional[bool] = True, 
    allow_X_nd: Optional[bool] = False, 
    allow_y_multi_output: Optional[bool] = False, 
    ensure_min_features: Optional[int] = 1
):
    """
        Input validation for standard estimators.
        Checks X and y for consistent length, enforces X to be 2D and y 1D. By
        default, X is checked to be non-empty and containing only finite values.
        Standard input checks are also applied to y, such as checking that y
        does not have np.nan or np.inf targets. For multi-label y, set
        multi_output=True to allow 2D 
    """
    no_val_X = isinstance(X, str) and X == "no_validation"
    no_val_y = y is None or isinstance(y, str) and y == "no_validation"

    if no_val_X and no_val_y: 
        raise ValueError("""Validation should be done on X,y or both.""")
    elif not no_val_X and no_val_y:
        res = check_array(
            X,
            ensure_2d=ensure_X_2d,
            allow_nd=allow_X_nd, 
            ensure_min_features=ensure_min_features
        )
    elif no_val_X and not no_val_y:
        pass
    else: 
        pass

    return res 

# taken from official sklearn repo; 
# TODO: simplify
def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
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
    elif hasattr(estimator, "__sklearn_is_fitted__"):
        fitted = estimator.__sklearn_is_fitted__()
    else:
        fitted = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]

    if not fitted:
        raise NotFittedError(msg % {"name": type(estimator).__name__})


def check_consistent_length(*arrays):
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


def _num_samples(x):
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
