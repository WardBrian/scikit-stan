import numbers
import warnings
from inspect import isclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from ..exceptions import NotFittedError


""" GENERAL LINK MAP
     identity - 0
     log - 1
     inverse - 2
     sqrt - 3
     inverse-square - 4
     logit - 5
     probit - 6
     cloglog - 7
     cauchit - 8
                    """

# NOTE: linear model regression assumes constant variance,

GAUSSIAN_LINKS = {"identity": 0, "log": 1, "inverse": 2}


# corresponding to logistic, normal and Cauchy CDFs respectively
BINOMIAL_LINKS = {"log": 1, "logit": 5, "probit": 6, "cloglog": 7, "cauchit": 8}

# NOTE: the Gamma regression is on parameterized Gamma(mu, alpha)
# where alpha is considered fixed as linear models all assume constant variance
GAMMA_LINKS = {
    "identity": 0,
    "log": 1,
    "inverse": 2,
}


POISSON_LINKS = {"identity": 0, "log": 1, "sqrt": 3}


INVERSE_GAUSSIAN_LINKS = {"identity": 0, "log": 1, "inverse": 2, "inverse-square": 4}


FAMILY_LINKS_MAP = {
    "gaussian": GAUSSIAN_LINKS,
    "binomial": BINOMIAL_LINKS,
    "gamma": GAMMA_LINKS,
    "poisson": POISSON_LINKS,
    "inverse-gaussian": INVERSE_GAUSSIAN_LINKS,
    # "binomial" : BINOMIAL_LINKS
}

SLOPE_PRIOR_DEFAULTS_INFO = {
    "normal": "normal(0, 2.5 * sd(y) / sd(X)) if Gaussian else normal(0, 2.5)",
    # NOTE: the laplace distribution is translation invariant
    "laplace": "laplace(0, 2.5)",
}

INTERCEPT_PRIOR_DEFAULTS_INFO = {
    "normal": "normal(mu(y), 2.5 * sd(y)) if Gaussian else normal(0, 2.5)",
    # NOTE: the laplace distribution is translation invariant
    "laplace": "double_exponential(0, 2.5)",
}

PRIORS_MAP = {
    "normal": 0,  # normal distribution, requires location (mu) and scale (sigma)
    # or else defaults to rstanarm's default:
    # https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html
    "laplace": 1,  # laplace distribution, requires location (mu) and scale (sigma)
    # or else defaults to double_exponential(0, 1)
    # for suggestions on this prior, refer to:
    #  https://www.jstor.org/stable/1403571#metadata_info_tab_contents
}

PRIORS_AUX_MAP = {
    "exponential": 0,  # exponential distribution, requires only beta parameter
    "chi2": 1,  # chi-squared distribution, requires only nu parameter
}


# NOTE: family and link combinations match R families package
# package: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family
def validate_family(family: str, link: Optional[str]) -> None:
    """Validate family and link combination choice.

    Parameters
    ----------
    family : str
        Name of chosen family. Only the following families are supported:
        "gaussian", "binomial", "gamma", "poisson", "inverse-gaussian".
    link : Optional[str]
        Name of chosen link function. Only the following combinations are supported,
        following the R families package:

            * "gaussian":
                * "identity" - Identity link function,
                * "log" - Log link function,
                * "inverse" - Inverse link function
            * "gamma":
                * "identity" - Identity link function,
                * "log" - Log link function,
                * "inverse" - Inverse link function
            * "inverse-gaussian":
                * "identity" - Identity link function,
                * "log" - Log link function,
                * "inverse" - Inverse link function,
                * "inverse-square" - Inverse square link function
            * "poisson":
                * "identity" - Identity link function,
                * "log" - Log link function,
                * "sqrt" - Square root link function
            * "binomial":
                * "log" - Log link function,
                * "logit" - Logit link function,
                * "probit" - Probit link function,
                * "cloglog" - Complementary log-log link function,
                * "cauchit" - Cauchit link function

        If an invalid combination of family and link is passed, a ValueError is raised.

    Raises
    ------
    ValueError
        Passed family is not supported.
    ValueError
        Passed link is not supported or is not valid for the chosen family.

    """

    if not link:
        raise ValueError(f"Link function must be specified for family {family!r}")

    if family not in FAMILY_LINKS_MAP:
        raise ValueError(
            f"""Family {family} not supported.
            Supported families: {list(FAMILY_LINKS_MAP.keys())}"""
        )

    if link not in FAMILY_LINKS_MAP[family].keys():
        raise ValueError(
            f"""Link {link} not supported for family {family}.
            These links are supported for {family}: {FAMILY_LINKS_MAP[family].keys()}."""
        )


# adapted from sklearn's data validation scheme which is distributed under the 3-Clause BSD License.
def check_array(
    X: ArrayLike,
    ensure_2d: bool = True,
    allow_nd: bool = False,
    dtype: type = np.float64,
) -> NDArray[Union[np.float64, np.int64]]:
    """Input validation on an array, list, sparse matrix or similar.
    By default, the input is checked to be a non-empty 2D array containing
    only finite values.

    Parameters
    ----------
    X : ArrayLike
        Array-like, list, sparse matrix, or similar of data to be checked.
    ensure_2d : bool, optional
        Whether to ensure that the array is 2D
    allow_nd : bool, optional
        Whether to allow the array to be an n-dimensional matrix where n > 2
    dtype : type, optional
        Dtype of the data; regressions only supported on
        float64 or int64 arrays

    Returns
    -------
    NDArray[Union[np.float64, np.int64]]
        Verified set of data that can be used for regression.

    Raises
    ------
    ValueError
        Sparse, complex, or otherwise invalid data type passed for X.
    ValueError
        Invalid number of dimensions in data passed for X, or otherwise data that
        cannot be recast to satisfy dimension requirements

    """
    # NOTE: cmdstanpy automatically deals with Pandas dataframes
    if sp.issparse(X):
        raise ValueError(
            """Estimator does not currently support sparse data
             entry; consider extracting with .data."""
        )

    if np.any(np.iscomplex(X)):  # type: ignore
        raise ValueError("""Complex data not supported.""")

    array_res: NDArray[Union[np.float64, np.int64]] = np.asarray(X, dtype=dtype)

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

        if array_res.ndim == 0:
            raise ValueError(
                f"""Expected 2D array, got scalar array instead:\narray={X!r}.\n
                    Reshape your data either using array.reshape(-1, 1) if
                    your data has a single feature or array.reshape(1, -1)
                    if it contains a single sample."""
            )
        if array_res.ndim == 1:
            warnings.warn(
                """Passed data is one-dimensional, while estimator expects"""
                + """ it to be at at least two-dimensional."""
            )
            array_res = np.asanyarray(X)[:, None]

    if not allow_nd and array_res.ndim > 2:
        raise ValueError(
            f"""
            Passed array with {array_res.ndim!r} dimensions. Estimator expected <= 2.
            """
        )

    return array_res


def _check_y(
    y: ArrayLike, dtype: type = np.float64
) -> NDArray[Union[np.float64, np.int64]]:
    return check_array(y, ensure_2d=False, dtype=dtype)


# adapted from sklearn's check_X_y validation scheme which
# is distributed under the 3-Clause BSD License.
def check_X_y(
    X: ArrayLike,
    y: ArrayLike,
    ensure_X_2d: bool = True,
    allow_nd: bool = False,
    dtype: type = np.float64,
) -> Tuple[NDArray[Union[np.float64, np.int64]], NDArray[Union[np.float64, np.int64]]]:
    X_checked = check_array(X, ensure_2d=ensure_X_2d, allow_nd=allow_nd, dtype=dtype)
    y_checked = _check_y(y, dtype=dtype)

    return X_checked, y_checked


# NOTE: This is derived from sk-learn's validation checks, which
# are distributed under the 3-Clause BSD License.
def check_is_fitted(
    estimator: Any,
    attributes: Optional[List[str]] = None,
    *,
    msg: Optional[str] = None,
    all_or_any: Callable[[Iterable[object]], bool] = all,
) -> None:
    """
    Perform is_fitted validation for estimator.
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
        If ``None``, ``estimator`` is considered fitted if there exist an
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


def validate_prior(prior_spec: Dict[str, Any], coeff_type: str) -> Dict[str, Any]:
    """
    Perform validation on given prior dictionary for prior on either slope or intercept.
    This is only called when there is a prior to check.

    Parameters
    ----------
    prior_spec : Dict[str, Any]
        Proposed prior dictionary, can be either for slope or intercept.
    coeff_type : str
        Specify whether the prior is for slope or intercept - should only be
        'slope' or 'intercept'.

    Returns
    -------
    Dict[str, Any]
        Validated dictionary of parameters for the given prior.

    Raises
    ------
    ValueError
        Validating a non-(slope or intercept) prior type.
    ValueError
        Prior distribution is not specified.
    ValueError
        Not all parameters for prior set-up are specified.
    ValueError
        Prior sigma is negative.
    """
    if coeff_type not in ["slope", "intercept"]:
        raise ValueError(
            "coeff_type should be either 'slope' or 'intercept', "
            "got {}".format(coeff_type)
        )

    config_keys = prior_spec.keys()

    if "prior_" + coeff_type + "_dist" not in config_keys:
        raise ValueError(
            f"""prior_{coeff_type}_dist must be specified in prior given by {prior_spec}."""
        )

    dist_key = prior_spec["prior_" + coeff_type + "_dist"]

    if dist_key not in PRIORS_MAP.keys():
        raise ValueError(
            f"Prior {dist_key} in prior specification {prior_spec} not supported."
        )

    if "prior_" + coeff_type + "_mu" not in config_keys:
        raise ValueError(
            f"""prior_{coeff_type}_mu must be specified in prior given by {prior_spec}."""
        )

    if "prior_" + coeff_type + "_sigma" not in config_keys:
        raise ValueError(
            f"""prior_{coeff_type}_sigma must be specified in prior given by {prior_spec}."""
        )

    sigmas = prior_spec["prior_" + coeff_type + "_sigma"]

    if isinstance(sigmas, Sequence):
        if any(x < 0 for x in sigmas):
            raise ValueError(
                f"""prior_{coeff_type}_sigma must be positive in prior given by {prior_spec}."""
            )
    else:
        if sigmas < 0:
            raise ValueError(
                f"""prior_{coeff_type}_sigma must be positive in prior given by {prior_spec}."""
            )

    return {
        "prior_" + coeff_type + "_dist": PRIORS_MAP[dist_key],
        "prior_" + coeff_type + "_mu": prior_spec["prior_" + coeff_type + "_mu"],
        "prior_" + coeff_type + "_sigma": prior_spec["prior_" + coeff_type + "_sigma"],
    }


def validate_aux_prior(aux_prior_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates passed configuration for prior on auxiliary parameters.
    This does not perform parameter autoscaling.

    Parameters
    ----------
    aux_prior_spec : Dict[str, Any]
        Dictionary containing configuration for prior on auxiliary parameters.
        Currently supported priors are: "exponential" and "chi2", which are
        both parameterized by a single scalar.

        Priors here with more parameters are a future feature.
        For single-parameter priors, this field is a dictionary with the following keys

            + "prior_aux_dist": distribution of the prior on this parameter
            + "prior_aux_param": parameter of the prior on this parameter

        For example, to specify a chi2 prior with nu=2.5, pass::

            {"prior_aux_dist": "chi2", "prior_aux_param": 2.5}

    Returns
    -------
    Dict[str, Any]
        Dictionary containing validated configuration for prior on auxiliary parameters.

    Raises
    ------
    ValueError
        Prior's distribution is not specified.
    ValueError
        Unsupported prior distribution for auxiliary parameter.
    ValueError
        Prior distribution parameters are not specified.
    """
    config_keys = aux_prior_spec.keys()

    if "prior_aux_dist" not in config_keys:
        raise ValueError(
            f"""prior_aux_dist must be specified in auxiliary prior given by {aux_prior_spec}."""
        )

    dist_key = aux_prior_spec["prior_aux_dist"]

    if dist_key not in PRIORS_AUX_MAP.keys():
        raise ValueError(
            f"Prior {dist_key} in auxiliary prior specification {aux_prior_spec} not supported."
        )

    prior_aux_clean = {
        "prior_aux_dist": PRIORS_AUX_MAP[dist_key],
    }

    if dist_key == "exponential":
        if "prior_aux_param" not in config_keys:
            raise ValueError(
                f"""prior_aux_param must be specified in
                exponential auxiliary prior given by {aux_prior_spec}."""
            )

        prior_aux_clean["prior_aux_param"] = aux_prior_spec["prior_aux_param"]
    elif dist_key == "chi2":
        if "prior_aux_param" not in config_keys:
            raise ValueError(
                f"""prior_aux_param must be specified in
                chi2 auxiliary prior given by {aux_prior_spec}."""
            )

        prior_aux_clean["prior_aux_param"] = aux_prior_spec["prior_aux_param"]

    return prior_aux_clean
