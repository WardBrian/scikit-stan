"""Folder for Python side of Beta regression model, following classical treatment by Ferrari and Cribari-Neto(2004)."""



import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
from numpy.typing import ArrayLike, NDArray

from scikit_stan.modelcore import CoreEstimator
from scikit_stan.utils.stan import load_stan_model
from scikit_stan.utils.validation import (
    FAMILY_LINKS_MAP,
    GLM_FAMILIES,
    check_array,
    check_is_fitted,
    method_dict,
    validate_aux_prior,
    validate_family,
    validate_prior,
)


class BetaReg(CoreEstimator) :
    r"""
    Beta regression model following the original statement by Ferrari and Cribari-Neto (2004)
    and the extension by Simas, Barreto-Souza, and Rocha (2010). The utility of this model lies
    in problems with continuous dependent variables that are restricted to the unit interval,
    especially resulting from rates or proportions.

    The model is beta-distributed using a parameterization of mu and precision parameter phi.
    As with the GLM in this package, the mean is linked to responses via a link function and
    linear predictor. Analogously, the precision parameter phi can be linked to another
    possibly overlapping set of regressors through an additional link function, which
    leads to a model with variable dispersion.

    The default link is "log" unless no precision model is specified, in which case
    the "identity" link is used.

    link : str, optional
        Specifies the link function used in the model for mu through the values in the design matrix.
        The following links are supported:

            * "logit", "probit", "cloglog", "cauchit", "log", and "loglog"

    link_phi : Optional[str], optional
        Specifies the link funtion used in the model for phi, the precision parameter, through the values in z,
        which is passed to fit() alongside the exogenous data.

        The following links are supported:

            * "identity", "log" (default) and "sqrt"

        Note that the "sqrt" link function is notoriously unstable, so it is advisable to use an alternative.

    phi_regression : bool, optional
        Whether or not to have a linear predictor on phi, the precision parameter of the model.
        If this is False, then phi is streated as a scalar parameter. Otherwise, as is by default,
        with this being True, there is a linear predictor for phi based on the value of z passed to fit().

    priors : Optional[Dict[str, Union[int, float, List]]], optional
        Dictionary for configuring prior distribution on coefficients.
        Currently supported priors are: "normal", and "laplace".

        By default, all regression coefficient priors are set to

        .. math:: \beta \sim \text{normal}(0, 2.5 \cdot \text{sd}(y) / \text{sd}(X))

        if Gaussian, else

        .. math:: \beta \sim \text{normal}(0, 2.5 / \text{sd}(X))

        if auto_scale is True, otherwise

        .. math:: \beta \sim \text{normal}(0, 2.5 \cdot \text{sd}(y))

        if Gaussian else

        .. math:: \beta \sim \text{normal}(0, 2.5)

        If an empty dictionary {} is passed, no priors are used and
        the default Stan uniform(-inf, inf) prior is used for all coefficients.

        The number of specified prior parameters cannot exceed the number of predictors in
        the data as each parameter is associated with a single coefficient.
        If the number of specified priors is less than the number of predictors,
        the remaining coefficients are set to the default prior.
        The prior on all regression coefficients is set with the following keys:

            + "prior_slope_dist": distribution of the prior for each coefficient
            + "prior_slope_mu": list of location parameters of the prior for each coefficient
            + "prior_slope_sigma": list of scale parameters of the prior for each coefficient

        Thus, to specify a standard normal prior on the first feature,
        the dictionary should be passed as::

            {
                "prior_slope_dist": "normal",
                "prior_slope_mu": [0.],
                "prior_slope_sigma": [1.]
            }

        Any unspecified priors will be set to the default.

        Also note that choosing a Laplace prior is equivalent to L1 regularization:
        https://stats.stackexchange.com/questions/177210/why-is-laplace-prior-producing-sparse-solutions/177217#177217

    prior_intercept : Optional[Dict[str, Any]], optional
        Prior for the intercept alpha parameter for GLM.
        If this is not specified, the default is

        .. math:: \alpha \sim \text{normal}(\text{mu}(y), 2.5 \cdot \text{sd}(y))

        if Gaussian family else

        .. math:: \alpha \sim \text{normal}(0, 2.5)

        If an empty dictionary {} is passed, the default Stan uniform(-inf, inf) prior is used.

        To set this prior explicitly, pass a dictionary with the following keys:

            + "prior_intercept_dist": str, distribution of the prior from the list of supported
              prior distributions: "normal", "laplace"
            + "prior_intercept_mu": float, location parameter of the prior distribution
            + "prior_intercept_sigma": float, error scale parameter of the prior distribution

        Thus, for example, passing::

            {
                "prior_intercept_dist": "normal",
                "prior_intercept_mu": 0,
                "prior_intercept_sigma": 1
            }

        results in

        .. math:: \alpha \sim \text{normal}(0, 1)

        by default (without autoscaling, see below).

        Also note that choosing a Laplace prior is equivalent to L1 regularization:
        https://stats.stackexchange.com/questions/177210/why-is-laplace-prior-producing-sparse-solutions/177217#177217

    priors_z : Optional[Dict[str, Union[int, float, List]]], optional

        Works the same as priors field described above. Defaults and autoscaling are the same.

    prior_intercept_z : Optional[Dict[str, Any]], optional

        Works the same as priors field described above. Defaults and autoscaling are the same.
    """
