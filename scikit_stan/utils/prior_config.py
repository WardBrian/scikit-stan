"""Correct mapping and set-up of customizable parameters for Stan models.
    Prior parameters must either be fully specified or left as defaults;
    partial specification is not allowed.
"""

import warnings
from typing import Any, Dict, Optional

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


def map_priors(prior_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map given dictionary of prior with or without all distribution parameters
    specified to the respective Stan configuration of inputs.

    Examples:
        prior_config = {}
        This sets the default prior configuration of the intercept prior distribution
        to normal(mu(y), 2.5 * sd(y)) and the slope prior distribution to
        normal(0, 2.5 * sd(y) / sd(X)), where mu(y) is the mean of the y-values
        and sd(y) is the standard deviation of the y-values, respectively.
        sd(X) is the standard deviation of the X-values.

        prior_config = {"prior_intercept_dist": "normal", "prior_intercept_mu": 0,
        "prior_intercept_sigma": 1, "prior_slope_dist": "normal",
        "prior_slope_mu": 0, "prior_slope_sigma": 1}
        This sets the intercept prior distribution to normal(0, 1)
        and the slope prior distribution to normal(0, 1).
        The error scale will follow the default of exponential(1/sd(y))
        where sd(y) is the standard deviation of the response variable y.

        prior_config = {"prior_intercept_dist": "normal",
                        "prior_intercept_mu": 0, "prior_intercept_sigma": 1}
        This sets the intercept prior distribution to normal(0,1) while the
        slope prior distribution follows the default: normal(0, 2.5 * sd(y) / sd(X))
        where sd(y) and sd(X) are the standard deviations of the response variable y
        and the predictor variable X, respectively.
        The error scale will follow the default of exponential(1/sd(y)) where sd(y)
        is the standard deviation of the response variable y.

        prior_config = {"prior_slope_dist": "laplace", "prior_slope_mu": 0, "prior_slope_sigma": 1}
        This sets the slope prior distribution to laplace(0, 1) while
        the intercept prior distribution follows the default: normal(0, 2.5).
        The error scale will follow the default of exponential(1).

        prior_config = {"prior_intercept_dist": "laplace", "prior_intercept_mu": 0,
        "prior_intercept_sigma": 1, "prior_slope_dist": "laplace",
        "prior_slope_mu": 0, "prior_slope_sigma": 1}
        This sets the intercept prior distribution to laplace(0, 1) and
        the slope prior distribution to laplace(0, 1).
        The error scale will follow the default of exponential(1).
    """
    # no user specs for prior configuration given, default to rstanarm's default:
    # https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html
    if prior_config is None or len(prior_config) == 0:
        return {}

    config_keys = prior_config.keys()

    # only one of the following two is possible at a given time;
    # prior_config is known to contain at least one of the two keys
    if "prior_intercept_dist" in config_keys:
        dist_key_intercept = prior_config["prior_intercept_dist"]
    else:
        warnings.warn(
            """
            No prior distribution on intercept was specified, defaulting to normal."""
        )
        dist_key_intercept = "normal"

    if "prior_slope_dist" in config_keys:
        dist_key_slope = prior_config["prior_slope_dist"]
    else:
        warnings.warn(
            """
            No prior distribution on slope was specified, defaulting to normal."""
        )
        dist_key_slope = "normal"

    # ensure that specified intercept and slope priors are valid
    if dist_key_intercept not in PRIORS_MAP.keys():
        raise ValueError(f"Prior {dist_key_intercept} not supported.")

    if dist_key_slope not in PRIORS_MAP.keys():
        raise ValueError(f"Prior {dist_key_slope} not supported.")

    res_priors: Dict[str, Any] = {
        "prior_intercept_dist": PRIORS_MAP[dist_key_intercept],
        "prior_slope_dist": PRIORS_MAP[dist_key_slope],
    }

    # ensure that all required parameters are specified in dictionary
    # for the intercept -- location & scale must be scalars
    # for other regression coefficients, location & scale must either
    #   1) be vectors of length equal to the number of coefficients
    #       excluding the intercept or
    #   2) be scalars, in which case the same values will be
    #       repeated for each coefficient
    if not (
        "prior_intercept_mu" in config_keys and "prior_intercept_sigma" in config_keys
    ):
        warnings.warn(
            f"No location or scale specified for intercept prior {dist_key_intercept}"
            f", defaulting to\n{INTERCEPT_PRIOR_DEFAULTS_INFO[dist_key_intercept]}."
        )
        res_priors["prior_intercept_mu"], res_priors["prior_intercept_sigma"] = (
            "default",
            "default",
        )
    else:
        # both location & scale are specified by user
        res_priors["prior_intercept_mu"], res_priors["prior_intercept_sigma"] = (
            prior_config["prior_intercept_mu"],
            prior_config["prior_intercept_sigma"],
        )

    if not ("prior_slope_mu" in config_keys and "prior_slope_sigma" in config_keys):
        warnings.warn(
            f"""No location or scale specified for slope prior {dist_key_slope}, defaulting to
            {SLOPE_PRIOR_DEFAULTS_INFO[dist_key_slope]}."""
        )
        res_priors["prior_slope_mu"], res_priors["prior_slope_sigma"] = (
            "default",
            "default",
        )
    else:
        # both location & scale are specified by user
        res_priors["prior_slope_mu"], res_priors["prior_slope_sigma"] = (
            prior_config["prior_slope_mu"],
            prior_config["prior_slope_sigma"],
        )

    return res_priors
