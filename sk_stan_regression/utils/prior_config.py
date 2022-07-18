"""Correct mapping and set-up of customizable parameters for Stan models.
    Prior parameters must either be fully specified or left as defaults;
    partial specification is not allowed.
"""

import warnings
from typing import Any, Dict, Optional, Union

PRIORS_MAP = {
    "normal": 0,  # normal distribution, requires location (mu) and scale (sigma)
    # or else defaults to
    "laplace": 1,  # laplace distribution, requires location (mu) and scale (sigma)
    # or else defaults to
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
    dist_key_intercept, dist_key_slope = (
        prior_config["prior_intercept_dist"],
        prior_config["prior_slope_dist"],
    )

    if dist_key_intercept not in PRIORS_MAP.keys():
        raise ValueError(f"Prior {dist_key_intercept} not supported.")

    if dist_key_slope not in PRIORS_MAP.keys():
        raise ValueError(f"Prior {dist_key_slope} not supported.")

    res_priors = {
        "prior_intercept_dist": dist_key_intercept,
        "prior_slope_dist": dist_key_slope,
    }

    # ensure that all required parameters are specified in dictionary
    # for the intercept -- location & scale must be scalars
    # for other regression coefficients, location & scale must either
    #   1) be vectors of length equal to the number of coefficients
    #       excluding the intercept or
    #   2) be scalars, in which case the same values will be
    #       repeated for each coefficient
    # TODO: length check happens right before everything is fed to model method in fit()!
    if dist_key_intercept == "normal":
        # normal is parameterized by location and scale
        if not (
            "prior_intercept_mu" in config_keys
            or "prior_intercept_sigma" in config_keys
        ):
            warnings.warn(
                f"""No location or scale specified for intercept prior {dist_key_intercept}, defaulting to 
                alpha ~ normal(mu(y), 2.5 * sd(y)) 
            ."""
            )
            res_priors["prior_intercept_mu"], res_priors["prior_intercept_sigma"] = (
                "default",
                "default",
            )
        elif (
            "prior_intercept_mu" not in config_keys
            and "prior_intercept_sigma" in config_keys
        ):
            warnings.warn(
                f"""Location parameter not specified for intercept prior {dist_key_intercept}, 
                so the default location parameter will be used. 
                The prior will be
                    alpha ~ normal(mu(y), {prior_config["prior_intercept_sigma"]})
                ."""
            )
            res_priors["prior_intercept_mu"], res_priors["prior_intercept_sigma"] = (
                "default",
                prior_config["prior_intercept_sigma"],
            )
        elif (
            "prior_intercept_mu" in config_keys
            and "prior_intercept_sigma" not in config_keys
        ):
            warnings.warn(
                f"""Scale parameter not specified for intercept prior {dist_key_intercept}, 
                so the default scale parameter will be used. 
                The prior will be
                    alpha ~ normal({prior_config["prior_intercept_mu"]}, 2.5 * sd(y))
                ."""
            )
            res_priors["prior_intercept_mu"], res_priors["prior_intercept_sigma"] = (
                prior_config["prior_intercept_mu"],
                "default",
            )
        else:
            # both location & scale are specified by user
            res_priors["prior_intercept_mu"], res_priors["prior_intercept_sigma"] = (
                prior_config["prior_intercept_mu"],
                prior_config["prior_intercept_sigma"],
            )

    if dist_key_slope == "normal":
        # normal is parameterized by location and scale
        if not ("prior_slope_mu" in config_keys or "prior_slope_sigma" in config_keys):
            warnings.warn(
                f"""No location or scale specified for slope prior {dist_key_slope}, defaulting to 
                    beta ~ normal(0, 2.5 * sd(y) / sd(X)) 
                for all components."""
            )
            res_priors["prior_slope_mu"], res_priors["prior_slope_sigma"] = (
                "default",
                "default",
            )
        elif "prior_slope_mu" not in config_keys and "prior_slope_sigma" in config_keys:
            warnings.warn(
                f"""Location parameter not specified for slope prior {dist_key_slope}, 
                so the default location parameter will be used. 
                The prior will be
                    beta ~ normal(0, {prior_config["prior_slope_sigma"]})
                for all components."""
            )
            res_priors["prior_slope_mu"], res_priors["prior_slope_sigma"] = (
                "default",
                prior_config["prior_slope_sigma"],
            )
        elif "prior_slope_mu" in config_keys and "prior_slope_sigma" not in config_keys:
            warnings.warn(
                f"""Scale parameter not specified for slope prior {dist_key_slope}, 
                so the default scale parameter will be used. 
                The prior will be
                    beta ~ normal({prior_config["prior_slope_mu"]}, 2.5 * sd(y) / sd(X))
                for all components."""
            )
            res_priors["prior_slope_mu"], res_priors["prior_slope_sigma"] = (
                prior_config["prior_slope_mu"],
                "default",
            )
        else:
            # both location & scale are specified by user
            res_priors["prior_slope_mu"], res_priors["prior_slope_sigma"] = (
                prior_config["prior_slope_mu"],
                prior_config["prior_slope_sigma"],
            )

    # elif dist_key_intercept == "laplace":
    #
    # elif dist_key == "laplace":
    #    # laplace is parameterized by location and scale
    #    if not ("location" in config_keys or "scale" in config_keys):
    #        warnings.warn(
    #            f"""No location or scale specified for prior {dist_key}, defaulting to
    #            beta ~ double_exponential(0, 2.5 * sd(y) / sd(X))
    #            alpha ~ double_exponential(mu(y), 2.5 * sd(y))
    #        ."""
    #        )
    #    elif not ("location" in config_keys):
    #        raise ValueError(
    #            "Location parameter (mu) `location` must be specified for laplace distribution."
    #        )
    #    elif not ("scale" in config_keys):
    #        raise ValueError(
    #            "Scale parameter (sigma) `scale` must be specified for laplace distribution."
    #        )
    #
    return res_priors
