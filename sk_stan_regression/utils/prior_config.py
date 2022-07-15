"""Correct mapping and set-up of customizable parameters for Stan models.
    Prior parameters must either be fully specified or left as defaults;
    partial specification is not allowed.
"""

from typing import Any, Dict, Optional, Union
import warnings


PRIORS_MAP = { 
    "normal": 0,    # normal distribution, requires location (mu) and scale (sigma) or else defaults to 
    "laplace": 1,   # laplace distribution, requires location (mu) and scale (sigma) or else defaults to 
}

def map_priors(prior_config : Optional[Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
    """Map given dictionary of prior with or without all distribution parameters
    specified to the respective Stan configuration of inputs."""
    # no user specs for prior configuration given, default to rstanarms default:
    # https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html
    if prior_config is None or len(prior_config) == 0: 
        return "default"

    config_keys = prior_config.keys()
    dist_key = prior_config["distribution"]

    if dist_key not in PRIORS_MAP.keys(): 
        raise ValueError(f"Prior {dist_key} not supported.")

    # ensure that all required parameters are specified in dictionary 
    # for the intercept, location & scale must be scalars
    # for other coefficients, location & scale must either 
    #   1) be vectors of length equal to the number of coefficients 
    #       excluding the intercept or 
    #   2) be scalars, in which case the same values will be 
    #       repeated for each coefficient 
    # TODO: length check happens right before everything is fed to model method in fit()! 
    if dist_key == "normal": 
        # normal is parameterized by location and scale 
        if not ("location" in config_keys or "scale" in config_keys):
            warnings.warn(f"""No location or scale specified for prior {dist_key}, defaulting to 
                beta ~ normal(0, 2.5 * sd(y) / sd(X)) 
                alpha ~ normal(mu(y), 2.5 * sd(y)) 
            .""")
        elif not ("location" in config_keys):
            raise ValueError("Location parameter (mu) `location` must be specified for normal distribution.")
        elif not ("scale" in config_keys):
            raise ValueError("Scale parameter (sigma) `scale` must be specified for normal distribution.")
    elif dist_key == "laplace":
        # laplace is parameterized by location and scale 
        if not ("location" in config_keys or "scale" in config_keys):
            warnings.warn(f"""No location or scale specified for prior {dist_key}, defaulting to 
                beta ~ double_exponential(0, 2.5 * sd(y) / sd(X)) 
                alpha ~ double_exponential(mu(y), 2.5 * sd(y)) 
            .""")
        elif not ("location" in config_keys):
            raise ValueError("Location parameter (mu) `location` must be specified for laplace distribution.")
        elif not ("scale" in config_keys):
            raise ValueError("Scale parameter (sigma) `scale` must be specified for laplace distribution.")

    
