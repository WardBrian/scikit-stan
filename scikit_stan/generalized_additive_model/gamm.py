"""Generalized Additive Linear Model with sk-learn type API."""

from pathlib import Path

from scikit_stan.modelcore import CoreEstimator
from scikit_stan.utils.validation import (
    FAMILY_LINKS_MAP,
    check_array,
    check_is_fitted,
    validate_aux_prior,
    validate_family,
    validate_prior,
)


class GAMM(CoreEstimator):
    """
    Generalized linear additive model with flexible priors.
    """
