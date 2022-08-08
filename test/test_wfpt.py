"""Tests for consistency and adherence to sk-learn style of first-hitting-time model for Wiener process."""

import numpy as np
import pytest
import scipy.stats as stats  # type: ignore
from data import _gen_fam_dat_continuous, _gen_fam_dat_discrete, bcdata_dict
from sklearn.utils.estimator_checks import check_estimator  # type: ignore

from scikit_stan.generalized_linear_regression import WienerFPT
from scikit_stan.modelcore import CoreEstimator

@pytest.mark.slow
#@pytest.mark.parametrize("estimator", [GLM()])
def test_compatible_estimator() -> None:
    """Ensure that WFPT Estimator is sk-learn compatible."""
    check_estimator(WienerFPT())
