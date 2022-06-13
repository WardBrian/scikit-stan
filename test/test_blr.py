"""Tests with confirmation from sklearn for estimators."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from sklearn.utils.estimator_checks import check_estimator

from sk_stan_regression.bayesian_lin_reg import BLR_Estimator


@pytest.mark.parametrize(
    "estimator", 
    [BLR_Estimator()]
)
def test_compatible_estimator(estimator):
        check_estimator(estimator)
