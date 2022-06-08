import sys

from sk_stan_regression.utils.validation import check_is_fitted
from sk_stan_regression.bayesian_lin_reg import BLR_Estimator

# TODO!
def test_compatible_estimator(estimator, check):
    check(estimator)
