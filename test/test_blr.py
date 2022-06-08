"""Tests with confirmation from sklearn for linear regression."""

import os
import sys

import sk_stan_regression

print(os.getcwd())

import unittest

from sklearn.utils.estimator_checks import check_estimator

from sk_stan_regression.bayesian_lin_reg import BLR_Estimator


class TestBLREstimator(unittest.TestCase):
    def test_compatible_estimator(self):
        check_estimator(BLR_Estimator())


if __name__ == "__main__":
    unittest.main()
