"""Tests with confirmation from sklearn for linear regression."""

import os
import sys

from pathlib import Path 

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from sklearn.utils.estimator_checks import check_estimator

from sk_stan_regression.bayesian_lin_reg import BLR_Estimator


class TestBLREstimator(unittest.TestCase):
    def test_compatible_estimator(self):
        check_estimator(BLR_Estimator())


if __name__ == "__main__":
    unittest.main()
