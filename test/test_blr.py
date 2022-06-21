"""Tests with confirmation from sklearn for estimators."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from sk_stan_regression.bayesian_lin_reg import BLR_Estimator

# from sklearn.utils.estimator_checks import check_estimator  # type: ignore


# est.mark.parametrize("estimator", [BLR_Estimator()])
# test_compatible_estimator(estimator: "CoreEstimator") -> None:
# check_estimator(estimator)


def test_notfittederror_blr() -> None:
    blr = BLR_Estimator()
    with pytest.raises(Exception) as e_info:
        blr.predict(X=np.array([2, 4, 8, 16]))


if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"]))
