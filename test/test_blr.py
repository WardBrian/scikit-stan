"""Tests with confirmation from sklearn for estimators."""

import sys
from pathlib import Path
import numpy as np

from sk_stan_regression.modelcore import CoreEstimator

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from sklearn.utils.estimator_checks import check_estimator  # type: ignore

from sk_stan_regression.bayesian_lin_reg import BLR_Estimator


@pytest.mark.parametrize("estimator", [BLR_Estimator()])
def test_compatible_estimator(estimator: "CoreEstimator") -> None:
    check_estimator(estimator)


def test_notfittederror_blr() -> None:
    blr = BLR_Estimator()
    with pytest.raises(Exception) as e_info:
        blr.predict(X=np.array([2, 4, 8, 16]))


if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"]))
