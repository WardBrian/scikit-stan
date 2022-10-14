"""Tests for consistency of generalized linear model and adherence to sklearn style."""

from typing import Callable

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from scikit_stan.generalized_linear_regression import GLM
from scikit_stan.modelcore import CoreEstimator  # type: ignore


@pytest.mark.slow
@parametrize_with_checks([GLM()])
def test_sklearn_compatible_estimator(
    estimator: CoreEstimator, check: Callable[[CoreEstimator], None]
) -> None:
    """Ensure that GLM Estimator is sk-learn compatible."""
    check(estimator)


def test_notfittederror_glm() -> None:
    glm = GLM()
    with pytest.raises(Exception):
        glm.predict(X=[2, 4, 8, 16])
