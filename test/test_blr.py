"""Tests with confirmation from sklearn for estimators."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator  # type: ignore

from sk_stan_regression.bayesian_lin_reg import BLR_Estimator
from sk_stan_regression.modelcore import CoreEstimator

FAKE_DATA = Path(__file__).parent / "data" / "fake_data.json"


@pytest.mark.parametrize("estimator", [BLR_Estimator()])
def test_compatible_estimator(estimator: "CoreEstimator") -> None:
    check_estimator(estimator)


def test_notfittederror_blr() -> None:
    blr = BLR_Estimator()
    with pytest.raises(Exception) as e_info:
        blr.predict(X=np.array([2, 4, 8, 16]))


# TODO: use scipy rvs to remove dependence on json input
@pytest.mark.parametrize("algorithm", ["HMC-NUTS", "Variational", "MLE"])
def test_fake_data_1d_gaussI_algos(algorithm: str) -> None:
    blr = BLR_Estimator(algorithm=algorithm)
    with open(FAKE_DATA) as file:
        jsondat = json.load(file)

    xdat = np.array(jsondat["x"])
    ydat = np.array(jsondat["y"])

    blr.fit(X=xdat, y=ydat)

    reg_coeffs = np.array([])  # type: ignore
    for val in [blr.alpha_, blr.beta_, blr.sigma_]:
        reg_coeffs = np.append(reg_coeffs, val)

    np.testing.assert_allclose(
        reg_coeffs, np.array([0.6, 0.2, 0.3]), rtol=1e-1, atol=1e-1
    )


if __name__ == "__main__":
    from scipy.special import expit  # type: ignore

    rng = np.random.default_rng(1234)
    with open(FAKE_DATA) as file:
        jsondat = json.load(file)

    xdat = np.array(jsondat["x"])
    ydat = np.array(jsondat["y"])
    #
    # kby2 = np.column_stack((xdat, xdat))  # type: ignore
    # print(kby2.shape)

    # NOTE: rate parameter sometimes becomes negative for poisson?
    blr = BLR_Estimator(family="bernoulli", link="cauchit")
    # blr = BLR_Estimator()
    # blr.fit(X=xdat, y=ydat, show_console=False)
    # blr.predict(X=xdat, show_console=True)

    # true params
    β0_true = 0.7
    β1_true = 0.4
    ## number of yes/no questions
    n = 1
    sample_size = 30
    x = np.linspace(-10, 20, sample_size)
    ## Linear model
    μ_true = β0_true + β1_true * x
    ## transformation (inverse logit function = expit)
    p_true = expit(μ_true)
    ## Generate data
    y = rng.binomial(n, p_true)
    # print(y)
    blr.fit(X=x, y=y)
    print(blr.predict(X=x))
#
# print(blr.fit(X=xdat, y=ydat).__dict__)
# blr.predict(X=xdat)

# print(blr.fit(kby2, ydat).__dict__)
# blr.predict(X=kby2)

# blrfamlink = BLR_Estimator(family="gaussian", link="inverse")
# blrfamlink.fit(xdat, ydat)

# blr2 = BLR_Estimator(algorithm="Variational")
# blr2.fit(xdat, ydat)
