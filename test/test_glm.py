"""Tests with confirmation from sklearn for estimators."""

import numpy as np
import pytest
from data import _gen_fam_dat
from sklearn.utils.estimator_checks import check_estimator  # type: ignore

from sk_stan_regression.generalized_linear_regression import GLM
from sk_stan_regression.modelcore import CoreEstimator


@pytest.mark.parametrize("estimator", [GLM()])
@pytest.mark.slow
def test_compatible_estimator(estimator: "CoreEstimator") -> None:
    check_estimator(estimator)


def test_notfittederror_glm() -> None:
    glm = GLM()
    with pytest.raises(Exception) as e_info:
        glm.predict(X=np.array([2, 4, 8, 16]))


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ["HMC-NUTS", "Variational", "MLE"])
def test_default_gauss_gen_predictions(algorithm: str) -> None:
    """
    GLM is fitted on randomly generated data with alpha=0.6, beta=0.2, sigma=0.3,
    and the predictions are performed on a different set of data that were generated
    with the same parameters.
    """
    glm1 = GLM(algorithm=algorithm)
    fake_data_1_X, fake_data_1_y = _gen_fam_dat(
        "gaussian", Nsize=1000, alpha=0.6, beta=0.2, sigma=0.3
    )
    glm1.fit(X=fake_data_1_X, y=fake_data_1_y)

    glm2 = GLM()
    fake_data_2_X, fake_data_2_y = _gen_fam_dat(
        "gaussian", Nsize=1000, alpha=0.6, beta=0.2, sigma=0.3
    )
    glm2.fit(X=fake_data_2_X, y=fake_data_2_y)

    # use one GLM to predict results on other set of data
    glm1.predict(X=fake_data_2_X)
    glm2.predict(X=fake_data_1_X)

    reg_coeffs1 = np.array([])  # type: ignore
    for val in [glm1.alpha_, glm1.beta_, glm1.sigma_]:
        reg_coeffs1 = np.append(reg_coeffs1, val)

    reg_coeffs2 = np.array([])  # type: ignore
    for val in [glm1.alpha_, glm1.beta_, glm1.sigma_]:
        reg_coeffs2 = np.append(reg_coeffs2, val)

    # each individual GLM has the correct coefficients
    np.testing.assert_allclose(
        reg_coeffs1, np.array([0.6, 0.2, 0.3]), rtol=1e-1, atol=1e-1
    )

    np.testing.assert_allclose(
        reg_coeffs2, np.array([0.6, 0.2, 0.3]), rtol=1e-1, atol=1e-1
    )

    # the GLMs have similar predictions
    np.testing.assert_allclose(reg_coeffs1, reg_coeffs2, rtol=1e-1, atol=1e-1)


# def test_gamma_scipy_gen() -> None:
#    glm = GLM()
#

if __name__ == "__main__":
    # from scipy.special import expit  # type: ignore
    from data import bcdata_dict
    rng = np.random.default_rng(1234)

    # NOTE: rate parameter sometimes becomes negative for poisson?
    # blr = GLM(family="bernoulli")
    blr = GLM(family="gamma", link="inverse")
    #gamma_dat_X, gamma_dat_Y = _gen_fam_dat("gamma", Nsize=1000, alpha=0.6, beta=0.2)
    bc_data_y, bc_data_X = np.log(bcdata_dict['u']), np.column_stack((bcdata_dict['lot1'], bcdata_dict['lot2']))
    print(bc_data_X.shape, bc_data_y.shape)
    #blr.fit(X=gamma_dat_X, y=gamma_dat_Y, show_console=True)
    blr.fit(X=bc_data_X, y=bc_data_y, show_console=True)
    # print(blr.predict(X=xdat, show_console=False))
    print(blr.alpha_, blr.beta_) #-1.68296667742 [-0.03430016  0.07737138]
    # print(blr.fit(X=xdat, y=ydat, show_console=True))
    # print(blr.predict(X=xdat, show_console=True))

    # true params
    # β0_true = 0.7
    # β1_true = 0.4
    ### number of yes/no questions
    # n = 1
    # sample_size = 30
    # x = np.linspace(-10, 20, sample_size)
    ### Linear model
    # μ_true = β0_true + β1_true * x
    ### transformation (inverse logit function = expit)
    # p_true = expit(μ_true)
    ### Generate data
    # y = rng.binomial(n, p_true)
    ## print(y)
    # blr.fit(X=x, y=y)
    # print(blr.predict(X=x))
#
# print(blr.fit(X=xdat, y=ydat).__dict__)
# blr.predict(X=xdat)

# print(blr.fit(kby2, ydat).__dict__)
# blr.predict(X=kby2)

# blrfamlink = BLR_Estimator(family="gaussian", link="inverse")
# blrfamlink.fit(xdat, ydat)

# blr2 = BLR_Estimator(algorithm="Variational")
# blr2.fit(xdat, ydat)
