"""Tests for consistency of generalized linear model and adherence to sklearn style."""

import numpy as np
import pytest
from data import _gen_fam_dat
from sklearn.utils.estimator_checks import check_estimator  # type: ignore

from sk_stan_regression.generalized_linear_regression import GLM
from sk_stan_regression.modelcore import CoreEstimator

from data import bcdata_dict


@pytest.mark.slow
@pytest.mark.parametrize("estimator", [GLM()])
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


@pytest.mark.parametrize("family", ["gaussian", "gamma", "inverse_gaussian"])
def test_auto_canonical_link_continuous(family: str) -> None:
    """
    Test that the canonical link is automatically chosen for the family.
    """
    canonical_links = {
        "gaussian": "identity",
        "gamma": "inverse",
        "inverse_gaussian": "inverse-square",
    }
    glm = GLM(family=family)
    glm.fit(X=np.array([[1, 2, 3], [4, 5, 6]]), y=np.array([1, 2]))

    assert glm.link == canonical_links[family]


# TODO: add predict...
# TODO: parameters for gamma in this test should be random !!
def test_gamma_scipy_gen() -> None:
    glm_gamma = GLM(family="gamma", link="identity")  # canonical link function
    gamma_dat_X, gamma_dat_Y = _gen_fam_dat("gamma", Nsize=1000, alpha=0.9, beta=0.3)
    glm_gamma.fit(X=gamma_dat_X, y=gamma_dat_Y)

    reg_coeffs = np.array([])
    for val in [glm_gamma.alpha_, glm_gamma.beta_]:
        reg_coeffs = np.append(reg_coeffs, val)

    np.testing.assert_allclose(reg_coeffs, np.array([0.9, 0.3]), rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("lotnumber", ["lot1", "lot2"])
def test_gamma_bloodclotting(lotnumber: str) -> None:
    glm_gamma = GLM(family="gamma", link="inverse")

    bc_data_X, bc_data_y = np.log(bcdata_dict["u"]), bcdata_dict[lotnumber]

    glm_gamma.fit(X=bc_data_X, y=bc_data_y, show_console=False)

    reg_coeffs = np.array([])
    for val in [glm_gamma.alpha_, glm_gamma.beta_]:
        reg_coeffs = np.append(reg_coeffs, val)

    # ensure that the results are close to
    # McCullagh & Nelder (1989), chapter 8.4.2 p 301-302
    if lotnumber == "lot1":
        np.testing.assert_allclose(
            reg_coeffs, np.array([-0.01655, 0.01534]), rtol=1e-2, atol=1e-2
        )
    else:
        np.testing.assert_allclose(
            reg_coeffs, np.array([-0.02391, 0.02360]), rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    # from scipy.special import expit  # type: ignore
    import matplotlib.pyplot as plt

    # from data import bcdata_dict
    # NOTE: rate parameter sometimes becomes negative for poisson?
    # blr = GLM(family="bernoulli")
    # blr = GLM(family="gamma", link="inverse")
    glm = GLM(family="gamma", link="inverse")

    gamma_dat_X, gamma_dat_Y = _gen_fam_dat(
        "gamma", Nsize=1000, alpha=0.9, beta=0.3, sigma=1.9
    )
    # gauss_dat_X, gauss_dat_y = _gen_fam_dat(
    #    "gaussian", Nsize=1000, alpha=0.9, beta=0.3
    # )
    # bc_data_y, bc_data_X = np.log(bcdata_dict["u"]), np.column_stack(
    #    (bcdata_dict["lot1"], bcdata_dict["lot2"])
    # )
    # bc_data_X, bc_data_y = np.log(bcdata_dict["u"]), bcdata_dict["lot2"]
    # blr.fit(X=bc_data_X, y=bc_data_y, show_console=True)
    glm.fit(X=gamma_dat_X, y=gamma_dat_Y, show_console=False)
    print(glm.alpha_, glm.beta_, glm.sigma_)
    # glm.fit(X=gauss_dat_X, y=gauss_dat_y, show_console=True)
    # blr.fit(X=bc_data_X, y=bc_data_y, show_console=True)
    # predics = glm.predict(X=gauss_dat_X)
    predics = glm.predict(X=gamma_dat_X)
    # plt.scatter(gauss_dat_X, gauss_dat_y)
    # plt.scatter(gauss_dat_X, predics)
    plt.hist(gamma_dat_Y, density=True, histtype="stepfilled", alpha=0.2)
    plt.hist(predics, density=True, histtype="stepfilled", alpha=0.2)
    # plt.scatter(gamma_dat_X, gamma_dat_Y)
    # plt.scatter(gamma_dat_X, predics)

    plt.show()
    # print(blr.predict(X=bc_data_X, show_console=False))
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
