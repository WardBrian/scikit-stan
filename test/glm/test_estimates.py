### TODO: Test against the MLE of frequentist tools like R's GLM. Needs flat priors
### OR, test full model optimize against either rstanarm or hand-coded stan MLES
### for this we could also use sampling and check mcmc standard error intervals overlap

# In either case, generate data, fit parameters, save data+param to disk.


"""Tests estimates from the GLM. """

import numpy as np
import pytest
import scipy.stats as stats
from data import _gen_fam_dat_continuous, _gen_fam_dat_discrete, bcdata_dict
from test_log_lik import assert_log_lik

from scikit_stan.generalized_linear_regression import GLM


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ["sample", "variational", "optimize"])
def test_default_gauss_gen_predictions(algorithm: str) -> None:
    """
    GLM is fitted on randomly generated data with alpha=0.6, beta=0.2, sigma=0.3,
    and the predictions are performed on a different set of data that were generated
    with the same parameters.
    """
    glm1 = GLM(algorithm=algorithm, seed=999999)
    fake_data_1_X, fake_data_1_y = _gen_fam_dat_continuous(
        "gaussian", Nsize=1000, alpha=0.6, beta=0.2, sigma=0.3, link="identity"
    )
    glm1.fit(X=fake_data_1_X, y=fake_data_1_y)

    glm2 = GLM(seed=999999)
    fake_data_2_X, fake_data_2_y = _gen_fam_dat_continuous(
        "gaussian", Nsize=1000, alpha=0.6, beta=0.2, sigma=0.3, link="identity"
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


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
def test_gaussian_link_scipy_gen(link: str):
    if link == "inverse":
        pytest.skip(
            reason="Gaussian + inverse is known not to work with default priors"
        )

    glm = GLM(family="gaussian", link=link, seed=1234, save_log_lik=True)

    gaussian_dat_X, gaussian_dat_Y = _gen_fam_dat_continuous(
        family="gaussian", link=link, Nsize=1000
    )

    glm.fit(X=gaussian_dat_X, y=gaussian_dat_Y)

    assert_log_lik(glm, gaussian_dat_X, gaussian_dat_Y, link, "gaussian")

    reg_coeffs = np.array([])
    for val in [glm.alpha_, glm.beta_]:
        reg_coeffs = np.append(reg_coeffs, val)

    np.testing.assert_allclose(reg_coeffs, np.array([0.6, 0.2]), rtol=1e-1, atol=1e-1)

    # very rough posterior predictive
    out_y = glm.predict(gaussian_dat_X)
    np.testing.assert_allclose(
        gaussian_dat_Y.mean(), out_y.mean(), rtol=1e-1, atol=1e-1
    )


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
def test_gamma_link_scipy_gen(link: str) -> None:
    glm = GLM(
        family="gamma", link=link, seed=1234, algorithm_params={}, save_log_lik=True
    )

    gamma_dat_X, gamma_dat_Y = _gen_fam_dat_continuous(
        family="gamma", link=link, Nsize=1000
    )

    fitted = glm.fit(X=gamma_dat_X, y=gamma_dat_Y)

    assert (
        fitted.fitted_samples_.summary()["5%"]["alpha[1]"] - 0.01
        <= 0.6
        <= fitted.fitted_samples_.summary()["95%"]["alpha[1]"]
    )
    assert (
        fitted.fitted_samples_.summary()["5%"]["beta[1]"] - 0.1
        <= 0.2
        <= fitted.fitted_samples_.summary()["95%"]["beta[1]"] + 0.1
    )

    # very rough posterior predictive
    out_y = glm.predict(gamma_dat_X)
    np.testing.assert_allclose(gamma_dat_Y.mean(), out_y.mean(), rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("lotnumber", ["lot1", "lot2"])
def test_gamma_bloodclotting(lotnumber: str) -> None:
    glm_gamma = GLM(family="gamma", link="inverse", seed=1234, save_log_lik=True)

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


@pytest.mark.parametrize("link", ["identity", "log", "inverse", "inverse-square"])
def test_invgaussian_link_scipy_gen(link: str):
    if link == "identity" or link == "inverse-square":
        pytest.skip(reason="Inverse Gaussian needs special data generation")

    glm = GLM(family="inverse-gaussian", link=link, seed=1234, autoscale=True)

    invgaussian_dat_X, invgaussian_dat_Y = _gen_fam_dat_continuous(
        family="inverse-gaussian", link=link
    )

    fitted = glm.fit(X=invgaussian_dat_X, y=invgaussian_dat_Y)

    assert (
        fitted.fitted_samples_.summary()["5%"]["alpha[1]"] - 0.02
        <= 0.6
        <= fitted.fitted_samples_.summary()["95%"]["alpha[1]"] + 0.02
    )
    assert (
        fitted.fitted_samples_.summary()["5%"]["beta[1]"] - 0.02
        <= 0.2
        <= fitted.fitted_samples_.summary()["95%"]["beta[1]"] + 0.02
    )


@pytest.mark.parametrize("link", ["identity", "log", "sqrt"])
def test_poisson_link_scipy_gen(link: str):
    if link == "identity":
        pytest.skip(
            reason="""Poisson + identity is known not to work with default priors;
             also, identity link leads to potentially negative lambda..."""
        )
    glm = GLM(family="poisson", link=link, seed=1234, save_log_lik=True)

    if link == "identity":
        rng = np.random.default_rng(seed=9999)

        poisson_dat_X = stats.norm.rvs(10, 1, size=(1000,))
        poisson_dat_Y = rng.poisson(0.6 + 0.2 * poisson_dat_X)
    else:
        poisson_dat_X, poisson_dat_Y = _gen_fam_dat_discrete(
            family="poisson", link=link
        )

    fitted = glm.fit(X=poisson_dat_X, y=poisson_dat_Y)

    assert_log_lik(glm, poisson_dat_X, poisson_dat_Y, link, "poisson")

    assert (
        fitted.fitted_samples_.summary()["5%"]["alpha[1]"]
        <= 0.6
        <= fitted.fitted_samples_.summary()["95%"]["alpha[1]"]
    )
    assert (
        fitted.fitted_samples_.summary()["5%"]["beta[1]"]
        <= 0.2
        <= fitted.fitted_samples_.summary()["95%"]["beta[1]"]
    )

    # very rough posterior predictive
    out_y = np.median(glm.predict_distribution(poisson_dat_X), axis=0)
    np.testing.assert_array_almost_equal(
        np.median(poisson_dat_Y), np.median(out_y), decimal=0
    )


# confirming that coefficients of regression line up with
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html
# this uses the canonical log link
def test_poisson_sklearn_poissonregressor():
    glm_poisson = GLM(family="poisson", link="log", seed=1234)

    X = [[1, 2], [2, 3], [3, 4], [4, 3]]
    y = [12, 17, 22, 21]

    fitted = glm_poisson.fit(X=X, y=y)

    reg_coeffs = np.array([])

    for val in [glm_poisson.alpha_, glm_poisson.beta_]:
        reg_coeffs = np.append(reg_coeffs, val)

    assert (
        fitted.fitted_samples_.summary()["5%"]["alpha[1]"]
        <= 2.088
        <= fitted.fitted_samples_.summary()["95%"]["alpha[1]"]
    )
    assert (
        fitted.fitted_samples_.summary()["5%"]["beta[1]"]
        <= 0.121
        <= fitted.fitted_samples_.summary()["95%"]["beta[1]"]
    )
    assert (
        fitted.fitted_samples_.summary()["5%"]["beta[2]"]
        <= 0.158
        <= fitted.fitted_samples_.summary()["95%"]["beta[2]"]
    )


def test_poisson_rstanarm_data():
    # NOTE: this data comes from rstanarm tests
    X = np.array(
        [
            [1, 1, 1],
            [1, 2, 1],
            [1, 3, 1],
            [1, 1, 2],
            [1, 2, 2],
            [1, 3, 2],
            [1, 1, 3],
            [1, 2, 3],
            [1, 3, 3],
        ]
    )

    y = np.array([18, 17, 15, 20, 10, 20, 25, 13, 12])

    glm_poisson = GLM(family="poisson", link="log", seed=1234, save_log_lik=True)

    glm_poisson.fit(X=X, y=y)

    assert_log_lik(glm_poisson, X, y, "log", "poisson")
