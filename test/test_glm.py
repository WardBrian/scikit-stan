"""Tests for consistency of generalized linear model and adherence to sklearn style."""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.stats as stats
from data import _gen_fam_dat_continuous, _gen_fam_dat_discrete, bcdata_dict
from sklearn.utils.estimator_checks import check_estimator  # type: ignore

from scikit_stan.generalized_linear_regression import GLM


def test_sparse() -> None:
    """Test sparse X"""

    glm = GLM(family="gaussian", seed=1234)

    gaussian_dat_X, gaussian_dat_Y = _gen_fam_dat_continuous(
        family="gaussian", link="identity"
    )

    glm.fit(X=sp.csr_matrix(gaussian_dat_X[:, np.newaxis]), y=gaussian_dat_Y)

    reg_coeffs = np.array([])
    for val in [glm.alpha_, glm.beta_]:
        reg_coeffs = np.append(reg_coeffs, val)

    np.testing.assert_allclose(reg_coeffs, np.array([0.6, 0.2]), rtol=1e-1, atol=1e-1)

    glm = GLM(family="poisson", link="log", seed=1234)

    poisson_dat_X, poisson_dat_Y = _gen_fam_dat_discrete(family="poisson", link="log")

    fitted = glm.fit(X=poisson_dat_X, y=poisson_dat_Y)

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


def test_no_intercept_regression() -> None:
    """Ensure that no intercept is added if fit_intercept is False
    and that the default behavior is fit_intercept=True."""
    X, y = _gen_fam_dat_continuous(family="gaussian", link="log", seed=1234321)

    glm_no_intercept = GLM(
        family="gaussian", link="log", fit_intercept=False, seed=1234
    )
    glm_intercept = GLM(family="gaussian", link="log", fit_intercept=True, seed=1234)
    glm_no_intercept.fit(X, y)
    glm_intercept.fit(X, y)

    assert "alpha[1]" not in list(
        glm_no_intercept.fitted_samples_.summary().head().index
    )
    assert "alpha[1]" in list(glm_intercept.fitted_samples_.summary().head().index)


@pytest.mark.slow
# @pytest.mark.parametrize("estimator", [GLM()])
def test_compatible_estimator() -> None:
    """Ensure that GLM Estimator is sk-learn compatible."""
    check_estimator(GLM())


@pytest.mark.parametrize(
    "alg_args",
    [
        {},
        {
            "iter_warmup": 100,
            "iter_sampling": 100,
        },
        {"chains": 2},
    ],
)
def test_GLM_alg_params_correct(alg_args) -> None:
    """
    Verify that validation occurs from within GLM class correctly.
    """
    glm = GLM(
        algorithm="sample",
        algorithm_params=alg_args,
        family="gamma",
        link="log",
        seed=1234,
    )

    gamma_dat_X, gamma_dat_Y = _gen_fam_dat_continuous(
        family="gamma", link="log", Nsize=100
    )

    glm.fit(X=gamma_dat_X, y=gamma_dat_Y)


def test_notfittederror_glm() -> None:
    glm = GLM(seed=1234)
    with pytest.raises(Exception) as e_info:
        glm.predict(X=np.array([2, 4, 8, 16]))


@pytest.mark.parametrize("prior_config", [None, {}])
def test_prior_config_default_nongaussian(prior_config) -> None:
    """Test that the default prior config is used if no prior config is provided."""
    glm = GLM(family="gamma", link="log", seed=1234, priors=prior_config)
    X, y = _gen_fam_dat_continuous(family="gamma", link="log", seed=1234321)

    fitted = glm.fit(X=X, y=y)

    if prior_config is None:
        assert fitted.priors_ == {
            "prior_slope_dist": 0,
            "prior_slope_mu": [0.0],
            "prior_slope_sigma": [2.5],
        }
    else:
        assert fitted.priors_["prior_slope_dist"] == -1

    assert fitted.prior_intercept_ == {
        "prior_intercept_dist": 0,
        "prior_intercept_mu": 0.0,
        "prior_intercept_sigma": 2.5,
    }

    assert (
        fitted.fitted_samples_.summary()["5%"]["alpha[1]"] - 0.01
        <= 0.6
        <= fitted.fitted_samples_.summary()["95%"]["alpha[1]"]
    )


@pytest.mark.parametrize("prior_config", [{}, {}])
def test_prior_config_default_gaussian(prior_config) -> None:
    """Test that the default prior config is used if no prior config is provided."""
    glm = GLM(
        family="gaussian", link="log", seed=1234, priors=prior_config, autoscale=True
    )
    X, y = _gen_fam_dat_continuous(family="gaussian", link="log", seed=1234321)

    fitted = glm.fit(X=X, y=y)

    assert fitted.priors_["prior_slope_dist"] == -1

    assert fitted.prior_intercept_ == {
        "prior_intercept_dist": 0,
        "prior_intercept_mu": 0.0,
        "prior_intercept_sigma": 2.5 * np.std(y),
    }

    assert (
        fitted.fitted_samples_.summary()["5%"]["alpha[1]"] - 0.01
        <= 0.6
        <= fitted.fitted_samples_.summary()["95%"]["alpha[1]"]
    )


def test_laplace_default_setup() -> None:
    glm = GLM(
        family="gaussian",
        link="log",
        seed=1234,
        priors={
            "prior_slope_dist": "laplace",
            "prior_slope_mu": [0.0],
            "prior_slope_sigma": [2.5],
        },
    )
    X, y = _gen_fam_dat_continuous(family="gaussian", link="log", seed=1234321)

    fitted = glm.fit(
        X=X,
        y=y,
    )

    assert fitted.priors_["prior_slope_dist"] == 1
    assert fitted.prior_intercept_["prior_intercept_dist"] == 0


@pytest.mark.parametrize(
    "unsupported_prior",
    [
        " ",
        "default",
        dict(
            prior_intercept_dist=" ",
            prior_intercept_mu=[0.0],
            prior_intercept_sigma=[2.5],
        ),
    ],
)
def test_prior_intercept_erroneous(unsupported_prior):
    """Confirm that error is raised on priors that are not supported."""
    glm = GLM(family="gamma", link="log", seed=1234, prior_intercept=unsupported_prior)

    gamma_dat_X, gamma_dat_Y = _gen_fam_dat_continuous(
        family="gamma", link="log", Nsize=100
    )

    with pytest.raises(Exception) as e_info:
        fitted = glm.fit(X=gamma_dat_X, y=gamma_dat_Y)


@pytest.mark.parametrize(
    "unsupported_prior",
    [
        " ",
        "default",
        dict(prior_slope_dist=" ", prior_slope_mu=[0.0], prior_slope_sigma=[2.5]),
    ],
)
def test_priors_erroneous(unsupported_prior):
    """Confirm that error is raised on priors that are not supported."""
    glm = GLM(family="gamma", link="log", seed=1234, priors=unsupported_prior)

    gamma_dat_X, gamma_dat_Y = _gen_fam_dat_continuous(
        family="gamma", link="log", Nsize=100
    )

    with pytest.raises(Exception) as e_info:
        fitted = glm.fit(X=gamma_dat_X, y=gamma_dat_Y)


@pytest.mark.parametrize(
    "prior_intercept_config,prior_slope_config",
    [
        (
            {},
            {
                "prior_intercept_dist": "normal",
                "prior_intercept_mu": [0.0],
                "prior_intercept_sigma": [1.0],
            },
        ),
        ({}, {}),
    ],
)
def test_prior_config_custom_normal(prior_slope_config, prior_intercept_config) -> None:
    """Test that partial & full set-up of priors with all-normal priors."""
    if prior_slope_config == {
        "prior_intercept_dist": "normal",
        "prior_intercept_mu": [0.0],
        "prior_intercept_sigma": [1.0],
    }:
        pytest.skip(reason="pytest misconfiguration ")
    glm = GLM(
        family="gamma",
        link="log",
        seed=1234,
        prior_intercept=prior_intercept_config,
        priors=prior_slope_config,
    )
    X, y = _gen_fam_dat_continuous(family="gamma", link="log", seed=1234321)

    fitted = glm.fit(X=X, y=y)

    if len(prior_slope_config) == 0:
        assert fitted.prior_intercept_["prior_intercept_dist"] == -1
    else:
        assert fitted.prior_intercept_["prior_intercept_dist"] == 0
    if len(prior_intercept_config) == 0:
        assert fitted.priors_["prior_slope_dist"] == -1
    else:
        assert fitted.priors_["prior_slope_dist"] == 0


def test_prior_setup_full() -> None:
    glm = GLM(
        family="gamma",
        link="log",
        seed=1234,
        prior_intercept={
            "prior_intercept_dist": "normal",
            "prior_intercept_mu": 0,
            "prior_intercept_sigma": 1,
        },
        priors={
            "prior_slope_dist": "normal",
            "prior_slope_mu": [0.0],
            "prior_slope_sigma": [1.0],
        },
    )

    X, y = _gen_fam_dat_continuous(family="gamma", link="log", seed=1234321)

    fitted = glm.fit(X=X, y=y)

    assert fitted.prior_intercept_["prior_intercept_dist"] == 0
    assert fitted.priors_["prior_slope_dist"] == 0


def test_prior_setup_half() -> None:
    glm = GLM(
        family="gamma",
        link="log",
        seed=1234,
        priors={
            "prior_slope_dist": "normal",
            "prior_slope_mu": [0.0],
            "prior_slope_sigma": [1.0],
        },
        prior_intercept={},
    )

    X, y = _gen_fam_dat_continuous(family="gamma", link="log", seed=1234321)

    fitted = glm.fit(X=X, y=y)

    assert fitted.prior_intercept_["prior_intercept_dist"] == -1
    assert fitted.priors_["prior_slope_dist"] == 0


@pytest.mark.parametrize("algorithm", ["sample", "optimize", "variational"])
def test_custom_seed_all_algs(algorithm: str) -> None:
    """Ensure that user-set seed persists for each algorithm."""
    glm = GLM(algorithm=algorithm, seed=999999)
    X, y = _gen_fam_dat_continuous(family="gamma", link="log", seed=999999)

    glm.fit(X=X, y=y)

    assert glm.fitted_samples_.metadata.cmdstan_config["seed"] == 999999


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


@pytest.mark.parametrize("family", ["gaussian", "gamma", "inverse-gaussian"])
def test_auto_canonical_link_continuous(family: str) -> None:
    """
    Test that the canonical link is automatically chosen for the family.
    """
    canonical_links = {
        "gaussian": "identity",
        "gamma": "inverse",
        "inverse-gaussian": "inverse-square",
    }
    glm = GLM(family=family, seed=1234)
    glm.fit(X=np.array([[1, 2, 3], [4, 5, 6]]), y=np.array([1, 2]))

    assert glm.link_ == canonical_links[family]


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
def test_gaussian_link_scipy_gen(link: str):
    if link == "inverse":
        pytest.skip(
            reason="Gaussian + inverse is known not to work with default priors"
        )

    glm = GLM(family="gaussian", link=link, seed=1234)

    gaussian_dat_X, gaussian_dat_Y = _gen_fam_dat_continuous(
        family="gaussian", link=link, Nsize=1000
    )

    glm.fit(X=gaussian_dat_X, y=gaussian_dat_Y)

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
    glm = GLM(family="gamma", link=link, seed=1234, algorithm_params={})

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
    glm_gamma = GLM(family="gamma", link="inverse", seed=1234)

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


@pytest.mark.skip(reason="Inverse Gaussian LLH computation may be unstable")
@pytest.mark.slow
@pytest.mark.parametrize("link", ["identity", "log", "inverse", "inverse-square"])
def test_invgaussian_link_scipy_gen(link: str):
    if link == "identity":
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


@pytest.mark.parametrize(
    "prior_aux",
    [
        None,
        {},
        {"prior_aux_dist": "exponential", "beta": 0.5},
        {"prior_aux_dist": "chi2", "nu": 2.5},
    ],
)
def test_glm_prior_aux_setup(prior_aux) -> None:
    """Test that auxiliary prior in continuous case is setup correctly."""
    glm = GLM(family="gaussian", link="inverse", seed=1234, prior_aux=prior_aux)

    gaussian_dat_X, gaussian_dat_Y = _gen_fam_dat_continuous(
        family="gaussian", link="inverse"
    )

    glm.fit(X=gaussian_dat_X, y=gaussian_dat_Y)

    if prior_aux is None:
        """Default unscaled prior."""
        assert glm.prior_aux_ == {
            "num_prior_aux_params": 1,
            "prior_aux_dist": 0,
            "prior_aux_params": [1.0],
        }
    elif len(prior_aux) == 0:
        assert glm.prior_aux_ == {
            "num_prior_aux_params": 1,
            "prior_aux_dist": -1,
            "prior_aux_params": [0.0],
        }
    else:
        if prior_aux["prior_aux_dist"] == "exponential":
            assert glm.prior_aux_ == {
                "num_prior_aux_params": 1,
                "prior_aux_dist": 0,
                "prior_aux_params": [0.5],
            }
        else:
            assert glm.prior_aux_ == {
                "num_prior_aux_params": 1,
                "prior_aux_dist": 1,
                "prior_aux_params": [2.5],
            }


# NOTE: for the identity link, the generated data may lead to a negative lambda
@pytest.mark.parametrize("link", ["identity", "log", "sqrt"])
def test_poisson_link_scipy_gen(link: str):
    if link == "identity":
        pytest.skip(
            reason="""Poisson + identity is known not to work with default priors;
             also, identity link leads to potentially negative lambda..."""
        )
    glm = GLM(family="poisson", link=link, seed=1234)

    if link == "identity":
        rng = np.random.default_rng(seed=9999)

        poisson_dat_X = stats.norm.rvs(10, 1, size=(1000,))
        poisson_dat_Y = rng.poisson(0.6 + 0.2 * poisson_dat_X)
    else:
        poisson_dat_X, poisson_dat_Y = _gen_fam_dat_discrete(
            family="poisson", link=link
        )

    fitted = glm.fit(X=poisson_dat_X, y=poisson_dat_Y)

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

    glm_poisson = GLM(family="poisson", link="log", seed=1234, algorithm="optimize")

    glm_poisson.fit(X=X, y=y)

    assert True
