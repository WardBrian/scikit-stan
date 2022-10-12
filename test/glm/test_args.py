"""Test that the various arguments of the GLM are validated properly"""

import numpy as np
import pytest
import scipy.sparse as sp
from data import _gen_fam_dat_continuous, _gen_fam_dat_discrete

from scikit_stan.generalized_linear_regression import GLM


def test_trials_non_binomial():
    """Test passing the trials argument to
    something other than a Binomial family."""

    glm = GLM(family="gaussian", seed=1234)

    with pytest.raises(ValueError):
        glm.fit([[0.1]], [0.1], trials=[0])


def test_trials_float():
    """Test with trials bigger which aren't ints"""

    glm = GLM(family="binomial", seed=1234)
    with pytest.raises(ValueError):
        glm.fit([[1.5]], [1], trials=[0.5])


def test_trials_too_big():
    """Test with trials bigger than i32"""

    glm = GLM(family="binomial", seed=1234)
    with pytest.raises(ValueError):
        glm.fit([[1.5]], [1], trials=[int(2**45)])


def test_sparse() -> None:
    """Test sparse X"""

    glm = GLM(family="gaussian", seed=1234)

    gaussian_dat_X, gaussian_dat_Y = _gen_fam_dat_continuous(
        family="gaussian", link="identity", seed=12345
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
    "prior_slope_config,prior_intercept_config",
    [
        (
            {},
            {
                "prior_intercept_dist": "normal",
                "prior_intercept_mu": 0.0,
                "prior_intercept_sigma": 1.0,
            },
        ),
        ({}, {}),
    ],
)
def test_prior_config_custom_normal(prior_slope_config, prior_intercept_config) -> None:
    """Test that partial & full set-up of priors with all-normal priors."""

    glm = GLM(
        family="gamma",
        link="log",
        seed=1234,
        prior_intercept=prior_intercept_config,
        priors=prior_slope_config,
    )
    X, y = _gen_fam_dat_continuous(family="gamma", link="log", seed=1234321)

    glm.fit(X=X, y=y)

    if len(prior_intercept_config) == 0:
        assert glm.prior_intercept_["prior_intercept_dist"] == -1
    else:
        assert glm.prior_intercept_["prior_intercept_dist"] == 0
    if len(prior_slope_config) == 0:
        assert glm.priors_["prior_slope_dist"] == -1
    else:
        assert glm.priors_["prior_slope_dist"] == 0


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
    X, y = _gen_fam_dat_continuous(family="gaussian", link="log", seed=1234)

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
