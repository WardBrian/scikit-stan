import numpy as np
import scipy.stats as stats
from data import _link_mu

from scikit_stan import GLM


def family_log_lik(family: str, y: np.ndarray, mu: np.ndarray, aux: np.ndarray):
    y = y[:, np.newaxis]
    if family == "gaussian":
        return stats.norm.logpdf(y, mu, aux)
    if family == "gamma":
        return stats.gamma.logpdf(y, aux / mu, aux)
    if family == "poisson":
        return stats.poisson.logpmf(y, mu)


def assert_log_lik(model: GLM, X: np.ndarray, y: np.ndarray, link: str, family: str):
    X = np.atleast_2d(X.T).T
    loglik = model.fitted_samples_.log_lik

    beta = model.beta_samples_
    mu = X @ beta.T
    if model.fit_intercept:
        mu += model.alpha_samples_.T

    mu_linked = _link_mu(link, mu)

    sigma = model.sigma_samples_ if hasattr(model, "sigma_samples_") else np.array([])

    computed_log_lik = family_log_lik(family, y, mu_linked, sigma)

    np.testing.assert_allclose(loglik, computed_log_lik.T, rtol=1e-5, atol=0.01)
