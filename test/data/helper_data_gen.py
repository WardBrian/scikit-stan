import numpy as np
import scipy.stats as stats  # type: ignore
from scipy.special import expit


def _link_mu(link: str, mu: np.ndarray):
    if link == "identity":
        return mu
    elif link == "log":
        return np.exp(mu)
    elif link == "inverse":
        return 1 / (mu)
    elif link == "inverse-square":
        return 1 / (mu**2)
    elif link == "logit":
        return expit(mu)
    elif link == "probit":
        return stats.norm.cdf(mu)
    elif link == "cloglog":
        pass
    elif link == "cauchit":
        return np.atan(mu) / np.pi + 0.5
    elif link == "sqrt":
        return np.square(mu)
    else:
        raise ValueError(f"bad link {link}")


# TODO: make multidimensional size input & output
def _gen_fam_dat_continuous(
    family: str,
    link: str,
    Nsize=100,
    alpha=0.6,
    beta=0.2,
    sigma=0.3,
    mu=0.145,
    seed=1234,
):
    """
    Generate data for a given family.
    """
    rng = np.random.default_rng(seed=seed)

    X = rng.random(size=Nsize)

    mu = alpha + beta * X
    mu_linked = _link_mu(link, mu)

    if family == "gaussian":
        y = stats.norm.rvs(mu_linked, sigma, random_state=rng)
    elif family == "gamma":
        y = rng.gamma(mu_linked)
    elif family == "inverse-gaussian":
        y = stats.invgauss.rvs(mu_linked, random_state=rng)
    else:
        raise ValueError(f"Family {family} not supported.")

    return X, y


# TODO: make this multimdimensional, the number of trials
# should be component-wise...
def _gen_fam_dat_discrete(
    family: str,
    link: str,
    alpha=0.6,
    beta=0.2,
    sample_size: int = 30,
    seed: int = 1234,
) -> np.ndarray:
    """
    Generate data for a given discrete family and link.
    """
    rng = np.random.default_rng(seed=seed)
    X = np.linspace(-10, 20, sample_size)

    mu = alpha + beta * X
    mu_linked = _link_mu(link, mu)

    if family == "poisson":
        y = rng.poisson(mu_linked, size=sample_size)
        # data gen follows sklearn example
        # https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_0_23_0.html#sphx-glr-auto-examples-release-highlights-plot-release-highlights-0-23-0-py
        # rng = np.random.RandomState(0)
        # X = rng.randn(1000, 20)
        # y = rng.poisson(lam=np.exp(X[:, 5]) / 2)

        # another alternative?
        # X = stats.poisson.rvs(poisson_lambda, size=sample_size)
        # y = stats.poisson.rvs(alpha + beta * X)

    else:
        raise ValueError(f"Family {family} not supported.")

    return X, y
