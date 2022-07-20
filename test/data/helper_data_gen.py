import numpy as np
import scipy.stats as stats  # type: ignore
from scipy.special import expit


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

    X = stats.norm.rvs(0, 1, size=Nsize, random_state=rng)

    if family == "gaussian":
        if link == "identity":
            Y = stats.norm.rvs(alpha + beta * X, sigma, random_state=rng)
        elif link == "log":
            Y = stats.norm.rvs(np.exp(alpha + beta * X), sigma, random_state=seed)
        elif link == "inverse":
            Y = stats.norm.rvs(1 / (alpha + beta * X), sigma, random_state=rng)
    elif family == "gamma":
        # NOTE: for the identity or inverse link, the generated data may lead to a negative lambda
        if link == "identity":
            Y = rng.gamma(alpha + beta * X)
        elif link == "log":
            Y = rng.gamma(np.exp(alpha + beta * X))
        elif link == "inverse":
            Y = rng.gamma(1 / (alpha + beta * X))
    elif family == "inverse-gaussian":
        if link == "identity":
            Y = stats.invgauss.rvs(alpha + beta * X, random_state=rng)
        elif link == "log":
            Y = stats.invgauss.rvs(np.exp(alpha + beta * X), random_state=rng)
        elif link == "inverse":
            Y = stats.invgauss.rvs(1 / (alpha + beta * X), random_state=rng)
        elif link == "inverse-square":
            Y = stats.invgauss.rvs(1 / (alpha + beta * X) ** 2, random_state=rng)
    else:
        raise ValueError(f"Family {family} not supported.")

    return X, Y


# TODO: make this multimdimensional, the number of trials
# should be component-wise...
def _gen_fam_dat_discrete(
    family: str,
    link: str,
    num_yn=1000,
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

    if family == "binomial":
        if link == "log":
            y = stats.binom.rvs(num_yn, np.exp(alpha + beta * X))
        elif link == "logit":
            y = stats.binom.rvs(num_yn, expit(alpha + beta * X))
        elif link == "probit":
            y = stats.binom.rvs(num_yn, stats.norm.cdf(alpha + beta * X))
        elif link == "cloglog":
            pass
        elif link == "cauchit":
            pass
    elif family == "poisson":
        if link == "identity":
            y = rng.poisson(alpha + beta * X, size=sample_size)
        elif link == "log":
            y = rng.poisson(np.exp(alpha + beta * X), size=sample_size)
        # NOTE: using sqrt as link is dangerous, results in multimodal distribution
        elif link == "sqrt":
            y = rng.poisson(np.square(alpha + beta * X), size=sample_size)

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
