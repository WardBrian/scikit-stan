import numpy as np
import scipy.stats as stats  # type: ignore
from scipy.special import expit


# TODO: make multidimensional size input & output
# TODO: generalize to include different link functions
def _gen_fam_dat_continuous(
    family: str, Nsize=1000, alpha=0.6, beta=0.2, sigma=0.3, mu=0.145, seed=1234
):
    """
    Generate data for a given family.
    """
    np.random.seed(seed=seed)

    if family == "gaussian":
        X = stats.norm.rvs(0, 1, size=Nsize)
        Y = stats.norm.rvs(alpha + beta * X, sigma)
    elif family == "gamma":
        X = stats.gamma.rvs(sigma, size=Nsize)
        Y = stats.gamma.rvs(alpha + beta * X, size=Nsize)
    elif family == "inverse-gaussian":
        X = stats.invgauss.rvs(mu, size=Nsize)
        Y = stats.invgauss.rvs(alpha + beta * X, size=Nsize)
    else:
        raise ValueError(f"Family {family} not supported.")

    return X, Y


# TODO: make this multimdimensional, the number of trials
# should be component-wise...
def _gen_fam_dat_discrete(
    family: str,
    link: str,
    alpha: float,
    beta: np.ndarray,
    num_yn: int,
    sample_size: int,
    poisson_lambda: float,
    seed: int = 1234,
) -> np.ndarray:
    """
    Generate data for a given discrete family and link.
    """
    rng = np.random.default_rng(seed=seed)

    if family == "binomial":
        X = np.linspace(-10, 20, sample_size)
        mu_true = alpha + beta * X
        p_true = expit(mu_true)
        y = rng.binomial(num_yn, p_true)
    elif family == "poisson":

        # data gen follows sklearn example
        # https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_0_23_0.html#sphx-glr-auto-examples-release-highlights-plot-release-highlights-0-23-0-py
        rng = np.random.RandomState(0)
        X = rng.randn(1000, 20)
        y = rng.poisson(lam=np.exp(X[:, 5]) / 2)

        # X = stats.poisson.rvs(poisson_lambda, size=sample_size)
        # y = stats.poisson.rvs(alpha + beta * X)
        # X = np.linspace(-10, 20, sample_size)
        # lambda_true = alpha + beta * X
        # y = rng.poisson(lambda_true, size=sample_size)

    else:
        raise ValueError(f"Family {family} not supported.")

    return X, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X, y = _gen_fam_dat_discrete("binomial", "a", 0.7, np.array([0.4]), 20, 30)
    plt.scatter(X, y, color="k")
    plt.show()
