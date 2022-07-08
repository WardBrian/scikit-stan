import numpy as np
import scipy.stats as stats  # type: ignore


# TODO: make multidimensional size input & output
# TODO: generalize to include different link functions
def _gen_fam_dat(
    family: str,
    Nsize=1000,
    alpha=0.6,
    beta=0.2,
    sigma=0.3,
    num_binom=100,
    discrete_prob=0.5,
    mu=0.145,
):
    """
    Generate data for a given family.
    """
    if family == "gaussian":
        X = stats.norm.rvs(0, 1, size=Nsize)
        Y = stats.norm.rvs(alpha + beta * X, sigma)
    # elif family == "binomial":
    #    X = stats.binom.rvs(Nsize, discrete_prob)
    #    # TODO: how to generate Y data?
    elif family == "gamma":
        X = stats.gamma.rvs(sigma, size=Nsize)
        Y = stats.gamma.rvs(alpha + beta * X, size=Nsize)
    elif family == "poisson":
        X = np.random.poisson(mu, size=Nsize)
        Y = np.random.poisson(alpha + beta * X, size=Nsize)
    elif family == "inverse-gaussian":
        X = stats.invgauss.rvs(mu, size=Nsize)
        Y = stats.invgauss.rvs(alpha + beta * X, size=Nsize)
    else:
        raise ValueError(f"Family {family} not supported.")

    return X, Y
