import numpy as np 
import scipy.stats as stats 

def gen_fam_dat(family: str, Nsize, alpha, beta, sigma=0.3): 
    """
    Generate data for a given family.
    """
    if family == "gaussian":
        X = stats.norm.rvs(0, 1, size=Nsize)
        Y = stats.norm.rvs(alpha + beta * X, sigma)
    elif family == "binomial":
        X= np.random.binomial(1, 0.5, size=Nsize)
    elif family == "gamma":
        X = np.random.gamma(1.9, size=Nsize)
        Y = np.random.gamma(alpha + beta * X, size=Nsize)
    elif family == "poisson":
        X= np.random.poisson(1, size=Nsize)
    elif family == "inverse_gaussian":
        X= stats.invgauss.rvs(0.45, size=Nsize)
    else:
        raise ValueError(f"Family {family} not supported.")

    return X, Y