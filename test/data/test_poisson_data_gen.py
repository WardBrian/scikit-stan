import numpy as np
import scipy.stats as stats  # type: ignore

N = 100
K = 5

X = stats.norm.rvs(0, 1, size=(N, K))
beta = stats.norm.rvs(0, 1, size=K)
alpha = stats.norm.rvs(0, 1, size=1)

rng = np.random.default_rng(seed=1234)

y = rng.poisson(np.exp(alpha + X @ beta))

# print(X.shape)
# print(y.shape)
# print(np.exp(alpha + X @ beta))

from pathlib import Path

GLM_POISSON_FOLDER = Path(__file__).parent / "poisson_reg.stan"

from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file=GLM_POISSON_FOLDER)
dat = {"X": X, "y": y, "N": N, "K": K}
fit = model.sample(data=dat)

print(fit.summary())
print(beta, alpha)
