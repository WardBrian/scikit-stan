import json

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from blrmodel import BLR_Estimator
from cmdstanpy import CmdStanModel
from scipy import stats

# cmdstanpy_data = az.from_cmdstanpy(
#    posterior=stan_fit,
#    posterior_predictive="y_hat",
#    observed_data={"y": eight_school_data["y"]},
#    log_likelihood="log_lik",
#    coords={"school": np.arange(eight_school_data["J"])},
#    dims={
#        "theta": ["school"],
#        "y": ["school"],
#        "log_lik": ["school"],
#        "y_hat": ["school"],
#        "theta_tilde": ["school"],
#    },
# )
# print(cmdstanpy_data)


# def posterior(Phi, t, alpha, beta, return_inverse=False):
#    """Computes mean and covariance matrix of the posterior distribution."""
#    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
#    S_N = np.linalg.inv(S_N_inv)
#    m_N = beta * S_N.dot(Phi.T).dot(t)
#
#    if return_inverse:
#        return m_N, S_N, S_N_inv
#    else:
#        return m_N, S_N
#
# def plot_posterior(mean, cov, w0, w1):
#    resolution = 100
#
#    grid_x = grid_y = np.linspace(-1, 1, resolution)
#    grid_flat = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)
#
#    densities = stats.multivariate_normal.pdf(grid_flat, mean=mean.ravel(), cov=cov).reshape(resolution, resolution)
#    plt.imshow(densities, origin='lower', extent=(-1, 1, -1, 1))
#    plt.scatter(w0, w1, marker='x', c="r", s=20, label='Truth')
#
#    plt.xlabel('w0')
#    plt.ylabel('w1')
#
# def identity_basis_function(x):
#    return x


# def expand(x, bf, bf_args=None):
#    print(x)
#    if bf_args is None:
#        return np.concatenate([np.ones(x.shape), bf(x)], axis=1)
#    else:
#        return np.concatenate([np.ones(x.shape)] + [bf(x, bf_arg) for bf_arg in bf_args], axis=1)
#

alphatrue = 0.6
betatrue = 0.2

stan_file = "./nvblinplot.stan"
stan_model = CmdStanModel(stan_file=stan_file)
stan_model.compile()

eight_school_data = {
    "J": 8,
    "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
    "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
}

stan_fit = stan_model.sample(data=eight_school_data)

cmdstanpy_data = az.from_cmdstanpy(
    posterior=stan_fit,
    posterior_predictive="y_hat",
    observed_data={"y": eight_school_data["y"]},
    log_likelihood="log_lik",
    coords={"school": np.arange(eight_school_data["J"])},
    dims={
        "theta": ["school"],
        "y": ["school"],
        "log_lik": ["school"],
        "y_hat": ["school"],
        "theta_tilde": ["school"],
    },
)

print(cmdstanpy_data)

# with open("../data/fake_data.json") as file:
#    jsondat = json.load(file)
#
# xdat_test = jsondat["x"]
# ydat_test = jsondat["y"]
#
# blr = BLR_Estimator()
#
# blr.fit(X=xdat_test, y=ydat_test)
#
# ysim = blr.predict(X=xdat_test)


# phi_test = expand(np.array(xdat_test), identity_basis_function)

# m, s = posterior(phi_test, ydat_test, blr.alpha, blr.beta)
