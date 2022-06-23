"""Vectorized BLR model with sk-learn type API"""

import json
from pathlib import Path

import numpy as np
import scipy.stats as stats  # type: ignore
from cmdstanpy import CmdStanModel  # type: ignore
from numpy.typing import ArrayLike, NDArray

from sk_stan_regression.modelcore import CoreEstimator
from sk_stan_regression.utils.validation import check_array, check_is_fitted, validate_family

BLR_FOLDER = Path(__file__).parent
DEFAULT_FAKE_DATA = BLR_FOLDER.parent / "data" / "fake_data.json"

method_dict = {
    "HMC-NUTS": CmdStanModel.sample,
    "MLE": CmdStanModel.optimize,
    "Variational": CmdStanModel.variational,
}


class BLR_Estimator(CoreEstimator):
    """
    Vectorized, multidimensional version of the BLR Estimator above.
    Note that the intercept alpha and error scale sigma remain as scalar values
    while beta becomes a vector.

    :param alpha_: posterior mean of intercept of the linear regression
    :param alpha_samples_: samples generated from the posterior for model intercept
    :param beta_: posterior mean of slope of the linear regression
    :param beta_samples_: samples generated from the posterior for model slope
    :param sigma_: posterior mean of error scale of the linear regression
    :param sigma_samples_: samples generated from the posterior for model error scale

    :param algorithm: algorithm that performs an operation on the posterior
    """

    # alpha_: float = None
    # alpha_samples_: ArrayLike = None
    # beta_: Optional[ArrayLike] = None
    # beta_samples_: Optional[ArrayLike] = None
    # sigma_: Optional[float] = None
    # sigma_samples_: Optional[ArrayLike] = None
    # is_fitted_: Optional[float] = None

    def __init__(
        self,
        algorithm: str = "HMC-NUTS",
        family: str = "gaussian", 
        link: str = "identity",
    ):
        self.algorithm = algorithm

        validate_family(family, link)

        self.family = family
        self.link = link 

    def __repr__(self) -> str:
        return (
            f"""<BLR_Estimator:
                        alpha={self.alpha_!r}, alpha_samples={self.alpha_samples_!r}, 
                        beta={self.beta_!r}, beta_samples={self.beta_samples_!r}, 
                        sigma={self.sigma_!r}, sigma_samples={self.sigma_samples_!r}>
                """
            if hasattr(self, "is_fitted_")
            else """<BLR_Estimator: unfitted>"""
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "CoreEstimator":
        """
        Fits current vectorized BLR object to the given data,
        with a default set of data.
        This model is considered fit once its alpha, beta,
        and sigma parameters are determined via a regression.

        Where N is the number of data items (rows) and
        K is the number of predictors (columns) in x:

        :param X: NxK predictor matrix, where K >= 0; if K = 1,
            then X is automatically reshaped to being 2D
        :param y: Nx1 outcome vector

        :return: self, an object
        """
        if y is None:
            raise ValueError(
                """This Bayesian Linear Regression
             estimator requires y to be passed, but it is None"""
            )

        if X is None:
            raise ValueError(
                """This Bayesian Linear Regression
             estimator requires X to be passed, but it is None"""
            )

        # TODO: test this functionality
        if self.algorithm not in method_dict.keys():
            raise ValueError(
                f"""Current Linear Regression created with algorithm 
                {self.algorithm!r}, which is not one of the supported 
                methods. Try with one of the following: (HMC-NUTS, MLE, 
                Variational)."""
            )

        X_clean, y_clean = self._validate_data(X=X, y=y, ensure_X_2d=True)

        self.model_ = CmdStanModel(stan_file=BLR_FOLDER / "blinreg_v.stan")

        dat = {
            "X": X_clean,
            "y": y_clean,
            "N": X_clean.shape[0],  # type: ignore
            "K": X_clean.shape[1],  # type: ignore
        }

        self.fitted_samples = method_dict[self.algorithm](
            self.model_, data=dat, show_console=False
        )

        stan_vars = self.fitted_samples.stan_variables()
        if self.algorithm == "HMC-NUTS":
            self.alpha_ = stan_vars["alpha"].mean(axis=0)
            self.beta_ = stan_vars["beta"].mean(axis=0)
            self.sigma_ = stan_vars["sigma"].mean(axis=0)

            self.alpha_samples_ = stan_vars["alpha"]
            self.beta_samples_ = stan_vars["beta"]
            self.sigma_samples_ = stan_vars["sigma"]
        else:
            self.alpha_ = stan_vars["alpha"]
            self.beta_ = stan_vars["beta"]
            self.sigma_ = stan_vars["sigma"]

        self.is_fitted_ = True

        return self

    def _predict_distribution(
        self,
        X: NDArray[np.float64],
        num_iterations: int = 1000,
        num_chains: int = 4,
    ) -> NDArray[np.float64]:
        """
        Predict using a fitted model after fit() has been applied.

        :param num_iterations: int
        :param num_chains: int number of chains for MCMC sampling

        :return: Return a dictionary mapping Stan program variable names
        to the corresponding numpy.ndarray containing the inferred values.
        """
        check_is_fitted(self)

        if X is None:
            raise ValueError(
                f"""This {self.__class__.__name__!r}
             estimator requires X to be passed, but it is None"""
            )

        # TODO: should be a call to self._validate_data
        X_clean = check_array(X=X, ensure_2d=True)

        if self.algorithm != "HMC-NUTS":
            return stats.norm.rvs(  # type: ignore
                self.alpha_ + np.dot(self.beta_, np.array(X_clean)),
                self.sigma_,
            )

        predictions = CmdStanModel(stan_file=BLR_FOLDER / "sample_normal_v.stan")

        dat = {
            "N": X_clean.shape[0],
            "K": X_clean.shape[1],
            "X": X_clean,
            "alpha": self.alpha_,
            "beta": self.beta_,
            "sigma": self.sigma_,
        }

        # known that fitted with HMC-NUTS, so fitted_samples is not None 
        # TODO: this generate_quantities call does not work!
        predicGQ = predictions.generate_quantities(dat, mcmc_sample=self.fitted_samples)

        #samples = predictions.sample(
        #    data=dat, iter_sampling=num_iterations, chains=num_chains
        #)

        return predicGQ.stan_variable("y_sim")  # type: ignore

    def predict(
        self,
        X: ArrayLike,
        num_iterations: int = 1000,
        num_chains: int = 4,
    ) -> np.float64:
        """
        Predict using a fitted model after fit() has been applied.

        :param num_iterations: int
        :param num_chains: int number of

        :return: Return a dictionary mapping Stan program variable
                names to the corresponding numpy.ndarray containing
                the inferred values.
        """
        X_clean, _ = self._validate_data(X=X, ensure_X_2d=True)
        # note the above errors out if X is None

        return self._predict_distribution(  # type: ignore
            X_clean, num_iterations, num_chains  # type: ignore
        ).mean(axis=0, dtype=np.float64)


if __name__ == "__main__":
    with open(DEFAULT_FAKE_DATA) as file:
        jsondat = json.load(file)

    xdat = np.array(jsondat["x"])
    ydat = np.array(jsondat["y"])

    kby2 = np.column_stack((xdat, xdat))

    blr = BLR_Estimator()
    blr.fit(xdat, ydat)
    print(blr.predict(X=xdat))

    #blr2 = BLR_Estimator(algorithm="Variational")
    #blr2.fit(xdat, ydat)
#
    #blr3 = BLR_Estimator(algorithm="MLE")
    #blr3.fit(xdat, ydat)
#
    #print(blr.predict(xdat))
    #print(blr._predict_distribution(xdat)) 
    #blr2 = BLR_Estimator()
    #blr2.fit(kby2, ydat)
#
    #blr2.predict(kby2)
    #res = blr2.predict(kby2)   
    ## check exceptions   
    #blr = BLR_Estimator()
    #blr.predict(X=xdat)

# blrvec = BLR_Estimator()
# blrvec.fit(X=xdat, y=ydat)
# print(blrvec.__repr__())
# ysim = blrvec.predict(X=xdat)
# print(ysim)

# blr2 = BLR_Estimator(algorithm="MLE")
# blr2.fit(X=xdat, y=ydat)
# ysim2 = blr2.predict(X=xdat)
# print(ysim2)

# kby2 = np.column_stack((xdat, xdat))

# blrvec = BLR_Estimator()
# blrvec.fit(kby2, ydat)
# ysim2 = blrvec.predict(X=kby2)
# print(ysim2)
