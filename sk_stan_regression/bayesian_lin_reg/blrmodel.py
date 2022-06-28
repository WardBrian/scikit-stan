"""Vectorized BLR model with sk-learn type API"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy.stats as stats  # type: ignore
# TODO: update cmdstanpy version to 1.0.2 for typing
from cmdstanpy import CmdStanModel  # type: ignore
from numpy.typing import ArrayLike, NDArray

from sk_stan_regression.modelcore import CoreEstimator
from sk_stan_regression.utils.validation import (
    FAMILY_LINKS_MAP,
    check_array,
    check_is_fitted,
    validate_family,
)

BLR_FOLDER = Path(__file__).parent
DEFAULT_FAKE_DATA = BLR_FOLDER.parent / "data" / "fake_data.json"

method_dict = {
    "HMC-NUTS": CmdStanModel.sample,
    "MLE": CmdStanModel.optimize,
    "Variational": CmdStanModel.variational,
}

BLR_FAMILIES = {
    "gaussian": 0,
    "binomial": 1,
    "gamma": 2,
    "poisson": 3,
    "inverse-gaussian": 4,
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

    # TODO: user defined seed ends up breaking the tests
    def __init__(
        self,
        algorithm: str = "HMC-NUTS",
        family: str = "gaussian",
        link: str = "identity",
        seed: Optional[int] = None,
        show_console: bool = False,
    ):
        self.algorithm = algorithm

        validate_family(family, link)
        self.family = family
        self.link = link

        self.seed = seed

        self.show_console = show_console

    def __repr__(self) -> str:
        return (
            f"""<BLR_Estimator:
                        alpha={self.alpha_!r}, alpha_samples={self.alpha_samples_!r}, 
                        beta={self.beta_!r}, beta_samples={self.beta_samples_!r}, 
                        sigma={self.sigma_!r}, sigma_samples={self.sigma_samples_!r}, 
                        seed={self.seed!r},>
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

        self.linkid_ = FAMILY_LINKS_MAP[self.family][self.link]
        self.familyid_ = BLR_FAMILIES[self.family]

        is_cont_dat = self.familyid_ in [0, 2, 4] # if true, continuous, else discrete

        X_clean, y_clean = self._validate_data(X=X, y=y, ensure_X_2d=True, dtype=np.float64 if is_cont_dat else np.int64)

        self.model_ = CmdStanModel(stan_file=BLR_FOLDER / "blinreg_v_continuous.stan") if is_cont_dat else CmdStanModel(stan_file=BLR_FOLDER / "blinreg_v_discrete.stan")

        print(type(X_clean[0][0]))

        # TODO: this is a hack, make validation cast to either ints or floats!!!
        dat = {
            "X": X_clean,
            "y": y_clean,
            "N": X_clean.shape[0],  # type: ignore
            "K": X_clean.shape[1],  # type: ignore
            "family": self.familyid_,
            "link": self.linkid_,
        }

        self.seed_ = self.seed

        self.fitted_samples_ = method_dict[self.algorithm](
            self.model_,
            data=dat,
            show_console=self.show_console,
            seed=self.seed_,
            sig_figs=9,
        )

        if self.seed_ is None:
            self.seed_ = self.fitted_samples_.metadata.cmdstan_config["seed"]

        stan_vars = self.fitted_samples_.stan_variables()
        if self.algorithm == "HMC-NUTS":
            self.alpha_ = stan_vars["alpha"].mean(axis=0)
            self.alpha_samples_ = stan_vars["alpha"]

            self.beta_ = stan_vars["beta"].mean(axis=0)
            self.beta_samples_ = stan_vars["beta"]

            # sigma error scale only for gaussian...
            if self.family == "gaussian":
                self.sigma_ = stan_vars["sigma"].mean(axis=0)
                self.sigma_samples_ = stan_vars["sigma"]
        else:
            self.alpha_ = stan_vars["alpha"]
            self.beta_ = stan_vars["beta"]
            
            if self.family == "gaussian":
                self.sigma_ = stan_vars["sigma"]

        self.is_fitted_ = True
        self.n_features_in_ = X_clean.shape[1]  # type: ignore

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
                self.alpha_ + np.dot(self.beta_, np.array(X_clean)),  # type: ignore
                self.sigma_,
                random_state=self.seed_,
            )

        predictions = CmdStanModel(stan_file=BLR_FOLDER / "sample_normal_v.stan")

        dat = {
            "N": X_clean.shape[0],
            "K": X_clean.shape[1],
            "X": X_clean,
            "family": self.familyid_,
            "link": self.linkid_,
            "alpha": self.alpha_,
            "beta": self.beta_,
            "sigma": self.sigma_,
        }

        # known that fitted with HMC-NUTS, so fitted_samples is not None
        predicGQ = predictions.generate_quantities(
            dat,
            mcmc_sample=self.fitted_samples_,
            seed=self.seed_,
            sig_figs=9,
            show_console=self.show_console,
        )

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

    def _more_tags(self) -> Dict[str, Any]:
        return {
            "_xfail_checks": {
                "check_methods_sample_order_invariance": "check is not applicable.",
                "check_methods_subset_invariance": "check is not applicable.",
                "check_fit_idempotent": """model is idempotent, but not to the required degree of accuracy as this is a 
                    probabilistic setting.""",
                "check_fit1d": """provided automatic cast from 1d to 2d in data validation.""",
                # NOTE: the expected behavior here is to raise a ValueError, the package intends
                # to give alternative default behavior in these scenarios!
                "check_fit2d_predict1d": """provided automatic cast from 1d to 2d in data validation.
                 STILL NEEDS TO BE INVESTIGATED FOR GQ ISSUE""",
                # NOTE: the expected behavior here is to raise a ValueError,
                #  the package intends to give alternative default behavior in these scenarios!
            }
        }

    @classmethod
    def _seed(self) -> Optional[int]:
        """
        Get the seed used to generate the samples.
        """
        return self.seed_ if self.seed_ == self.seed else self.seed
