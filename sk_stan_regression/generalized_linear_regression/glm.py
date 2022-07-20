"""Vectorized BLR model with sk-learn type API"""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import scipy.stats as stats  # type: ignore
from cmdstanpy import CmdStanModel
from numpy.typing import ArrayLike, NDArray

from sk_stan_regression.modelcore import CoreEstimator
from sk_stan_regression.utils.validation import (
    FAMILY_LINKS_MAP,
    check_array,
    check_is_fitted,
    validate_family,
    validate_prior,
)

GLM_STAN_FILES_FOLDER = Path(__file__).parent.parent / "stan_files"

method_dict = {
    "HMC-NUTS": CmdStanModel.sample,
    "L-BFGS": CmdStanModel.optimize,
    "ADVI": CmdStanModel.variational,
}

GLM_FAMILIES = {
    "gaussian": 0,
    "gamma": 1,
    "inverse-gaussian": 2,
    "poisson": 3,
    "binomial": 4,
    "negative-binomial": 5,
}

# pre-compile continuous & discrete models
# so they aren't compiled every time fit() is called
GLM_CONTINUOUS_STAN = CmdStanModel(
    stan_file=GLM_STAN_FILES_FOLDER / "glm_v_continuous.stan"
)

GLM_DISCRETE_STAN = CmdStanModel(
    stan_file=GLM_STAN_FILES_FOLDER / "glm_v_discrete.stan"
)


class GLM(CoreEstimator):
    """
    Vectorized, multidimensional version of the BLR Estimator above.
    Note that the intercept alpha and error scale sigma remain as scalar values
    while beta becomes a vector.

    :param algorithm: algorithm that performs an operation on the posterior
    :param family: GLM family function; all R Families are supported
    :param link: GLM link function; all R Families links are supported in admissible combinations
    :param seed: random seed used for probabilistic operations; chosen randomly if not specified
    :param priors: Dictionary of priors to use for the model.
    By default, all regression coefficient priors are set to
        beta ~ normal(0, 2.5 * sd(y) / sd(X)) if Gaussian else normal(0, 2.5)

    The number of specified priors cannot exceed the number of features in the data.
    Each prior is specified as a dictionary with the following keys:
        "prior_slope_dist": distribution of the prior on this coefficient
        "prior_slope_mu": location parameter of the prior on this coefficient
        "prior_slope_sigma": scale parameter of the prior on this coefficient
    The main passed dictionary indexes the priors by the row index of the feature starting at 0.
    Thus, to specify a prior on the first feature, the dictionary should be passed as
     {0: {"prior_slope_dist": "normal", "prior_slope_mu": 0, "prior_slope_sigma": 1}}.
    Any unspecified priors will be set to the default.

    :param prior_intercept: Prior for the intercept alpha parameter for GLM.
    If this is not specified, the default is
        alpha ~ normal(mu(y), 2.5 * sd(y)) if Gaussian family else normal(0, 2.5)

    To set this prior explicitly, pass a dictionary with the following keys:
        "prior_intercept_dist": str, distribution of the prior from the list
         of supported prior distributions: "normal", "laplace"
        "prior_intercept_mu": float, location parameter of the prior distribution
        "prior_intercept_sigma": float, error scale parameter of the prior distribution
    """

    def __init__(
        self,
        algorithm: str = "HMC-NUTS",
        family: str = "gaussian",
        link: Optional[str] = None,
        seed: Optional[int] = None,
        priors: Optional[Dict[int, Dict[str, Any]]] = None,
        prior_intercept: Optional[Dict[str, Any]] = None,
    ):
        self.algorithm = algorithm

        self.family = family
        self.link = link

        self.priors = priors
        self.prior_intercept = prior_intercept

        self.seed = seed

    # TODO: make intercept-less choice?
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        show_console: bool = False,
    ) -> "CoreEstimator":
        """
        Fits current vectorized BLR object to the given data,
        with a default set of data.
        This model is considered fit once its alpha, beta,
        and sigma parameters are determined via a regression.

        Where N is the number of data items (rows) and
        K is the number of predictors (columns) in x:

        :param X: NxK predictor matrix, where K >= 0; if K = 1,
            then X is automatically reshaped to being 2D; this raises a warning
        :param y: Nx1 outcome vector
        :param show_console: whether to show the Stan console output
        :param priors: dictionary of priors to use for the model.
        The prior for the intercept is set via "intercept" and the prior for the non-intercept
        regression coefficients is set via "beta".
        If only one is specified, then the other is set to the default.
        Currently supported priors are: "normal" (default) with parameters "location" and "scale",
        and "laplace" with parameters "location" and "scale". The error scale "sigma" for
        continuous models defaults to exponential(1/sy) where sy = sd(y) if the specified family is
        a Gaussian, and sy = 1 in all other cases. TODO: this should be user settable as well...
        For the intercept prior, "location" and "scale" must be scalars.
        For the prior on the coefficients, "location" and "scale" must be either
            1) be vectors of length equal to the number of coefficients
          excluding the intercept (the number of features in your dataset) or
            2) be scalars, in which case the same values will be
          repeated for each coefficient
          TODO: improve docs for this... should the parameters for each individual
          prior for each coefficient be customizable?

        NOTE: the usual prior-selection advice holds. See for example:
        http://www.stat.columbia.edu/~gelman/research/published/entropy-19-00555-v2.pdf

        :return: self, an object of type GLM
        """
        if y is None:
            raise ValueError(
                """This Generalized Linear Model requires a response variable y, but it is None."""
            )

        if X is None:
            raise ValueError(
                """This Generalized Linear Model requires predictors X, but it is None."""
            )

        if self.algorithm not in method_dict.keys():
            raise ValueError(
                f"""Current Linear Regression created with algorithm 
                {self.algorithm!r}, which is not one of the supported 
                methods. Try with one of the following: (HMC-NUTS, L-BFGS, ADVI)."""
            )

        self.is_cont_dat_ = self.family in [
            "gaussian",
            "gamma",
            "inverse-gaussian",
        ]

        X_clean, y_clean = self._validate_data(
            X=X,
            y=y,
            ensure_X_2d=True,
            dtype=np.float64 if self.is_cont_dat_ else np.int64,
        )

        self.link_ = self.link
        # set the canonical link function for each family if
        # user does not specify the link function
        if not self.link_:  # link has not been set
            if self.family == "gaussian":
                self.link_ = "identity"
            elif self.family == "gamma":
                self.link_ = "inverse"
            elif self.family == "inverse-gaussian":
                self.link_ = "inverse-square"
            elif self.family == "poisson":
                self.link_ = "log"
            elif any(self.family == x for x in ["bernoulli", "binomial"]):
                self.link_ = "logit"

            warnings.warn(
                f"""
                    Link function not specified. Using default link function {self.link_!r}
                    for family {self.family!r}.
                """
            )

        validate_family(self.family, self.link_)

        if not self.is_cont_dat_ and self.link_ == "identity":
            self.link_ = "logit" if self.family == "bernoulli" else "log"

        # link_ is already verified to not be None
        self.linkid_ = FAMILY_LINKS_MAP[self.family][self.link_]  # type: ignore
        self.familyid_ = GLM_FAMILIES[self.family]

        K = X_clean.shape[1]

        # data common to all prior choices for intercept and coefficients
        dat = {
            "X": X_clean,
            "y": y_clean,
            "N": X_clean.shape[0],
            "K": K,
            "family": self.familyid_,
            "link": self.linkid_,
            "predictor": 0,
            "prior_intercept_dist": None,
            "prior_intercept_mu": None,
            "prior_intercept_sigma": None,
            "prior_slope_dist": [None] * K,
            "prior_slope_mu": [None] * K,
            "prior_slope_sigma": [None] * K,
        }

        # set up common prior parameters; this computation is
        # unnecessary if the user supplies all prior parameters
        # TODO: clean up so these computations are not performed in
        # that scenario
        if self.familyid_ == 0:  # gaussian
            sdy, sdx, my = np.std(y_clean), np.std(X_clean), np.mean(y_clean)
        else:
            sdy, sdx, my = 1.0, 1.0, 0.0

        if sdy == 0.0:
            sdy = 1.0

        dat["sdy"] = sdy

        # likely to be reused across multiple features
        # default prior selection follows:
        # https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html
        DEFAULT_SLOPE_PRIOR = (
            {
                "prior_slope_dist": 0,
                "prior_slope_mu": 0.0,
                "prior_slope_sigma": 2.5 * sdy / sdx,
            }
            if self.family == "gaussian"
            else {
                "prior_slope_dist": 0,
                "prior_slope_mu": 0.0,
                "prior_slope_sigma": 2.5,
            }
        )

        priors_ = {}

        # user did not specify any priors
        if self.priors is None or len(self.priors) == 0:
            priors_ = {idx: DEFAULT_SLOPE_PRIOR for idx in np.arange(K)}
        else:
            for idx in np.arange(K):
                if idx in self.priors:
                    priors_[idx] = validate_prior(self.priors[idx], "slope")
                else:
                    priors_[idx] = DEFAULT_SLOPE_PRIOR

        self.priors_ = priors_

        # TODO: add functionality for GLM to not have an intercept at all
        # set up default prior for intercept if not user-specified
        if self.prior_intercept is None or len(self.prior_intercept) == 0:
            warnings.warn(
                """Prior on intercept not specified. Using default prior.
                alpha ~ normal(mu(y), 2.5 * sd(y)) if Gaussian family else normal(0, 2.5)"""
            )
            self.prior_intercept_ = {
                "prior_intercept_dist": 0,  # normal
                "prior_intercept_mu": my,
                "prior_intercept_sigma": 2.5 * sdy,
            }
        else:
            self.prior_intercept_ = validate_prior(self.prior_intercept, "intercept")

        dat["prior_intercept_dist"] = self.prior_intercept_["prior_intercept_dist"]
        dat["prior_intercept_mu"] = self.prior_intercept_["prior_intercept_mu"]
        dat["prior_intercept_sigma"] = self.prior_intercept_["prior_intercept_sigma"]

        # set up of vectors for intercept and coefficients for Stan data
        for idx in np.arange(K):
            dat["prior_slope_dist"][idx] = self.priors_[idx]["prior_slope_dist"]
            dat["prior_slope_mu"][idx] = self.priors_[idx]["prior_slope_mu"]
            dat["prior_slope_sigma"][idx] = self.priors_[idx]["prior_slope_sigma"]

        if self.is_cont_dat_:
            self.model_ = GLM_CONTINUOUS_STAN
        else:
            self.model_ = GLM_DISCRETE_STAN
            # TODO: this shouldn't be a repeat, this should be different for every component?
            dat["trials"] = np.repeat(y_clean.shape[0], X_clean.shape[0])

        self.seed_ = self.seed

        self.fitted_samples_ = method_dict[self.algorithm](  # type: ignore
            self.model_,
            data=dat,
            show_console=show_console,
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

            # sigma error scale only for continuous models
            if self.is_cont_dat_:
                self.sigma_ = stan_vars["sigma"].mean(axis=0)
                self.sigma_samples_ = stan_vars["sigma"]
        else:
            self.alpha_ = stan_vars["alpha"]
            self.beta_ = stan_vars["beta"]

            if self.is_cont_dat_:
                self.sigma_ = stan_vars["sigma"]

        self.is_fitted_ = True
        self.n_features_in_ = X_clean.shape[1]

        return self

    def predict_distribution(
        self,
        X: NDArray[Union[np.float64, np.int64]],
        show_console: bool = False,
    ) -> NDArray[Union[np.float64, np.int64]]:
        """
        Predicts the distribution of the response variable after model has been fit.
        This is distinct from predict(), which returns the mean of distribution predictions.

        :param X: predictor matrix or array to use as basis for prediction

        :return: predictions made by fitted model an NDArray of shape (n_samples, 1)
        """
        check_is_fitted(self)

        if X is None:
            raise ValueError(
                f"""This {self.__class__.__name__!r}
             estimator requires X to be passed, but it is None"""
            )

        X_clean = check_array(
            X=X, ensure_2d=True, dtype=np.float64 if self.is_cont_dat_ else np.int64
        )

        # TODO: link functions???
        # TODO: discrete families
        # NOTE: in a future Stan release, generate quantities() will not be restricted
        # to requiring an MCMC sample, so the following will be obsolete
        if self.algorithm != "HMC-NUTS":
            if self.family == "gaussian":
                return stats.norm.rvs(  # type: ignore
                    self.alpha_ + np.dot(self.beta_, X_clean),  # type: ignore
                    self.sigma_,
                    random_state=self.seed_,
                )
            elif self.family == "gamma":
                return stats.gamma.rvs(  # type: ignore
                    self.alpha_ + np.dot(self.beta_, X_clean),  # type: ignore
                    self.sigma_,
                    random_state=self.seed_,
                )
            elif self.family == "inverse-gaussian":
                return stats.invgauss.rvs(  # type: ignore
                    self.alpha_ + np.dot(self.beta_, X_clean),  # type: ignore
                    self.sigma_,
                    random_state=self.seed_,
                )

        dat = {
            "N": X_clean.shape[0],
            "K": X_clean.shape[1],
            "X": X_clean,
            "y": [],
            "trials": [],
            "family": self.familyid_,
            "link": self.linkid_,
            "predictor": 1,
            "prior_intercept_dist": 0,
            "prior_intercept_mu": 1.0,
            "prior_intercept_sigma": 2.5,
            "prior_slope_dist": [0] * X_clean.shape[1],
            "prior_slope_mu": [0.0] * X_clean.shape[1],
            "prior_slope_sigma": [0.0] * X_clean.shape[1],
            "sdy": 1.0,
        }

        # known that fitted with HMC-NUTS, so fitted_samples is not None
        predicGQ = self.model_.generate_quantities(
            dat,
            mcmc_sample=self.fitted_samples_,
            seed=self.seed_,
            sig_figs=9,
            show_console=show_console,
        )

        return predicGQ.stan_variable("y_sim")

    def predict(
        self,
        X: ArrayLike,
        show_console: bool = False,
    ) -> NDArray[np.float64]:
        """
        Predict using a fitted model after fit() has been applied.
        Computes the mean of the predicted distribution, given by y_sim.

        :param X: predictor matrix or array to use as basis for prediction

        :return: an NDArray of shape (n_samples, 1)
        """
        check_is_fitted(self)

        if X is None:
            raise ValueError(
                f"""This {self.__class__.__name__!r}
             estimator requires X to be passed, but it is None"""
            )

        X_clean, _ = self._validate_data(X=X, ensure_X_2d=True)

        return self.predict_distribution(  # type: ignore
            X_clean,
            show_console=show_console,
        ).mean(axis=0, dtype=np.float64)

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        show_console: bool = False,
    ) -> float:
        """
        Computes the coefficient of determination R^2 of the prediction,
        as do other sklearn estimators.

        :param X: array-like of shape (n_samples, n_features) containing test samples
        :param y: array-like of shape (n_samples,) containing test target values
        :show_console: verbose display of CmdStanPy console output

        :return: (float) R^2 of the prediction versus the given target values;
                 this is the mean accuracy of self.predict(X) with respect to y
        """
        check_is_fitted(self)

        # ensure that y vector works plays nicely with np
        y_clean = check_array(
            y, ensure_2d=False, dtype=np.float64 if self.is_cont_dat_ else np.int64
        )

        predictions = self.predict(X=X, show_console=show_console)

        mean_obs = np.sum(y_clean) / len(y_clean)
        ssreg: float = np.sum((predictions - mean_obs) ** 2)
        sstot: float = np.sum((y_clean - mean_obs) ** 2)

        return 1 - ssreg / sstot

    def _more_tags(self) -> Dict[str, Any]:
        """
        Sets tags for current model that exclude certain sk-learn estimator
        checks that are not applicable to this model.
        """
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
