"""Generalized Linear Model with prior control for regression in scikit-learn type API."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
from numpy.typing import ArrayLike, NDArray

from scikit_stan.modelcore import CoreEstimator
from scikit_stan.utils.stan import load_stan_model
from scikit_stan.utils.validation import (
    FAMILY_LINKS_MAP,
    GLM_FAMILIES,
    check_array,
    check_is_fitted,
    method_dict,
    validate_aux_prior,
    validate_family,
    validate_prior,
)

GLM_CONTINUOUS_STAN = load_stan_model("glm_v_continuous")
GLM_DISCRETE_STAN = load_stan_model("glm_v_discrete")


class GLM(CoreEstimator):
    r"""
    A generalized linear model estimator with several options for families, links,
    and priors on regression coefficients, the intercept, and error scale,
    done in an scikit-learn style.
    This class also provides an autoscaling feature of the priors.
    For deterministic behavior from this model, the class's seed can be set and is then
    passed to Stan computations.

    Parameters
    ----------
    algorithm : str, optional
        Algorithm to be used by the Stan model. The following are supported:

            * sample - runs the HMC-NUTS sampler,
            * optimize - produces a likelihood estimate of model parameters,
            * variational - runs Stan's variational inference algorithm to compute the posterior.

    algorithm_params : Dict[str, Any], optional
        Parameters for the selected algorithm. The key words
        and values are derived from CmdStanPy's API,
        so please refer to
        this documentation for more information:
        https://mc-stan.org/cmdstanpy/api.html#cmdstanmodel

        Customizing these fields is done via a passed dictionary, which is validated
        on the level of CmdStan. As an example, to specify the number of chains for
        the HMC-NUTS sampler to run, it is sufficient to pass::

            {
                "chains": 2
            }

        or to specify the number of warmup and sampling
        iterations, pass::

            {
                "iter_warmup": 100,
                "iter_sampling": 100,
            },

        Default Stan parameters are used if nothing is passed.

    fit_intercept : bool, optional
        Whether the GLM should fit a global intercept. Some hierarchical models operate
        without an intercept, so setting this to False will fill the first column of the
        predictor matrix with ones.

        By default, the GLM has an intercept term -- the first column of the predictor
        matrix X is used for inference of the intercept.

    family : str, optional
        Distribution family used for linear regression. All R Families package are supported:

            * "gaussian" - Gaussian distribution,
            * "gamma" - Gamma distribution,
            * "inverse-gaussian" - Inverse Gaussian distribution,
            * "poisson" - Poisson distribution,
            * "binomial" - Binomial distribution

    link : Optional[str], optional
        Distribution link function used for linear regression,
        following R's family-link combinations.
        These are family-specific:

            * "gaussian":
                + "identity" - Identity link function,
                + "log" - Log link function,
                + "inverse" - Inverse link function
            * "gamma":
                + "identity" - Identity link function,
                + "log" - Log link function,
                + "inverse" - Inverse link function
            * "inverse-gaussian":
                + "identity" - Identity link function,
                + "log" - Log link function,
                + "inverse" - Inverse link function,
                + "inverse-square" - Inverse square link function
            * "poisson":
                + "identity" - Identity link function,
                + "log" - Log link function,
                + "sqrt" - Square root link function
            * "binomial":
                + "log" - Log link function,
                + "logit" - Logit link function,
                + "probit" - Probit link function,
                + "cloglog" - Complementary log-log link function,
                + "cauchit" - Cauchit link function

        If an invalid combination of family and link is passed, a ValueError is raised.

        When no link is specified, these are family-specific defaults for the link function:

            * "gaussian": "identity",
            * "gamma": "inverse",
            * "inverse-gaussian": "inverse",
            * "poisson": "identity",
            * "binomial": "logit"

    seed : Optional[int], optional
        Seed for random number generator. Must be an integer between 0 an 2^32-1.
        If this is left unspecified, then a random seed will be used for all chains
        via :class:`numpy.random.RandomState`.
        Specifying this field will yield the same result for multiple uses if
        all other parameters are held the same.

    priors : Optional[Dict[str, Union[int, float, List]]], optional
        Dictionary for configuring prior distribution on coefficients.
        Currently supported priors are: "normal", and "laplace".

        By default, all regression coefficient priors are set to

        .. math:: \beta \sim \text{normal}(0, 2.5 \cdot \text{sd}(y) / \text{sd}(X))

        if Gaussian, else

        .. math:: \beta \sim \text{normal}(0, 2.5 / \text{sd}(X))

        if auto_scale is True, otherwise

        .. math:: \beta \sim \text{normal}(0, 2.5 \cdot \text{sd}(y))

        if Gaussian else

        .. math:: \beta \sim \text{normal}(0, 2.5)

        If an empty dictionary {} is passed, no priors are used and
        the default Stan uniform(-inf, inf) prior is used for all coefficients.

        The number of specified prior parameters cannot exceed the number of predictors in
        the data as each parameter is associated with a single coefficient.
        If the number of specified priors is less than the number of predictors,
        the remaining coefficients are set to the default prior.
        The prior on all regression coefficients is set with the following keys:

            + "prior_slope_dist": distribution of the prior for each coefficient
            + "prior_slope_mu": list of location parameters of the prior for each coefficient
            + "prior_slope_sigma": list of scale parameters of the prior for each coefficient

        Thus, to specify a standard normal prior on the first feature,
        the dictionary should be passed as::

            {
                "prior_slope_dist": "normal",
                "prior_slope_mu": [0.],
                "prior_slope_sigma": [1.]
            }

        Any unspecified priors will be set to the default.

        Also note that choosing a Laplace prior is equivalent to L1 regularization:
        https://stats.stackexchange.com/questions/177210/why-is-laplace-prior-producing-sparse-solutions/177217#177217

    prior_intercept : Optional[Dict[str, Any]], optional
        Prior for the intercept alpha parameter for GLM.
        If this is not specified, the default is

        .. math:: \alpha \sim \text{normal}(\text{mu}(y), 2.5 \cdot \text{sd}(y))

        if Gaussian family else

        .. math:: \alpha \sim \text{normal}(0, 2.5)

        If an empty dictionary {} is passed, the default Stan uniform(-inf, inf) prior is used.

        To set this prior explicitly, pass a dictionary with the following keys:

            + "prior_intercept_dist": str, distribution of the prior from the list of supported
              prior distributions: "normal", "laplace"
            + "prior_intercept_mu": float, location parameter of the prior distribution
            + "prior_intercept_sigma": float, error scale parameter of the prior distribution

        Thus, for example, passing::

            {
                "prior_intercept_dist": "normal",
                "prior_intercept_mu": 0,
                "prior_intercept_sigma": 1
            }

        results in

        .. math:: \alpha \sim \text{normal}(0, 1)

        by default (without autoscaling, see below).

        Also note that choosing a Laplace prior is equivalent to L1 regularization:
        https://stats.stackexchange.com/questions/177210/why-is-laplace-prior-producing-sparse-solutions/177217#177217

    prior_aux : Optional[Dict[str, Any]], optional
        Prior on the auxiliary parameter for the family used in
        the regression: for example, the std for Gaussian, shape for gamma, etc...
        Currently supported priors are: "exponential", "chi2", "gamma", and "inv_gamma".
        These are parameterized by the following distribution parameters:
        "beta", "nu", "alpha" and "beta", and "alpha" and "beta", respectively.

        Priors here with more parameters are a future feature.

        If an empty dictionary {} is passed, the default Stan uniform(-inf, inf) prior is used.

        For single-parameter priors, this field is a dictionary with the following keys

            + "prior_aux_dist": distribution of the prior on this parameter
            + "prior_aux_param": parameter of the prior on this parameter
                                - see above for admissible tags

        For example, to specify a chi2 prior with nu=2.5, pass::

            {
                "prior_aux_dist": "chi2",

                "nu": 2.5
            }

        The default un-scaled prior is ``exponential(1)``, the default scaled prior is
        ``exponential(1/sy)`` where ``sy = sd(y)`` if the specified family is a Gaussian,
        and ``sy = 1`` in all other cases. Setting ``autoscale=True`` results in division by
        sy.

    autoscale : bool, optional
        Enable automatic scaling of priors. Autoscaling is performed the same as in `rstanarm
        <https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html>`__.

        This procedure does not happen by default.

    Notes
    -----
    The usual prior-selection advice holds. See these discussions about prior selection:
        - https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
        - http://www.stat.columbia.edu/~gelman/research/published/entropy-19-00555-v2.pdf
    """

    def __init__(
        self,
        algorithm: str = "sample",
        algorithm_params: Optional[Dict[str, Any]] = None,
        fit_intercept: bool = True,
        family: str = "gaussian",
        link: Optional[str] = None,
        seed: Optional[int] = None,
        priors: Optional[Dict[str, Union[int, float, List[Union[float, int]]]]] = None,
        prior_intercept: Optional[Dict[str, Any]] = None,
        prior_aux: Optional[Dict[str, Any]] = None,
        autoscale: bool = False,
    ):
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params

        self.fit_intercept = fit_intercept

        self.family = family
        self.link = link

        self.priors = priors
        self.prior_intercept = prior_intercept
        self.prior_aux = prior_aux
        self.autoscale = autoscale

        self.seed = seed

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        show_console: bool = False,
    ) -> "CoreEstimator":
        """
        Fits GLM model to the given data.
        This model is considered fit once its alpha, beta,
        and sigma parameters are determined via a regression.

        Parameters
        ----------
        X : ArrayLike
            NxK matrix of predictors, where K >= 0. If K = 1,
            then X is automatically reshaped to being 2D and raises a warning.
            X can be sparse. It will be converted to CSR format if it is not already.
        y : ArrayLike
            Nx1 outcome vector where each row is a response corresponding to
            the same row of predictors in X.
        show_console : bool, optional
            Printing output of default CmdStanPy console during Stan operations.

        Returns
        -------
        CoreEstimator
            Abstract class for all estimator-type models in this package.
            The GLM class is a subclass of CoreEstimator.

        Raises
        ------
        ValueError
            X is required to have at least 2 columns and have the same number of rows as y.
        ValueError
            y is required to have exactly 1 column and the same number of rows as X.
        ValueError
            Algorithm choice in model set-up is not supported.

        Notes
        -----
        Other ValueErrors may be raised by additional validation checks.
        These include invalid prior set-up or invalid data.
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
                methods. Try with one of the following: (sample, optimize, variational)."""
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

        # ensure that identity link function is not used for discrete regressions
        if not self.is_cont_dat_ and self.link_ == "identity":
            self.link_ = "logit" if self.family == "bernoulli" else "log"

        # link_ is already verified to not be None
        self.linkid_ = FAMILY_LINKS_MAP[self.family][self.link_]  # type: ignore
        self.familyid_ = GLM_FAMILIES[self.family]

        K = X_clean.shape[1]

        # data common to all prior choices for intercept and coefficients
        dat = {
            "y": y_clean,
            "N": X_clean.shape[0],
            "K": K,
            "family": self.familyid_,
            "link": self.linkid_,
            "predictor": 0,
            "fit_intercept": self.fit_intercept,
            "prior_intercept_dist": None,
            "prior_intercept_mu": None,
            "prior_intercept_sigma": None,
            "prior_slope_dist": [None] * K,
            "prior_slope_mu": [None] * K,
            "prior_slope_sigma": [None] * K,
        }

        prepare_X_data(dat, X_clean)

        if sp.issparse(X_clean):
            # .std does not work on sparse matrices, so do this instead
            sdx = np.sqrt((X_clean.power(2)).mean() - np.square(X_clean.mean()))
        else:
            sdx = X_clean.std()
        if self.familyid_ == 0:  # gaussian
            my = np.mean(y_clean) if self.linkid_ == 0 else 0.0
            sdy = np.std(y_clean)
        else:
            my = 0.0
            sdy = 1.0

        if sdy == 0.0:
            sdy = 1.0

        DEFAULT_SLOPE_PRIOR = {
            "prior_slope_dist": 0,
            "prior_slope_mu": [0.0] * K,
        }
        # likely to be reused across multiple features
        # default prior selection follows:
        # https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html
        # NOTE: this sets up default priors for all features
        if self.family == "gaussian" and self.autoscale:
            DEFAULT_SLOPE_PRIOR["prior_slope_sigma"] = [2.5 * sdy / sdx] * K
        else:
            if self.autoscale:
                DEFAULT_SLOPE_PRIOR["prior_slope_sigma"] = [2.5 / sdx] * K
            else:
                DEFAULT_SLOPE_PRIOR["prior_slope_sigma"] = [2.5] * K

        priors_ = {}

        # user did not specify any regression coefficient priors
        if self.priors is None:
            priors_ = DEFAULT_SLOPE_PRIOR
        else:
            # set slope priors to be default flat as in Stan:
            # uniform(-infinity, +infinity)
            if len(self.priors) == 0:
                priors_ = {
                    "prior_slope_dist": -1,  # no prior for intercept
                    "prior_slope_mu": [0] * K,
                    "prior_slope_sigma": [0] * K,
                }
            else:
                priors_ = validate_prior(self.priors, "slope")
                if (
                    len(self.priors["prior_slope_mu"]) != K  # type: ignore
                    or len(self.priors["prior_slope_sigma"]) != K  # type: ignore
                ):
                    raise ValueError(
                        "Length of prior_slope_mu and prior_slope_sigma must be "  # type: ignore
                        "equal to the number of features in X.\n"
                        f"Got {len(self.priors['prior_slope_mu'])} "
                        f"and {len(self.priors['prior_slope_sigma'])} respectively."
                    )

        self.priors_ = priors_

        # set up default prior for intercept if not user-specified
        if self.prior_intercept is None:
            warnings.warn(
                """Prior on intercept not specified. Using default prior.
                alpha ~ normal(mu(y), 2.5 * sd(y)) if Gaussian family else normal(0, 2.5)"""
            )

            self.prior_intercept_ = {
                "prior_intercept_dist": 0,  # normal
                "prior_intercept_mu": my
                if self.family == "gaussian" and self.link == "identity"
                else 0.0,
            }

            self.prior_intercept_["prior_intercept_sigma"] = (
                2.5 * sdy if self.autoscale else 2.5
            )
        else:
            # set intercept prior to be default flat as in Stan:
            # uniform(-infinity, +infinity)
            if len(self.prior_intercept) == 0:
                self.prior_intercept_ = {
                    "prior_intercept_dist": -1,
                    "prior_intercept_mu": 0.0,
                    "prior_intercept_sigma": 0.0,
                }
            else:
                self.prior_intercept_ = validate_prior(
                    self.prior_intercept, "intercept"
                )

        self.prior_aux_: Dict[str, Any] = {}

        # validate auxiliary parameter prior
        if self.prior_aux is None:
            self.prior_aux_ = {
                "num_prior_aux_params": 1,
                "prior_aux_dist": 0,  # exponential
            }

            if self.autoscale:
                warnings.warn(
                    """Prior on auxiliary parameter not specified. Using default scaled prior
                        sigma ~ exponential(1 / sd(y))
                    """
                )
                self.prior_aux_["prior_aux_params"] = [1.0 / sdy]
            else:
                warnings.warn(
                    """Prior on auxiliary parameter not specified. Using default unscaled prior
                        sigma ~ exponential(1)
                    """
                )

                self.prior_aux_["prior_aux_params"] = [1.0]
        else:
            # set auxiliary parameter prior to be default flat as in Stan:
            # uniform(-infinity, +infinity)
            if len(self.prior_aux) == 0:
                self.prior_aux_ = {
                    "num_prior_aux_params": 1,
                    "prior_aux_dist": -1,
                    "prior_aux_params": [0.0],
                }
            else:
                self.prior_aux_ = validate_aux_prior(self.prior_aux)

        dat["prior_intercept_dist"] = self.prior_intercept_["prior_intercept_dist"]
        dat["prior_intercept_mu"] = self.prior_intercept_["prior_intercept_mu"]
        dat["prior_intercept_sigma"] = self.prior_intercept_["prior_intercept_sigma"]

        # set up of vectors for intercept and coefficients for Stan data
        dat["prior_slope_dist"] = self.priors_["prior_slope_dist"]
        dat["prior_slope_mu"] = self.priors_["prior_slope_mu"]
        dat["prior_slope_sigma"] = self.priors_["prior_slope_sigma"]

        dat["prior_aux_dist"] = self.prior_aux_["prior_aux_dist"]
        dat["num_prior_aux_params"] = self.prior_aux_["num_prior_aux_params"]
        dat["prior_aux_params"] = self.prior_aux_["prior_aux_params"]

        if self.is_cont_dat_:
            self.model_ = GLM_CONTINUOUS_STAN
        else:
            self.model_ = GLM_DISCRETE_STAN
            # XXX this seems like mostly nonsense?
            # needs to be separate keyword argument or make Y 2d?
            # binomial should be its own model class
            dat["trials"] = np.repeat(y_clean.shape[0], X_clean.shape[0])

        self.seed_ = self.seed

        self.fitted_samples_ = method_dict[self.algorithm](  # type:ignore
            self.model_,
            data=dat,
            show_console=show_console,
            seed=self.seed_,
            sig_figs=9,
            **self.algorithm_params if self.algorithm_params else {},
        )

        if self.seed_ is None:
            self.seed_ = self.fitted_samples_.metadata.cmdstan_config["seed"]

        stan_vars = self.fitted_samples_.stan_variables()
        if self.algorithm == "sample":
            if self.fit_intercept:
                self.alpha_ = stan_vars["alpha"].mean(axis=0)
                self.alpha_samples_ = stan_vars["alpha"]

            self.beta_ = stan_vars["beta"].mean(axis=0)
            self.beta_samples_ = stan_vars["beta"]

            # sigma error scale only for continuous models
            if self.is_cont_dat_:
                self.sigma_ = stan_vars["sigma"].mean(axis=0)
                self.sigma_samples_ = stan_vars["sigma"]
        else:
            if self.fit_intercept:
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
        """Predicts the distribution of the response variable after model has been fit.
        This is distinct from predict(), which returns the mean of distribution
        predictions generated by this method.

        Parameters
        ----------
        X : NDArray[Union[np.float64, np.int64]]
            Predictor matrix or array of data to use for prediction.
            X can be sparse. It will be converted to CSR format if it is not already.
        show_console : bool, optional
            Printing output of default CmdStanPy console during Stan operations.

        Returns
        -------
        NDArray[Union[np.float64, np.int64]]
            Set of draws generated by Stan generate quantities method.

        Raises
        ------
        ValueError
            Method requires data X to be supplied.
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

        # NOTE: in a future Stan release, generate quantities() will not be restricted
        # to requiring an MCMC sample, so the following will be obsolete
        if self.algorithm != "sample":
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

        dat: Dict[str, Any] = {
            "N": X_clean.shape[0],
            "K": X_clean.shape[1],
            "y": [],
            # XXX: setting empty breaks discrete GQ
            "trials": [],
            "family": self.familyid_,
            "link": self.linkid_,
            "predictor": 1,
            "fit_intercept": self.fit_intercept,
            "prior_intercept_dist": 0,
            "prior_intercept_mu": 1.0,
            "prior_intercept_sigma": 2.5,
            "prior_slope_dist": 0,
            "prior_slope_mu": [0.0] * X_clean.shape[1],
            "prior_slope_sigma": [0.0] * X_clean.shape[1],
            "prior_aux_dist": 0,  # these don't affect anything when generating predictions
            "prior_aux_param": 1.0,  # these don't affect anything when generating predictions
        }

        prepare_X_data(dat, X_clean)

        dat["num_prior_aux_params"] = self.prior_aux_["num_prior_aux_params"]
        dat["prior_aux_params"] = self.prior_aux_["prior_aux_params"]

        # known that fitted with sampling, so fitted_samples is not None
        predicGQ = self.model_.generate_quantities(
            dat,
            mcmc_sample=self.fitted_samples_,
            seed=self.seed_,
            sig_figs=9,
            show_console=show_console,
        )

        return predicGQ.y_sim

    def predict(
        self,
        X: ArrayLike,
        return_std: bool = False,
        show_console: bool = False,
    ) -> Union[
        NDArray[Union[np.float64, np.int64]],
        Tuple[
            NDArray[Union[np.float64, np.int64]], NDArray[Union[np.float64, np.int64]]
        ],
    ]:
        """Compute predictions from supplied data using a fitted model.
            This computes the mean of the predicted distribution,
            given by y_sim in predict_distribution().

        A key issue is that predict() will not utilize Stan's generate quantities method
        if the model as not fit with HMC-NUTS. Instead, a Python-based rng is used with
        coefficients and intercept derived from the fitted model. This will change in a
        future Stan release.

        Parameters
        ----------
        X : ArrayLike
            Predictor matrix or array of data to use for prediction.
            X can be sparse. It will be converted to CSR format if it is not already.
        return_std : bool, optional, default=False
            If True, return standard deviation of predictions.
        show_console : bool, optional, default=False
            Printing output of default CmdStanPy console during Stan operations.

        Returns
        -------
        NDArray[np.float64]
            Array of predictions of shape (n_samples, 1) made by fitted model.

        NDArray[np.float64]
            Standard deviation of predictive distribution points.

        """
        check_is_fitted(self)

        if X is None:
            raise ValueError(
                f"""This {self.__class__.__name__!r}
             estimator requires X to be passed, but it is None"""
            )

        X_clean, _ = self._validate_data(X=X, ensure_X_2d=True)

        y_predict_mean = self.predict_distribution(
            X_clean,
            show_console=show_console,
        ).mean(axis=0, dtype=np.float64)

        if return_std:
            sigmas_squared_data = (np.dot(X_clean, self.sigma_) * X_clean).sum(  # type: ignore
                axis=1
            )
            y_std = np.sqrt(sigmas_squared_data + (1.0 / self.alpha_))

            return y_predict_mean, y_std
        else:
            return y_predict_mean  # type: ignore

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        show_console: bool = False,
    ) -> float:
        """Computes the coefficient of determination R2 of the prediction,
        like other sklearn estimators.

        Parameters
        ----------
        X : ArrayLike
            Matrix or array of predictors having shape (n_samples, n_predictors)
            that consists of test data.
            X can be sparse. It will be converted to CSR format if it is not already.
        y : ArrayLike
            Array of shape (n_samples,) containing the target values
            corresponding to given test dat X.
        show_console : bool, optional
            Printing output of default CmdStanPy console during Stan operations.

        Returns
        -------
        float
            R2 of the prediction versus the given target values.
                This is the mean accuracy of self.predict(X) with respect to y
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


def prepare_X_data(
    data: Dict[str, Any],
    X: Union[NDArray[np.int64], NDArray[np.float64], sp.csr_matrix],
) -> None:
    """
    Populate data dictionary with the (possibly sparse) X matrix.
    """
    if sp.issparse(X):
        assert isinstance(X, sp.csr_matrix)
        data["X_dense"] = 0
        data["X"] = []
        data["X_data"] = X.data
        data["X_nz"] = X.nnz
        data["X_idxs"] = X.indices + 1
        data["X_indptr"] = X.indptr + 1
    else:
        data["X_dense"] = 1
        data["X"] = X
        data["X_nz"] = 0
        data["X_data"] = []
        data["X_idxs"] = []
        data["X_indptr"] = []
