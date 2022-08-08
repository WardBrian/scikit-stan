"""Model of Wiener First Passage Time for Threshold Regression with sk-learn type API."""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import scipy.stats as stats
from cmdstanpy import CmdStanModel, set_cmdstan_path
from numpy.typing import ArrayLike, NDArray

from scikit_stan.modelcore import CoreEstimator
from scikit_stan.utils.validation import (
    method_dict, 
    check_array,
    check_is_fitted,
    validate_aux_prior,
    validate_family,
    validate_prior,
)

STAN_FILES_FOLDER = Path(__file__).parent.parent / "stan_files"
CMDSTAN_VERSION = "2.30.1"


class WienerFPT(CoreEstimator): 
    r"""
    Regression model for first-hitting-time regression based on the Wiener process. 
    
    For a Wiener process with positive initial value and negative mean, the time required for
    the process to reach the zero level for the first time is given by the inverse Gaussian distribution. 



    The model returns the first passage time of the accumulation process over the upper
    boundary only. In order to obtain the result of the lower boundary, use the parameterization with 

    .. math:: \alpha \mapsto \alpha, \tau \mapsto \tau, \beta \mapsto 1 - \beta, \delta \mapsto - \delta

    where the mapping is from the default upper boundary parameterization to the lower boundary parameterization.

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

        Customizing these fields occurs as a passed dictionary, which is validated
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



    seed : Optional[int], optional
        Seed for random number generator. Must be an integer between 0 an 2^32-1.
        If this is left unspecified, then a random seed will be used for all chains
        via :class:`numpy.random.RandomState`.
        Specifying this field will yield the same result for multiple uses if
        all other parameters are held the same.

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
        prior_separation: Optional[Dict[str, Any]] = None, 
        prior_nd_time: Optional[Dict[str, Any]] = None, 
        prior_drift_rate: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        autoscale: bool = False,
    ):
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params

        self.prior_separation = prior_separation
        self.prior_nd_time = prior_nd_time
        self.prior_drift_rate = prior_drift_rate

        self.seed = seed
        self.autoscale = autoscale


    def fit(self, 
            X: ArrayLike,
            y: ArrayLike,
            bias: float = 0.5, 
            show_console: bool = False,
        ) -> "CoreEstimator":
        """
        Fits the Wiener process model. 
        This model is considered fit once its alpha, tau, and delta 
        parameters are determined via a regression. 

        The a-priori bias is assumed to be 0.5 by default and can be changed at model fitting time. 

        Note that by construction of the model, values of the data cannot be negative or 
        be smaller than the admissible range of values for the non-decision time tau.  

        Parameters
        ----------
        X : ArrayLike
            NxK matrix of predictors, where K >= 0. If K = 1,
            then X is automatically reshaped to being 2D and raises a warning.
        y : ArrayLike
            Nx1 outcome vector where each row is a response corresponding to
            the same row of predictors in X.
        show_console : bool, optional
            Printing output of default CmdStanPy console during Stan operations.

        Returns
        -------
        CoreEstimator
            Abstract class for all estimator-type models in this package.
            The Wiener First Passage Time class is a subclass of CoreEstimator.
        """
        if y is None:
            raise ValueError(
                """This Generalized Linear Model requires a response variable y, but it is None."""
            )

        if y is None:
            raise ValueError(
                """Wiener First Passage Time Model requires a response variable y, but it is None."""
            )

        X_clean, y_clean = self._validate_data(
            X=X,
            y=y,
            ensure_X_2d=True,
            dtype=np.float64 if self.is_cont_dat_ else np.int64,
        )


        if self.algorithm not in method_dict.keys():
            raise ValueError(
                f"""Current Linear Regression created with algorithm
                {self.algorithm!r}, which is not one of the supported
                methods. Try with one of the following: (sample, optimize, variational)."""
            )

        
        

        return self 


    def predict(self, X: ArrayLike) -> NDArray:
        """
        Predict the response.
        """
        pass


    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Score the model.
        """
        pass
