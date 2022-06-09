"""Non-vectorized BLR model with sk-learn type fit() API"""

import json
import sys
from pathlib import Path

from typing import Optional, Union
from numpy.typing import ArrayLike

from cmdstanpy import CmdStanMCMC, CmdStanMLE, CmdStanModel, CmdStanVB

from sk_stan_regression.modelcore import CoreEstimator

sys.path.insert(0, str(Path(__file__).parent.parent))

# TODO: mover this to ./test/
#from sk_stan_regression.utils.validation import (
#    check_consistent_length,
#    check_is_fitted,
#)


# TODO: should create an abstract class to manage these things instead of importing from sklearn. what kinds of fucntionality should the abstract classes have?

BLR_FOLDER = Path(__file__).parent
BLR_STAN_FILE = BLR_FOLDER / "nvblinreg.stan"
BLR_VECTORIZED_STAN_FILE = BLR_FOLDER / "blinregvectorized.stan"
BLR_NORMAL_NV_SAMPLE_STAN_FILE = BLR_FOLDER / "sample_normal_nv.stan"
BLR_NORMAL_V_SAMPLE_STAN_FILE = BLR_FOLDER / "sample_normal_v.stan"
DEFAULT_FAKE_DATA = BLR_FOLDER.parent / "data" / "fake_data.json"


method_dict = {
    "HMC-NUTS": CmdStanModel.sample,
    "MLE": CmdStanModel.optimize,
    "Variational": CmdStanModel.variational,
}


class BLR_Estimator(CoreEstimator):
    def __init__(
        self,
        posterior_function: Optional[str] = "HMC-NUTS",
    ) -> None:
        """
        Initialization of non-vectorized BLR from given data and chosen posterior operation algorithm. TODO needs greater granularity
        See https://mc-stan.org/docs/2_29/stan-users-guide/linear-regression.html.
        The model is defined as yn = alpha + beta*xn + sigman, where each variable is defined below.

        :param alpha: posterior mean of intercept of the linear regression
        :param alpha_samples: samples generated from the posterior for model intercept
        :param beta: posterior mean of slope of the linear regression
        :param beta_samples: samples generated from the posterior for model slope
        :param sigma: posterior mean of error scale of the linear regression
        :param sigma_samples: samples generated from the posterior for model error scale
        :param posterior_func: algorithm that performs an operation on the posterior
        """
        # NOTE: in accordance with sk-learn rules, an Estimator subclass should not set any attribute aside from parameters in the constructor. 
        #self.alpha_: Optional[float] = None  # posterior mean of the slope
        #self.alpha_samples_: Optional[ArrayLike] = None  # slope draws
        #self.beta_: Optional[float] = None
        #self.beta_samples_: Optional[ArrayLike] = None
        #self.sigma_: Optional[float] = None
        #self.sigma_samples_: Optional[ArrayLike] = None
#
        #self.Xtrain_ = None
        #self.ytrain_ = None

        self.posterior_function = posterior_function
        #self.pfunctag: str = posterior_function
        #self.posterior_function_callable_ : Callable = method_dict[self.posterior_function]


    # NOTE: not really needed since super() gives a representation method
    def __repr__(self) -> str:
        return "<BLR_Estimator: alpha={}, alpha_samples={}, beta={}, beta_samples={}, sigma={}, sigma_samples={}>".format(
            self.alpha_,
            self.alpha_samples_,
            self.beta_,
            self.beta_samples_,
            self.sigma_,
            self.sigma_samples_,
        )

    # NOTE: fit parameters should be restricted to directly data dependent variables
    def fit(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
    ) -> Union[CmdStanMCMC, CmdStanVB, CmdStanMLE]:
        """
        Fits the BLR object to given data, with the default being the fake data set from 6/6. This model is considered fit once its alpha, beta, and sigma parameters are determined via a regression on some data.

        :param X:
        :param y:

        :return: an object in this construct: Union[CmdStanMCMC, CmdStanVB, CmdStanMLE]. Note that GenGQ requires an MCMC samples
                 in order to function and is thus not provided in the fit() function; could be included as some chain from sample() -> GQ?
        """
        self.model_ = CmdStanModel(stan_file=BLR_STAN_FILE)
        # NOTE: currently only for MCMC, but others can be supported with other methods by passing another method string in, like
        # in the mapping set up above
        #try:
        #    check_consistent_length(X, y)
        #except ValueError:
        #    return

        # ensure that the X and y that are passed in are the primary data fields being used

        # TODO: this should be a check
        if len(X) and len(y):
            vb_fit = method_dict[self.posterior_function](
                self.model_,
                data={"x": X, "y": y, "N": len(X)}, 
            )

            self.Xtrain_ = X
            self.ytrain_ = y

        stan_vars = vb_fit.stan_variables()
        if self.posterior_function in "HMC-NUTS":
            summary_df = vb_fit.summary()
            self.alpha_ = summary_df.at["alpha", "Mean"]
            self.beta_ = summary_df.at["beta", "Mean"]
            self.sigma_ = summary_df.at["sigma", "Mean"]

            self.alpha_samples_ = stan_vars["alpha"]
            self.beta_samples_ = stan_vars["beta"]
            self.sigma_samples_ = stan_vars["sigma"]

        else:
            self.alpha_ = stan_vars["alpha"]
            self.beta_ = stan_vars["beta"]
            self.sigma_ = stan_vars["sigma"]
            # estimators require an is_fitted_ field post-fit

        self.is_fitted_ = True

        return self

    def predict(
        self,
        X: Optional[ArrayLike] = None,
        num_iterations: Optional[int] = 1000,
        num_chains: Optional[int] = 4,
    ):
        """
        Utilizes a fitted model with previous data to generate additional quantities. The default behavior is to use the data that was previously used to train the model.

        :param X:
        """
        # TODO
        # try:
        #    check_is_fitted(self, "is_fitted_")

        # except NotFittedError:
        #    return

        if not len(X):
            # this defines default behavior for predict();
            # if no data is passed, then just generate
            # additional data from the data used to fit
            X = self.Xtrain_

        data = {
            "N": len(X),
            "X": X,
            "alpha": self.alpha_,
            "beta": self.beta_,
            "sigma": self.sigma_,
        }

        sm = CmdStanModel(stan_file=BLR_NORMAL_NV_SAMPLE_STAN_FILE)

        samples = sm.sample(
            data=data, iter_sampling=num_iterations, chains=num_chains
        )

        return samples.stan_variables()

    @property
    def alpha(self) -> Optional[float]:
        """Posterior mean for regression intercept."""
        return self.alpha_

    @property
    def alpha_samples(self) -> Optional[ArrayLike]:
        """Samples generated from posterior for regression intercept."""
        return self.alpha_samples_

    @property
    def beta(self) -> Optional[float]:
        """Posterior mean for regression slope."""
        return self.beta_

    @property
    def beta_samples(self) -> Optional[ArrayLike]:
        """Samples generated from posterior for regression slope."""
        return self.beta_samples_

    @property
    def sigma(self) -> Optional[float]:
        """Posterior mean for regression error scale."""
        return self.sigma_

    @property
    def sigma_samples(self) -> Optional[ArrayLike]:
        """Samples generated from posterior for regression error scale."""
        return self.sigma_samples_


class BLR_Estimator_V:
    """
    Vectorized, multidimensional version of the BLR Estimator above. Note that the intercept alpha and error scale sigma remain as scalar values while beta becomes a vector.

    This should supersede the class above (?) as it is a special case -- K = 1 in the above class.


    """

    def __init__(self, posterior_function):
        #self.alpha_: Optional[float] = None  # posterior mean of the slope
        #self.alpha_samples_: Optional[ArrayLike] = None  # slope draws
        #self.beta_: Optional[ArrayLike] = None
        #self.beta_samples_: Optional[ArrayLike] = None
        #self.sigma_: Optional[float] = None
        #self.sigma_samples_: Optional[ArrayLike] = None
#
        #self.Xtrain_ = None
        #self.ytrain_ = None

        self.posterior_function = posterior_function
        #self.pfunctag: str = posterior_function
        #self.posterior_function: Callable = method_dict[self.pfunctag]

        #self.is_fitted_ = None
#
        #self.model_ = CmdStanModel(stan_file=BLR_STAN_FILE)

    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike,
    ):
        """
        Fits current vectorized BLR object to the given data, with a default set of data. This model is considered fit once its alpha, beta, and sigma parameters are determined via a regression. 

        Where N is the number of data items (rows) and K is the number of predictors (columns) in x: 
        :param X: NxK predictor matrix
        :param y: Nx1 outcome vector

        :return: self, an object
        """
        self.model_ = CmdStanModel(stan_file=BLR_VECTORIZED_STAN_FILE)

        # TODO: validate inputs... 

        # TODO: ensure that data is reshaped appropriately; as long as dimensions are the same, it seems that stan does this automatically? 
        
        self.Xtrain_ = X 
        self.yTrain_ = y 
        
        vb_fit  = method_dict[self.posterior_function](data={"x": X, "y": y, "N": X.shape[0], "K": X.shape[1]}, show_console=True)

        stan_vars = vb_fit.stan_variables() 
        if self.posterior_function == "HMC-NUTS":  
            summary_df = vb_fit.summary() 
            self.alpha_ = summary_df.at["alpha", "Mean"]
            self.beta_ = summary_df.at["beta", "Mean"]
            self.sigma_ = summary_df.at["sigma", "Mean"]

            self.alpha_samples_ = stan_vars["alpha"]
            self.beta_samples_ = stan_vars["beta"]
            self.sigma_samples_ = stan_vars["sigma"]
        else:
            self.alpha_ = stan_vars["alpha"]
            self.beta_ = stan_vars["beta"]
            self.sigma_ = stan_vars["sigma"]
            # estimators require an is_fitted_ field post-fit

        self.is_fitted_ = True

        return self

    def predict(self, X):
        """
        Predict using a fitted model.


        """

        # TODO: better validation on input dimensions 
        if not len(X): 
            X = self.Xtrain_ 

        data = { 
            "N": X.shape[0], 
            "K": X.shape[1],
            "x": X, 
            "alpha": self.alpha, 
            "beta": self.beta, 
            "sigma": self.sigma
        }

        sm = CmdStanModel(stan_file=BLR_NORMAL_V_SAMPLE_STAN_FILE)
        samples = sm.sample(
            data=data,
        )

        return samples.stan_variables()

    @property
    def alpha(self) -> Optional[float]:
        """Posterior mean for regression intercept."""
        return self.alpha_

    @property
    def alpha_samples(self) -> Optional[ArrayLike]:
        """Samples generated from posterior for regression intercept."""
        return self.alpha_samples_

    @property
    def beta(self) -> Optional[ArrayLike]:
        """Posterior mean for regression slope."""
        return self.beta_

    @property
    def beta_samples(self) -> Optional[ArrayLike]:
        """Samples generated from posterior for regression slope."""
        return self.beta_samples_

    @property
    def sigma(self) -> Optional[float]:
        """Posterior mean for regression error scale."""
        return self.sigma_

    @property
    def sigma_samples(self) -> Optional[ArrayLike]:
        """Samples generated from posterior for regression error scale."""
        return self.sigma_samples_


if __name__ == "__main__":
    print(DEFAULT_FAKE_DATA)
    with open(DEFAULT_FAKE_DATA) as file:
        jsondat = json.load(file)

    xdat = jsondat["x"]
    ydat = jsondat["y"]

    blrpred = BLR_Estimator()
    blrpred.fit(X=xdat, y=ydat)
    ysim = blrpred.predict(X=xdat)


#
# blrsimdefault = BLR_Estimator()
# blrsimdefault.fit(X=xdat, y=ydat)
# print(blrsimdefault.__repr__())
#
# bsimvi = BLR_Estimator(posterior_function="Variational")
# bsimvi.fit()
# print(bsimvi.__repr__())
#
# bsimmle = BLR_Estimator(posterior_function="MLE")
# bsimmle.fit(X=xdat, y=ydat)
# print(bsimmle.__repr__())
#
# bexception = BLR_Estimator()
# bexception.predict(
#    xdat
# )  # expected failure, might as well start writing a test suite at some point TODO

# bsimviexception = BLR_Estimator(posterior_function="Variational")
# bsimviexception.fit()
# bsimviexception.predict(xdat)  # expected failure
