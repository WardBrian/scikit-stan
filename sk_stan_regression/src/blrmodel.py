"""Non-vectorized BLR model with sk-learn type fit() API"""

from cmdstanpy import CmdStanMLE, CmdStanVB, CmdStanModel, CmdStanMCMC

from typing import Optional, Union, List, Callable

import json

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_consistent_length,
)
from sklearn.utils.multiclass import unique_labels

if __name__ == "__main__":
    BLR_STAN_FILE = (
        "./stanfiles/nvblinreg.stan"  # basic non-vectorized linear regression
    )
    DEFAULT_FAKE_DATA = "../data/fake_data.json"  # simulated data
    BLR_NORMAL_SAMPLE_FILE = "./stanfiles/sample_normal.stan"
else:
    BLR_STAN_FILE = "../sk_stan_regression/src/stanfiles/nvblinreg.stan"  # basic non-vectorized linear regression
    DEFAULT_FAKE_DATA = "../data/fake_data.json"  # simulated data

method_dict = {
    "HMC-NUTS": CmdStanModel.sample,
    "MLE": CmdStanModel.optimize,
    "Variational": CmdStanModel.variational,
}


class BLR_Estimator(BaseEstimator):
    def __init__(self, posterior_function: Optional[str] = "HMC-NUTS",) -> None:
        """
        Initialization of non-vectorized BLR from given data and chosen posterior operation algorithm. TODO needs greater granularity  
        See https://mc-stan.org/docs/2_29/stan-users-guide/linear-regression.html. 
        The model is defined as yn = alpha + beta*xn + sigman, where each variable is defined below.

        :param alpha: posterior mean of intercept of the linear regression
        :param alpha_samples: samples generated from the posterior for model intercept 
        :param beta: posterior mean of slope of the linear regression
        :param beta_samples: samples generated from the posterior for model slope
        :param sigma: posterior mean of error scale of the linear regressoin 
        :param sigma_samples: samples generated from the posterior for model error scale
        :param posterior_func: algorithm that performs an operation on the posterior 
        """
        self._alpha: Optional[float] = None  # posterior mean of the slope
        self._alpha_samples: Optional[List] = None  # slope draws
        self._beta: Optional[float] = None
        self._beta_samples: Optional[List] = None
        self._sigma: Optional[float] = None
        self._sigma_samples: Optional[List] = None

        self.Xtrain = None
        self.ytrain = None

        self.pfunctag: str = posterior_function
        self.posterior_function: Callable = method_dict[self.pfunctag]

        self.model = CmdStanModel(stan_file=BLR_STAN_FILE)

    # NOTE: not really needed since super() gives a representation method
    def __repr__(self) -> str:
        return "<BLR_Estimator: alpha={}, alpha_samples={}, beta={}, beta_samples={}, sigma={}, sigma_samples={}>".format(
            self.alpha,
            self.alpha_samples,
            self.beta,
            self.beta_samples,
            self.sigma,
            self.sigma_samples,
        )

    # NOTE: fit parameters should be restricted to directly data dependent variables
    # TODO: use typing.ArrayLike... some import version issue present atm
    def fit(
        self,
        X: Optional[List] = None,
        y: Optional[List] = None,
        data_path: Optional[str] = DEFAULT_FAKE_DATA,
    ) -> Union[CmdStanMCMC, CmdStanVB, CmdStanMLE]:
        """
        Fits the BLR object to given data, with the default being the fake data set from 6/6. 

        :param X: 
        :param y: 
        :param data_file: (optional) path to data source in the form of rows containing x and y labels of simulated data 

        :return: an object in this construct: Union[CmdStanMCMC, CmdStanVB, CmdStanMLE]. Note that GenGQ requires an MCMC samples 
                 in order to function and is thus not provided in the fit() function; could be included as some chain from sample() -> GQ?
        """
        # NOTE: currently only for MCMC, but others can be supported with other methods by passing another method string in, like
        # in the mapping set up above
        try:
            check_consistent_length(X, y)
        except ValueError:
            return

        # TODO: give possibility to just pass in a json?
        # ensure that the X and y that are passed in are the primary data fields being used

        if X and y:
            vb_fit = self.posterior_function(
                self.model, data={"x": X, "y": y, "N": len(X)}, show_console=True
            )
            self.Xtrain = X
            self.ytrain = y
        else:
            vb_fit = self.posterior_function(
                self.model, data=data_path, show_console=True
            )
            self.Xtrain = json.load(data_path)["x"]
            self.ytrain = json.load(data_path)["y"]


        stan_vars = vb_fit.stan_variables()
        if self.pfunctag in "HMC-NUTS":
            summary_df = vb_fit.summary()
            self._alpha = summary_df.at["alpha", "Mean"]
            self._beta = summary_df.at["beta", "Mean"]
            self._sigma = summary_df.at["sigma", "Mean"]

            self._alpha_samples = stan_vars["alpha"]
            self._beta_samples = stan_vars["beta"]
            self._sigma_samples = stan_vars["sigma"]

            # sk-learn estimators require an is_fitted_ field post-fit()
            self.is_fitted_ = True
        else:
            self._alpha = stan_vars["alpha"]
            self._beta = stan_vars["beta"]
            self._sigma = stan_vars["sigma"]

        return self

    def predict(
            self, 
            X: Optional[List] = None,
            num_iterations: Optional[int] = 1000,
            num_chains: Optional[int] = 4
        ):
        """
        Utilizes a fitted model with previous data to generate additional quantities.

        :param: 
        """
        try:
            check_is_fitted(self, "is_fitted_")
        except NotFittedError:
            # TODO: can perform this by default and keep some store of data from some previous interaction
            print(
                "No MCMC samples generated for this instance, execute .fit() with input data and HMC-NUTS."
            )
            return
        
        # can't really set the default value of X in the 
        # argument list to self.Xtrain - recursion
        if not X: 
            X = self.Xtrain
        
        data = {
                "N": len(X), 
                "X": X, 
                "alpha": self.alpha, 
                "beta": self.beta, 
                "sigma": self.sigma 
                }

        sm = CmdStanModel(stan_file=BLR_NORMAL_SAMPLE_FILE)

        samples = sm.sample(data=data, iter_sampling=num_iterations, chains=num_chains)


        print(samples.stan_variables())




        pass

    @property
    def alpha(self) -> Optional[float]:
        """Posterior mean for regression intercept."""
        return self._alpha

    @property
    def alpha_samples(self) -> Optional[list]:
        """Samples generated from posterior for regression intercept."""
        return self._alpha_samples

    @property
    def beta(self) -> Optional[float]:
        """Posterior mean for regression slope."""
        return self._beta

    @property
    def beta_samples(self) -> Optional[list]:
        """Samples generated from posterior for regression slope."""
        return self._beta_samples

    @property
    def sigma(self) -> Optional[float]:
        """Posterior mean for regression error scale."""
        return self._sigma

    @property
    def sigma_samples(self) -> Optional[list]:
        """Samples generated from posterior for regression error scale."""
        return self._sigma_samples


if __name__ == "__main__":
    with open("../data/fake_data.json") as file:
        jsondat = json.load(file)

    xdat = jsondat["x"]
    ydat = jsondat["y"]

    blrpred = BLR_Estimator()
    blrpred.fit(X=xdat, y=ydat)
    blrpred.predict(X=xdat)

#
    #blrsimdefault = BLR_Estimator()
    #blrsimdefault.fit(X=xdat, y=ydat)
    #print(blrsimdefault.__repr__())
#
    #bsimvi = BLR_Estimator(posterior_function="Variational")
    #bsimvi.fit()
    #print(bsimvi.__repr__())
#
    #bsimmle = BLR_Estimator(posterior_function="MLE")
    #bsimmle.fit(X=xdat, y=ydat)
    #print(bsimmle.__repr__())
#
    #bexception = BLR_Estimator()
    #bexception.predict(
    #    xdat
    #)  # expected failure, might as well start writing a test suite at some point TODO

    #bsimviexception = BLR_Estimator(posterior_function="Variational")
    #bsimviexception.fit()
    #bsimviexception.predict(xdat)  # expected failure
#
    # NOTE: the data generation example is dependent on a previous MCMC object, can't be called on the raw model, encapsulate in predict()
    # bsimgen = BLRsim(posterior_function="Gen")
    # bsimgen.fit()
    # print(bsimgen.__repr__())
