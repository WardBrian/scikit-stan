"""Non-vectorized BLR model with sk-learn type fit() API"""

from cmdstanpy import (
    CmdStanMLE, 
    CmdStanVB, 
    CmdStanModel, 
    CmdStanMCMC
)

from typing import Optional, Union, List, Callable

BLR_STAN_FILE = './stanfiles/nvblinreg.stan' # basic non-vectorized linear regression
DEFAULT_FAKE_DATA = '../data/fake_data.json' # simulated data

method_dict = { 
    "HMC-NUTS"          : CmdStanModel.sample, 
    "MLE"               : CmdStanModel.optimize, 
    "Variational"       : CmdStanModel.variational
}


class BLRsim: 
    def __init__(self, 
        posterior_function:Optional[str]= "HMC-NUTS",
    ) -> None:  
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
        self._alpha         : Optional[float] = None      #posterior mean of the slope
        self._alpha_samples : Optional[List]  = None      #slope draws
        self._beta          : Optional[float] = None
        self._beta_samples  : Optional[List]  = None
        self._sigma         : Optional[float] = None
        self._sigma_samples : Optional[List]  = None

        self.pfunctag       : str             = posterior_function
        self.posterior_function  : Callable        = method_dict[self.pfunctag] 

        self.model = CmdStanModel(stan_file=BLR_STAN_FILE)


    def __repr__(self) -> str: 
        return "<BLRSim: alpha={}, alpha_samples={}, beta={}, beta_samples={}, sigma={}, sigma_samples={}>".format(
                    self.alpha, self.alpha_samples, self.beta, self.beta_samples, self.sigma, self.sigma_samples)


    # NOTE: fit parameters should be restricted to directly data dependent variables
    def fit(
            self, 
            data_path:Optional[str] = DEFAULT_FAKE_DATA,
        ) -> Union[CmdStanMCMC, CmdStanVB, CmdStanMLE]: 
        """
        Fits the BLR object to given data, with the default being the fake data set from 6/6. 

        :param data_file: path to data source in the form of rows containing x and y labels of simulated data 
        :param alg: specified posterior sampling or approximating algorithm 
        :param method: (not implemented), see comment in email 

        :return: an object in this construct: Union[CmdStanMCMC, CmdStanVB, CmdStanMLE]. Note that GenGQ requires an MCMC samples 
                 in order to function and is thus not provided in the fit() function; could be included as some chain from sample() -> GQ?
        """
        #NOTE: currently only for MCMC, but others can be supported with other methods by passing another method string in, like
        # in the mapping set up above 
        vb_fit = self.posterior_function(self.model, data=data_path, show_console=True)

        stan_vars = vb_fit.stan_variables()
        if self.pfunctag in "HMC-NUTS": 
            summary_df = vb_fit.summary()
            self._alpha = summary_df.at['alpha', 'Mean']
            self._beta = summary_df.at['beta', 'Mean']
            self._sigma = summary_df.at['sigma', 'Mean']

            self._alpha_samples = stan_vars['alpha']
            self._beta_samples  = stan_vars['beta']
            self._sigma_samples = stan_vars['sigma']
        else: 
            self._alpha = stan_vars['alpha']
            self._beta  = stan_vars['beta']
            self._sigma = stan_vars['sigma']

        return vb_fit


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


if __name__ == '__main__': 
    blrsimdefault = BLRsim()
    blrsimdefault.fit()
    print(blrsimdefault.__repr__())

    bsimvi = BLRsim(posterior_function="Variational")
    bsimvi.fit()
    print(bsimvi.__repr__())

    bsimmle = BLRsim(posterior_function="MLE")
    bsimmle.fit()
    print(bsimmle.__repr__())

    # NOTE: the data generation example is dependent on a previous MCMC object, can't be called on the raw model, encapsulate in predict() 
    #bsimgen = BLRsim(posterior_function="Gen")
    #bsimgen.fit() 
    #print(bsimgen.__repr__())