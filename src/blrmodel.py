"""Non-vectorized BLR model with sk-learn type fit() API"""

from cmdstanpy import (
    CmdStanGQ, 
    CmdStanMLE, 
    CmdStanVB, 
    CmdStanModel, 
    CmdStanMCMC
)

from typing import Optional, Union

BLR_STAN_FILE = './stanfiles/nvblinreg.stan' # basic non-vectorized linear regression
DEFAULT_FAKE_DATA = '../data/fake_data.json' # simulated data

# TODO: adding additional methods to the fit (?) can be done with a mapping like this and something similar for the output 
method_dict = { 
    "MCMC"  : CmdStanModel.sample, 
    "MLE"   : CmdStanModel.optimize, 
    "GQ"    : CmdStanModel.generate_quantities,
    "VB"    : CmdStanModel.variational
}


class BLRsim: 
    def __init__(self) -> None:  
        """
        Initialization of non-vectorized BLR from given data and chosen sampling algorithm. 

        """

        self._alpha         = None      #posterior mean of the slope
        self._alpha_samples = None      #slope draws
        self._beta          = None
        self._beta_samples  = None
        self._sigma         = None
        self._sigma_samples = None

        self.model = CmdStanModel(stan_file=BLR_STAN_FILE)

    def __repr__(self) -> str: 
        return "<BLRSim: alpha={}, alpha_samples={}, beta={}, beta_samples={}, sigma={}, sigma_samples={}>".format(
                    self.alpha, self.alpha_samples, self.beta, self.beta_samples, self.sigma, self.sigma_samples
                )

    def fit(
        self, 
        data_path:Optional[str] = DEFAULT_FAKE_DATA,
        method:Optional[str] = None
        ) -> Union[CmdStanMCMC, CmdStanVB, CmdStanMLE, CmdStanGQ]: 
        """
        Fits the BLR object to given data, with the default being the fake data set from 6/6. 

        :param data_file: path to data source in the form of rows containing x and y labels of simulated data 
        :param alg: specified posterior sampling or approximating algorithm 
        :param method: (not implemented), see comment in email 

        :return: currently only CmdStanMCMC, but we could offer other sampling/fitting methods and the corresponding CmdStan objects
                    would be returned
        """
        #NOTE: currently only for MCMC, but others can be supported with other methods by passing another method string in, like
        # in the mapping set up above 
        vb_fit = self.model.sample(data=data_path, show_console=True)

        # set up approximated parameters 
        summary_df = vb_fit.summary()
        self._alpha = summary_df.at['alpha', 'Mean']
        self._beta = summary_df.at['beta', 'Mean']
        self._sigma = summary_df.at['sigma', 'Mean']

        stan_vars = vb_fit.stan_variables()
        self._alpha_samples = stan_vars['alpha']
        self._beta_samples  = stan_vars['beta']
        self._sigma_samples = stan_vars['sigma']

        # for example of the more general solution with using any of the other sampling/fitting methods 
        #vb_fit = self.model.variational(data=data_file, show_console=True, require_converged=False)
        #print(vb_fit.stan_variables())
        #print(vb_fit.column_names)

        return vb_fit


    @property
    def alpha(self) -> Optional[float]: 
        return self._alpha


    @property 
    def alpha_samples(self) -> Optional[list]:
        return self._alpha_samples


    @property 
    def beta(self) -> Optional[float]:
        return self._beta


    @property 
    def beta_samples(self) -> Optional[list]:
        return self._beta_samples


    @property 
    def sigma(self) -> Optional[float]:
        return self._sigma


    @property 
    def sigma_samples(self) -> Optional[list]: 
        return self._sigma_samples 


if __name__ == '__main__': 
    blrsim = BLRsim()
    blrsim.fit()
    print(blrsim.__repr__())
