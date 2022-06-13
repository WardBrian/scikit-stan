"""Non-vectorized BLR model with sk-learn type fit() API"""

import json
import sys
from pathlib import Path
from typing import Optional

from cmdstanpy import CmdStanModel
from numpy.typing import ArrayLike
import scipy.stats as stats

from sk_stan_regression.modelcore import CoreEstimator

sys.path.insert(0, str(Path(__file__).parent.parent))

# TODO: mover this to ./test/
# from sk_stan_regression.utils.validation import (
#    check_consistent_length,
#    check_is_fitted,
# )


# TODO: should create an abstract class to manage these things
# instead of importing from sklearn.
# what kinds of fucntionality should the abstract classes have?

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
    """
    Vectorized, multidimensional version of the BLR Estimator above.
    Note that the intercept alpha and error scale sigma remain as scalar values
    while beta becomes a vector.

    This should supersede the class above (?) as it is a special case
        -- K = 1 in the above class.


    :param alpha: posterior mean of intercept of the linear regression
    :param alpha_samples: samples generated from the posterior
                            for model intercept
    :param beta: posterior mean of slope of the linear regression
    :param beta_samples: samples generated from the posterior
                            for model slope
    :param sigma: posterior mean of error scale of the linear regression
    :param sigma_samples: samples generated from the posterior
                            for model error scale
    :param algorithm: algorithm that performs an operation on the posterior
    """

    def __init__(
        self,
        algorithm: Optional[str] = "HMC-NUTS",
    ):
        # self.alpha_: Optional[float] = None  # posterior mean of the slope
        # self.alpha_samples_: Optional[ArrayLike] = None  # slope draws
        # self.beta_: Optional[ArrayLike] = None
        # self.beta_samples_: Optional[ArrayLike] = None
        # self.sigma_: Optional[float] = None
        # self.sigma_samples_: Optional[ArrayLike] = None
        #
        # self.Xtrain_ = None
        # self.ytrain_ = None

        self.algorithm = algorithm
        # self.pfunctag: str = posterior_function
        # self.posterior_function: Callable = method_dict[self.pfunctag]

        # self.is_fitted_ = None

    #
    # self.model_ = CmdStanModel(stan_file=BLR_STAN_FILE)

    def __repr__(self) -> str:
        return f"""<BLR_Estimator:
                        alpha={self.alpha}, alpha_samples={self.alpha_samples}, 
                        beta={self.beta}>, beta_samples={self.beta_samples}, 
                        sigma={self.sigma}, sigma_samples={self.sigma_samples}>
                """

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ):
        """
        Fits current vectorized BLR object to the given data,
        with a default set of data.
        This model is considered fit once its alpha, beta,
        and sigma parameters are determined via a regression.

        Where N is the number of data items (rows) and
        K is the number of predictors (columns) in x:

        :param X: NxK predictor matrix
        :param y: Nx1 outcome vector

        :return: self, an object
        """
        try:
            datakval = X.shape[1]
            dat = {"x": X, "y": y, "N": X.shape[0], "K": datakval}

            self.model_ = CmdStanModel(stan_file=BLR_VECTORIZED_STAN_FILE)
        except IndexError:
            datakval = 1
            dat = {"x": X, "y": y, "N": X.shape[0]}

            self.model_ = CmdStanModel(stan_file=BLR_STAN_FILE)

        vb_fit = method_dict[self.algorithm](self.model_, data=dat, show_console=True)

        # TODO: validate inputs...

        # TODO: ensure that data is reshaped appropriately;
        # as long as dimensions are the same, it seems that stan does this automatically?

        stan_vars = vb_fit.stan_variables()
        if self.algorithm == "HMC-NUTS":
            summary_df = vb_fit.summary()
            self.alpha_ = summary_df.at["alpha", "Mean"]

            if datakval == 1:
                self.beta_ = summary_df.at["beta", "Mean"]
            else:
                self.beta_ = np.array([])

                for idx in range(datakval):
                    self.beta_ = np.append(
                        self.beta_, [summary_df.at[f"beta[{idx+1}]", "Mean"]]
                    )

            self.sigma_ = summary_df.at["sigma", "Mean"]

            self.alpha_samples_ = stan_vars["alpha"]
            self.beta_samples_ = stan_vars["beta"]
            self.sigma_samples_ = stan_vars["sigma"]
        else:
            self.alpha_ = stan_vars["alpha"]
            self.beta_ = stan_vars["beta"]
            self.sigma_ = stan_vars["sigma"]

        self.is_fitted_ = True

        return self

    def predict(self, X: ArrayLike, num_iterations: int = 1000, num_chains: int = 4):
        """
        Predict using a fitted model.


        """
        if self.algorithm != "HMC-NUTS":
            return stats.norm.rvs(
                self.alpha + np.dot(self.beta, np.array(X)), self.sigma
            )

        try:
            dat = {
                "N": X.shape[0],
                "K": X.shape[1],
                "X": X,
                "alpha": self.alpha,
                "beta": self.beta,
                "sigma": self.sigma,
            }

            sm = CmdStanModel(stan_file=BLR_NORMAL_V_SAMPLE_STAN_FILE)
        except IndexError:
            dat = {
                "N": len(X),
                "X": X,
                "alpha": self.alpha,
                "beta": self.beta,
                "sigma": self.sigma,
            }

            sm = CmdStanModel(stan_file=BLR_NORMAL_NV_SAMPLE_STAN_FILE)

        samples = sm.sample(data=dat, iter_sampling=num_iterations, chains=num_chains)

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

    import numpy as np

    xdat = np.array(jsondat["x"])
    ydat = np.array(jsondat["y"])

    # blrvec = BLR_Estimator()
    # blrvec.fit(X=xdat, y=ydat)
    # print(blrvec.__repr__())
    # ysim = blrvec.predict(X=xdat)
    # print(ysim)

    # blr2 = BLR_Estimator(algorithm="MLE")
    # blr2.fit(X=xdat, y=ydat)
    # ysim2 = blr2.predict(X=xdat)
    # print(ysim2)

    kby2 = np.column_stack((xdat, xdat))

    blrvec = BLR_Estimator()
    blrvec.fit(kby2, ydat)
    ysim2 = blrvec.predict(X=kby2)
    print(ysim2)

    # import matplotlib.pyplot as plt
    # plt.scatter(xdat, ysim2)
    # plt.show()

    # blrpred = BLR_Estimator()
    # blrpred.fit(X=xdat, y=ydat)
    # ysim = blrpred.predict(X=xdat)
