import sys 

from sklearn.utils.estimator_checks import parametrize_with_checks

sys.path.append("../")

from sk_stan_regression.src import BLR_Estimator


@parametrize_with_checks([BLR_Estimator()])
def test_sklearn_compatible_estimator(estimator, check): 
    check(estimator)
