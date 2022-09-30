import numpy as np
from functions import stanfns  # noqa


def test_inv_cauchit(stanfns):  # noqa
    x = np.random.random(100)
    ans = np.arctan(x) / np.pi + 0.5
    np.testing.assert_allclose(ans, stanfns.inv_cauchit(x))
