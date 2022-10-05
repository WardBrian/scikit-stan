import numpy as np
import pytest
from data import _link_mu
from functions import stanfns  # noqa

from scikit_stan.utils.validation import FAMILY_LINKS_MAP


def test_inv_cauchit(stanfns):  # noqa
    x = np.random.random(100)
    ans = np.arctan(x) / np.pi + 0.5
    np.testing.assert_allclose(ans, stanfns.inv_cauchit(x))


ALL_LINKS = {}
for d in FAMILY_LINKS_MAP.values():
    ALL_LINKS.update(d)


@pytest.mark.parametrize("link,id", ALL_LINKS.items())
def test_inv_links(stanfns, link, id):  # noqa
    x = np.random.random(100)
    ans = _link_mu(link, x)
    np.testing.assert_allclose(ans, stanfns.common_invert_link(x, id))
