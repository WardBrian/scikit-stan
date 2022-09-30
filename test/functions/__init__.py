import os
import sys
from glob import glob

import pytest
from functions.expose import expose

from scikit_stan.utils.stan import STAN_FILES_FOLDER

try:
    import pybind11  # noqa
except ImportError:
    pytest.skip("PyBind11 not installed!", allow_module_level=True)

if sys.platform.startswith("win"):
    pytest.skip("Exposed function tests cannot run on Windows", allow_module_level=True)


@pytest.fixture(scope="session")
def stanfns():
    module = expose(str(STAN_FILES_FOLDER / "tests.stanfunctions"))
    yield module
    for file in glob(str(STAN_FILES_FOLDER / "tests.*")):
        if not file.endswith(".stanfunctions"):
            os.remove(file)
