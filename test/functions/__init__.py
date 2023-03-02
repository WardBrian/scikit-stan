import os
import sys
from glob import glob
from types import ModuleType
from typing import Generator

import pytest

from scikit_stan.utils.stan import STAN_FILES_FOLDER

if sys.platform.startswith("win"):
    pytest.skip("Exposed function tests cannot run on Windows", allow_module_level=True)

try:
    from pybind_stan_fns import expose
except ImportError:
    pytest.skip("pybind_stan_fns not installed!", allow_module_level=True)


@pytest.fixture(scope="session")
def stanfns() -> Generator[ModuleType, None, None]:
    module = expose(str(STAN_FILES_FOLDER / "tests.stanfunctions"))
    yield module
    for file in glob(str(STAN_FILES_FOLDER / "tests.*")):
        if not file.endswith(".stanfunctions"):
            os.remove(file)
