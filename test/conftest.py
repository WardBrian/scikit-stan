"""The global configuration for the test suite"""
# import atexit
# import os
# import shutil
# import subprocess
# from typing import Generator
#
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# NOTE: adapted from cmdstanpy:
# https://github.com/stan-dev/cmdstanpy/blob/3777785e88b41d1781a797301b37e9000ebc1197/test/conftest.py#L16-L26
# TODO: what should be the resturn type on this???
# @pytest.fixture(scope="session", autouse=True)
# def cleanup_test_files() -> Generator:
#
#    import cmdstanpy  # type: ignore
#
#    # see https://github.com/pytest-dev/pytest/issues/5502
#    atexit.unregister(cmdstanpy._cleanup_tmpdir)
#
#    yield
#
#    shutil.rmtree(cmdstanpy._TMPDIR, ignore_errors=True)
#
