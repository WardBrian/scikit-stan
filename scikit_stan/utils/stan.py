"""
Module for handling local CmdStan installation and pre-compiled models.
"""
import shutil
import warnings
from pathlib import Path

from cmdstanpy import CmdStanModel, set_cmdstan_path

STAN_FILES_FOLDER = Path(__file__).parent.parent / "stan_files"
CMDSTAN_VERSION = "2.30.1"


def init_local_cmdstan() -> None:
    local_cmdstan = STAN_FILES_FOLDER / f"cmdstan-{CMDSTAN_VERSION}"
    if local_cmdstan.exists():
        set_cmdstan_path(str(local_cmdstan.resolve()))


def load_stan_model(name: str) -> CmdStanModel:
    try:
        model = CmdStanModel(
            exe_file=STAN_FILES_FOLDER / f"{name}.exe",
            stan_file=STAN_FILES_FOLDER / f"{name}.stan",
            compile=False,
        )
    except ValueError:
        warnings.warn(f"Failed to load pre-built model '{name}.exe', compiling")
        model = CmdStanModel(
            stan_file=STAN_FILES_FOLDER / f"{name}.stan",
            stanc_options={"O1": True},
        )
        shutil.copy(
            model.exe_file,  # type: ignore
            STAN_FILES_FOLDER / f"{name}.exe",
        )

    return model
