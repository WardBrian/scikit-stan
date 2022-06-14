import json
import sys
from pathlib import Path
import numpy as np 

from cmdstanpy import CmdStanModel  # type: ignore

BLR_FOLDER = Path(__file__).parent
DEFAULT_FAKE_DATA = BLR_FOLDER.parent / "data" / "fake_data.json"

if __name__ == "__main__":  
    with open(DEFAULT_FAKE_DATA) as file:
        jsondat = json.load(file)

    xdat = np.array(jsondat["x"])
    ydat = np.array(jsondat["y"])

    model= CmdStanModel(stan_file=BLR_FOLDER / "blinregvectorized.stan")

    dat = {"x": xdat[:,None], "y": ydat, "N": len(xdat), "K": 1}

    model.sample(data=dat, show_console=True)