import pandas as pd
import numpy as np

# ATTRIBUTION: McCullagh & Nelder (1989), chapter 8.4.2 p 301-302
bcdata_dict = {
    'u': np.array([5,10,15,20,30,40,60,80,100]),
    'lot1': np.array([118,58,42,35,27,25,21,19,18]), 
    'lot2': np.array([69,35,26,21,18,16,13,12,12])
}

bcdata_pandas = pd.DataFrame.from_dict(bcdata_dict)