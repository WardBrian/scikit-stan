from typing import Dict


# inspired by sklearn's utilities for parameter validation 
def validate_param_constraints(param_constraints: Dict, params: Dict, caller: str):
    if params.keys() != param_constraints.keys():
        raise ValueError(
            f"The parameter constraints {list(param_constraints.keys())} do not "
            f"match the parameters to validate {list(params.keys())}."
        )
