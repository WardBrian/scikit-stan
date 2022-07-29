"""Abstract classes for different model types that conform to sk-learn style."""

from collections import defaultdict
from inspect import signature
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scikit_stan.utils.validation import _check_y

from .utils import check_array, check_X_y


# NOTE: This estimator class derives class methods from scikit-learn,
# which is distributed under the 3-Clause BSD License.
class CoreEstimator:
    """
    Abstract class for all estimator-type models in this package.
    """

    @classmethod
    def _get_param_names(cls) -> List[str]:
        """Get parameter names for the estimator"""
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []

        init_signature = signature(init)

        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]

        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )

        return sorted([p.name for p in parameters])

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out: Dict[str, Any] = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params: Dict[str, Dict[str, Any]]) -> "CoreEstimator":
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # type: ignore
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    @overload
    def _validate_data(
        self,
        X: ArrayLike,
        y: ArrayLike,
        ensure_X_2d: bool = True,
        allow_X_nd: bool = False,
        dtype: type = np.float64,
    ) -> Tuple[
        NDArray[Union[np.float64, np.int64]],
        NDArray[Union[np.float64, np.int64]],
    ]:
        ...

    @overload
    def _validate_data(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        ensure_X_2d: bool = True,
        allow_X_nd: bool = False,
        dtype: type = np.float64,
    ) -> Tuple[
        NDArray[Union[np.float64, np.int64]],
        Optional[NDArray[Union[np.float64, np.int64]]],
    ]:
        ...

    # custom function adapted from sklearn's validations,
    # which are distributed under the 3-Clause BSD License.
    def _validate_data(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        ensure_X_2d: bool = True,
        allow_X_nd: bool = False,
        dtype: type = np.float64,
    ) -> Tuple[
        Optional[NDArray[Union[np.float64, np.int64]]],
        Optional[NDArray[Union[np.float64, np.int64]]],
    ]:
        """
        Input validation for standard estimators.
        Checks X and y for consistent length, enforces X to be 2D and y 1D. By
        default, X is checked to be non-empty and containing only finite values.
        Standard input checks are also applied to y, such as checking that y
        """
        no_X, no_y = X is None, y is None

        if no_X and no_y:
            raise ValueError("""Validation should be done on X,y or both.""")
        elif not no_X and no_y:
            res_X = check_array(
                X,  # type: ignore
                ensure_2d=ensure_X_2d,
                allow_nd=allow_X_nd,
                dtype=dtype,
            )
            res_y = None
        elif no_X and not no_y:
            res_y = _check_y(y, dtype=dtype)  # type:ignore
            res_X = None
        else:
            res_X, res_y = check_X_y(X, y, dtype=dtype)  # type:ignore

        return res_X, res_y
