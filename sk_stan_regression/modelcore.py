"""Abstract classes for different model types, conforming to sk-learn style."""

from collections import defaultdict
from inspect import signature
from typing import TypeVar, Optional
from numpy.typing import ArrayLike

from .utils import check_array, check_X_y


# TODO: why does this exist and why doesn't mypy like it?
# from typing_extensions import Self
# TODO: how to properly type annotate methods that return self?
CE = TypeVar("CE", bound="CoreEstimator")


# NOTE: these are the same as sk-learn's three methods
class CoreEstimator:
    """
    Abstract class for all estimator-type models in this package.
    """

    @classmethod
    def _get_param_names(cls) -> list:
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
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
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True) -> dict:
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
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params) -> CE:
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
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
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

        return

    # custom function adapted from sklearn's validations
    def _validate_data(
        self,
        X="no-validation",
        y="no-validation",
        ensure_X_2d: Optional[bool] = True,
        allow_X_nd: Optional[bool] = False,
        allow_y_multi_output: Optional[bool] = False,
        ensure_min_features: Optional[int] = 1,
    ):
        """
        Input validation for standard estimators.
        Checks X and y for consistent length, enforces X to be 2D and y 1D. By
        default, X is checked to be non-empty and containing only finite values.
        Standard input checks are also applied to y, such as checking that y
        does not have np.nan or np.inf targets. For multi-label y, set
        multi_output=True to allow 2D
        """
        no_val_X = isinstance(X, str) and X == "no_validation"
        no_val_y = y is None or isinstance(y, str) and y == "no_validation"
        print(no_val_X, no_val_y)
        res_X = X
        res_y = y

        if no_val_X and no_val_y:
            raise ValueError("""Validation should be done on X,y or both.""")
        elif not no_val_X and no_val_y:
            res_X = check_array(
                X,
                ensure_2d=ensure_X_2d,
                allow_nd=allow_X_nd,
                ensure_min_features=ensure_min_features,
            )
        elif no_val_X and not no_val_y:
            pass
        else:
            # TODO: add separate validation of X and y? !!!!!
            res_X, res_y = check_X_y(X, y)

        return res_X, res_y
