"""Custom exception types, as in sklearn."""


try:
    from sklearn.exceptions import NotFittedError
except ImportError:

    class NotFittedError(ValueError, AttributeError):  # type:ignore
        """Exception class to raise if estimator is used before fitting.
        This class inherits from both ValueError and AttributeError to help with
        exception handling and backward compatibility.
        """


__all__ = ["NotFittedError"]
