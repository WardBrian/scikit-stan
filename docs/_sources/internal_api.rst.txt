.. py:currentmodule:: scikit_stan

Internal API Reference
======================

Here is a documentation of methods internal to the package, which are subject to considerable change between releases. No promises of backwards compatibility are made with these methods.

The package consists of a single general class for estimators, which is modelled after
sk-learn's Estimator class.

*******
Classes
*******

CoreEstimator
--------------

.. autoclass:: scikit_stan.modelcore.CoreEstimator
    :members:


******************
Validation Methods
******************

.. currentmodule:: scikit_stan.utils.validation

Validating Family-Link Choice
-----------------------------

.. autofunction:: validate_family


Note that the package has a consistent internal numbering scheme for families and links alike. Specifically, since Stan does not support strings, families, links, and priors
have internal numeric representations.

Families are mapped as follows:
    * "gaussian": 0,
    * "gamma": 1,
    * "inverse-gaussian": 2,
    * "poisson": 3,
    * "binomial": 4,
    * "negative-binomial": 5,

Link functions are mapped as follows:
    * identity - 0
    * log - 1
    * inverse - 2
    * sqrt - 3
    * inverse-square - 4
    * logit - 5
    * probit - 6
    * cloglog - 7
    * cauchit - 8

Validating Input Data
----------------------

.. autofunction:: check_array

.. autofunction:: check_is_fitted

.. autofunction:: check_consistent_length

.. autofunction:: _num_samples


Validating Priors
-----------------

.. autofunction:: validate_prior

.. autofunction:: validate_aux_prior


