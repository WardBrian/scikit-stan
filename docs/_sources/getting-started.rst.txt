.. py:currentmodule:: scikit_stan

Getting Started
===============

SciKit Stan is a Python package of generalized linear models in the Stan with the familiar sk-learn interface. With scikit_stan, you can:

+ Compile a GLM from a highly-customizeable Stan model with control over family, link, priors, and scaling,

+ Perform sk-learn style model fitting with :meth:`~GLM.fit` to perform regressions based on an inference conditioned on your data. This can be done using one of Stan's inference algorithsm:

    + `HMC-NUTS <https://mc-stan.org/docs/reference-manual/hmc.html>`_ for exact Bayesian estimation,

    + `ADVI <https://mc-stan.org/docs/reference-manual/vi-algorithms.html>`_ for approximate Bayesian estimation,

    + `L-BFGS <https://mc-stan.org/docs/reference-manual/optimization-algorithms.html>`_ for MAP estimation,

+ Generate posterior predictive samples from the fitted model with :meth:`~GLM.predict`,

+ Quantify prediction quality with R-squared metric via :meth:`~GLM.score`

    + This enables hyperparameter searching with, for example, sk-learn's :class:`~sklearn.model_selection.GridSearchCV`

`scikit_stan` wraps the `CmdStanPy <https://mc-stan.org/cmdstanpy/index.html>`_
Python interface into Stan and provides a base for developing probabilistic models on top of Stan in Python.

This package is designed to provide a sk-learn type interface to models written in Stan.
Concretely, the `sk-learn methods system <https://scikit-learn.org/stable/developers/develop.html>`_
is the same here, with :meth:`~GLM.fit`, :meth:`~GLM.predict` and :meth:`~GLM.score` methods, among others,
having the same purpose in their respective contexts. 
