# scikit-stan

Scikit-Stan is a package of [Stan](mc-stan.org) models wrapped in a
[Scikit-Learn](https://scikit-learn.org/stable/) style interface.

This package is currently under active development and should be treated as beta software.

Documentation is available at https://brianward.dev/scikit-stan/

## Installation

Pre-compiled wheels for the package are available for MacOS, Windows, and Linux systems via `pip install scikit_stan`.

Source installation requires a working installation of [CmdStan](https://mc-stan.org/docs/cmdstan-guide/index.html).

## Basic usage

```python
from scikit_stan import GLM

m = GLM(family='gamma') # Gamma family distribution with canonical inverse link
m.fit(X, y) # runs HMC-NUTS
predictions = m.predict(X) # generates new predictions from fitted model
score = m.score(X, y) # computes the R2 score of the fitted model on the data X and observations y
```

## Attribution

This package is licensed under the BSD 3-clause license.

It is inspired by existing packages in the Stan ecosystem like
[rstanarm](https://github.com/stan-dev/rstanarm).

This package was initially developed at the Simons Foundation by Alexey Izmailov during
a summer 2022 internship under the mentorship of Brian Ward.
