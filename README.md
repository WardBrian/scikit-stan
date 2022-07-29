# scikit-stan

Scikit-Stan is a package of [Stan](mc-stan.org) models wrapped in a
[Scikit-Learn](https://scikit-learn.org/stable/) style interface.


## Installation

Pre-compiled wheels for the package are available for MacOS, Windows, and Linux systems via `pip`.

Source installation requires a working installation of [CmdStan](https://mc-stan.org/docs/cmdstan-guide/index.html).

## Basic usage

```python
from scikit_learn import GLM

m = GLM(familiy='gamma')
m.fit(X, y) # runs HMC-NUTs
m.predict(X) # generates new predictions from fitted model
```

## Attribution

This package is licensed under the BSD 3-clause license.

It is inspired by existing packages in the Stan ecosystem like
[rstanarm](https://github.com/stan-dev/rstanarm).

This package was initially developed at the Simons Foundation by Alexey Izmailov during
a summer 2022 internship under the mentorship of Brian Ward.
