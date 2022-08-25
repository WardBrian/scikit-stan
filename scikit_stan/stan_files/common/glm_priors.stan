  int<lower=-1> prior_intercept_dist;     // distribution for intercept
  real prior_intercept_mu;                // mean of the prior for intercept
  real prior_intercept_sigma;             // error scale of the prior for intercept
  int<lower=-1> prior_slope_dist;         // distribution for regression coefficients
  vector[K] prior_slope_mu;               // mean of the prior for each regression coefficient
  vector[K] prior_slope_sigma;            // error scale of the prior for each  regression coefficient
