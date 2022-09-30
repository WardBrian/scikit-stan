// GLM for Binomial, Bernoulli, and Poisson regressions
functions {
  #include /likelihoods/discrete.stan
}
data {
  // control flags and X_data
  #include /common/glm_data.stan
  
  array[(predictor > 0) ? 0 : N] int<lower=0> y; // outcome vector
  array[N] int<lower=0> trials; // number of trials
  
  int<lower=0> link; // link function of the model
  
  // set up for user-defineable priors
  #include /common/glm_priors.stan
}
parameters {
  array[fit_intercept] real alpha; // intercept
  vector[K] beta; // coefficients for predictors
}
model {
  // expected values / linear predictor
  #include /common/make_mu.stan
  
  // default prior selection follows:
  // https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html
  if (prior_intercept_dist == 0) {
    // normal prior; has mu and sigma
    alpha ~ normal(prior_intercept_mu, prior_intercept_sigma);
  } else if (prior_intercept_dist == 1) {
    // laplace prior; has mu and sigma
    alpha ~ double_exponential(prior_intercept_mu, prior_intercept_sigma);
  }
  
  // NOTE: these are vectorized operations
  if (prior_slope_dist == 0) {
    // normal prior
    beta ~ normal(prior_slope_mu, prior_slope_sigma);
  } else if (prior_slope_dist == 1) {
    // laplace prior
    beta ~ double_exponential(prior_slope_mu, prior_slope_sigma);
  }
  
  target += binomial_llh(y, trials, mu, link);
}
generated quantities {
  array[predictor * N] real y_sim;
  vector[save_log_lik * N] log_lik;
  {
    if (predictor || save_log_lik) {
      // expected values / linear predictor
      #include /common/make_mu.stan
      
      if (predictor) {
        vector[N] mu_unlinked = common_invert_link(mu, link); // reverse link function
        y_sim = binomial_rng(trials, mu_unlinked);
      }
      if (save_log_lik) {
        for (n in 1 : N) {
          log_lik[n] = binomial_llh(y[n : n], trials[n : n], mu[n : n], link);
        }
      }
    }
  }
}
