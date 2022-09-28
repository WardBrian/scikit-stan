// GLM for Binomial, Bernoulli, and Poisson regressions
functions {
  #include /likelihoods/discrete.stan
}
data {
  // control flags and X_data
  #include /common/glm_data.stan
  
  array[(predictor > 0) ? 0 : N] int<lower=0> y; // outcome vector
  
  // assume validation performed externally
  int<lower=3, upper=5> family; // family of the model
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
  
  if (family == 3) {
    // Poisson
    y ~ poisson(common_invert_link(mu, link));
  } else if (family == 4) {
    // TODO neg_binomial family
    // TODO aux parameter and priors
    // y ~ neg_binomial_2(common_invert_link(mu, link), aux);
  } else if (family == 5) {
    // bernoulli
    if (link == 5) {
      // logit
      // efficient Stan function for this family-link combination
      y ~ bernoulli_logit(mu);
    } else {
      y ~ bernoulli(common_invert_link(mu, link));
    }
  }
}
generated quantities {
  array[predictor * N] real y_sim;
  vector[(predictor > 0) ? 0 : N] log_lik;
  {
    // expected values / linear predictor
    #include /common/make_mu.stan
    
    vector[N] mu_unlinked = common_invert_link(mu, link); // reverse link function
    
    if (family == 3) {
      // Poisson
      if (predictor) {
        y_sim = poisson_rng(mu_unlinked);
      } else {
        for (n in 1 : N) {
          log_lik[n] = poisson_lpmf(y[n] | mu_unlinked[n]);
        }
      }
    }
    if (family == 4) {
      // neg_binomial_2
      // y_sim = neg_binomial_2_rng(mu_unlinked, aux);
    } else if (family == 5) {
      // bernoulli
      if (predictor) {
        y_sim = bernoulli_rng(mu_unlinked);
      } else {
        if (link == 5) {
          for (n in 1 : N) {
            log_lik[n] = bernoulli_logit_lpmf(y[n] | mu[n]);
          }
        } else {
          for (n in 1 : N) {
            log_lik[n] = bernoulli_lpmf(y[n] | mu_unlinked[n]);
          }
        }
      }
    }
  }
}
