// GLM for Gaussian, Gamma, inverse Gaussian, and Beta models
functions {
  #include /likelihoods/continuous.stan
  #include /common/functions.stan
}
data {
  // control flags and X_data
  #include /common/glm_data.stan

  vector[(predictor > 0) ? 0 : N] y; // outcome vector; change to N*(1-predictor)

  // assume validation performed externally to Stan
  int<lower=0, upper=2> family; // family of the model
  int<lower=0, upper=4> link; // link function of the model

  // set up for user-defineable priors
  #include /common/glm_priors.stan

  // validation on parameters for each distribution occurs Python-side
  int<lower=-1> prior_aux_dist; // distribution for auxiliary parameter (sigma):
  // -1 is default uniform(-inf, inf), 0 is exponential, 1 is chi2
  int<lower=1> num_prior_aux_params; // number of parameters in the prior for auxiliary parameter
  array[num_prior_aux_params] real<lower=0> prior_aux_params; // distribution parameter for the prior for sigma
}
transformed data {
  real s_log_y = sum(log(y));
  vector[rows(y)] sqrt_y = sqrt(y);
}
parameters {
  array[fit_intercept] real alpha; // regression intercept alpha; empty if fit_intercept = 0
  vector[K] beta; // regression coefficients beta
  real<lower=0> sigma; // error scale OR variance of the error distribution
}
model {
  #include /common/make_mu.stan

  vector[N] mu_unlinked = common_invert_link(mu, link); // reverse link function

  // default prior selection follows:
  // https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html
  if (prior_intercept_dist == 0) {
    // normal prior; has mu and sigma
    alpha ~ normal(prior_intercept_mu, prior_intercept_sigma);
  } else if (prior_intercept_dist == 1) {
    // laplace prior; has mu and sigma
    alpha ~ double_exponential(prior_intercept_mu, prior_intercept_sigma);
  }

  if (prior_slope_dist == 0) {
    // normal prior, has mu and sigma vectors
    beta ~ normal(prior_slope_mu, prior_slope_sigma);
  } else if (prior_slope_dist == 1) {
    // laplace prior, has mu and sigma vectors
    beta ~ double_exponential(prior_slope_mu, prior_slope_sigma);
  }

  if (prior_aux_dist == 0) {
    // exponential
    sigma ~ exponential(prior_aux_params[1]);
  } else if (prior_aux_dist == 1) {
    // chi2
    sigma ~ chi_square(prior_aux_params[1]);
  } else if (prior_aux_dist == 2) {
    // gamma, alpha & beta
    sigma ~ gamma(prior_aux_params[1], prior_aux_params[2]);
  } else if (prior_aux_dist == 3) {
    // inverse gamma, alpha & beta
    sigma ~ inv_gamma(prior_aux_params[1], prior_aux_params[2]);
  }
  // additional auxiliary parameter priors go here
  // NOTE: the current set up shows how to add multivariable priors,
  // ones with more parameters just need to index the prior_aux_params array as needed

  if (family == 0) {
    // Gaussian
    y ~ normal(mu_unlinked, sigma);
  } else if (family == 1) {
    // Gamma
    target += gamma_llh(y, s_log_y, mu, sigma, link);
  } else if (family == 2) {
    target += inv_gaussian_llh(y, s_log_y, mu_unlinked, sigma, sqrt_y);
  }
  // add additional continuous families here
}
generated quantities {
  array[predictor * N] real y_sim;
  vector[(predictor > 0) ? 0 : N] log_lik;

  {
    #include /common/make_mu.stan

    vector[N] mu_unlinked = common_invert_link(mu, link); // reverse link function

    if (family == 0) {
      // Gaussian
      if (predictor) {
        y_sim = normal_rng(mu_unlinked, sigma);
      } else {
        for (n in 1 : N) {
          log_lik[n] = normal_lpdf(y[n] | mu_unlinked[n], sigma);
        }
      }
    } else if (family == 1) {
      // Gamma
      if (predictor) {
        y_sim = gamma_rng(sigma, sigma ./ mu_unlinked);
      } else {
        for (n in 1 : N) {
          log_lik[n] = gamma_llh(y[n:n+1], s_log_y, mu[n:n+1], sigma, link);
        }
      }
    } else {
      // inverse Gaussian; in the future this should be changed to a vectorized library function
      if (predictor) {
        for (n in 1 : N) {
          y_sim[n] = inv_gaussian_rng(mu_unlinked[n], sigma);
        }
      } else {
        for (n in 1 : N) {
          log_lik[n] = inv_gaussian_llh(y[n:n+1], s_log_y, mu_unlinked[n:n+1], sigma, sqrt_y);
        }
      }
    }
  }
}
