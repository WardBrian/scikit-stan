// GLM for Gaussian, Gamma, inverse Gaussian, and Beta models
functions {
  #include /likelihoods/continuous.stan
  #include ./common.stan
}
data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors/features
  int<lower=0, upper=1> predictor; // 0: fitting run, 1: prediction run
  matrix[N, K] X;   // predictor matrix
  vector[(predictor > 0) ? 0 : N] y;      // outcome vector; change to N*(1-predictor)
  // assume validation performed externally to Stan
  int<lower=0, upper=2> family; // family of the model
  int<lower=0, upper=4> link; // link function of the model

  // set up for user-defineable priors
  int<lower=-1> prior_intercept_dist;    // distribution for intercept
  real prior_intercept_mu;              // mean of the prior for intercept
  real prior_intercept_sigma;           // error scale of the prior for intercept
  int<lower=-1> prior_slope_dist;        // distribution for regression coefficients
  vector[K] prior_slope_mu;             // mean of the prior for each regression coefficient
  vector[K] prior_slope_sigma;          // error scale of the prior for each  regression coefficient
  int<lower=-1> prior_aux_dist;          // distribution for auxiliary parameter (sigma): 0 is exponential, 1 is chi2

  real<lower=0> prior_aux_param;      // distribution parameter for the prior for sigma
  real sdy;
}
transformed data {
  real s_log_y = sum(log(y));
  vector[rows(y)] sqrt_y = sqrt(y);
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale OR variance of the error distribution
}
transformed parameters {
  vector[N] mu = alpha + X * beta; // expected values / linear predictor
  vector[N] mu_unlinked = common_invert_link(mu, link);
}
model {
  // default prior selection follows:
  // https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html
  if (prior_intercept_dist == 0) { // normal prior; has mu and sigma
      alpha ~ normal(prior_intercept_mu, prior_intercept_sigma);
  }
  else if (prior_intercept_dist == 1) { // laplace prior; has mu and sigma
      alpha ~ double_exponential(prior_intercept_mu, prior_intercept_sigma);
  }

  // NOTE: these operations are vectorized 
  if (prior_slope_dist == 0) { // normal prior, has mu and sigma vectors 
    beta ~ normal(prior_slope_mu, prior_slope_sigma);
  }
  else if (prior_slope_dist == 1) { // laplace prior, has mu and sigma vectors 
    beta ~ double_exponential(prior_slope_mu, prior_slope_sigma); 
  }

  // NOTE: prior_aux_param is a placeholder value and this
  // should be a loop once more general prior distributions are supported
  if (prior_aux_dist == 0) { // exponential
    sigma ~ exponential(prior_aux_param);
  }
  else if (prior_aux_dist == 1) { // chi2
    sigma ~ chi_square(prior_aux_param);
  }

  if (family == 0) { // Gaussian
    //Increment target log probability density with
    // normal_lpdf( y | mu, sigma) dropping constant additive terms.
    y ~ normal(mu_unlinked, sigma);
  }
  else if (family == 1) { // Gamma
    target += gamma_llh(y, s_log_y, mu, sigma, link);
  }
  else if (family == 2)
   {
    target += inv_gaussian_llh(y, s_log_y, mu_unlinked, sigma, sqrt_y);
  }
  // add additional families here
}
generated quantities {
  array[predictor * N] real y_sim;
  if (predictor) {
      if (family == 0) { // Gaussian
        y_sim = normal_rng(mu_unlinked, sigma);
      }
      else if (family == 1) { // Gamma
        y_sim = gamma_rng(sigma, sigma ./ mu_unlinked);
      }
      else { // inverse Gaussian; in the future this should be changed to a vectorized library function 
        for (n in 1:N) {
          y_sim[n] = inv_gaussian_rng(mu_unlinked[n], sigma);
        }
      }
  }
}
