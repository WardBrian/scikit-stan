// GLM for Binomial, Bernoulli, and Poisson regressions
functions {
    #include /likelihoods/discrete.stan
    //#include ./common.stan
}
data {
  int<lower=0> N;                   // number of data items
  int<lower=0> K;                   // number of predictors
  matrix[N, K] X;                   // predictor matrix
  int<lower=0, upper=1> predictor;  // 0: fitting run, 1: prediction run
  array[(predictor > 0) ? 0 : N] int<lower=0> y;   // outcome vector
  array[N] int<lower=0> trials;           // number of trials

  // assume validation performed externally
  int<lower=3, upper=5> family;     // family of the model
  int<lower=0> link;                // link function of the model

  // set up for user-defineable priors
  // TODO lengths of these set by ternary based on whether priors are default?
  int<lower=-1> prior_intercept_dist;    // distribution for intercept
  real prior_intercept_mu;              // mean of the prior for intercept
  real prior_intercept_sigma;           // error scale of the prior for intercept
  int<lower=-1> prior_slope_dist;        // distribution for regression coefficients
  vector[K] prior_slope_mu;             // mean of the prior for regression coefficients
  vector[K] prior_slope_sigma;          // error regression coefficients
  real sdy;
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
}
transformed parameters {
    vector[N] mu = alpha + X * beta; // linear predictor
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

    // NOTE these are vectorized operations 
    if (prior_slope_dist == 0) { // normal prior
        beta ~ normal(prior_slope_mu, prior_slope_sigma);
    }
    else if (prior_slope_dist == 1) { // laplace prior 
        beta ~ double_exponential(prior_slope_mu, prior_slope_sigma);
    }

    if (family == 3) { // Poisson
        y ~ poisson(common_invert_link(mu, link));
    }
    else if (family == 4) { // binomial
        target += binomial_llh(y, trials, mu, link);
    }
}
generated quantities {
    array[predictor * N] real y_sim;

    if (predictor) {
        vector[N] mu_unlinked = common_invert_link(mu, link);

        if (family == 3) { // Poisson
            y_sim = poisson_rng(mu_unlinked);
        }
        else if (family == 4) { // binomial
            y_sim = binomial_rng(trials, mu_unlinked);
        }
    }
}
