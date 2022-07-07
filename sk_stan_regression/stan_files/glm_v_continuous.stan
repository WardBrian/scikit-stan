// GLM for Gaussian, Gamma, inverse Gaussian, and Beta models 
functions { 
  #include /likelihoods/continuous.stan 
  #include /likelihoods/common.stan 
}
data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  int<lower=0, upper=1> predictor; 
  matrix[N, K] X;   // predictor matrix
  vector[(predictor > 0) ? 0 : N] y;      // outcome vector
  int<lower=0, upper=2> family; // family of the model
  int<lower=0, upper=4> link; // link function of the model 
  // validation performed outside of this
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale OR variance of the error distribution
}
transformed parameters {
  real s_log_y = sum(log(y)); 
  vector[N] sqrt_y = sqrt(y); 

  vector[N] mu; // expected values / linear predictor

  mu = alpha + X * beta; 
}
model {
  if (family == 0) { // Gaussian
    //Increment target log probability density with
    // normal_lpdf( y | mu, sigma) dropping constant additive terms.
    y ~ normal(common_invert_link(mu, link), sigma); 
  } 
  else if (family == 1) { // Gamma
    #alpha ~ cauchy(0,10); //prior for the intercept following Gelman 2008
    #sigma ~ exponential(1); //prior for inverse dispersion parameter
#
    ##if (size(beta) > 1) { 
    #beta[1:] ~ cauchy(0,2.5); //prior for the slopes following Gelman 2008
    #}

    target += gamma_llh(y, s_log_y, common_invert_link(mu, link), sigma);
  }
  else if (family == 2) { // inverse Gaussian
    target += inv_gaussian_llh(y, s_log_y, common_invert_link(mu, link), sigma, sqrt_y);
  }
 
  //} else if (family == 5) { // Inverse Gaussian
  //  if (link == 0) {  // identity link
  //    y ~ inverse_gaussian(alpha + X * beta);
  //  } else if (link == 1) { // log link
  //    y ~ inverse_gaussian(log(alpha + X * beta));
  //  } else if (link == 2) { // inverse link
  //    y ~ inverse_gaussian(inv(alpha + X * beta));
  //  }
  //}
}
generated quantities {
  real y_sim[N]; 
  vector[N] mu_unlinked = common_invert_link(mu, link); 

  if (family == 0) { // Gaussian
    y_sim = normal_rng(mu_unlinked, sigma); 
  }
  else if (family == 1) { // Gamma
    y_sim = gamma_rng(sigma, sigma ./ mu_unlinked); 
  }


}