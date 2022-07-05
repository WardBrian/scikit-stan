data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  matrix[N, K] X;   // predictor matrix
  vector[N] y;      // outcome vector
  int<lower=0> family; // family of the model
  int<lower=0> link; // link function of the model 
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale OR variance of the error distribution
}
transformed parameters {
  vector[N] mu; // expected values / linear predictor

  mu = alpha + X * beta; 
  vector[N] beta_internal; // rate parameter for gamma distribution 
  // family-link combination has been validated up to this point
  // only links corresponding to continuous families 
  if (link == 1) { // using log link  
    mu = exp(mu); 
  }
  else if (link == 2) { // using inverse link  
    mu = inv(mu);
  }
  else if (link == 3) { // using sqrt link
    mu = square(mu); 
  }
  # TODO: 1/mu^2 link? combination of square and inverse

  # TODO: this isn't necessary for every model? 
  beta_internal = rep_vector(sigma, N) ./ mu;  
}
model {
  # TODO: move variable transformation to transformed parameters
  if (family == 0) { // Gaussian
    y ~ normal(mu, sigma); 
  } 
  // TODO: add other families
  //else if (family == 1) { // Binomial
    //if (link == 0) {  // identity link
    //  y ~ binomial(alpha + X * beta);
    //} else if (link == 1) { // log link
    //  y ~ binomial(log(alpha + X * beta));
    //} else if (link == 2) { // inverse link
    //  y ~ binomial(inv(alpha + X * beta));
    //} 
  //}
  else if (family == 2) { // Gamma
    #alpha ~ cauchy(0,10); //prior for the intercept following Gelman 2008
    #sigma ~ exponential(1); //prior for inverse dispersion parameter
#
    ##if (size(beta) > 1) { 
    #beta[1:] ~ cauchy(0,2.5); //prior for the slopes following Gelman 2008
    #}

    y ~ gamma(beta_internal, sigma); 
    #if (link == 0) {  // identity link
    #  y ~ gamma(sigma, (sigma ./ (mu)));
    #} else if (link == 1) { // inverse link
    #  y ~ gamma(sigma, (sigma ./ inv(mu)));
    #} else if (link == 2) { // log link
    #  //print(sigma); 
    #  y ~ gamma(sigma, beta_internal);
    #}
  }
  // else if (family == 3) { // Poisson
  //  if (link == 0) {  // identity link
  //    y ~ poisson(exp(alpha + X * beta));
  //  } else if (link == 1) { // log link
  //    y ~ poisson(exp(log(alpha + X * beta)));
  //  } else if (link == 2) { // sqrt link
  //    y ~ poisson(exp(sqrt(alpha + X * beta)));
  //  } 
  //}
  //else if (family == 3) { // Negative Binomial
  //  if (link == 0) {  // identity link
  //    y ~ negative_binomial(alpha + X * beta);
  //  } else if (link == 1) { // log link
  //    y ~ negative_binomial(log(alpha + X * beta));
  //  } else if (link == 2) { // inverse link
  //    y ~ negative_binomial(inv(alpha + X * beta));
  //  } 
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
//generated quantities {
//   real y_sim[N]; 
//
//   y_sim = normal_rng(mu, sigma); 
//   
//}