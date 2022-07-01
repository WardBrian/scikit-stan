data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  matrix[N, K] X;   // predictor matrix
  vector[N] y;      // outcome vector
  int<lower=0> family; // family of the model
  int<lower=0> link; // link function of the model 
  // 0: Gaussian, | 0: identity, 1: log, 2: inverse 
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale OR variance of the error distribution !!!
}
transformed parameters {
  vector[N] mu; // expected values / linear predictor

  mu = alpha + X * beta; 
  vector[N] beta_internal; // rate parameter for gamma distribution 
  if (family == 2) { // Gamma
    mu = exp(mu); // using log link 

    beta_internal = rep_vector(sigma, N) ./ mu;  
  }
}
//transformed parameters {
//  vector[N] mu;         
//  vector[N] beta_t; 
//
//  if (family == 2) { // Gamma
//    if (link == 2) { // log link  
//      mu = exp(alpha + X * beta); 
//    }
//  }
//
//  print("transformed variables");
//  print("sigma: ", sigma);
//  print("alpha: ", alpha);
//  print("beta: ", beta);
//  print("dims beta:", dims(beta_t));   
//  print("dims mu:", dims(mu));
//  print("dims rep", dims(rep_vector(sigma, N)));
//  beta_t = rep_vector(sigma, N) ./ beta;
//  print(mu); 
//  print(beta_t);
//}
model {
  // see Gaussian links: https://cran.r-project.org/web/packages/GlmSimulatoR/vignettes/exploring_links_for_the_gaussian_distribution.html 
  if (family == 0) { // Gaussian
    if (link == 0) {  // identity link
      y ~ normal(mu, sigma);
    } else if (link == 1) { // log link
      y ~ normal(exp(mu), sigma);
    } else if (link == 2) { // inverse link
      y ~ normal(inv(mu), sigma);
    } 
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
    alpha ~ cauchy(0,10); //prior for the intercept following Gelman 2008
    sigma ~ exponential(1); //prior for inverse dispersion parameter

    #if (size(beta) > 1) { 
    beta[1:] ~ cauchy(0,2.5); //prior for the slopes following Gelman 2008
    #}

    if (link == 0) {  // identity link
      y ~ gamma(sigma, (sigma ./ (mu)));
    } else if (link == 1) { // inverse link
      y ~ gamma(sigma, (sigma ./ inv(mu)));
    } else if (link == 2) { // log link
      print(sigma); 
      y ~ gamma(sigma, beta_internal);
    }
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
