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
  real<lower=0> sigma;  // error scale
}
model {
  // see Gaussian links: https://cran.r-project.org/web/packages/GlmSimulatoR/vignettes/exploring_links_for_the_gaussian_distribution.html 
  if (family == 0) { // Gaussian
    if (link == 0) {  // identity link
      y ~ normal(alpha + X * beta, sigma);
    }
    //else if (link == 1) { // log link
    //  y ~ normal(log(alpha + X * beta), sigma);
    //}
    // else if (link == 2) { // inverse link
    //  y ~ (1 / (normal(alpha + X * beta, sigma)));
    //}
  }
}
