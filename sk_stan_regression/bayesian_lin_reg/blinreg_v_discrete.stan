data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  matrix[N, K] X;   // predictor matrix
  int y[N];      // outcome vector
  int<lower=0> family; // family of the model
  int<lower=0> link; // link function of the model 
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
}
transformed parameters {
    vector[N] mu; 

    mu = alpha + X * beta; // lienar predictor 
}
model {
    print(y);
    // TODO: add error scales for alpha & beta?
    //alpha ~ normal(0., sigma); 
    //beta ~ normal(0., sigma);

    // likelihood
    // binomial with logit link 
    y ~ bernoulli_logit(mu); 
}
//generated quantities {
//   vector[N] y_sim; 
//
//   y_sim ~ bernoulli_rng(mu);
//}
