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
    // TODO: add error scales for alpha & beta?
    //alpha ~ normal(0., sigma); 
    //beta ~ normal(0., sigma);

    // likelihood
    // bernoulli family 
    //if (family == 1) { 
    //    if (link == 0) { // logit link
    //        y ~ binomial(inv_logit(mu)); 
    //    } else if (link == 1) { // probit link
    //        y ~ binomial(Phi(mu));
    //    } else if (link == 2) { // cauchit link
    //        y ~ binomial(0.5 + atan(mu) / pi()); 
    //    } else if (link == 3) // log link
    //        y ~ binomial(exp(mu));
    //    else { // cloglog link 
    //        y ~ binomial(inv_cloglog(mu));
    //    }
    //} 
    if (family == 3) { 
        if (link == 0) { // identity link 
            y ~ poisson(mu); 
        } else if (link == 1) { // log link  
            y ~ poisson(exp(mu));
        } else { // sqrt link 
            y ~ poisson(square(mu)); 
        }
    }
}
