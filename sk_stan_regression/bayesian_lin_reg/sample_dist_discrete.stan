data { 
    int<lower=0> N; 
    int<lower=0> K;  
    matrix[N, K] X; 
    int<lower=0> family; 
    int<lower=0> link; 
}
parameters { 
    real alpha;           // intercept
    vector[K] beta;       // coefficients for predictors
}
transformed parameters {
   vector[N] mu; 

   mu = alpha + X * beta; 
}
generated quantities {
   vector[N] y_sim;

    if (family == 1) { // Bernoulli 
        if (link == 0) { // logit link
            y_sim = bernoulli_rng(inv_logit(mu)); 
        } else if (link == 1) { // probit link 
            y_sim = bernoulli_rng(Phi(mu));
        } else if (link == 3) { // log link 
            y_sim = bernoulli_rng(exp(mu));
        } else { // cloglog link  
            y_sim = bernoulli_rng(inv_cloglog(mu));
        }
    }
}