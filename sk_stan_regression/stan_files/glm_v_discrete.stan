// GLM for Binomial, Bernoulli, and Poisson regressions
functions { 
    #include /likelihoods/discrete.stan 
    //#include ./common.stan 
}
data {
  int<lower=0> N;                   // number of data items
  int<lower=0> K;                // number of predictors
  matrix[N, K] X;   // predictor matrix
  int<lower=0, upper=1> predictor;  // 0: fitting run, 1: prediction run
  int<lower=0> y[(predictor > 0) ? 0 : N];   // outcome vector
  int<lower=0> trials[N];           // number of trials 
  int<lower=3, upper=5> family;     // family of the model
  int<lower=0> link;                // link function of the model 
  // assume validation performed externally 
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale OR variance of the error distribution
}
transformed parameters {
    vector[N] mu; 

    mu = alpha + X * beta; // linear predictor 
}
model {
    // TODO: add error scales for alpha & beta?
    //alpha ~ normal(0., sigma); 
    //beta ~ normal(0., sigma);

    if (family == 3) { // Poisson  
        
    }
    else if (family == 4) { // binomial
        target += binomial_llh(y, trials, mu, link); 
    }
    // TODO: cholesky factor necessary? 
    // TOOD: weighted LLH?
    //if (family == 3) { // poisson  
    //    // NOTE: two different parameterizations: 
    //    // poisson_log_lpmf uses log-alpha parameterization, 
    //    // see: 
    //    // https://mc-stan.org/docs/functions-reference/poisson-distribution-log-parameterization.html
    //    // while log_lpmf does not, see 
    //    // https://mc-stan.org/docs/functions-reference/poisson.html
    //    if (link == 0) { // identity link  
    //        target += poisson_log_lpmf(y | mu); 
    //    } 
    //    #else { 
    //    #    
    //    #}
    //}

    // TODO: the following implementations are clunky, clean them up.
    //else if (family == 4) { // binomial  
    //    if (link == 1) { // log link 
    //        target += y * mu; 
    //        target += (trial_results - y) * log1m_exp(mu); 
    //        target += lchoose(trials, y); 
    //        #for (n in 1:num_elements(y)) { 
    //        #    target += y[n] * mu[n]; 
    //        #    target += (trial_results[n] - y[n]) * log1m_exp(mu[n]);
    //        #    target += lchoose(trials[n], y[n]);
    //        #}
    //    }
    //    else if (link == 5) { // logit 
    //        target += binomial_logit_lpmf(y | trial_results, mu);
    //    }    
    //    else if (link == 7) { // cloglog link 
    //        real n_exp_mu = -exp(mu);
    //        target += y * log1m_exp(n_exp_mu);
    //        target += (trial_results - y) * n_exp_mu;
    //        target += lchoose(trial_results, y);
    //    }
    //    else { 
    //        vector[N] mu_unlinked = common_invert_link(mu, link); 
    //        target += binomial_llh(y, mu_unlinked);
    //    }
//
    //}
    //else if (family == 3) { // poisson family 
    //    if (link == 0) { // identity link 
    //        y ~ poisson(mu); 
    //    } else if (link == 1) { // log link  
    //        y ~ poisson(exp(mu));
    //    } else { // sqrt link 
    //        y ~ poisson(square(mu)); 
    //    }
    //}
}
//generated quantities {
//    
//}
