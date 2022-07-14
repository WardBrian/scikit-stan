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
  int<lower=0> y[(predictor > 0) ? 0 : N];   // outcome vector
  int<lower=0> trials[N];           // number of trials 
  int<lower=3, upper=5> family;     // family of the model
  int<lower=0> link;                // link function of the model 
  // assume validation performed externally 
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
}
transformed parameters {
    // default prior selection follows: 
    // https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html
    real sdy = (family == 0) ? sd(y) : 1;
    real sdx = (family == 0) ? sd(X) : 1;
    real my = (family == 0) ? mean(y) : 0; 

    vector[N] mu = alpha + X * beta; // linear predictor 
}
model {
    beta ~ normal(0, 2.5 * sdy / sdx); 
    alpha ~  normal(my, 2.5 * sdy); 

    if (family == 3) { // Poisson  
        y ~ poisson(common_invert_link(mu, link));
    }
    else if (family == 4) { // binomial
        target += binomial_llh(y, trials, mu, link); 
    }
}
generated quantities {
    real y_sim[predictor * N];

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
