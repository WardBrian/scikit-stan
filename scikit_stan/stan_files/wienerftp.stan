// Regression model for first-hitting-time of the Wiener process
functions {  
    #include /likelihoods/continuous.stan
    #include ./common.stan
}
data { 
    int<lower=0> N;   // number of data items
    int<lower=0> K;   // number of predictors/features
    int<lower=0, upper=1> predictor; // 0: fitting run, 1: prediction run
    matrix[N, K] X;   // predictor matrix
    vector[(predictor > 0) ? 0 : N] y;      // outcome vector; change to N*(1-predictor)
    // assume validation performed externally to Stan
    int<lower=0, upper=2> family; // family of the model
    int<lower=0, upper=4> link; // link function of the model
}
parameters { 
    real alpha; // intercept
    vector[K] beta; // coefficients
}
transformed parameters { 
    
}
model { 
    // default prior selection follows:
    // https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html


}
// model cannot currently support generated quantities since Stan core does not have an rng for this distribution 
// generated quantities {
// 
// }