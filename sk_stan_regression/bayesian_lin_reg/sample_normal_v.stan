data { 
    int<lower=0> N; 
    int<lower=0> K;  
    matrix[N, K] X; 
    int<lower=0> family; // family of the model
    int<lower=0> link; // link function of the model 
  // 0: Gaussian, | 0: identity, 1: log, 2: inverse 
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale
}
generated quantities {
    real y_sim[N]; 

     // see Gaussian links: https://cran.r-project.org/web/packages/GlmSimulatoR/vignettes/exploring_links_for_the_gaussian_distribution.html 
    if (family == 0) { // Gaussian
      if (link == 0) {  // identity link
        y_sim = normal_rng(alpha + X * beta, sigma);
      } else if (link == 1) { // log link
        y_sim = normal_rng(log(alpha + X * beta), sigma);
      } else if (link == 2) { // inverse link
        y_sim = normal_rng(inv(alpha + X * beta), sigma);
      }
    }
   
    //y_sim = normal_rng(alpha + X*beta, sigma); 
}
