// GLM for Gaussian, Gamma, inverse Gaussian, and Beta models 
functions { 
  #include /likelihoods/continuous.stan 
  #include ./common.stan 
}
data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  int<lower=0, upper=1> predictor; // 0: fitting run, 1: prediction run
  matrix[N, K] X;   // predictor matrix
  vector[(predictor > 0) ? 0 : N] y;      // outcome vector; change to N*(1-predictor)
  int<lower=0, upper=2> family; // family of the model
  int<lower=0, upper=4> link; // link function of the model 
  // assume validation performed externally to Stan 
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale OR variance of the error distribution
}
transformed parameters {
  // default prior selection follows: 
  // https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html
  real sdy; 
  real my; 
  // NOTE: if this is a prediction run, y is not specified! 
  if (predictor == 1) { 
      sdy = 1.;
      my = 0.;  
  }
  else { 
      sdy = (family == 0) ? sd(y) : 1.;
      // safeguard to ensure that scale parameter on scale for prior is positive
      if (sdy <= 0.) { 
        sdy = 1.;
      }
      my = (family == 0) ? mean(y) : 0.;
  }
  real sdx = (family == 0) ? sd(X) : 1.;

  real s_log_y = sum(log(y)); 
  vector[rows(y)] sqrt_y = sqrt(y); 

  vector[N] mu; // expected values / linear predictor
  mu = alpha + X * beta; 
  vector[N] mu_unlinked = common_invert_link(mu, link); 
}
model {  
  if (family == 1) { // Gamma  
    target += gamma_llh(y, s_log_y, mu, sigma, link);
  } 
  else { 
    beta ~ normal(0, 2.5 * sdy / sdx); 
    alpha ~  normal(my, 2.5 * sdy); 

    //sigma ~ exponential(1 / sdy); // std for Gaussian, shape for gamma

    if (family == 0) { // Gaussian
      //Increment target log probability density with
      // normal_lpdf( y | mu, sigma) dropping constant additive terms.
      y ~ normal(mu_unlinked, sigma); 
    } 
    else if (family == 2) { // inverse Gaussian
      target += inv_gaussian_llh(y, s_log_y, mu_unlinked, sigma, sqrt_y);
    }
  }
}
generated quantities {
  real y_sim[predictor * N]; 
  if (predictor) { 
      if (family == 0) { // Gaussian
        y_sim = normal_rng(mu_unlinked, sigma); 
      }
      else if (family == 1) { // Gamma
        y_sim = gamma_rng(sigma, sigma ./ mu_unlinked); 
      } 
      else { // inverse Gaussian  
        for (n in 1:N) { 
          y_sim[n] = inv_gaussian_rng(mu_unlinked[n], sigma);
        }
      }
  } 
}
