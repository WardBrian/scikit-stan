data {
  int N; //the number of observations
  int K; //the number of columns in the model matrix
  real y[N]; //the response
  matrix[N,K] X; //the model matrix
}
parameters {
  vector[K] betas; //the regression parameters
  real<lower=0> inverse_phi; //the variance parameter
}
transformed parameters {
  vector[N] mu; //the expected values (linear predictor)
  vector[N] beta; //rate parameter for the gamma distribution
  
  mu <- exp(X*betas); //using the log link 
  beta <- rep_vector(inverse_phi, N) ./ mu;
}
model {  
  betas[1] ~ cauchy(0,10); //prior for the intercept following Gelman 2008

  betas[2:] ~ cauchy(0,2.5);//prior for the slopes following Gelman 2008
  
  inverse_phi ~ exponential(1); // prior on inverse dispersion parameter
  y ~ gamma(inverse_phi,beta);
}
generated quantities {
  vector[N] y_rep;
  for(n in 1:N){
  y_rep[n] <- gamma_rng(inverse_phi,beta[n]); //posterior draws to get posterior predictive checks
 }
}