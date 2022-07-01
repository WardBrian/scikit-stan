data {
  int N; //the number of observations
  int K; //the number of columns in the model matrix
  real y[N]; //the response
  matrix[N, K] X; //the model matrix
}
parameters {
  vector[K] betas; //the regression parameters
  real phi; //the variance parameter
}
transformed parameters {
  vector[N] mu; //the expected values (linear predictor)
  vector[N] alpha; //shape parameter for the gamma distribution
  vector[N] beta; //rate parameter for the gamma distribution
  
  mu <- exp(X*betas); //using the log link 
  alpha <- mu .* mu / phi; 
  beta <- mu / phi;
}
model {  
  betas[1] ~ cauchy(0,10); //prior for the intercept following Gelman 2008

  for(i in 2:K)
   betas[i] ~ cauchy(0,2.5);//prior for the slopes following Gelman 2008
  
  y ~ gamma(alpha,beta);
}