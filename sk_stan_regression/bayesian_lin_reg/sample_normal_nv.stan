data {
  int N;
  real X[N]; 
  real alpha; 
  real beta; 
  real sigma; 
}

generated quantities {
  real y[N];

  for (i in 1:N) {
    y[i] = normal_rng(alpha + beta * X[i], sigma);
  }
}