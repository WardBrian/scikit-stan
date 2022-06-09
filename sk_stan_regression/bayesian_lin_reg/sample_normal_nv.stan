data {
  int N;
  real X[N]; // should be redone as matrix for multi-dimensional
  real alpha; 
  real beta; 
  real sigma; 
}

//normal_rng(alpha + beta * x_new, sigma)
generated quantities {
  real y[N];

  for (i in 1:N) {
    y[i] = normal_rng(alpha + beta * X[i], sigma);
  }
}