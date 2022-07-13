data { 
    int<lower=0> N; 
    int K; 
    matrix[N, K] X; 
    array[N] int y; 
}
parameters { 
    vector[K] beta; 
    real alpha; 
}
model { 
    alpha ~ normal(0, 1); 
    beta ~ normal(0, 1);
    y ~ poisson_log(alpha + X * beta);
}
