data { 
    int<lower=0> N; 
    int<lower=0> K;  
    matrix[N, K] X; 
    real alpha; 
    vector[K] beta;
    real<lower=0> sigma; 
}

generated quantities {
    real y[N]; 

    y = normal_rng(alpha + X*beta, sigma); 
}