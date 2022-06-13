data { 
    int<lower=0> N; 
    int<lower=0> K;  
    matrix[N, K] X; 
    real alpha; 
    vector[K] beta;
    real<lower=0> sigma; 
}

//normal_rng(alpha + beta * x_new, sigma)
//the vectorized problem is faster in Stan than 
//if K = 1
generated quantities {
    real y[N]; 

    for (i in 1:N) { 
        y[i] = normal_rng(alpha + beta * X[i], sigma); 
    }
}