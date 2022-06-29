data { 
    int<lower=0> N; 
    int<lower=0> K;  
    matrix[N, K] X; 
    int<lower=0> family; 
    int<lower=0> link; 
}
parameters { 
    
}