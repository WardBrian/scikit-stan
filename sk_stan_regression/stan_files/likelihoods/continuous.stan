/* Log-likelihood functions for different families. */

/**
* Log-likelihood function for the Gaussian distribution.

* Valid links are: 
* 0 - identity 
* 1 - log
* 2 - inverse 
* Assumes proper validation has occurred prior to this call. 

*/
//real gaussian_llh(vector y, real sigma, vector mu, int link) {
//    real L = -0.5 * rows(y) * log(2 * pi() * sigma);
//
//    if (link == 0) { // identity link   
//        L -= 0.5 * square((y - mu) / sigma); 
//    }
//    else if (link == 1) { // log link 
//        L -= 0.5 * square((y - exp(mu)) / sigma);
//    }
//    else if (link == 2) { // inverse link  
//        L -= 0.5 * square((y - inv(mu)) / sigma);
//    }
//
//    return L;
//}


/**
* Log-likelihood function for the Gamma distribution.
* Because linear regression assumesc onstant variance, 
* the shape parameter of the distribution is considered constant
* and the regression occurs through mean mu and shape, where the
* parameterization is mu = alpha / beta and dispersion phi = 1 / alpha 
* resuling in a regression with Gamma(mu, phi). 

* See https://math.stackexchange.com/questions/310487/likelihood-function-of-a-gamma-distributed-sample

* Valid links are: 
*   0 - identity
*   1 - log
*   2 - inverse 
* Assumes proper validation has occurred prior to this call. 

*/ 
real gamma_llh(vector y, real s_log_y,
                vector mu, real alpha) {
    real L = (alpha - 1) * s_log_y + 
                rows(y) * alpha * (log(alpha) - lgamma(alpha)) - 
                                    (sum(y ./ mu) + sum(log(mu)));

    return L;                
}

/**

* See: https://math.stackexchange.com/questions/2888717/how-to-find-the-mle-of-the-parameters-of-an-inverse-gaussian-distribution
*/

real inv_gaussian_llh(vector y, real s_log_y, 
                        vector mu, real sigma, 
                        vector sqrt_y) {
    real L = 0.5 * ( rows(y) * log(sigma / (2 * pi() - 
                        sigma * dot_self((y-mu) ./ (mu .* sqrt_y))))) 
                        - 1.5 * s_log_y;
    return L; 
}

/* (P)RNG for Inverse Gaussian distribution
*  
* parameters of distribution
* @param mu: mean of distribution 
* @param lambda: shape parameter of distribution
* See this Wikipedia algorithm for sampling: 
* https://www.wikiwand.com/en/Inverse_Gaussian_distribution#/Sampling_from_an_inverse-Gaussian_distribution
*/
real inv_gaussian_rng(real mu, real lambda) {
    real nu = uniform_rng(0,1); 
    real y = nu * nu; 
    real x = mu + 
                    mu / (2 * lambda) * (mu * y - sqrt(4 * mu * lambda * y + square(mu) * square(y)));
    
    real z = uniform_rng(0,1);
    if (z < mu / (mu + x)) {
        return x;
    }
    else {
        return mu * mu / x;
    }
}

// TODO: vectorize (overload, really) the above algorithm, 
// either here or a PR to Stan math core

/* Vectorized (P)RNG for Inverse Gaussian distribution
*  
* parameters of distribution
* @param mu: mean of distribution 
* @param lambda: shape parameter of distribution
* See this Wikipedia algorithm for sampling: 
* https://www.wikiwand.com/en/Inverse_Gaussian_distribution#/Sampling_from_an_inverse-Gaussian_distribution
*/
//vector inv_gaussian_rng(vector mu, real lambda) {
//    int N = rows(mu); 
//    real nu[rows(mu)] = uniform_rng(0,1, N); 
//    real y = nu * nu; 
//    real x = mu + 
//                    mu / (2 * lambda) * (mu * y - sqrt(4 * mu * lambda * y + square(mu) * square(y)));
//    
//    real z = uniform_rng(0,1);
//    if (z < mu / (mu + x)) {
//        return x;
//    }
//    else {
//        return mu * mu / x;
//    }
//}