/* Log-likelihood functions for different families. */

/**
* Log-likelihood function for the Gaussian distribution.

* Valid links are: 
* 0 - identity 
* 1 - log
* 2 - inverse 
* Assumes proper validation has occurred prior to this call. 

*/
real gaussian_llh(vector y, real sigma, vector mu, int link) {
    real L = -0.5 * rows(y) * log(2 * pi() * sigma);

    if (link == 0) { // identity link   
        L -= 0.5 * square((y - mu) / sigma); 
    }
    else if (link == 1) { // log link 
        L -= 0.5 * square((y - exp(mu)) / sigma);
    }
    else if (link == 2) { // inverse link  
        L -= 0.5 * square((y - inv(mu)) / sigma);
    }

    return L;
}


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
                 vector mu, real alpha, 
                 int link) {
    real L = (alpha - 1) * s_log_y + 
                rows(y) * (alpha * log(alpha) - lgamma(shape));

    if (link == 0) { // identity link 
        L -= alpha * (sum(y ./ mu) + sum(log(mu))); 
    }
    else if (link == 1) { // log link 
        L -= alpha * (sum(y ./ exp(mu)) + sum(mu)); 
    }
    else if (link == 2) { // inverse link  
        L +=  alpha * (- dot_product(mu, y) + sum(log(mu))); 
    }

    return L; 
                
}