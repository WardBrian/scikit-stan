/* Log-likelihood functions for different continuous families. */

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
* Assumes proper validation has occurred prior to being called. 
* Also assumes that the inverse fo the link function has already been applied 
* to mu; see the function calls in /glm_v_continuous.stan.

@param y: vector of observations
@param s_log_y: sum of the log of the observations
@param mu: vector of means (linear predictor) 
@param alpha: shape parameter for the Gamma distribution 
@return scalar for log-likelihood 
*/ 
real gamma_llh(vector y, real s_log_y,
                vector mu, real alpha, int link) {
    real L = (alpha - 1) * s_log_y + 
                rows(y) * (alpha * log(alpha) - lgamma(alpha));

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

/**
* Log-likelihood function for the inverse Gaussian distribution (this will be made
* a library function in the future). This is parameterized by mu, the linear predictor, 
* and by lambda, the dispersion parameter (see the distribution's single-parameter form).

* Valid links are:
* 0 - identity
* 1 - log
* 2 - inverse
* 4 - inverse squared

* See: https://math.stackexchange.com/questions/2888717/how-to-find-the-mle-of-the-parameters-of-an-inverse-gaussian-distribution

* Assumes proper validation has occurred prior to being called.
* Also assumes that the inverse fo the link function has already been applied 
* to mu; see the function calls in /glm_v_continuous.stan.

@param y: vector of observations
@param s_log_y: sum of the log of the observations
@param mu: vector of means (linear predictor) 
@param lambda: 
@return scalar for log-likelihood 
*/

real inv_gaussian_llh(vector y, real s_log_y, 
                        vector mu, real lambda, 
                        vector sqrt_y) {
    return 0.5 * rows(y) * log(lambda / (2 * pi())) -
            1.5 * s_log_y - 
            0.5 * lambda * dot_self((y - mu) ./ (mu .* sqrt_y));
}

/**
* (P)RNG for Inverse Gaussian distribution
* The distribution is assumed to be parameterized by mu, the linear predictor,
* and by lambda, the dispersion parameter (see the distribution's two-parameter form).

* See this Wikipedia algorithm for sampling: 
* https://www.wikiwand.com/en/Inverse_Gaussian_distribution#/Sampling_from_an_inverse-Gaussian_distribution

@param mu: mean of distribution 
@param lambda: shape parameter of distribution
@return single sample from distribution 
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
