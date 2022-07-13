#include "./common.stan"
/* Log-likelihood functions for different discrete families. */

// TODO: the following implementations are clunky, clean them up.

real bernoulli_llh(vector y, real p) {
    return log(p) * sum(y) + log(1 - p) * (rows(y) - sum(y));
}


/* Log-likelihood for binomial distribution. 

Valid links are: 
* 1 - log 
* 5 - logit 
* 6 - probit 
* 7 - cloglog
* 8 - cauchit

The link validation is assumed to have been performed outside of this function.

@param y: response vector/vector of observations 
@param trial_results: 
@param mu: vector of means (linear predictor) 
@param link: link function
*/
real binomial_llh(int[] y, int[] trial_results, vector mu, int link) {
    if (link == 5) { // logit  
        return binomial_logit_lpmf(y | trial_results, mu); 
    }
    else if (link == 6 || link == 8) { // probit or cauchit 
        return binomial_lpmf(y | trial_results, common_invert_link(mu, link)); 
    } 
    else if (link == 1) { // log 
        real L; 
        for (n in 1:num_elements(y)) {
            L += y[n] * mu[n];
            L += (trial_results[n] - y[n]) * log1m_exp(mu[n]);
            L += lchoose(trial_results[n], y[n]);
        } 
        return L;
    } 
    else { // cloglog 
        vector[num_elements(y)] neg_exp_mu = - exp(mu); 
        real L; 
        for (n in 1:num_elements(y)) { 
            L += y[n] * log1m_exp(neg_exp_mu[n]);
            L += (trial_results[n] - y[n]) * neg_exp_mu[n];
            L += lchoose(trial_results[n], y[n]);
        }
        return L;
    }
}

//real poisson_llh(vector y, real theta)