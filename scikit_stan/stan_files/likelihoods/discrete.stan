#include "./common.stan"
/* Log-likelihood functions for different discrete families. */

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
real binomial_llh(array[] int y, array[] int trial_results, vector mu, int link) {
    if (link == 5) { // logit
        return binomial_logit_lpmf(y | trial_results, mu);
    }
    else if (link == 6 || link == 8) { // probit or cauchit
        return binomial_lpmf(y | trial_results, common_invert_link(mu, link));
    }
    else if (link == 1) { // log
        real L;
        //for (n in 1:num_elements(y)) {
        //    L += y[n] * mu[n];
        //    L += (trial_results[n] - y[n]) * log1m_exp(mu[n]);
        //    L += lchoose(trial_results[n], y[n]);
        //}

        // NOTE: the following is equivalent to the above, but is vectorized and is more efficient in memory
        vector[num_elements(y)] y_v = to_vector(y);
        vector[num_elements(y)] trial_results_v = to_vector(trial_results);

        L = sum(y_v .* mu) + sum((trial_results_v - y_v) .* log1m_exp(mu)) + sum(lchoose(trial_results_v, y_v));

        return L;
    }
    else { // cloglog
        vector[num_elements(y)] neg_exp_mu = - exp(mu);
        real L;
        //for (n in 1:num_elements(y)) {
        //    L += y[n] * log1m_exp(neg_exp_mu[n]);
        //    L += (trial_results[n] - y[n]) * neg_exp_mu[n];
        //    L += lchoose(trial_results[n], y[n]);
        //}

        // NOTE: the following is equivalent to the above, but is vectorized and is more efficient in memory
        vector[num_elements(y)] y_v = to_vector(y);
        vector[num_elements(y)] trial_results_v = to_vector(trial_results);

        L = sum(y_v .* log1m_exp(neg_exp_mu)) + sum((trial_results_v - y_v) .* neg_exp_mu) + sum(lchoose(trial_results_v, y_v));

        return L;
    }
}
