/* Log-likelihood functions for different discrete families. */
// TODO: the following implementations are clunky, clean them up.
functions { 
    #include "../common.stan"
}

real bernoulli_llh(vector y, real p) {
    return log(p) * sum(y) + log(1 - p) * (n - sum(y));
}

real binomial_llh(int[] y, int[] trial_results, vector mu, int link) {
    if (link == 6) { // probit
        return binomial_lpmf(y | trial_results, Phi(mu))
    } 
    else { // cauchit
        return binomial_lpmf(y | trial_results, atan(mu) / pi() + 0.5);
    }
    //return sum(y) * log(p) + (n - sum(y)) * log(1 - p);
}

real poisson_llh()

//real poisson_llh(vector y, real theta)