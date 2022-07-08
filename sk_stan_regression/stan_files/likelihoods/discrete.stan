/* Log-likelihood functions for different discrete families. */

real bernoulli_llh(vector y, real p) {
    return log(p) * sum(y) + log(1 - p) * (n - sum(y));
}

real binomial_llh(vector y, real n, real p) {
    return sum(y) * log(p) + (n - sum(y)) * log(1 - p);
}

//real poisson_llh(vector y, real theta)