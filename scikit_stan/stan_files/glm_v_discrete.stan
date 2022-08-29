// GLM for Binomial, Bernoulli, and Poisson regressions
functions {
    #include /likelihoods/discrete.stan
}
data {
    // control flags and X_data
    #include /common/glm_data.stan

    array[(predictor > 0) ? 0 : N] int<lower=0> y;   // outcome vector
    array[N] int<lower=0> trials;           // number of trials

    // assume validation performed externally
    int<lower=3, upper=6> family;     // family of the model
    int<lower=0> link;                // link function of the model

    // set up for user-defineable priors
    #include /common/glm_priors.stan
}
parameters {
    array[fit_intercept] real alpha;           // intercept
    vector[K] beta;                      // coefficients for predictors
}
model {
    // expected values / linear predictor
    #include /common/make_mu.stan

    // default prior selection follows:
    // https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html
    if (prior_intercept_dist == 0) { // normal prior; has mu and sigma
        alpha ~ normal(prior_intercept_mu, prior_intercept_sigma);
    }
    else if (prior_intercept_dist == 1) { // laplace prior; has mu and sigma
        alpha ~ double_exponential(prior_intercept_mu, prior_intercept_sigma);
    }

    // NOTE: these are vectorized operations
    if (prior_slope_dist == 0) { // normal prior
        beta ~ normal(prior_slope_mu, prior_slope_sigma);
    }
    else if (prior_slope_dist == 1) { // laplace prior
        beta ~ double_exponential(prior_slope_mu, prior_slope_sigma);
    }

    if (family == 3) { // Poisson
        y ~ poisson(common_invert_link(mu, link));
    }

    // TODO binomial should be separate model since it uses different data
    else if (family == 4) { // binomial
        target += binomial_llh(y, trials, mu, link);
    }

    // TODO neg_binomial family

    else if (family == 6) { // bernoulli
        if (link == 5) { // logit
            // efficient Stan function for this family-link combination
            y ~ bernoulli_logit(mu);
        }
        else {
            y ~ bernoulli(common_invert_link(mu, link));
        }
    }
}
generated quantities {
    array[predictor * N] real y_sim;
    {
        if (predictor) {
            // expected values / linear predictor
            #include /common/make_mu.stan
            vector[N] mu_unlinked = common_invert_link(mu, link); // reverse link function

            if (family == 3) { // Poisson
                y_sim = poisson_rng(mu_unlinked);
            }
            else if (family == 4) { // binomial
                y_sim = binomial_rng(trials, mu_unlinked);
            }
            else if (family == 6) { // bernoulli
                y_sim = bernoulli_rng(mu_unlinked);
            }
        }
    }
}
