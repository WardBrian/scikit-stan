  vector[N] mu;         // expected values / linear predictor
  if (X_dense){
    mu = X * beta;
  } else {
    mu = csr_matrix_times_vector(N, K, X_data, X_idxs, X_indptr, beta);
  }

  if (fit_intercept) {
    mu = mu + alpha[1];
  }
