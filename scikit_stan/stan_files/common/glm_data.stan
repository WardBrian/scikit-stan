  int<lower=0, upper=1> predictor; // 0: fitting run, 1: prediction run
  int<lower=0, upper=1> fit_intercept; // 0: no intercept, 1: intercept

  int<lower=0, upper=1> X_dense;
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors/features
  matrix[N*X_dense, K*X_dense] X;   // predictor matrix

  int<lower=0> X_nz; // number of non-zero elements in X
  vector[X_nz] X_data; // non-zero elements
  array[X_nz] int<lower=1, upper=K> X_idxs; // column indices for w_X
  // where the non-zeros start in each row of X
  array[X_dense ? 0 : (N + 1)] int<lower=1, upper=rows(X_data) + 1> X_indptr;
