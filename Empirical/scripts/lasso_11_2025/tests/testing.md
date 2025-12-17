Tests for the function used in the empirical part

- [x] Lasso Estimation gets the paramter values under correct DGP(1_testing_lasso_base_function.ipynb)

- [x] Lasso rolling window gets the right paramters when the parameters are unchanged over time (2_testing_lasso_rolling_window.ipynb)
    - [x] With only current time relationship (no lags) [there was an error in the function, so that we were not using the future x, but the current one X[end:end+1]] takes x[end] and not x[end+1]
    - [x] Create lagged fature works (reversed the function to align it with treating features as x_t, x_t-1, x_t-2 instead of x_t-2, x_t-1, x_t)
    - [x] With Lagged feature

- [ ] estimate_single_config works on simulated data
    - [x] compute_stage2_r_squared works on simulated data
    - [ ] generate actual plausible data

- [ ] check that standard errors are correct

# 1) estimating beta_t on regression of r_{t-1} on x_{t-2}
# 2) need two predictions: beta_t x_t and beta_{t-1} x_{t-1} to generate r_t
# but I think we are doing beta_t x_{t-1} and beta_{t-1} x_{t-2}