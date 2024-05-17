//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

data {
  int<lower=0> N;  // Total number of households
  int<lower=0> J;  // Number of boroughs
  int<lower=1, upper=J> borough[N];  // Borough index for each household
  int<lower=0> K;  // Total number of predictors
  int<lower=0> K_r;  // Number of household-level predictors with random slopes
  int<lower=0> K_f;  // Number of household-level predictors with fixed effects
  matrix[N, K_r] X_household_random;  // Household-level predictors with random slopes
  matrix[N, K_f] X_fixed;  // Household-level predictors with fixed effects
  int<lower=0, upper=1> rodent2[N];  // Outcome: presence of rodents (0 or 1)
}

parameters {
  matrix[K_r, J] z_beta_household_random;  // Non-centered parameterization
  vector<lower=0>[K_r] sigma_beta_random;  // SDs of the random slopes
  vector[K_f] beta_fixed;  // Fixed effects predictors coefficient
  vector[J] z_alpha;  // Non-centered random intercepts
  real<lower=0> sigma_alpha;  // SD of borough intercepts
}

transformed parameters {
  matrix[K_r, J] beta_household_random;  // Transformed parameters for random slopes
  vector[J] alpha;  // Transformed random intercepts
  
  for (k in 1:K_r) {
    for (j in 1:J) {
      beta_household_random[k, j] = sigma_beta_random[k] * z_beta_household_random[k, j];
    }
  }
  
  for (j in 1:J) {
    alpha[j] = sigma_alpha * z_alpha[j];
  }
}

model {
  // Priors
  for (k in 1:K_r) {
    for (j in 1:J) {
      z_beta_household_random[k, j] ~ normal(0, 1);  // Standard normal for non-centered parameterization
    }
  }
  sigma_beta_random ~ cauchy(0, 2);  // Prior for the SDs of the random slopes
  beta_fixed ~ normal(0, 2.5);  // Priors for fixed effects
  z_alpha ~ normal(0, 1);  // Standard normal for non-centered parameterization
  sigma_alpha ~ cauchy(0, 2);  // Prior for the SD of borough intercepts

  // Likelihood
  for (n in 1:N) {
    real household_effects_random = dot_product(X_household_random[n, :], beta_household_random[:, borough[n]]);  // Random effects
    real effects_fixed = dot_product(X_fixed[n, :], beta_fixed);  // Fixed effects
    real logit_p = alpha[borough[n]] + household_effects_random + effects_fixed;
    rodent2[n] ~ bernoulli_logit(logit_p);
  }
}

generated quantities {
  vector[N] y_pred;
  for (n in 1:N) {
    real household_effects_random = dot_product(X_household_random[n, :], beta_household_random[:, borough[n]]);
    real effects_fixed = dot_product(X_fixed[n, :], beta_fixed);
    y_pred[n] = bernoulli_logit_rng(alpha[borough[n]] + household_effects_random + effects_fixed);
  }
}

