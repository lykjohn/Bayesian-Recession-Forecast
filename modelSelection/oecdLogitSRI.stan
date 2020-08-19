// No-U-Turn Sampling with Simple Random Intercept(SRI) Logistic Regression Model

data {
    int N; // total number of observation
    int C; // number of countries
    int K; // number of predictors
    int y[N]; // economic outcome
    matrix[N,5] x; // a matrix of predictors
    int g[N]; // map economic outcomes to countries
}
parameters {
    real alpha; // intercept parameter of the simple random intercept model
    real w[N]; // randomness parameter of the simple random intercept model
    vector[K] beta; // coefficients of the predictors
    real<lower=0,upper=10> sigma; // standard error of the randomness parameter
}
model {
  alpha ~ normal(0,10); // a prior on alpha
  w ~ normal(0,sigma); // a hyperprior on the randomeness paramenter
  beta ~ normal(0,2.5); //a prior on beta
  for(n in 1:N) {
    // fitting the logit regression model on binary outcomes
    y[n] ~ bernoulli(inv_logit(alpha + w[g[n]] + x[n]*beta));
  }
}

generated quantities{
  vector[N] log_lik;
  for (n in 1:N){
    log_lik[n] = bernoulli_logit_lpmf(y[n] | alpha + w[g[n]] + x[n]*beta);
  }
}
