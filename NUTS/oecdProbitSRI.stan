// No-U-Turn Sampling with Simple Random Intercept(SRI) Probit Regression Model

data {
    int N; // number of econonmic observation
    int C; // number of countries
    int K; // number of predictors
    int y[N]; // economic outcome
    matrix[N,5] x; // a matrix of predictors
    int g[N]; // map economic outcomes to countries
}
parameters {
    real alpha; // intercept parameter of the simple random intercept model
    real w[N];  // randomness parameter of the simple random intercept model
    vector[K] beta; // coefficients of the predictors
    real<lower=0,upper=10> sigma;  // standard error of the randomness parameter
}
model {
  alpha ~ normal(0,10); // a prior on alpha
  w ~ normal(0,sigma); // a hyperprior on the randomeness paramenter
  beta ~ normal(0,2.5); //a prior on beta
  for(n in 1:N) {
    // fitting the probit regression model on binary outcomes
    y[n] ~ bernoulli(Phi(alpha + w[g[n]] + x[n]*beta));
  }
}

