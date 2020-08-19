// No-U-Turn Sampling with Weighted- Logistic Regression Model

data {
  int <lower=0> N; // total number of observation
  matrix[N,6] x; // a matrix of predictors (time inclusive)
  int<lower=0,upper=1> y[N]; // economic outcome
}

parameters {
  real alpha; // intercept parameter of the weighted-time intercept model
  vector[6] beta; // coefficients of the predictors
  
}
model { 
  
  alpha~normal(0,10); // a prior on alpha
  beta~normal(0,2.5); // a prior on beta
  
  for(n in 1:N){
    // fitting the logit regression model on binary outcomes
    y[n]~bernoulli_logit(alpha+x[n]*beta);
  }
}

generated quantities{
  // for model selection
  vector[N] log_lik;
  for (n in 1:N){
    log_lik[n]=bernoulli_logit_lpmf(y[n] | alpha+x[n]*beta);
  }
}






