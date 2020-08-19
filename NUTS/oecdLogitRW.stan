// No-U-Turn Sampling with Random Walk(RW) Logistic Regression Model

data {
    int N; // total number of observations
    int C; // number of countries
    int T; // number of time steps
    int K; // number of predictors
    int y[C,T]; // economic outcome
    matrix[N,5] x; // a matrix of predictors
    
}
parameters {
    real alpha[T]; // noise parameter for a random walk model 
    matrix[5,T] beta; // a matrix coefficients of the predictors 
    
}
model {
  alpha[1] ~ normal(0,10); // a prior on alpha at the first time step
  beta[1,] ~ normal(0,2.5); // a prior on beta at the first time step
  
  for (t in 2:T){
    alpha[t]~normal(alpha[t-1],10); // sampling alpha's for the rest of the time steps
    beta[,t]~normal(beta[,t-1],2.5); // sampling beta's for the remaining time steps
  }
  
  for(c in 1:C){ 
    for(t in 1:T) {
      // fitting the logistic regression model on binary outcomes
      y[c,t] ~ bernoulli(inv_logit(alpha[t] + x[((c-1)*T+1):(c*T),][t]*beta[,t]));
    }
  }
}


