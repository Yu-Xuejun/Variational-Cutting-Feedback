data {            // data block for data declarations
  int<lower=0> N;
  int Y11[N];
  int Y12[N];
  int Y21[N];
  int Y22[N];
}
parameters {      // parameter block for parameter declarations
  //vector<lower=0, upper=1>[N] eta;
  real theta21;
  real theta22;
  vector<lower=0, upper=1>[N] theta1;
}
//transformed parameters  {
//   vector[N] theta1;
//   for (n in 1:N)
    //theta1[n] = 1/(1 + exp(-eta[n]));
//    theta1[n] = log(eta[n]/(1-eta[n]));
//}
model {  // model block: priors for parameters and model for data
  for (n in 1:N)
    theta1[n] ~ uniform(0,1);
    //eta[n] ~ uniform(0,1);
  theta21 ~ normal(0, sqrt(1000));
  theta22 ~ normal(0, sqrt(1000));
  Y11 ~ binomial(Y12, theta1);
  for (n in 1:N)
    Y21[n] ~ poisson(exp(theta21 + theta1[n] * theta22 + log(Y22[n] * 1e-3)));
}
