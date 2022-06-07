data {            // data block for data declarations
  int<lower=0> N;
  int<lower=0> S;
  matrix<lower=0, upper=1>[S, N] theta1;
  int Y21[N];
  int Y22[N];
}
parameters {      // parameter block for parameter declarations
  vector[S] theta21;
  vector[S] theta22;
}
model {  // model block: priors for parameters and model for data
  theta21 ~ normal(0, sqrt(1000));
  theta22 ~ normal(0, sqrt(1000));
  for (s in 1:S)
    for (n in 1:N)
      Y21[n] ~ poisson(exp(theta21[s] + theta1[s,n] * theta22[s] + log(Y22[n] * 1e-3)));
}
