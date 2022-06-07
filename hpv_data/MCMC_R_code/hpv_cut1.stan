data {            // data block for data declarations
  int<lower=0> N;
  int Y11[N];
  int Y12[N];
}
parameters {      // parameter block for parameter declarations
  vector<lower=0, upper=1>[N] theta1;
}
model {  // model block: priors for parameters and model for data
  for (n in 1:N)
    theta1[n] ~ uniform(0,1);
  Y11 ~ binomial(Y12, theta1);
}
