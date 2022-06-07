# 3.4h/55min /14min(1 chain)
rm(list = ls())
Sys.setenv(USE_CXX14 = 1)
library(rstan)

set.seed(0)
start_time <- Sys.time()

Y11 = c(7, 6, 10, 10, 1, 1, 10, 4, 35, 0, 10, 8, 4)
Y12 = c(111, 71, 162, 188, 145, 215, 166, 37, 173, 143, 229, 696, 93)
beta_a = 1 + Y11
beta_b = 1 + Y12 - Y11
theta1_sample_0 = matrix(data=NA, nrow=4000, ncol = 13)
for(i in 1:13){
  theta1_sample_0[,i] = rbeta(4000, beta_a[i], beta_b[i], ncp = 0)
}

# print(colMeans(theta1_sample_0))
# 
# hpv_data_cut1 <- list(
#   N = 13,
#   Y11 = c(7, 6, 10, 10, 1, 1, 10, 4, 35, 0, 10, 8, 4),
#   Y12 = c(111, 71, 162, 188, 145, 215, 166, 37, 173,
#           143, 229, 696, 93)
# )
# 
# 
# fit_cut1 <- stan(file = 'hpv_cut1.stan',
#             data = hpv_data_cut1,
#             iter = 2000, chains = 4)
# 
# e1 <- extract(fit_cut1,permuted = TRUE) # return a list of arrays
# 
# theta1_sample = e1$theta1  #[4000,13]
# print(colMeans(theta1_sample))
# print(colMeans(theta1_sample) - colMeans(theta1_sample_0))

hpv_data_cut2 <- list(
  N = 13,
  S = 4000,
  theta1 = theta1_sample_0,
  Y21 = c(16, 215, 362, 97, 76, 62, 710, 56, 133,28, 62, 413, 194),
  Y22 = c(26983, 250930, 829348, 157775, 150467, 352445, 553066,
          26751, 75815, 150302, 354993, 3683043, 507218)
)

fit_cut2 <- stan(file = 'hpv_cut2.stan', 
                 data = hpv_data_cut2,
                 iter = 2000, chains = 1)

e2 <- extract(fit_cut2,permuted = TRUE) # return a list of arrays

end_time <- Sys.time()
print(end_time - start_time)

theta2_sample = rbind(e2$theta21[1000,], e2$theta22[1000,])
plot(e2$theta21[-1,], e2$theta22[-1,],xlim=c(-4.5,3),ylim=c(0,40))
write.csv(theta2_sample,'hpv-cut.csv')
