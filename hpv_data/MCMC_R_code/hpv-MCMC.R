# 1.5min/32s
rm(list = ls())
Sys.setenv(USE_CXX14 = 1)
library(rstan)

set.seed(0)

hpv_data <- list(
  N = 13,
  Y11 = c(7, 6, 10, 10, 1, 1, 10, 4, 35, 0, 10, 8, 4),
  Y12 = c(111, 71, 162, 188, 145, 215, 166, 37, 173,
           143, 229, 696, 93),
  Y21 = c(16, 215, 362, 97, 76, 62, 710, 56, 133,28, 62, 413, 194),
  Y22 = c(26983, 250930, 829348, 157775, 150467, 352445, 553066,
           26751, 75815, 150302, 354993, 3683043, 507218)
)

initf1 <- function() {
  #list(theta21 = -5, theta22 = 20, theta1 = rep(0.1,13))
  list(theta21 = runif(1,-7,-3), theta22 = runif(1,18,22), theta1 = rep(0.1,13))
}

start_time <- Sys.time()
fit <- stan(file = 'hpv.stan', 
            data = hpv_data,
            init = initf1,
            warmup = 19000,
            iter = 20000, 
            chains = 4)
# extract samples
e <- extract(fit,permuted = TRUE) # return a list of arrays

traceplot(fit, pars = c("theta21", "theta22","theta1[9]"), inc_warmup = TRUE, nrow = 3)

end_time <- Sys.time()
print(end_time - start_time)

theta2_sample = rbind(e$theta21,e$theta22)
plot(e$theta21, e$theta22,xlim=c(-4.5,3),ylim=c(0,40))
write.csv(theta2_sample,'hpv-full.csv')

