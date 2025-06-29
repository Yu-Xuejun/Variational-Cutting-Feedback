import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# "cut_second.npy" file saves 1000 MCMC samples of the cut posterior eta
cut_MCMC = np.load("cut_second.npy")
gamma_MCMC = cut_MCMC[:,0]

# "Phi_sample.npy" file saves 1000 SMC samples of the cut posterior eta
full_eta = np.load("Phi_sample.npy")  # I called eta Phi in my code. eta and Phi are exactly the same thing.
full_gamma_MCMC = full_eta[:,0]

print(cut_MCMC.shape)
print(full_eta.shape)

# both the samples have shape 1000*9,
# with 1000 the number of samples, and 9 the dimension of vector eta
# eta = (\gamma, \alpha_{low}, \alpha_{med}, \xi(dim=5), \sigma_\xi)



######################################################
# beta samples -- full model
# full_beta_MCMC has size 1000*4, and contains 1000 samples of beta_1 to beta_4
# you can save "full_beta_MCMC" for future use.
full_rho = np.load("rho_sample_20240929.npy")
full_beta_MCMC = full_rho[:,:4]
print(full_beta_MCMC.shape)


######################################################
# beta samples -- cut model
# cut_beta_MCMC has size 1000*4, and contains 1000 samples of beta_1 to beta_4
# you can save "cut_beta_MCMC" for future use.
cut_beta_chain1 = np.load("beta_chain1_20241005.npy")
cut_beta_chain2 = np.load("beta_chain2_20241005.npy")
cut_beta_chain1 = cut_beta_chain1[-10000:]  # keep the last 10000 samples
cut_beta_chain2 = cut_beta_chain2[-10000:]  # keep the last 10000 samples
cut_beta_chain1 = cut_beta_chain1[::20]     # thin the sample every 20 and keep 500 samples
cut_beta_chain2 = cut_beta_chain2[::20]     # thin the sample every 20 and keep 500 samples
cut_beta_MCMC = np.concatenate((cut_beta_chain1, cut_beta_chain2))
print(cut_beta_MCMC.shape)


######################################################
# here is the density plots of beta_1 to beta_4 samples from full model
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
sns.distplot(full_beta_MCMC[:,0], hist = False, kde = True,
            kde_kws = {'linewidth': 3}, label = 'MCMC')
plt.xlabel("beta_1_full", fontsize=20)
plt.subplot(2, 2, 2)
sns.distplot(full_beta_MCMC[:,1], hist = False, kde = True,
            kde_kws = {'linewidth': 3}, label = 'MCMC')
plt.xlabel("beta_2_full", fontsize=20)
plt.subplot(2, 2, 3)
sns.distplot(full_beta_MCMC[:,2], hist = False, kde = True,
            kde_kws = {'linewidth': 3}, label = 'MCMC')
plt.xlabel("beta_3_full", fontsize=20)
plt.subplot(2, 2, 4)
sns.distplot(full_beta_MCMC[:,3], hist = False, kde = True,
            kde_kws = {'linewidth': 3}, label = 'MCMC')
plt.xlabel("beta_4_full", fontsize=20)
plt.show()

######################################################
# here is the density plots of beta_1 to beta_4 samples from cut model
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
sns.distplot(cut_beta_MCMC[:,0], hist = False, kde = True,
            kde_kws = {'linewidth': 3}, label = 'MCMC')
plt.xlabel("beta_1_cut", fontsize=20)
plt.subplot(2, 2, 2)
sns.distplot(cut_beta_MCMC[:,1], hist = False, kde = True,
            kde_kws = {'linewidth': 3}, label = 'MCMC')
plt.xlabel("beta_2_cut", fontsize=20)
plt.subplot(2, 2, 3)
sns.distplot(cut_beta_MCMC[:,2], hist = False, kde = True,
            kde_kws = {'linewidth': 3}, label = 'MCMC')
plt.xlabel("beta_3_cut", fontsize=20)
plt.subplot(2, 2, 4)
sns.distplot(cut_beta_MCMC[:,3], hist = False, kde = True,
            kde_kws = {'linewidth': 3}, label = 'MCMC')
plt.xlabel("beta_4_cut", fontsize=20)
plt.show()



