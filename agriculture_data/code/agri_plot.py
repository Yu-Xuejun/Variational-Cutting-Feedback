import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import scipy.stats as stats
import seaborn as sns
import matplotlib.mlab as mlab


cut_MCMC = np.load("cut_second.npy")
gamma_MCMC = cut_MCMC[:,0]
gamma_mu_VB = torch.load('gamma_mu.pt')
gamma_sd_VB = torch.load('gamma_sd.pt')

sns.distplot(gamma_MCMC, hist = False, kde = True,
            kde_kws = {'linewidth': 3}, label = 'MCMC')
x = np.linspace(-1, 1, 1000)
plt.plot(x, stats.norm.pdf(x, gamma_mu_VB, gamma_sd_VB),linewidth=3,label = 'Variational approximation')
plt.ylim((0,3.8))
plt.ylabel("Density", fontsize=20)
plt.xlabel("$\gamma$", fontsize=20)
plt.legend(prop={'size': 15}, loc ='upper left')
plt.savefig('gamma.pdf')


plt.clf()
full_Phi = np.load("Phi_sample.npy")
full_gamma_MCMC = full_Phi[:,0]
full_gamma_mu_VB = torch.load('full_gamma_mu_adam.pt')
full_gamma_sd_VB = torch.load('full_gamma_sd_adam.pt')

sns.distplot(full_gamma_MCMC, hist = False, kde = True,
            kde_kws = {'linewidth': 3}, label = 'SMC')
x = np.linspace(-5, 2, 1000)
plt.plot(x, stats.norm.pdf(x, full_gamma_mu_VB, full_gamma_sd_VB),linewidth=3,label = 'Variational approximation')
plt.ylim((0,0.9))
plt.ylabel("Density", fontsize=20)
plt.xlabel("$\gamma$", fontsize=20)
plt.legend(prop={'size': 15}, loc ='upper left')
plt.savefig('gamma_full_comp.pdf')


#conflict check
M_A_cut = torch.load('M_A_history_23.pt')
M_A_cut = M_A_cut[:, :, -10000:]
M_A_full = torch.load('M_A_history_full.pt')
M_A_full = M_A_full[:, :, -10000:]

cut1 = (10000 - np.sum(M_A_cut,(1,2)))/10000
cut2 = np.sum(M_A_cut[:,0,:],1)/10000
cut3 = np.sum(M_A_cut[:,1,:],1)/10000
full1 = (10000 - np.sum(M_A_full,(1,2)))/10000
full2 = np.sum(M_A_full[:,0,:],1)/10000
full3 = np.sum(M_A_full[:,1,:],1)/10000


plt.clf()
plt.scatter(cut1, full1)
plt.xlim((0,1))
plt.ylim((0,1))
plt.ylabel("Full",fontsize=18)
plt.xlabel("Cut",fontsize=18)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints,linestyle='-', color='k', lw=3, scalex=False, scaley=False)
plt.savefig('check1.pdf')
plt.clf()
plt.scatter(cut2, full2)
plt.xlim((0,1))
plt.ylim((0,1))
plt.ylabel("Full",fontsize=20)
plt.xlabel("Cut",fontsize=20)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints,linestyle='-', color='k', lw=3, scalex=False, scaley=False)
plt.savefig('check2.pdf')
plt.clf()
plt.scatter(cut3, full3)
plt.xlim((0,1))
plt.ylim((0,1))
plt.ylabel("Full",fontsize=20)
plt.xlabel("Cut",fontsize=20)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints,linestyle='-', color='k', lw=3, scalex=False, scaley=False)
plt.savefig('check3.pdf')


