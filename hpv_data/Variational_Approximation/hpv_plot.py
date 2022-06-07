import numpy as np
import matplotlib.pyplot as plt
import csv



with open('hpv-full.csv','r') as full_MCMC:
    reader = csv.reader(full_MCMC)
    next(reader)
    theta21_full_MCMC = next(reader)
    theta22_full_MCMC = next(reader)
    theta21_full_MCMC = [float(i) for i in theta21_full_MCMC[1:]]
    theta22_full_MCMC = [float(i) for i in theta22_full_MCMC[1:]]

with open('hpv-cut.csv','r') as cut_MCMC:
    reader = csv.reader(cut_MCMC)
    next(reader)
    theta21_cut_MCMC = next(reader)
    theta22_cut_MCMC = next(reader)
    theta21_cut_MCMC = [float(i) for i in theta21_cut_MCMC[1:]]
    theta22_cut_MCMC = [float(i) for i in theta22_cut_MCMC[1:]]

mu_full = np.load('aver_mu_full.npy')
V_full = np.load('aver_V_full.npy')
mu_cut = np.load('aver_mu_cut.npy')
V_cut = np.load('aver_V_cut.npy')

mu_cut_trans = mu_cut
mu_cut_trans[0:13] = 1/(1 + np.exp(- mu_cut[0:13]))
np.save('mu_cut_trans.npy',mu_cut_trans)

mu_full_trans = mu_full
mu_full_trans[0:13] = 1/(1 + np.exp(- mu_full[0:13]))

def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                 mux=0.0, muy=0.0, sigmaxy=0.0):
    Xmu = X-mux
    Ymu = Y-muy
    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                        constrained_layout=False)
x = np.arange(-3.2, -0.5, 0.1)
y = np.arange(0, 40, 0.1)
X, Y = np.meshgrid(x, y)
var_full_samples = bivariate_normal(X, Y, np.sqrt(V_full[13,13]), np.sqrt(V_full[14,14]), mu_full[13], mu_full[14], V_full[13,14])
CS1 = ax1.contour(X, Y, var_full_samples,4, linewidths=1, alpha= 1, colors='royalblue')
ax1.scatter(theta21_full_MCMC, theta22_full_MCMC, alpha=0.3,label = 'MCMC full posterior',color='darkorange')
labels = ['Variational full posterior']
CS1.collections[0].set_label(labels[0])
ax1.legend(prop={'size': 10})
ax1.set_xlabel('$θ_{21}$')
ax1.set_ylabel('$θ_{22}$')
ax1.set_title("Joint Full Posterior")

x = np.arange(-3.2, -0.5, 0.1)
y = np.arange(0, 40, 0.1)
X, Y = np.meshgrid(x, y)
var_cut_samples = bivariate_normal(X, Y, np.sqrt(V_cut[13,13]), np.sqrt(V_cut[14,14]), mu_cut[13], mu_cut[14], V_cut[13,14])
CS2 = ax2.contour(X, Y, var_cut_samples,4, linewidths=1, alpha= 1, colors='royalblue')
ax2.scatter(theta21_cut_MCMC, theta22_cut_MCMC, alpha=0.3,label = 'MCMC cut posterior',color='darkorange')
labels = ['Variational cut posterior']
CS2.collections[0].set_label(labels[0])
ax2.set_xlabel('$θ_{21}$')
ax2.legend(prop={'size': 10})
ax2.set_title("Joint Cut Posterior")
plt.savefig('HPV_JOINT.pdf')

plt.clf()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6),
                        constrained_layout=False)
s_full = np.random.multivariate_normal(mu_full[13:].squeeze(), V_full[13:,13:], 4000)
s_cut = np.random.multivariate_normal(mu_cut[13:].squeeze(), V_cut[13:,13:], 4000)
ax1.scatter(s_full[:,0], s_full[:,1], alpha=0.6,label = 'full')
ax1.scatter(s_cut[:,0], s_cut[:,1], alpha=0.6,label = 'cut')
ax1.set_xlim(-3,-1)
ax1.set_ylim(3,40)
ax1.tick_params(axis='both', which='major', labelsize=13)
ax1.set_xlabel('$\eta_1$', fontsize=20)
ax1.set_ylabel('$\eta_2$', fontsize=20)
ax1.legend(prop={'size': 15})
ax1.set_title("Variational Joint Distribution",fontsize=22)

ax2.scatter(theta21_full_MCMC, theta22_full_MCMC, alpha=0.6,label = 'full')
ax2.scatter(theta21_cut_MCMC, theta22_cut_MCMC, alpha=0.6,label = 'cut')
ax2.set_xlim(-3,-1)
ax2.set_ylim(3,40)
ax2.tick_params(axis='both', which='major', labelsize=13)
ax2.set_xlabel('$\eta_1$', fontsize=20)
ax2.set_title("MCMC Joint Distribution",fontsize=22)
ax2.legend(prop={'size': 15})
plt.savefig('HPV_JOINT2.pdf')
#plt.show()