import numpy as np
from numpy import zeros
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(3)

n1 = 100
n2 = 1000
lambda1 = 1
lambda2 = 100
theta1_true = 0
theta2_true = 1
y1 = np.random.normal(theta1_true, 1, n1)
y2 = np.random.normal(theta2_true, 1, n2)


E = 100
# Initializing
mu_q_theta1 = 0
sigmasq_q_theta1 = 1
mu_q_theta2 = 0
sigmasq_q_theta2 = 1
q_theta1 = zeros([E,2])
q_theta2 = zeros([E,2])
LB = zeros([E,1])
q_theta1_cut = zeros([E,2])
q_theta2_cut = zeros([E,2])
LB2 = zeros([E,1])


#full posterior
for e in range(E):
    sigmasq_q_theta2 = 1 / (lambda2 + n2)
    mu_q_theta2 = sigmasq_q_theta2 * n2 * (np.average(y2) - mu_q_theta1)
    sigmasq_q_theta1 = 1 / (lambda1 + n1 + n2)
    mu_q_theta1 = sigmasq_q_theta1 * (np.sum(y1) + np.sum(y2) - n2 * mu_q_theta2)

    q_theta1[e, :] = [mu_q_theta1, sigmasq_q_theta1]
    q_theta2[e, :] = [mu_q_theta2, sigmasq_q_theta2]

    LB[e] = -(n1 + n2 + 2) / 2 * math.log(2 * math.pi) + math.log(lambda1) / 2 + math.log(lambda2) / 2 \
            - (np.sum(np.square(y1)) - 2 * np.sum(y1) * mu_q_theta1 + n1 * (sigmasq_q_theta1 + np.square(mu_q_theta1)))/2 \
            - (np.sum(np.square(y2)) - 2 * np.sum(y2) * (mu_q_theta1 + mu_q_theta2) \
               + n2 * (sigmasq_q_theta1 + np.sum(mu_q_theta1) + sigmasq_q_theta2 + np.sum(mu_q_theta2)) + 2 * n2 * mu_q_theta1 * mu_q_theta2)/2 \
            + math.log(sigmasq_q_theta1) / 2 + math.log(sigmasq_q_theta2) / 2 + math.log(2 * math.pi) + 1

#plt.plot(q_theta2[:,0], label='theta1_mu')
#plt.plot(q_theta2[:,1], label='theta1_sigmasq')
#plt.plot(LB, label='lower bound')
#plt.legend()
#plt.show()

sigmasq_q_theta1_cut = 1 / (lambda1 + n1)
mu_q_theta1_cut = sigmasq_q_theta1_cut * np.sum(y1)

#cut posterior
for e in range(E):
    sigmasq_q_theta2_cut = 1 / (lambda2 + n2)
    mu_q_theta2_cut = sigmasq_q_theta2_cut * n2 * (np.average(y2) - mu_q_theta1_cut)

    q_theta1_cut[e, :] = [mu_q_theta1_cut, sigmasq_q_theta1_cut]
    q_theta2_cut[e, :] = [mu_q_theta2_cut, sigmasq_q_theta2_cut]

    LB2[e] = -(n1 + n2 + 2) / 2 * math.log(2 * math.pi) + math.log(lambda1) / 2 + math.log(lambda2) / 2 \
            - (np.sum(np.square(y1)) - 2 * np.sum(y1) * mu_q_theta1_cut + n1 * (sigmasq_q_theta1_cut + np.square(mu_q_theta1_cut)))/2 \
            - (np.sum(np.square(y2)) - 2 * np.sum(y2) * (mu_q_theta1_cut + mu_q_theta2_cut) \
               + n2 * (sigmasq_q_theta1_cut + np.sum(mu_q_theta1_cut) + sigmasq_q_theta2_cut + np.sum(mu_q_theta2_cut)) + 2 * n2 * mu_q_theta1_cut * mu_q_theta2_cut)/2 \
            + math.log(sigmasq_q_theta1_cut) / 2 + math.log(sigmasq_q_theta2_cut) / 2 + math.log(2 * math.pi) + 1


#exact posterior
Sigma_inv = np.array([[n1+lambda1+n2, n2], [n2, n2+lambda2]])
Sigma = np.linalg.inv(Sigma_inv)
b = np.array([[n1*np.mean(y1)+n2*np.mean(y2)],[n2*np.mean(y2)]])
mu = np.dot(Sigma,b)
mu_cut1 = n1*np.mean(y1)/(n1+lambda1)
mu_cut2 = mu[1,] + Sigma[0,1] * Sigma[0,0]**(-1) * (mu_cut1 - mu[0,])
mu_cut = np.array([[mu_cut1],mu_cut2])
Sigma_cut11 = 1/(n1+lambda1)
Sigma_cut22 = Sigma[1,1] - Sigma[0,1]**2 /Sigma[0,0] + (n2/(n2+lambda2))**2 * Sigma_cut11
Sigma_cut12 = -n2/(n2+lambda2) * Sigma_cut11
Sigma_cut = np.array([[Sigma_cut11, Sigma_cut12],[Sigma_cut12,Sigma_cut22]])


#lam = np.array([mu_q_theta1_cut,sigmasq_q_theta1_cut,mu_q_theta2_cut,sigmasq_q_theta2_cut])
x = np.linspace(-2, 2, 10000)
q1_cut = stats.norm.pdf(x, mu_q_theta1_cut, np.sqrt(sigmasq_q_theta1_cut))
q2_cut = stats.norm.pdf(x, mu_q_theta2_cut, np.sqrt(sigmasq_q_theta2_cut))
q1_full = stats.norm.pdf(x, mu_q_theta1, np.sqrt(sigmasq_q_theta1))
q2_full = stats.norm.pdf(x, mu_q_theta2, np.sqrt(sigmasq_q_theta2))
exact_cut_theta1 = stats.norm.pdf(x, mu_cut1, np.sqrt(Sigma_cut11))
exact_cut_theta2 = stats.norm.pdf(x, mu_cut2, np.sqrt(Sigma_cut22))
exact_full_theta1 = stats.norm.pdf(x, mu[0,], np.sqrt(Sigma[0,0]))
exact_full_theta2 = stats.norm.pdf(x, mu[1,], np.sqrt(Sigma[1,1]))
plt.plot(x,q1_cut, label = 'variational cut', linestyle='-',color = "royalblue")
plt.plot(x,q1_full, label = 'variational full',linestyle='--',color = "royalblue")
plt.plot(x,exact_cut_theta1, label = 'exact cut', linestyle='-',color = "darkorange")
plt.plot(x,exact_full_theta1, label = 'exact full',linestyle='--',color = "darkorange")
plt.axvline(x=0, color ="black", label = 'true value')
plt.legend(prop={'size': 11.9},loc ='upper left')
plt.ylabel("Density",fontsize=20)
plt.xlabel("$\\varphi$",fontsize=20)
plt.xlim((-0.5,0.8))
plt.savefig('biased_marginal1.pdf')

plt.clf()
plt.plot(x,q2_cut, label = 'variational cut', linestyle='-', color = "royalblue")
plt.plot(x,q2_full, label = 'variational full',linestyle='--', color = "royalblue")
plt.plot(x,exact_cut_theta2, label = 'exact cut', linestyle='-', color = "darkorange")
plt.plot(x,exact_full_theta2, label = 'exact full',linestyle='--',color = "darkorange")
plt.axvline(x=1, color ="black",label = 'true value')
plt.legend(prop={'size': 11.9})
plt.ylabel("Density",fontsize=20)
plt.xlabel("$\eta$",fontsize=20)
plt.xlim((0.2,1.7))
plt.savefig('biased_marginal2.pdf')


# Perform the kernel density estimate
def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                 mux=0.0, muy=0.0, sigmaxy=0.0):
    Xmu = X-mux
    Ymu = Y-muy
    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom

plt.clf()
x = np.arange(-0.4, 0.7, 0.001)
y = np.arange(0.3, 1.3, 0.001)
X, Y = np.meshgrid(x, y)
exact_full_samples = bivariate_normal(X, Y, np.sqrt(Sigma[0,0]), np.sqrt(Sigma[1,1]), mu[0,], mu[1,], Sigma[0,1])
exact_cut_samples = bivariate_normal(X, Y, np.sqrt(Sigma_cut11), np.sqrt(Sigma_cut22), mu_cut1, mu_cut2, Sigma_cut12)
var_full_samples = bivariate_normal(X, Y, np.sqrt(sigmasq_q_theta1), np.sqrt(sigmasq_q_theta2), mu_q_theta1, mu_q_theta2, 0)
var_cut_samples = bivariate_normal(X, Y, np.sqrt(sigmasq_q_theta1_cut), np.sqrt(sigmasq_q_theta2_cut), mu_q_theta1_cut, mu_q_theta2_cut, 0)
CS1 = plt.contour(X, Y, exact_full_samples,9, label='Line 1',linestyles="dashed", linewidths=1, alpha= 1, colors='darkorange')
CS2 = plt.contour(X, Y, exact_cut_samples,9, label='Line 2',linewidths=1, alpha= 1, colors='darkorange')
CS3 = plt.contour(X, Y, var_full_samples,9, linestyles="dashed", linewidths=1, alpha= 1, colors='royalblue')
CS4 = plt.contour(X, Y, var_cut_samples,9, linewidths=1, alpha= 1,colors='royalblue')
plt.xlabel("$\\varphi$",fontsize=20)
plt.ylabel("$\eta$",fontsize=20)
labels = ['Exact full posterior', 'Exact cut posterior', 'Variational full posterior', 'Variational cut posterior']
CS1.collections[0].set_label(labels[0])
CS2.collections[0].set_label(labels[1])
CS3.collections[0].set_label(labels[2])
CS4.collections[0].set_label(labels[3])
plt.legend(prop={'size': 11.9})
plt.savefig('biased_contour.pdf')