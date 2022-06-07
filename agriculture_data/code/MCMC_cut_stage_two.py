#75min
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import time
from tqdm import tqdm
import scipy.stats as stats
import seaborn as sns

np.random.seed(123456)
torch.manual_seed(123456)

# import data
data_arc = pd.read_csv('data_arc.csv')
data_mod = pd.read_csv('data_mod.csv')

# deal with dummy variables
arc_indv = label_binarize(data_arc['Category'], classes=["barley", "wheat"])
site_A = label_binarize(data_arc['Site'], classes=sorted(data_arc['Site'].drop_duplicates()))
X2_A = np.array(site_A)
S_A = np.array(data_arc['Size']).reshape(-1, 1)


# first stage results
MA_chain = np.load("MA_chain.npy")
x = np.linspace(60000, 100000, 1001).astype(int)
MA_MCMC = MA_chain[x,:,:]


def cdf_p_M_all(alpha, Phi):
    gamma = Phi[0]
    xi = Phi[3:8].reshape(-1,1)
    sigma_xi_tilde = Phi[8]
    sigma_xi  = 3.5*np.exp(sigma_xi_tilde)/(1+np.exp(sigma_xi_tilde))
    mid = alpha - gamma * S_A - sigma_xi * np.matmul(X2_A, xi)
    p = 1/(1+np.exp(-mid))
    return p

def log_p_M_all(M_A, Phi):
    alpha_low = Phi[1]
    alpha_med_tilde = Phi[2]
    ind_M_A = np.zeros((269,3,1))
    ind_M_A[:, 0, 0] = np.logical_and(np.logical_not(M_A[:,0]), np.logical_not(M_A[:,1]))
    ind_M_A[:, 1, 0] = np.logical_and(M_A[:, 0], np.logical_not(M_A[:, 1]))
    ind_M_A[:, 2, 0] = np.logical_and(np.logical_not(M_A[:, 0]), M_A[:, 1])
    alpha1 = alpha_low
    alpha2 = np.exp(alpha_med_tilde) + alpha1
    p_low = cdf_p_M_all(alpha1, Phi)
    p_med = cdf_p_M_all(alpha2, Phi)
    p_matrix = np.concatenate((np.log(p_low), np.log(1e-5 + p_med - p_low), np.log(1e-5 + 1 - p_med)), 1)
    p_matrix = np.expand_dims(p_matrix, axis=1)
    p_M = np.matmul(p_matrix, ind_M_A)
    p_M = np.squeeze(p_M, axis = 2)
    return p_M

def log_p_M(M_A, Phi):
    return np.sum(log_p_M_all(M_A, Phi))

def log_f_Phi(Phi, M_A):
    gamma = Phi[0]
    alpha_low = Phi[1]
    alpha_med_tilde = Phi[2]
    xi = Phi[3:8].reshape(-1,1)
    sigma_xi_tilde = Phi[8]
    f1 = -0.5/4 * gamma**2
    f2 = -0.5 / 1.5 * alpha_low ** 2
    f3 = -0.5 / 7 * (alpha_med_tilde+5) ** 2
    f4 = -0.5 * np.matmul(np.transpose(xi), xi)
    f5 = np.log(np.exp(sigma_xi_tilde)*(1+np.exp(sigma_xi_tilde))-np.exp(sigma_xi_tilde)**2)- 2*np.log(1+np.exp(sigma_xi_tilde))
    f6 = log_p_M(M_A, Phi)
    f = f1 + f2 + f3 + f4 + f5 + f6
    return f



T = 10000
G = 1000
J = 1000
C_Phi = torch.load('aver_C_Phi.pt')
Phi_chain = np.ones([T+1, 9])
Phi_var = np.matmul(C_Phi, np.transpose(C_Phi))
final_states = np.zeros([J,9])

start_time = time.time()
for j in tqdm(range(J)):
    M_A = np.squeeze(MA_MCMC[j,:,:])
    for t in range(T):
        Phi_candidate = np.random.multivariate_normal(Phi_chain[t, :], Phi_var)
        alpha_Phi = np.exp( log_f_Phi(Phi_candidate, M_A) - log_f_Phi(Phi_chain[t,:], M_A) )
        U = np.random.uniform(0, 1)
        if U <= min(1, alpha_Phi):
            Phi_chain[t + 1, :] = Phi_candidate
        else:
            Phi_chain[t + 1, :] = Phi_chain[t, :]
    final_states[j,:] = Phi_chain[-1,:]


print("--- %s seconds ---" % (time.time() - start_time))

# check convergence
fig, axes = plt.subplots(3, 1)
axes[0].plot(Phi_chain[:,0], label='gamma')
axes[0].legend()
axes[1].plot(Phi_chain[:,1], label='alpha_low')
axes[1].legend()
axes[2].plot(Phi_chain[:,2], label='alpha_med_tilde')
axes[2].legend()
plt.savefig('MCMC_Second_Stage1.pdf')

plt.clf()
fig, axes = plt.subplots(3, 1)
axes[0].plot(Phi_chain[:,3], label='xi')
axes[0].legend()
axes[1].plot(Phi_chain[:,8], label='sigma_xi_tilde')
axes[1].legend()
axes[2].plot(Phi_chain[:,7], label='xi')
axes[2].legend()
plt.savefig('MCMC_Second_Stage2.pdf')

np.save("cut_second.npy",final_states)


