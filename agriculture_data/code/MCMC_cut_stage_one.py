#9h45min+11h21min
import pandas as pd
from sklearn.preprocessing import label_binarize
import numpy as np
import scipy.stats
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.distributions import Categorical


# PYDEVD_USE_FRAME_EVAL=NO
np.random.seed(123456)
torch.manual_seed(123456)

# import data
data_arc = pd.read_csv('data_arc.csv')
data_mod = pd.read_csv('data_mod.csv')

# deal with dummy variables
arc_indv = label_binarize(data_arc['Category'], classes=["barley", "wheat"])
mod_indv = label_binarize(data_mod['Category'], classes=["barley", "wheat"])
site_A = label_binarize(data_arc['Site'], classes=sorted(data_arc['Site'].drop_duplicates()))
site_M = label_binarize(data_mod['Site'], classes=sorted(data_mod['Site'].drop_duplicates()))
mod_M = label_binarize(data_mod['ManureLevel'], classes=["low", "medium", "high"])

# prepare data
Z1 = np.array(data_arc['normd15N'])
Z2 = np.array(data_mod['normd15N'])
Z = np.concatenate((Z1, Z2)).reshape(-1, 1)
R_M = np.array(data_mod['Rainfall']).reshape(-1, 1)  # log_R_M
S_A = np.array(data_arc['Size']).reshape(-1, 1)
M_M = np.array(mod_M[:, 1:3])  # (268,2)
X2_A = np.array(site_A)
X2_1 = np.concatenate((site_A, np.zeros((269, 19))), 1)
X2_2 = np.concatenate((np.zeros((268, 5)), site_M), 1)
X2 = np.concatenate((X2_1, X2_2))
Category = np.concatenate((arc_indv, mod_indv))
min_Rainfall = np.array(data_arc['Rainfall_min'])
max_Rainfall = np.array(data_arc['Rainfall_max'])
mean_Rainfall = 0.5*(min_Rainfall + max_Rainfall)
mean_Rainfall = mean_Rainfall.reshape((-1,1))
var_Rainfall = (1/12)*(max_Rainfall - min_Rainfall)**2
Var_Rainfall = np.diag(var_Rainfall)

def design_matrix(M_A, R_A):
    R = np.concatenate((R_A, R_M)).reshape(-1, 1)
    M = np.concatenate((M_A, M_M))
    X1 = np.concatenate((np.ones((537, 1)), R, M), 1)
    return X1

def Variance(C, v, sigmasq, sigmasq_zeta, X2):
    D_v = (1 - C) + v * C
    V = np.diag(np.squeeze(D_v))
    Sigma_zeta = np.eye(24)*sigmasq_zeta
    Var = sigmasq * V + np.matmul(X2, np.matmul(Sigma_zeta, np.transpose(X2)))
    return Var

def sample_beta(M_A, R_A, C, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde, X2):
    v = np.exp(v_tilde)
    sigmasq = (5*np.exp(sigmasq_tilde)+0.1)/(1+np.exp(sigmasq_tilde))
    sigmasq_zeta = (5*np.exp(sigmasq_zeta_tilde)+0.1)/(1+np.exp(sigmasq_zeta_tilde))
    X1 = design_matrix(M_A, R_A)
    Var = Variance(C, v, sigmasq, sigmasq_zeta, X2)
    Sigma_beta = np.linalg.inv(np.matmul(np.matmul(np.transpose(X1),np.linalg.inv(Var)),X1) + 1/G * np.eye(4))
    mu_beta = np.matmul(np.matmul(Sigma_beta, np.matmul(np.transpose(X1),np.linalg.inv(Var))), Z)
    mu_beta = np.squeeze(mu_beta)
    beta = np.random.multivariate_normal(mu_beta, Sigma_beta)
    return beta

def sample_LRA(beta, M_A, C, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde, X2):
    v = np.exp(v_tilde)
    sigmasq = (5*np.exp(sigmasq_tilde)+0.1)/(1+np.exp(sigmasq_tilde))
    sigmasq_zeta = (5*np.exp(sigmasq_zeta_tilde)+0.1)/(1+np.exp(sigmasq_zeta_tilde))
    Var = Variance(C, v, sigmasq, sigmasq_zeta, X2)
    Var = Var[:269,:269]
    MA1 = M_A[:,0].reshape(-1,1)
    MA2 = M_A[:,1].reshape(-1,1)
    Z_A = Z[:269,:]
    Sigma_LRA = np.linalg.inv(np.linalg.inv(Var_Rainfall) + beta[1,:]**2 * np.linalg.inv(Var))
    mu_LRA = np.matmul(Sigma_LRA, np.matmul(beta[1,:]*np.linalg.inv(Var), (Z_A - beta[0,:]*np.ones((269,1)) - beta[2,:]*MA1 - beta[3,:]*MA2)) + np.matmul(np.linalg.inv(Var_Rainfall), mean_Rainfall))
    mu_LRA = np.squeeze(mu_LRA)
    C_Sigma = np.linalg.cholesky(Sigma_LRA)
    epsilon = np.random.multivariate_normal(np.zeros([269]), np.eye(269))
    LRA = mu_LRA + np.matmul(C_Sigma, epsilon)
    return LRA

def cond_prob_M_A(beta, M_A, R_A, C, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde, X2):
    v = np.exp(v_tilde)
    sigmasq = (5*np.exp(sigmasq_tilde)+0.1)/(1+np.exp(sigmasq_tilde))
    sigmasq_zeta = (5*np.exp(sigmasq_zeta_tilde)+0.1)/(1+np.exp(sigmasq_zeta_tilde))
    Var = Variance(C, v, sigmasq, sigmasq_zeta, X2)
    Var = Var[:269,:269]
    D_v = np.diag(Var).reshape(-1,1)
    X1 = design_matrix(M_A, R_A)
    A = Z - np.matmul(X1, beta)
    A = A[:269,:]
    log_density = - (0.5/sigmasq) * A**2/D_v
    return log_density

M_A_low = np.zeros((269, 2))
M_A_med = np.concatenate((np.ones((269, 1)), np.zeros((269, 1))), 1)
M_A_high = np.concatenate((np.zeros((269, 1)), np.ones((269, 1))), 1)

def sample_M_A(beta, R_A, C, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde, X2):
    log_p1 = cond_prob_M_A(beta, M_A_low, R_A, C, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde, X2)
    log_p2 = cond_prob_M_A(beta, M_A_med, R_A, C, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde, X2)
    log_p3 = cond_prob_M_A(beta, M_A_high, R_A, C, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde, X2)
    p1 = 1 / (1 + np.exp(log_p2 - log_p1) + np.exp(log_p3 - log_p1))
    p2 = 1 / (1 + np.exp(log_p1 - log_p2) + np.exp(log_p3 - log_p2))
    p3 = 1 / (1 + np.exp(log_p1 - log_p3) + np.exp(log_p2 - log_p3))
    p = np.concatenate((p1,p2,p3),1)
    normal_p = p / np.sum(p, 1).reshape(-1,1)
    normal_p = torch.from_numpy(normal_p)
    dist = Categorical(probs=normal_p)
    sample = dist.sample()
    arc_M = label_binarize(sample, classes=[0, 1, 2])
    M_A = arc_M[:, 1:3]
    return M_A

def prior_unif(sigmasq_tilde):
    f = (5*np.exp(sigmasq_tilde)*(1+np.exp(sigmasq_tilde)) - (5*np.exp(sigmasq_tilde)+0.1)*np.exp(sigmasq_tilde))/(4.9*(1+np.exp(sigmasq_tilde))**2)
    return f

def data_loglikelihood(beta, M_A, R_A, C, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde, X2):
    v = np.exp(v_tilde)
    sigmasq = (5*np.exp(sigmasq_tilde)+0.1)/(1+np.exp(sigmasq_tilde))
    sigmasq_zeta = (5*np.exp(sigmasq_zeta_tilde)+0.1)/(1+np.exp(sigmasq_zeta_tilde))
    X1 = design_matrix(M_A, R_A)
    Var = Variance(C, v, sigmasq, sigmasq_zeta, X2)
    # sign, logdet
    f1 = np.linalg.slogdet(Var)[0]* np.linalg.slogdet(Var)[1] * (-1/2)
    A =  Z-np.matmul(X1, beta)
    f2 = -0.5*np.matmul(np.matmul(np.transpose(A),np.linalg.inv(Var)),A)
    f = f1 + f2
    return f

def log_f_sigmasq_tilde(sigmasq_tilde,beta, M_A, R_A, C, v_tilde,sigmasq_zeta_tilde, X2):
    f1 = np.log(prior_unif(sigmasq_tilde))
    f2 = data_loglikelihood(beta, M_A, R_A, C, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde, X2)
    f = f1 + f2
    return f

def log_f_sigmasq_zeta_tilde(sigmasq_zeta_tilde, beta, M_A, R_A, C, v_tilde, sigmasq_tilde, X2):
    f1 = np.log(prior_unif(sigmasq_zeta_tilde))
    f2 = data_loglikelihood(beta, M_A, R_A, C, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde, X2)
    f = f1 + f2
    return f

def log_f_v_tilde(v_tilde,beta, M_A, R_A, C,sigmasq_tilde, sigmasq_zeta_tilde, X2):
    f1 = -0.5/G * v_tilde**2
    f2 = data_loglikelihood(beta, M_A, R_A, C, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde, X2)
    f = f1 + f2
    return f



T = 100000
G = 1000

start_time = time.time()

for k in range(2):
    if k == 0:
        beta_chain = np.zeros([T+1,4])
        LRA_chain = np.ones([T+1,269])*0.1
        sigmasq_chain = np.ones([T+1,1])*(-0.5)
        MA_chain = np.zeros([T+1, 269, 2])
        sigmasq_zeta_chain = np.ones([T+1,1])*(-1.5)
        v_chain = np.ones([T+1,1])*2
    elif k == 1:
        beta_chain = np.ones([T+1,4])
        LRA_chain = np.ones([T+1,269])*0.7
        sigmasq_chain = np.ones([T+1,1])*(0.5)
        MA_chain = np.ones([T+1, 269, 2])
        sigmasq_zeta_chain = np.ones([T+1,1])*(-0.5)
        v_chain = np.ones([T+1,1])*1

    n_reject_sigmasq = 0
    n_reject_sigmasq_zeta = 0
    n_reject_v = 0
    C_rho = torch.load('C_rho_23.pt')
    sigmasq_tilde_var = np.abs(C_rho[4,4])
    sigmasq_zeta_tilde_var = np.abs(C_rho[5,5])
    v_tilde_var = np.abs(C_rho[6,6])

    for t in tqdm(range(T)):
        beta_candidate = sample_beta(np.squeeze(MA_chain[t,:,:]), LRA_chain[t,:].reshape(-1,1), Category, v_chain[t,:], sigmasq_chain[t,:], sigmasq_zeta_chain[t,:], X2)
        beta_chain[t+1,:] = beta_candidate
        LRA_candidate = sample_LRA(beta_chain[t+1,:].reshape(-1,1),np.squeeze(MA_chain[t,:,:]), Category, v_chain[t,:], sigmasq_chain[t,:], sigmasq_zeta_chain[t,:], X2)
        LRA_chain[t + 1, :] = LRA_candidate
        sigmasq_candidate = np.random.normal(sigmasq_chain[t,:], sigmasq_tilde_var)
        alpha_sigmasq = np.exp(log_f_sigmasq_tilde(sigmasq_candidate,beta_chain[t+1,:].reshape(-1,1), np.squeeze(MA_chain[t,:,:]), LRA_chain[t+1,:].reshape(-1,1), Category, v_chain[t,:],sigmasq_zeta_chain[t,:], X2)\
                        -log_f_sigmasq_tilde(sigmasq_chain[t,:],beta_chain[t+1,:].reshape(-1,1), np.squeeze(MA_chain[t,:,:]), LRA_chain[t+1,:].reshape(-1,1), Category, v_chain[t,:],sigmasq_zeta_chain[t,:], X2))
        U = np.random.uniform(0, 1)
        if U <= min(1, alpha_sigmasq):
            sigmasq_chain[t+1,:] = sigmasq_candidate
        else:
            sigmasq_chain[t+1,:] = sigmasq_chain[t,:]
            n_reject_sigmasq = n_reject_sigmasq + 1
        MA_candidate = sample_M_A(beta_chain[t+1,:].reshape(-1,1), LRA_chain[t+1,:].reshape(-1,1),  Category, v_chain[t,:], sigmasq_chain[t+1,:], sigmasq_zeta_chain[t,:], X2)
        MA_chain[t+1,:,:] = MA_candidate
        sigmasq_zeta_candidate = np.random.normal(sigmasq_zeta_chain[t,:], sigmasq_zeta_tilde_var)
        alpha_sigmasq_zeta = np.exp(log_f_sigmasq_zeta_tilde(sigmasq_zeta_candidate,beta_chain[t+1,:].reshape(-1,1), np.squeeze(MA_chain[t+1,:,:]), LRA_chain[t+1,:].reshape(-1,1), Category, v_chain[t,:], sigmasq_chain[t+1,:], X2)\
                             -log_f_sigmasq_zeta_tilde(sigmasq_zeta_chain[t,:],beta_chain[t+1,:].reshape(-1,1), np.squeeze(MA_chain[t+1,:,:]), LRA_chain[t+1,:].reshape(-1,1), Category, v_chain[t,:], sigmasq_chain[t+1,:], X2))
        U = np.random.uniform(0, 1)
        if U <= min(1, alpha_sigmasq_zeta):
            sigmasq_zeta_chain[t+1,:] = sigmasq_zeta_candidate
        else:
            sigmasq_zeta_chain[t+1,:] = sigmasq_zeta_chain[t,:]
            n_reject_sigmasq_zeta = n_reject_sigmasq_zeta + 1
        v_candidate = np.random.normal(v_chain[t,:], v_tilde_var)
        alpha_v = np.exp(log_f_v_tilde(v_candidate,beta_chain[t+1,:].reshape(-1,1),np.squeeze(MA_chain[t+1,:,:]),LRA_chain[t+1,:].reshape(-1,1), Category, sigmasq_chain[t+1,:], sigmasq_zeta_chain[t+1,:], X2)\
                  -log_f_v_tilde(v_chain[t,:],beta_chain[t+1,:].reshape(-1,1),np.squeeze(MA_chain[t+1,:,:]),LRA_chain[t+1,:].reshape(-1,1), Category, sigmasq_chain[t+1,:], sigmasq_zeta_chain[t+1,:], X2))
        U = np.random.uniform(0, 1)
        if U <= min(1, alpha_v):
            v_chain[t+1,:] = v_candidate
        else:
            v_chain[t+1,:] = v_chain[t,:]
            n_reject_v = n_reject_v + 1

    if k == 0:
        beta_chain1 = beta_chain
        LRA_chain1 = LRA_chain
        sigmasq_chain1 = sigmasq_chain
        MA_chain1 = MA_chain
        sigmasq_zeta_chain1 = sigmasq_zeta_chain
        v_chain1 = v_chain
    elif k == 1:
        beta_chain2 = beta_chain
        LRA_chain2 = LRA_chain
        sigmasq_chain2 = sigmasq_chain
        MA_chain2 = MA_chain
        sigmasq_zeta_chain2 = sigmasq_zeta_chain
        v_chain2 = v_chain


print("--- %s seconds ---" % (time.time() - start_time))

fig, axes = plt.subplots(3, 1)
axes[0].plot(beta_chain1[:,0], label='beta1')
axes[0].plot(beta_chain2[:,0], label='beta2')
axes[0].legend()
axes[1].plot(LRA_chain1[:,0],label='LAR1')
axes[1].plot(LRA_chain2[:,0],label='LAR2')
axes[1].legend()
axes[2].plot(sigmasq_chain1, label='sigmasq1')
axes[2].plot(sigmasq_chain2, label='sigmasq2')
axes[2].legend()
plt.savefig('MCMC_TRY.pdf')

plt.clf()
fig, axes = plt.subplots(3, 1)
axes[0].plot(v_chain1, label='v1')
axes[0].plot(v_chain2, label='v2')
axes[0].legend()
axes[1].plot(sigmasq_zeta_chain1, label='sigmasq_zeta1')
axes[1].plot(sigmasq_zeta_chain2, label='sigmasq_zeta2')
axes[1].legend()
axes[2].plot(beta_chain1[:,1], label='beta1')
axes[2].plot(beta_chain2[:,1], label='beta2')
axes[2].legend()
plt.savefig('MCMC_TRY2.pdf')



np.save("beta_chain.npy",beta_chain1)
np.save("LRA_chain.npy",LRA_chain1)
np.save("sigmasq_chain.npy", sigmasq_chain1)
np.save("sigmasq_zeta_chain.npy", sigmasq_zeta_chain1)
np.save("v_chain.npy", v_chain1)
np.save("MA_chain.npy",MA_chain1)
