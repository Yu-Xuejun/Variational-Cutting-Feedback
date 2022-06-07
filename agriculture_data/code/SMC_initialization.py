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
torch.set_default_tensor_type(torch.DoubleTensor)

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
Z1 = np.array(data_arc['normd15N']).reshape(-1,1)
Z2 = np.array(data_mod['normd15N']).reshape(-1,1)
R_M = np.array(data_mod['Rainfall']).reshape(-1, 1)  # log_R_M
S_A = np.array(data_arc['Size']).reshape(-1, 1)
M_M = np.array(mod_M[:, 1:3])  # (268,2)
X2_A = np.array(site_A)
X2_B = np.array(site_M)
Category = np.concatenate((arc_indv, mod_indv))

# Phi [1000,9]-------------------------------------------------
H = 1000
G = 1000
#Phi_1 = np.random.normal(0,np.sqrt(G), size=(H,8))
gamma = np.random.normal(0,np.sqrt(4), size=(H,1))
alpha_low = np.random.normal(0,np.sqrt(1.5), size=(H,1))
alpha_med_tilde = np.random.normal(-5,np.sqrt(7), size=(H,1))
xi = np.random.normal(0,np.sqrt(1), size=(H,5))
Phi_2 = np.random.uniform(0,3.5,size=(H,1))
Phi_ini = np.concatenate((gamma, alpha_low, alpha_med_tilde,xi,Phi_2),1)


# MA [1000,269,2]----------------------------------------------
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

M_A_low = np.zeros((269, 2))
M_A_med = np.concatenate((np.ones((269, 1)), np.zeros((269, 1))), 1)
M_A_high = np.concatenate((np.zeros((269, 1)), np.ones((269, 1))), 1)

def sample_M_A(Phi):
    log_p1 = log_p_M_all(M_A_low, Phi)
    log_p2 = log_p_M_all(M_A_med, Phi)
    log_p3 = log_p_M_all(M_A_high, Phi)
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

MA_ini = np.zeros((H, 269, 2))

for i in range(H):
    Phi = np.squeeze(Phi_ini[i, :])
    MA_ini[i,:,:] = sample_M_A(Phi)

# LRA [1000, 269]
min_Rainfall = np.array(data_arc['Rainfall_min'])
max_Rainfall = np.array(data_arc['Rainfall_max'])
mean_Rainfall = 0.5*(min_Rainfall + max_Rainfall)
#mean_Rainfall = mean_Rainfall.reshape((-1,1))
var_Rainfall = (1/12)*(max_Rainfall - min_Rainfall)**2
Var_Rainfall = np.diag(var_Rainfall)
LRA_ini = np.random.multivariate_normal(mean_Rainfall, Var_Rainfall, size=H)

# rho_ini [1000, 7]
def Variance(C, X2, v, sigmasq, sigmasq_zeta):
    D_v = (1 - C) + v * C
    V = np.diag(np.squeeze(D_v))
    Sigma_zeta = np.eye(19)*sigmasq_zeta
    Var = sigmasq * V + np.matmul(X2, np.matmul(Sigma_zeta, np.transpose(X2)))
    return Var

def sample_beta(Z, C, X2, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde):
    v = np.exp(v_tilde)
    sigmasq = (5*np.exp(sigmasq_tilde)+0.1)/(1+np.exp(sigmasq_tilde))
    sigmasq_zeta = (5*np.exp(sigmasq_zeta_tilde)+0.1)/(1+np.exp(sigmasq_zeta_tilde))
    X1_mod = np.concatenate((np.ones((268,1)),R_M, M_M),1)
    Var = Variance(C, X2, v, sigmasq, sigmasq_zeta)
    Sigma_beta = np.linalg.inv(np.matmul(np.matmul(np.transpose(X1_mod),np.linalg.inv(Var)),X1_mod) + 1/G * np.eye(4))
    mu_beta = np.matmul(np.matmul(Sigma_beta, np.matmul(np.transpose(X1_mod),np.linalg.inv(Var))), Z)
    mu_beta = np.squeeze(mu_beta)
    beta = np.random.multivariate_normal(mu_beta, Sigma_beta)
    return beta

def prior_unif(sigmasq_tilde):
    f = (5*np.exp(sigmasq_tilde)*(1+np.exp(sigmasq_tilde)) - (5*np.exp(sigmasq_tilde)+0.1)*np.exp(sigmasq_tilde))/(4.9*(1+np.exp(sigmasq_tilde))**2)
    return f

def data_loglikelihood(Z, C, X2, beta, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde):
    v = np.exp(v_tilde)
    sigmasq = (5*np.exp(sigmasq_tilde)+0.1)/(1+np.exp(sigmasq_tilde))
    sigmasq_zeta = (5*np.exp(sigmasq_zeta_tilde)+0.1)/(1+np.exp(sigmasq_zeta_tilde))
    X1 = np.concatenate((np.ones((268,1)),R_M, M_M),1)
    Var = Variance(C, X2, v, sigmasq, sigmasq_zeta)
    # sign, logdet
    f1 = np.linalg.slogdet(Var)[0]* np.linalg.slogdet(Var)[1] * (-1/2)
    A =  Z-np.matmul(X1, beta)
    f2 = -0.5*np.matmul(np.matmul(np.transpose(A),np.linalg.inv(Var)),A)
    f = f1 + f2
    return f

def log_f_sigmasq_tilde(sigmasq_tilde, Z, C, X2, beta, v_tilde,sigmasq_zeta_tilde):
    f1 = np.log(prior_unif(sigmasq_tilde))
    f2 = data_loglikelihood(Z, C, X2, beta, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde)
    f = f1 + f2
    return f

def log_f_sigmasq_zeta_tilde(sigmasq_zeta_tilde, Z, C, X2, beta, v_tilde, sigmasq_tilde):
    f1 = np.log(prior_unif(sigmasq_zeta_tilde))
    f2 = data_loglikelihood(Z, C, X2, beta, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde)
    f = f1 + f2
    return f

def log_f_v_tilde(v_tilde,Z, C, X2, beta, sigmasq_tilde, sigmasq_zeta_tilde):
    f1 = -0.5/G * v_tilde**2
    f2 = data_loglikelihood(Z, C, X2, beta, v_tilde, sigmasq_tilde, sigmasq_zeta_tilde)
    f = f1 + f2
    return f

T = 10000
beta_chain = np.zeros([T+1,4])
sigmasq_chain = np.ones([T+1,1])*(-0.5)
sigmasq_zeta_chain = np.ones([T+1,1])*(-1.5)
v_chain = np.ones([T+1,1])*2
C_rho = torch.load('C_rho_23.pt')
sigmasq_tilde_var = np.abs(C_rho[4,4])
sigmasq_zeta_tilde_var = np.abs(C_rho[5,5])
v_tilde_var = np.abs(C_rho[6,6])

start_time = time.time()
for t in tqdm(range(T)):
    #print("t=",t)
    beta_candidate = sample_beta(Z2, mod_indv, X2_B, v_chain[t,:], sigmasq_chain[t,:], sigmasq_zeta_chain[t,:])
    beta_chain[t+1,:] = beta_candidate
    sigmasq_candidate = np.random.normal(sigmasq_chain[t,:], sigmasq_tilde_var)
    alpha_sigmasq = np.exp(log_f_sigmasq_tilde(sigmasq_candidate,Z2, mod_indv, X2_B, beta_chain[t+1,:].reshape(-1,1), v_chain[t,:],sigmasq_zeta_chain[t,:])\
                    -log_f_sigmasq_tilde(sigmasq_chain[t,:],Z2, mod_indv, X2_B,beta_chain[t+1,:].reshape(-1,1), v_chain[t,:],sigmasq_zeta_chain[t,:]))
    U = np.random.uniform(0, 1)
    if U <= min(1, alpha_sigmasq):
        sigmasq_chain[t+1,:] = sigmasq_candidate
    else:
        sigmasq_chain[t+1,:] = sigmasq_chain[t,:]
    sigmasq_zeta_candidate = np.random.normal(sigmasq_zeta_chain[t,:], sigmasq_zeta_tilde_var)
    alpha_sigmasq_zeta = np.exp(log_f_sigmasq_zeta_tilde(sigmasq_zeta_candidate,Z2, mod_indv, X2_B,beta_chain[t+1,:].reshape(-1,1), v_chain[t,:], sigmasq_chain[t+1,:])\
                         -log_f_sigmasq_zeta_tilde(sigmasq_zeta_chain[t,:],Z2, mod_indv, X2_B,beta_chain[t+1,:].reshape(-1,1), v_chain[t,:], sigmasq_chain[t+1,:]))
    U = np.random.uniform(0, 1)
    if U <= min(1, alpha_sigmasq_zeta):
        sigmasq_zeta_chain[t+1,:] = sigmasq_zeta_candidate
    else:
        sigmasq_zeta_chain[t+1,:] = sigmasq_zeta_chain[t,:]
    v_candidate = np.random.normal(v_chain[t,:], v_tilde_var)
    alpha_v = np.exp(log_f_v_tilde(v_candidate,Z2, mod_indv, X2_B,beta_chain[t+1,:].reshape(-1,1), sigmasq_chain[t+1,:], sigmasq_zeta_chain[t+1,:])\
              -log_f_v_tilde(v_chain[t,:],Z2, mod_indv, X2_B,beta_chain[t+1,:].reshape(-1,1),  sigmasq_chain[t+1,:], sigmasq_zeta_chain[t+1,:]))
    U = np.random.uniform(0, 1)
    if U <= min(1, alpha_v):
        v_chain[t+1,:] = v_candidate
    else:
        v_chain[t+1,:] = v_chain[t,:]


print("--- %s seconds ---" % (time.time() - start_time))


# check convergence
fig, axes = plt.subplots(4, 1)
axes[0].plot(beta_chain, label='beta')
axes[0].legend()
axes[1].plot(v_chain,label='v')
axes[1].legend()
axes[2].plot(sigmasq_chain, label='sigmasq')
axes[2].legend()
axes[3].plot(sigmasq_zeta_chain, label='sigmasq_zeta')
axes[3].legend()
plt.show()


x = np.linspace(6004, 10000, 1000).astype(int)
beta_ini = beta_chain[x,:]
sigmasq_ini = sigmasq_chain[x,:]
sigmasq_zeta_ini = sigmasq_zeta_chain[x,:]
v_ini = v_chain[x,:]
rho_ini = np.concatenate((beta_ini,sigmasq_ini,sigmasq_zeta_ini,v_ini,LRA_ini),1)


# save posterior samples
np.save("SMC_rho_ini.npy",rho_ini)
np.save("SMC_Phi_ini.npy",Phi_ini)
np.save("SMC_MA_ini.npy",MA_ini)
