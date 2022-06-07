# final version 15h
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
n = 269
G = 1000
Z1 = np.array(data_arc['normd15N']).reshape(-1, 1)
Z2 = np.array(data_mod['normd15N']).reshape(-1, 1)
# Z = np.concatenate((Z1, Z2)).reshape(-1, 1)
R_M = np.array(data_mod['Rainfall']).reshape(-1, 1)  # log_R_M
S_A = np.array(data_arc['Size']).reshape(-1, 1)
M_M = np.array(mod_M[:, 1:3])  # (268,2)
X2_A = np.array(site_A)
X2_B = np.array(site_M)
C1 = arc_indv
C2 = mod_indv
min_Rainfall = np.array(data_arc['Rainfall_min'])
max_Rainfall = np.array(data_arc['Rainfall_max'])
mean_Rainfall = 0.5*(min_Rainfall + max_Rainfall)
mean_Rainfall = mean_Rainfall.reshape((-1,1))
var_Rainfall = (1/12)*(max_Rainfall - min_Rainfall)**2
Var_Rainfall = np.diag(var_Rainfall)


# Initialization
rho_ini = np.load("SMC_rho_ini.npy")  # [1000,276]
Phi_ini = np.load("SMC_Phi_ini.npy")  # [1000,8]
MA_ini = np.load("SMC_MA_ini.npy")  # [1000,269,2]
H = Phi_ini.shape[0]
# prepare data
Z_arc = np.tile(Z1, (H, 1, 1))
Z_mod = np.tile(Z2, (H, 1, 1))
C_arc = np.tile(C1, (H, 1, 1))
C_mod = np.tile(C2, (H, 1, 1))
X2_arc = np.tile(X2_A, (H, 1, 1))
X2_mod = np.tile(X2_B, (H, 1, 1))
S_arc = np.tile(S_A, (H, 1, 1))



# Reweighting
def datapoint_tempering_HM(rho, MA, Z_arc, C_arc, t):
    z_t = np.expand_dims(Z_arc[:, t, :], axis=1)
    c_t = np.expand_dims(C_arc[:, t, :], axis=1)
    v = np.exp(rho[:, 6]).reshape((1000, 1, 1))
    sigmasq_tilde = rho[:, 4].reshape((1000, 1, 1))
    sigmasq = (5 * np.exp(sigmasq_tilde) + 0.1) / (1 + np.exp(sigmasq_tilde))
    sigmasq_zeta_tilde = rho[:, 5].reshape((1000, 1, 1))
    sigmasq_zeta = (5 * np.exp(sigmasq_zeta_tilde) + 0.1) / (1 + np.exp(sigmasq_zeta_tilde))
    beta = np.expand_dims(rho[:, :4], axis=2)  # [1000,4,1]
    LRA = np.expand_dims(rho[:, 7 + t], axis=(1, 2))  # [1000,1,1]
    MA_t = np.expand_dims(MA[:, t, :], axis=1)  # [1000,1,2]
    X1 = np.concatenate((np.ones((H, 1, 1)), LRA, MA_t), 2)  # [1000,1,4]
    Var = sigmasq_zeta + sigmasq * (v * c_t + 1 - c_t)  # [1000,1,1]
    f1 = -0.5 * np.log(Var)  # [1000,1,1]
    A = z_t - np.matmul(X1, beta)  # [1000,1,1]
    f2 = -0.5 * (A ** 2) / Var  # [1000,1,1]
    f = f1 + f2
    f = np.squeeze(f)  # [1000,]
    return f


def cdf_p_M_t(alpha, Phi, S_arc, X2_arc, t):
    gamma = Phi[:, 0].reshape((1000, 1, 1))
    xi = Phi[:, 3:8].reshape((1000, 5, 1))
    sigma_xi_tilde = Phi[:, 8].reshape((1000, 1, 1))
    sigma_xi = 3.5 * np.exp(sigma_xi_tilde) / (1 + np.exp(sigma_xi_tilde))
    S_A_t = np.expand_dims(S_arc[:,t,:], axis=1)
    X2_t = np.expand_dims(X2_arc[:,t, :], axis=1) # [1000,1,5]
    mid = alpha - gamma * S_A_t - sigma_xi * np.matmul(X2_t, xi)
    p = 1 / (1 + np.exp(-mid))  # [1000,1,1]
    p = np.squeeze(p)  # [1000,]
    return p


def datapoint_tempering_PO(Phi, MA, S_arc, X2_arc, t):
    MA_t = MA[:, t, :]  # [1000,2]
    alpha_low = Phi[:, 1].reshape((1000, 1, 1))
    alpha_med_tilde = Phi[:, 2].reshape((1000, 1, 1))
    ind_M_A = np.zeros((1000, 3, 1))
    ind_M_A[:, 0, 0] = np.logical_and(np.logical_not(MA_t[:, 0]), np.logical_not(MA_t[:, 1]))
    ind_M_A[:, 1, 0] = np.logical_and(MA_t[:, 0], np.logical_not(MA_t[:, 1]))
    ind_M_A[:, 2, 0] = np.logical_and(np.logical_not(MA_t[:, 0]), MA_t[:, 1])
    alpha1 = alpha_low
    alpha2 = np.exp(alpha_med_tilde) + alpha1
    p_low = cdf_p_M_t(alpha1, Phi, S_arc, X2_arc, t).reshape(-1, 1)
    p_med = cdf_p_M_t(alpha2, Phi, S_arc, X2_arc, t).reshape(-1, 1)
    p_matrix = np.concatenate((np.log(p_low), np.log(1e-5 + p_med - p_low), np.log(1e-5 + 1 - p_med)), 1)
    p_matrix = np.expand_dims(p_matrix, axis=1)  # [1000,1,3]
    p_M = np.matmul(p_matrix, ind_M_A)  # [1000,1,1]
    p_M = np.squeeze(p_M)  # [1000,]
    return p_M


def reweight_loglikelihood(rho, MA, Phi, Z_arc, C_arc, S_arc, X2_arc, t):
    f1 = datapoint_tempering_HM(rho, MA, Z_arc, C_arc, t)
    f2 = datapoint_tempering_PO(Phi, MA, S_arc, X2_arc, t)
    return f1 + f2, f1, f2


# functions for Move
def Variance(C, X2, v, sigmasq, sigmasq_zeta):
    D_v = (1 - C) + v * C  # [1000,d,1]
    d = C.shape[1]
    V = np.zeros((H, d, d))
    for i in range(D_v.shape[0]):
        V[i, :, :] = np.diag(D_v[i, :, 0])
    l = X2.shape[-1]
    Sigma_zeta = np.tile(np.eye(l), (H, 1, 1)) * sigmasq_zeta
    Var = sigmasq * V + np.matmul(X2, np.matmul(Sigma_zeta, X2.transpose((0, 2, 1))))
    return Var


def sample_beta(rho, MA, Z_arc, Z_mod, C_arc, C_mod, X2_arc, X2_mod):
    d = Z_arc.shape[1]
    Z = np.concatenate((Z_arc, Z_mod),1)
    C = np.concatenate((C_arc, C_mod),1)
    X2_1 = np.concatenate((X2_arc, np.zeros((H, d, 19))), 2)
    X2_2 = np.concatenate((np.zeros((H, 268, 5)), X2_mod), 2)
    X2 = np.concatenate((X2_1, X2_2),1)
    v = np.exp(rho[:, 6]).reshape((1000, 1, 1))
    sigmasq_tilde = rho[:, 4].reshape((1000, 1, 1))
    sigmasq = (5 * np.exp(sigmasq_tilde) + 0.1) / (1 + np.exp(sigmasq_tilde))
    sigmasq_zeta_tilde = rho[:, 5].reshape((1000, 1, 1))
    sigmasq_zeta = (5 * np.exp(sigmasq_zeta_tilde) + 0.1) / (1 + np.exp(sigmasq_zeta_tilde))
    LRA = np.expand_dims(rho[:, 7:7 + d], axis=(2))  # [1000,d,1]
    MA = MA[:, :d, :]  # [1000,d,2]
    X1_A = np.concatenate((np.ones((H, d, 1)), LRA, MA), 2)  # [1000,d,4]
    X1_B = np.concatenate((np.ones((H, 268, 1)), np.tile(R_M, (H, 1, 1)), np.tile(M_M, (H, 1, 1))), 2)
    X1 = np.concatenate((X1_A, X1_B), 1)
    Var = Variance(C, X2, v, sigmasq, sigmasq_zeta)  # [1000,d,d]
    Sigma_beta = np.linalg.inv(
        np.matmul(np.matmul(np.transpose(X1, (0, 2, 1)), np.linalg.inv(Var)), X1) + 1 / G * np.tile(np.eye(4),
                                                                                                    (H, 1, 1)))
    mu_beta = np.matmul(np.matmul(Sigma_beta, np.matmul(np.transpose(X1, (0, 2, 1)), np.linalg.inv(Var))), Z)
    beta = np.zeros((H, 4))
    for i in range(H):
        beta[i, :] = np.random.multivariate_normal(np.squeeze(mu_beta[i, :, 0]), Sigma_beta[i, :, :])
    return beta  # [1000,4]


def sample_LRA(rho, MA, Z_arc, C_arc, X2_arc):
    d = Z_arc.shape[1]
    v = np.exp(rho[:, 6]).reshape((1000, 1, 1))
    sigmasq_tilde = rho[:, 4].reshape((1000, 1, 1))
    sigmasq = (5 * np.exp(sigmasq_tilde) + 0.1) / (1 + np.exp(sigmasq_tilde))
    sigmasq_zeta_tilde = rho[:, 5].reshape((1000, 1, 1))
    sigmasq_zeta = (5 * np.exp(sigmasq_zeta_tilde) + 0.1) / (1 + np.exp(sigmasq_zeta_tilde))
    beta = np.expand_dims(rho[:, :4], axis=2)  # [1000,4,1]
    MA1 = np.expand_dims(MA[:, :d, 0], axis=2)  # [1000,d,1]
    MA2 = np.expand_dims(MA[:, :d, 1], axis=2)
    Var = Variance(C_arc, X2_arc, v, sigmasq, sigmasq_zeta)  # [1000,d,d]
    Sigma_LRA = np.linalg.inv(np.tile(np.linalg.inv(Var_Rainfall[:d,:d]), (H, 1, 1)) + np.tile(np.expand_dims(beta[:, 1, :], axis=1), (1, d, d)) ** 2 * np.linalg.inv(Var))
    A = np.expand_dims(beta[:, 1, :], axis=1) * np.matmul(np.linalg.inv(Var), (Z_arc - np.tile(np.expand_dims(beta[:, 0, :], axis=1), (1, d, 1))- np.expand_dims(beta[:, 2, :], axis=1) * MA1 - np.expand_dims(beta[:, 3, :], axis=1) * MA2))
    B = np.tile(np.matmul(np.linalg.inv(Var_Rainfall[:d,:d]),mean_Rainfall[:d,:]),(H,1,1))
    mu_LRA = np.matmul(Sigma_LRA, A+B)
    LRA_1 = np.zeros((H, d))
    LRA_2 = np.zeros((H, 269 - d))
    if d == 269:
        for i in range(H):
            C_Sigma = np.linalg.cholesky(Sigma_LRA[i, :, :])
            epsilon = np.random.multivariate_normal(np.zeros([d]), np.eye(d))
            LRA_1[i, :] = mu_LRA[i, :, 0] + np.matmul(C_Sigma, epsilon)
    else:
        for i in range(H):
            C_Sigma = np.linalg.cholesky(Sigma_LRA[i, :, :])
            epsilon = np.random.multivariate_normal(np.zeros([d]), np.eye(d))
            LRA_1[i, :] = mu_LRA[i, :, 0] + np.matmul(C_Sigma, epsilon)
            LRA_2[i, :] = np.random.multivariate_normal(np.zeros([269 - d]), np.eye(269 - d))
    LRA = np.concatenate((LRA_1, LRA_2), 1)  # [1000,269]
    return LRA  # [1000,269]



def prior_unif(sigmasq_tilde):
    f = (5 * np.exp(sigmasq_tilde) * (1 + np.exp(sigmasq_tilde)) - (5 * np.exp(sigmasq_tilde) + 0.1) * np.exp(
        sigmasq_tilde)) / (4.9 * (1 + np.exp(sigmasq_tilde)) ** 2)
    f = np.squeeze(f)
    return f


def data_loglikelihood_HM(rho, MA, Z_arc, Z_mod, C_arc, C_mod, X2_arc, X2_mod):
    # d+268 dimensional
    d = Z_arc.shape[1]
    Z = np.concatenate((Z_arc, Z_mod),1)
    C = np.concatenate((C_arc, C_mod),1)
    X2_1 = np.concatenate((X2_arc, np.zeros((H, d, 19))), 2)
    X2_2 = np.concatenate((np.zeros((H, 268, 5)), X2_mod), 2)
    X2 = np.concatenate((X2_1, X2_2),1)
    beta = np.expand_dims(rho[:, :4], axis=2)  # [1000,4,1]
    v = np.exp(rho[:, 6]).reshape((1000, 1, 1))
    sigmasq_tilde = rho[:, 4].reshape((1000, 1, 1))
    sigmasq = (5 * np.exp(sigmasq_tilde) + 0.1) / (1 + np.exp(sigmasq_tilde))
    sigmasq_zeta_tilde = rho[:, 5].reshape((1000, 1, 1))
    sigmasq_zeta = (5 * np.exp(sigmasq_zeta_tilde) + 0.1) / (1 + np.exp(sigmasq_zeta_tilde))
    LRA = np.expand_dims(rho[:, 7:7 + d], axis=(2))  # [1000,d,1]
    MA = MA[:, :d, :]  # [1000,d,2]
    X1_A = np.concatenate((np.ones((H, d, 1)), LRA, MA), 2)  # [1000,d,4]
    X1_B = np.concatenate((np.ones((H, 268, 1)), np.tile(R_M, (H, 1, 1)), np.tile(M_M, (H, 1, 1))), 2)
    X1 = np.concatenate((X1_A, X1_B), 1)
    Var = Variance(C, X2, v, sigmasq, sigmasq_zeta)  # [1000,d,d]
    # sign, logdet
    f1 = np.linalg.slogdet(Var)[0] * np.linalg.slogdet(Var)[1] * (-1 / 2)
    A = Z - np.matmul(X1, beta)
    f2 = -0.5 * np.matmul(np.matmul(np.transpose(A, (0, 2, 1)), np.linalg.inv(Var)), A)
    f2 = np.squeeze(f2)
    f = f1 + f2
    return f  # [1000,]


def log_f_sigmasq_tilde(sigmasq_tilde, rho, MA, Z_arc, Z_mod, C_arc, C_mod, X2_arc, X2_mod):
    f1 = np.log(prior_unif(sigmasq_tilde.reshape((1000, 1, 1))))
    rho[:,4] = sigmasq_tilde
    f2 = data_loglikelihood_HM(rho, MA, Z_arc, Z_mod, C_arc, C_mod, X2_arc, X2_mod)
    f = f1 + f2
    return f  # [1000,]


def log_f_sigmasq_zeta_tilde(sigmasq_zeta,rho, MA, Z_arc, Z_mod, C_arc, C_mod, X2_arc, X2_mod):
    f1 = np.log(prior_unif(sigmasq_zeta.reshape((1000, 1, 1))))
    rho[:,5] = sigmasq_zeta
    f2 = data_loglikelihood_HM(rho, MA, Z_arc, Z_mod, C_arc, C_mod, X2_arc, X2_mod)
    f = f1 + f2
    return f


def log_f_v_tilde(v_tilde, rho, MA, Z_arc, Z_mod, C_arc, C_mod, X2_arc, X2_mod):
    f1 = -0.5 / G * v_tilde ** 2
    f1 = np.squeeze(f1)
    rho[:,6] = v_tilde
    f2 = data_loglikelihood_HM(rho, MA, Z_arc, Z_mod, C_arc, C_mod, X2_arc, X2_mod)
    f = f1 + f2
    return f


def cdf_p_M_all(alpha, Phi, S_arc, X2_arc):
    gamma = Phi[:, 0].reshape((1000, 1, 1))
    xi = Phi[:, 3:8].reshape((1000, 5, 1))
    sigma_xi_tilde = Phi[:, 8].reshape((1000, 1, 1))
    sigma_xi = 3.5 * np.exp(sigma_xi_tilde) / (1 + np.exp(sigma_xi_tilde))
    mid = alpha - gamma * S_arc - sigma_xi * np.matmul(X2_arc, xi)
    p = 1 / (1 + np.exp(-mid))  # [1000,d,1]
    return p


def log_p_M_all(MA, Phi, S_arc, X2_arc):
    d = MA.shape[1]
    alpha_low = Phi[:, 1].reshape((H, 1, 1))
    alpha_med_tilde = Phi[:, 2].reshape((H, 1, 1))
    ind_M_A = np.zeros((H, d, 3, 1))
    ind_M_A[:, :, 0, 0] = np.logical_and(np.logical_not(MA[:, :, 0]), np.logical_not(MA[:, :, 1]))
    ind_M_A[:, :, 1, 0] = np.logical_and(MA[:, :, 0], np.logical_not(MA[:, :, 1]))
    ind_M_A[:, :, 2, 0] = np.logical_and(np.logical_not(MA[:, :, 0]), MA[:, :, 1])
    alpha1 = alpha_low
    alpha2 = np.exp(alpha_med_tilde) + alpha1
    p_low = cdf_p_M_all(alpha1, Phi, S_arc, X2_arc)
    p_med = cdf_p_M_all(alpha2, Phi, S_arc, X2_arc)
    p_matrix = np.concatenate((np.log(p_low), np.log(1e-5 + p_med - p_low), np.log(1e-5 + 1 - p_med)), 2)
    p_matrix = np.expand_dims(p_matrix, axis=2)  # [1000,d,1,3]
    p_M = np.matmul(p_matrix, ind_M_A)  # [1000,d,1,1]
    p_M = np.squeeze(p_M, axis=2)  # [1000,d,1]
    return p_M


def cond_prob_MA(rho, MA, Phi, Z_arc, C_arc, X2_arc, S_arc):
    # data 1:d
    d = Z_arc.shape[1]
    v = np.exp(rho[:, 6]).reshape((H, 1, 1))
    sigmasq_tilde = rho[:, 4].reshape((H, 1, 1))
    sigmasq = (5 * np.exp(sigmasq_tilde) + 0.1) / (1 + np.exp(sigmasq_tilde))
    sigmasq_zeta_tilde = rho[:, 5].reshape((H, 1, 1))
    sigmasq_zeta = (5 * np.exp(sigmasq_zeta_tilde) + 0.1) / (1 + np.exp(sigmasq_zeta_tilde))
    beta = np.expand_dims(rho[:, :4], axis=2)  # [1000,4,1]
    # [1000,d,4]
    LRA = np.expand_dims(rho[:, 7:7 + d], axis=2)
    MA = MA[:, :d, :]  # [1000,d,2]
    X1 = np.concatenate((np.ones((H, d, 1)), LRA, MA), 2)
    Var = Variance(C_arc, X2_arc, v, sigmasq, sigmasq_zeta)  # [1000,d,d]
    D_v = np.zeros((H, d, 1))
    for i in range(H):
        D_v[i, :, 0] = np.diag(Var[i, :, :])
    mid = Z_arc - np.matmul(X1, beta)  # [1000,d,1]
    log_density = - 0.5 * mid ** 2 / D_v
    A = log_density
    B = log_p_M_all(MA, Phi, S_arc, X2_arc)
    return A + B  # [1000,d,1]


M_A_low = np.zeros((1000, 269, 2))
M_A_med = np.concatenate((np.ones((1000, 269, 1)), np.zeros((1000, 269, 1))), 2)
M_A_high = np.concatenate((np.zeros((1000, 269, 1)), np.ones((1000, 269, 1))), 2)


def sample_M_A(rho, Phi, Z_arc, C_arc, X2_arc_all,S_arc_all):
    d = Z_arc.shape[1]
    log_p1 = cond_prob_MA(rho, M_A_low[:,:d,:], Phi, Z_arc, C_arc, X2_arc_all[:,:d,:], S_arc_all[:,:d,:])
    log_p2 = cond_prob_MA(rho, M_A_med[:,:d,:], Phi, Z_arc, C_arc, X2_arc_all[:,:d,:], S_arc_all[:,:d,:])
    log_p3 = cond_prob_MA(rho, M_A_high[:,:d,:], Phi, Z_arc, C_arc, X2_arc_all[:,:d,:], S_arc_all[:,:d,:])
    log_p = np.concatenate((log_p1, log_p2, log_p3), 2) #[1000,d,3]
    log_p = log_p - np.tile(np.expand_dims(np.max(log_p,axis=2),axis=2),(1,1,3))
    p = np.exp(log_p)
    # MA ~ p(MA|Phi) for [d:]
    rest_log_p1 = log_p_M_all(M_A_low[:,d:,:], Phi, S_arc_all[:,d:,:], X2_arc_all[:,d:,:])
    rest_log_p2 = log_p_M_all(M_A_med[:, d:, :], Phi, S_arc_all[:,d:,:], X2_arc_all[:,d:,:])
    rest_log_p3 = log_p_M_all(M_A_med[:, d:, :], Phi, S_arc_all[:,d:,:], X2_arc_all[:,d:,:])
    rest_log_p = np.concatenate((rest_log_p1, rest_log_p2, rest_log_p3), 2)  # [1000,d,3]
    rest_log_p = rest_log_p - np.tile(np.expand_dims(np.max(rest_log_p, axis=2), axis=2), (1, 1, 3))
    rest_p = np.exp(rest_log_p)
    new_p = np.concatenate((p,rest_p),1)
    normal_p = new_p / np.sum(new_p, 2).reshape((H, n, 1))
    normal_p = torch.from_numpy(normal_p)
    dist = Categorical(probs=normal_p)
    sample = dist.sample()  # [1000,d]
    M_A = np.zeros((H, n, 2))
    for i in range(H):
        arc_M = label_binarize(sample[i, :], classes=[0, 1, 2])
        M_A[i, :, :] = arc_M[:, 1:3]
    return M_A #[H,269,2]


def log_f_Phi(Phi, MA, S_arc, X2_arc):
    d = S_arc.shape[1]
    gamma = Phi[:, 0].reshape((1000, 1, 1))
    xi = Phi[:, 3:8].reshape((1000, 5, 1))
    sigma_xi_tilde = Phi[:, 8].reshape((1000, 1, 1))
    alpha_low = Phi[:, 1].reshape((1000, 1, 1))
    alpha_med_tilde = Phi[:, 2].reshape((1000, 1, 1))
    f1 = -0.5 / 4 * gamma ** 2
    f2 = -0.5 / 1.5 * alpha_low ** 2
    f3 = -0.5 / 7 * (alpha_med_tilde+5) ** 2
    f4 = -0.5 * np.matmul(np.transpose(xi, axes=(0, 2, 1)), xi)
    f5 = np.log(np.exp(sigma_xi_tilde) * (1 + np.exp(sigma_xi_tilde)) - np.exp(sigma_xi_tilde) ** 2) - 2 * np.log(
        1 + np.exp(sigma_xi_tilde))
    f6 = np.sum(log_p_M_all(MA[:,:d,:], Phi, S_arc, X2_arc),1)
    f6 = np.squeeze(f6)
    f = np.squeeze(f1 + f2 + f3 + f4 + f5) + f6
    return f # [1000]




log_w = np.zeros(H)
rho = rho_ini
MA = MA_ini
Phi = Phi_ini
C_Phi = torch.load('aver_C_Phi.pt')
Phi_var = np.matmul(C_Phi, np.transpose(C_Phi))
C_rho = torch.load('C_rho_23.pt')
rho_var = np.matmul(C_rho, np.transpose(C_rho))

rho_history = np.ones([n, 276])
Phi_history = np.ones([n, 9])
MA_history = np.ones([n, 269, 2])
ESS_list = []

start_time = time.time()
for t in tqdm(range(n)):
    temp = log_w + reweight_loglikelihood(rho, MA, Phi, Z_arc, C_arc, S_arc, X2_arc, t)[0]
    log_w = temp - np.max(temp)
    w_tilde = np.exp(log_w) / np.sum(np.exp(log_w))
    ESS_Z = np.sum(np.exp(reweight_loglikelihood(rho, MA, Phi, Z_arc, C_arc, S_arc, X2_arc, t)[1]))**2/(np.sum(np.exp(reweight_loglikelihood(rho, MA, Phi, Z_arc, C_arc, S_arc, X2_arc, t)[1])**2))
    ESS_M = np.sum(np.exp(reweight_loglikelihood(rho, MA, Phi, Z_arc, C_arc, S_arc, X2_arc, t)[2]))**2/ (np.sum(np.exp(reweight_loglikelihood(rho, MA, Phi, Z_arc, C_arc, S_arc, X2_arc, t)[2]) ** 2))
    ESS = 1 / (np.sum(w_tilde ** 2))
    ESS_list.append(ESS)
    print("ESS=", ESS)
    if ESS < H / 2:
        index = np.random.choice(1000, size=(1000), p=w_tilde)
        MA_chain = np.zeros([11, H, 269, 2])
        MA_chain[0, :, :, :] = MA[index, :, :]
        Phi_chain = np.ones([11, H, 9])
        Phi_chain[0, :, :] = Phi[index, :]
        rho_chain = np.ones([11, H, 276])
        rho_chain[0, :, :] = rho[index, :]
        for i in range(10):
            MA_candidate = sample_M_A(rho_chain[i,:,:], Phi_chain[i,:,:], Z_arc[:,:t+1,:], C_arc[:,:t+1,:], X2_arc,S_arc)
            MA_chain[i + 1, :, :, :] = MA_candidate
            epsilon = np.random.multivariate_normal(np.zeros((9)), Phi_var, size=H)
            Phi_candidate = Phi_chain[i, :, :] + epsilon
            alpha_Phi = np.exp(log_f_Phi(Phi_candidate, MA_chain[i + 1, :, :,:], S_arc[:,:t+1,:], X2_arc[:,:t+1,:])
                               - log_f_Phi(Phi_chain[i, :, :],MA_chain[i + 1, :,:,:],S_arc[:,:t+1,:], X2_arc[:,:t+1,:]))
            U = np.random.uniform(0, 1, size=H)
            flag = (U <= np.minimum(np.ones(1000), alpha_Phi)).astype(int)
            flag = np.tile(flag.reshape(-1, 1), (1, 9))
            Phi_chain[i + 1, :, :] = (1 - flag) * Phi_chain[i, :, :] + flag * Phi_candidate

            rho_chain[i+1,:,:] = rho_chain[i,:,:]
            beta_candidate = sample_beta(rho_chain[i+1,:,:], MA_chain[i + 1, :, :,:], Z_arc[:,:t+1,:], Z_mod, C_arc[:,:t+1,:], C_mod, X2_arc[:,:t+1,:], X2_mod)
            rho_chain[i + 1, :,:4] = beta_candidate
            LRA_candidate = sample_LRA(rho_chain[i+1,:,:], MA_chain[i + 1, :, :,:], Z_arc[:,:t+1,:], C_arc[:,:t+1,:], X2_arc[:,:t+1,:])
            rho_chain[i + 1, :,7:] = LRA_candidate

            epsilon = np.random.normal(0, np.abs(C_rho[4,4]), size=H)
            sigmasq_candidate = rho_chain[i, :, 4] + epsilon
            alpha_sigmasq = np.exp(log_f_sigmasq_tilde(sigmasq_candidate, rho_chain[i+1,:,:], MA_chain[i + 1, :, :,:], Z_arc[:,:t+1,:], Z_mod, C_arc[:,:t+1,:], C_mod, X2_arc[:,:t+1,:], X2_mod)
                                   -log_f_sigmasq_tilde(rho_chain[i,:,4], rho_chain[i+1,:,:], MA_chain[i + 1, :, :,:], Z_arc[:,:t+1,:], Z_mod, C_arc[:,:t+1,:], C_mod, X2_arc[:,:t+1,:], X2_mod))
            U = np.random.uniform(0, 1, size=H)
            flag = (U <= np.minimum(np.ones(1000), alpha_sigmasq)).astype(int)
            rho_chain[i+1, :, 4] = (1 - flag) * rho_chain[i, :, 4] + flag * sigmasq_candidate

            epsilon = np.random.normal(0, np.abs(C_rho[5, 5]), size=H)
            sigmasq_zeta_candidate = rho_chain[i, :, 5] + epsilon
            alpha_sigmasq_zeta = np.exp(
                log_f_sigmasq_zeta_tilde(sigmasq_zeta_candidate, rho_chain[i + 1,:,:], MA_chain[i + 1, :, :, :], Z_arc[:, :t+1, :],
                                    Z_mod, C_arc[:, :t+1, :], C_mod, X2_arc[:, :t+1, :], X2_mod)
                - log_f_sigmasq_zeta_tilde(rho_chain[i, :, 5], rho_chain[i + 1,:,:], MA_chain[i + 1, :, :, :], Z_arc[:, :t+1, :],
                                      Z_mod, C_arc[:, :t+1, :], C_mod, X2_arc[:, :t+1, :], X2_mod))
            U = np.random.uniform(0, 1, size=H)
            flag = (U <= np.minimum(np.ones(1000), alpha_sigmasq_zeta)).astype(int)
            rho_chain[i+1, :, 5] = (1 - flag) * rho_chain[i, :, 5] + flag * sigmasq_zeta_candidate

            epsilon = np.random.normal(0, np.abs(C_rho[6, 6]), size=H)
            v_candidate = rho_chain[i, :, 6] + epsilon
            alpha_v = np.exp(
                log_f_sigmasq_tilde(v_candidate, rho_chain[i + 1,:,:], MA_chain[i + 1, :, :, :], Z_arc[:, :t+1, :],
                                    Z_mod, C_arc[:, :t+1, :], C_mod, X2_arc[:, :t+1, :], X2_mod)
                - log_f_sigmasq_tilde(rho_chain[i, :, 6], rho_chain[i + 1,:,:], MA_chain[i + 1, :, :, :], Z_arc[:, :t+1, :],
                                      Z_mod, C_arc[:, :t+1, :], C_mod, X2_arc[:, :t+1, :], X2_mod))
            U = np.random.uniform(0, 1, size=H)
            flag = (U <= np.minimum(np.ones(1000), alpha_v)).astype(int)
            rho_chain[i+1, :, 6] = (1 - flag) * rho_chain[i, :, 6] + flag * v_candidate
        rho = rho_chain[10, :, :]
        Phi = Phi_chain[10, :, :]
        MA = MA_chain[10, :, :, :]
        log_w = np.zeros(H)
    rho_history[t, :] = np.matmul(w_tilde.reshape(1, -1), rho)
    Phi_history[t, :] = np.matmul(w_tilde.reshape(1, -1), Phi)
    MA_history[t, :, 0] = np.matmul(w_tilde.reshape(1, -1), MA[:, :, 0])
    MA_history[t, :, 1] = np.matmul(w_tilde.reshape(1, -1), MA[:, :, 1])

print("--- %s seconds ---" % (time.time() - start_time))

#np.save("rho_history.npy",rho_history)
#np.save("Phi_history.npy",Phi_history)
#np.save("MA_history.npy",MA_history)
#np.save("rho_sample.npy", rho)
np.save("Phi_sample.npy", Phi)
#np.save("MA_sample.npy", MA)
np.save("ESS.npy", ESS_list)

plt.clf()
import seaborn as sns
sns.distplot(Phi[:,0], hist = False, kde = True, kde_kws = {'linewidth': 3})
plt.savefig('SMC_gamma.pdf')

plt.clf()
fig, axes = plt.subplots(3, 1)
axes[0].plot(Phi_history[:268,0], label='gamma')
axes[0].legend()
axes[1].plot(Phi_history[:268,1],label='alpha_low')
axes[1].legend()
axes[2].plot(Phi_history[:268,2], label='alpha_med')
axes[2].legend()
plt.savefig('SMC_Phi.pdf')

plt.clf()
ESS_list = np.load("ESS.npy")
plt.plot(ESS_list)
plt.savefig('ESS.pdf')
