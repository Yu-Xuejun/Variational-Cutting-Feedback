#360s/172s


import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from VB_cut_stage_one import *
import scipy.stats as stats
from tqdm import tqdm

#PYDEVD_USE_FRAME_EVAL=NO
np.random.seed(123456)
torch.manual_seed(123456)
torch.set_default_tensor_type(torch.DoubleTensor)

rho_history = torch.load('rho_history_23.pt')
M_A_history = torch.load('M_A_history_23.pt')
rho_history = rho_history[:, -10000:]
M_A_history = M_A_history[:, :, -10000:]
check_mean = np.mean(rho_history, 1)[0:6]


def cdf_p_M_all(alpha, Phi):
    gamma = Phi[0, :]
    xi = Phi[3:8, :]
    mid = alpha - gamma * S_A - X2_A @ xi
    p = 1/(1+torch.exp(-mid))
    return p

def log_p_M_all(M_A, Phi):
    M_A = M_A.detach()
    ind_M_A = torch.zeros((269,3,1))
    ind_M_A[:, 0, 0] = torch.logical_and(torch.logical_not(M_A[:,0]), torch.logical_not(M_A[:,1]))
    ind_M_A[:, 1, 0] = torch.logical_and(M_A[:, 0], torch.logical_not(M_A[:, 1]))
    ind_M_A[:, 2, 0] = torch.logical_and(torch.logical_not(M_A[:, 0]), M_A[:, 1])
    alpha1 = Phi[1, :]
    alpha2 = torch.exp(Phi[2, :]) + Phi[1, :]
    p_low = cdf_p_M_all(alpha1, Phi)
    p_med = cdf_p_M_all(alpha2, Phi)
    p_matrix = torch.cat((torch.log(p_low), torch.log(1e-5 + p_med - p_low), torch.log(1e-5 + 1 - p_med)), 1)
    p_matrix = p_matrix.unsqueeze(dim=1)
    p_M = torch.bmm(p_matrix, ind_M_A)
    p_M = p_M.squeeze(dim = 2)
    return p_M

def log_p_M(M_A, Phi):
    return torch.sum(log_p_M_all(M_A, Phi))

def log_g_2(Phi, M_A):
    B1 = log_p_M(M_A, Phi)
    if torch.isinf(B1) is True:
        print("B1=inf")
        print(Phi[2,:])
    B2 = log_normal(Phi[0, :], mu=0, sigmasq=4)
    B31 = log_normal(Phi[1, :], mu=0, sigmasq=1.5) #alpha1
    B32 = log_normal(Phi[2, :], mu=-5, sigmasq=7) #log(alpha2-alpha1)
    B4 = log_multinormal(Phi[3:8, :], mu=0, Sigma= (3.5/(1 + torch.exp(-Phi[8,:])))**2 * torch.eye(5))
    B5 = torch.log((torch.exp(Phi[8,:])*(1+torch.exp(Phi[8, :]))- torch.square(torch.exp(Phi[8,:])))/torch.square(1+torch.exp(Phi[8, :])))
    B = B1 + B2 + B31 + B32 + B4 + B5
    return B


# iteration for Phi
G = 1000
K2 = 100000
rate = 0.9
epsilon = 1e-6
mu_Phi = torch.tensor(0.5 * np.ones((9, 1))) + 2
mu_Phi[1,:] = 1
mu_Phi[2,:] = 0.1
C_Phi_diag = torch.tensor(0.99 * np.ones((9)))
C_Phi_diag[2] = 0.01
C_Phi = torch.diag(C_Phi_diag)

E_g_mu = torch.zeros([9, 1], dtype=torch.float64)
E_delta_mu = torch.zeros([9, 1], dtype=torch.float64)
E_g_C = torch.ones([9, 9], dtype=torch.float64) * 0.1
E_delta_C = torch.ones([9, 9], dtype=torch.float64) * 0.1

Phi_history = np.zeros((9,K2))
mu_Phi_history = np.zeros((9,K2))
delta_mu_Phi_history = np.zeros((9,K2))
C_Phi_history = np.zeros((9,9,K2))
delta_C_Phi_history = np.zeros((9,9,K2))
E_g_C_history = np.zeros((9,9,K2))
E_delta_C_history = np.zeros((9,9,K2))
grad_Phi_history = np.zeros((9,K2))


def delta(C):
    D = torch.diag(1 / C.diag())
    return D


start_time = time.time()
for t in tqdm(range(K2)):
    while True:
        z = np.random.multivariate_normal(np.zeros([9]), np.eye(9))
        z = torch.tensor(z).reshape((-1, 1))
        Phi = (torch.mm(C_Phi, z) + mu_Phi).clone().detach().requires_grad_(True)
        index = np.random.randint(10000)
        M_A = M_A_history[:,:,index]
        M_A = torch.tensor(M_A)
        gradient = log_g_2(Phi, M_A)
        if torch.isinf(gradient) == False:
            break
    gradient.backward()

    g_mu = Phi.grad
    E_g_mu = rate * E_g_mu + (1 - rate) * torch.mul(g_mu, g_mu)
    delta_mu = (torch.sqrt(E_delta_mu + epsilon) / torch.sqrt(E_g_mu + epsilon)) * g_mu
    E_delta_mu = rate * E_delta_mu + (1 - rate) * torch.mul(delta_mu, delta_mu)
    mu_Phi = mu_Phi + delta_mu


    g_C = torch.mm(Phi.grad, z.transpose(0, 1)) + delta(C_Phi)
    E_g_C = rate * E_g_C + (1 - rate) * torch.mul(g_C, g_C)
    delta_C = (torch.sqrt(E_delta_C + epsilon) / torch.sqrt(E_g_C + epsilon)) * g_C
    delta_C = torch.tril(delta_C)
    E_delta_C = rate * E_delta_C + (1 - rate) * torch.mul(delta_C, delta_C)
    C_Phi = C_Phi + delta_C

    Phi_history[:, t] = Phi.detach().numpy().squeeze()
    mu_Phi_history[:, t] = mu_Phi.squeeze()
    C_Phi_history[:,:, t] = C_Phi.squeeze()
    grad_Phi_history[:, t] = Phi.grad.detach().numpy().squeeze()

print("--- %s seconds ---" % (time.time() - start_time))


aver_mu = np.mean(mu_Phi_history[:, -10000:],1)
aver_C = np.mean(C_Phi_history[:,:, -10000:],2)
torch.save(aver_mu,"aver_mu_Phi.pt")
torch.save(aver_C,"aver_C_Phi.pt")
torch.save(aver_mu[0], 'gamma_mu.pt')
torch.save(np.abs(aver_C[0,0]), 'gamma_sd.pt')

fig, axes = plt.subplots(5, 3)
axes[0, 0].plot(Phi_history[0, :], label='Phi_gamma')
axes[0, 0].legend()
axes[0, 1].plot(mu_Phi_history[0, :], label='mu_gamma')
axes[0, 1].legend()
axes[0, 2].plot(C_Phi_history[0, 0, :], label='C_gamma')
axes[0, 2].legend()
axes[1, 0].plot(Phi_history[1, :], label='Phi_alpha')
axes[1, 0].legend()
axes[1, 1].plot(mu_Phi_history[1, :], label='mu_alpha')
axes[1, 1].legend()
axes[1, 2].plot(C_Phi_history[1, 1, :], label='C_alpha')
axes[1, 2].legend()
axes[2, 0].plot(Phi_history[2, :], label='Phi_alpha2')
axes[2, 0].legend()
axes[2, 1].plot(mu_Phi_history[2, :], label='mu_alpha2')
axes[2, 1].legend()
axes[2, 2].plot(C_Phi_history[2, 2, :], label='C_alpha2')
axes[2, 2].legend()
axes[3, 0].plot(Phi_history[3, :], label='Phi_xi')
axes[3, 0].legend()
axes[3, 1].plot(mu_Phi_history[3, :], label='mu_xi')
axes[3, 1].legend()
axes[3, 2].plot(C_Phi_history[3, 3, :], label='C_xi')
axes[3, 2].legend()
axes[4, 0].plot(Phi_history[8, :], label='Phi_sigmaxi')
axes[4, 0].legend()
axes[4, 1].plot(mu_Phi_history[8, :], label='mu_sigmaxi')
axes[4, 1].legend()
axes[4, 2].plot(C_Phi_history[8, 8, :], label='C_sigmaxi')
axes[4, 2].legend()
plt.savefig('Phi.pdf')