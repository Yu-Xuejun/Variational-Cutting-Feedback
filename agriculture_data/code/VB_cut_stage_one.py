#1h50min/93min

import pandas as pd
from sklearn.preprocessing import label_binarize
import numpy as np
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


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
Z1 = torch.tensor(data_arc['normd15N'])
Z2 = torch.tensor(data_mod['normd15N'])
Z = torch.cat((Z1, Z2)).reshape(-1, 1)
R_M = torch.tensor(data_mod['Rainfall']).reshape(-1, 1)  # log_R_M
S_A = torch.tensor(data_arc['Size']).reshape(-1, 1)
M_M = torch.tensor(mod_M[:, 1:3], dtype=torch.float64)  # (268,2)
X2_A = torch.tensor(site_A, dtype=torch.float64)
X2_1 = torch.cat((torch.tensor(site_A, dtype=torch.float64), torch.zeros((269, 19))), 1)
X2_2 = torch.cat((torch.zeros((268, 5)), torch.tensor(site_M, dtype=torch.float64)), 1)
X2 = torch.cat((X2_1, X2_2))
Category = torch.cat((torch.tensor(arc_indv, dtype=torch.float64), torch.tensor(mod_indv, dtype=torch.float64)))
min_Rainfall = torch.tensor(data_arc['Rainfall_min'])
max_Rainfall = torch.tensor(data_arc['Rainfall_max'])
mean_Rainfall = 0.5*(min_Rainfall + max_Rainfall)
mean_Rainfall = mean_Rainfall.reshape((-1,1))
var_Rainfall = (1/12)*(max_Rainfall - min_Rainfall)**2



def design_matrix(M_A, R_A):
    R = torch.cat((R_A, R_M)).reshape(-1, 1)
    M = torch.cat((M_A, M_M))
    X1 = torch.cat((torch.ones((537, 1), dtype=torch.float64), R, M), 1)
    return X1


def log_multinormal(x, mu, Sigma):
    N = torch.tensor(x.size())
    N = N[0]
    pi = torch.tensor(np.pi)
    f1 = - 0.5 * torch.logdet(Sigma)
    f2 = - 0.5 * N * torch.log(2 * pi)
    f3 = - 0.5 * torch.mm(torch.transpose((x - mu), 0, 1), torch.solve(x - mu, Sigma)[0])
    f = f1 + f2 + f3
    return f


def log_normal(x, mu, sigmasq):
    pi = torch.tensor(np.pi)
    f = - 0.5 * torch.log(2 * pi * sigmasq) - ((x - mu) ** 2) / (2 * sigmasq)
    return f


def Category_matrix(c, v):
    C = torch.where(c == 0, torch.tensor(1.0), v)
    return torch.squeeze(C)


def log_g_1(rho, M_A, G):
    V = torch.diag(Category_matrix(Category, torch.exp(rho[6, :])))
    Sigma_zeta = (5 * torch.exp(rho[5, :]) + 0.1) / (1 + torch.exp(rho[5, :])) * torch.eye(24)
    sigma_1 = (5 * torch.exp(rho[4, :]) + 0.1) / (1 + torch.exp(rho[4, :])) * V \
              + torch.mm(torch.mm(X2, Sigma_zeta),torch.transpose(X2, 0, 1))
    X1 = design_matrix(M_A, rho[7:,:])
    A1 = log_multinormal(Z, mu=torch.mm(X1, rho[0:4, :]), Sigma=sigma_1)
    A2 = log_multinormal(rho[0:4, :], mu=0, Sigma=G * torch.eye(4))  # beta
    A4 = - torch.log(torch.tensor(4.9)) - 2 * torch.log(1 + torch.exp(rho[4, :])) \
         + torch.log(5 * torch.exp(rho[4, :]) * (1 + torch.exp(rho[4, :]))
                     - (5 * torch.exp(rho[4, :]) + 0.1) * torch.exp(rho[4, :]))  # sigma_tuta
    A5 = - 2 * torch.log(1 + torch.exp(rho[5,:])) \
        + torch.log(5 * torch.exp(rho[5,:]) * (1 + torch.exp(rho[5,:])) - (5 * torch.exp(rho[5,:]) + 0.1) * torch.exp(rho[5,:])) # sigma_zeta_tuta
    A6 = log_normal(rho[6, :], mu=0, sigmasq=G)  # v
    A7 = log_multinormal(rho[7:, :], mu=mean_Rainfall, Sigma=torch.diag_embed(var_Rainfall))  # log_R_A
    A = A1 + A2 + A4 + A5 + A6 + A7
    return A


M_A_low = torch.zeros((269, 2))
M_A_med = torch.cat((torch.ones((269, 1), dtype=torch.float64), torch.zeros((269, 1), dtype=torch.float64)), 1)
M_A_high = torch.cat((torch.zeros((269, 1), dtype=torch.float64), torch.ones((269, 1))), 1)

def cond_prob_M_A(M_A, rho, Z1):
    V = torch.diag(Category_matrix(Category, torch.exp(rho[6, :])))
    Sigma_zeta = (5 * torch.exp(rho[5, :]) + 0.1) / (1 + torch.exp(rho[5, :])) * torch.eye(24)
    sigma_1 = (5 * torch.exp(rho[4, :]) + 0.1) / (1 + torch.exp(rho[4, :])) * V \
              + torch.mm(torch.mm(X2, Sigma_zeta),torch.transpose(X2, 0, 1))
    Sigma = torch.diag(sigma_1[:269,:269]).reshape(-1,1)
    X1 = design_matrix(M_A, rho[7:,:])[:269,:]
    Z1 = Z1.reshape(-1, 1)
    A = - 0.5 * (Z1 - torch.mm(X1, rho[0:4, :])) ** 2 / Sigma
    return A

def draw_M_A(rho):
    log_p1 = cond_prob_M_A(M_A_low, rho, Z1)
    log_p2 = cond_prob_M_A(M_A_med, rho, Z1)
    log_p3 = cond_prob_M_A(M_A_high, rho, Z1)
    p1 = 1 / (1 + torch.exp(log_p2 - log_p1) + torch.exp(log_p3 - log_p1))
    p2 = 1 / (1 + torch.exp(log_p1 - log_p2) + torch.exp(log_p3 - log_p2))
    p3 = 1 / (1 + torch.exp(log_p1 - log_p3) + torch.exp(log_p2 - log_p3))
    p = torch.cat((p1,p2,p3),1)
    normal_p = p / torch.sum(p,1).reshape(-1,1)
    dist = Categorical(probs=normal_p)
    sample = dist.sample()
    arc_M = label_binarize(sample, classes=[0, 1, 2])
    M_A = torch.tensor(arc_M, dtype=torch.float64)
    M_A = M_A[:, 1:3]
    return M_A



if __name__ == '__main__':


    # iteration for rho
    n1 = 7
    n2 = 269
    n = n1 + n2
    G = 1000
    K1 = 100000
    #K1=100
    rate = 0.95
    epsilon = 1e-6

    mu_rho = torch.ones((n1 + n2, 1))*0.1
    # theta_mcmc_ini = pd.read_csv('theta_mcmc_ini.csv')
    # theta_ini = np.array(theta_mcmc_ini["x"]).reshape(-1, 1)
    # R_A_ini = np.array(0.5 * data_arc['Rainfall_max'] + 0.5 * data_arc['Rainfall_min']).reshape(-1, 1)
    # rho_0 = theta_ini
    # rho_0[4,0] = np.log((rho_0[4,0] ** 2 - 0.1)/(5 - rho_0[4,0] ** 2))
    # rho_0[5,0] = np.log((rho_0[29,0] ** 2 - 0.1)/(5 - rho_0[29,0] ** 2))
    # rho_0[6,0] = np.log(rho_0[30,0])
    # rho_0 = rho_0[:7,]
    # rho_0 = np.concatenate((rho_0, R_A_ini))
    # rho_0 = rho_0 + 15
    # mu_rho = torch.tensor(rho_0)
    # initial value of C
    C1_rho_diag = torch.tensor(1 * np.ones(n1))
    C1_rho = torch.diag(C1_rho_diag)
    C2_rho_diag = torch.tensor(1 * np.ones(n2))
    C2_rho = C2_rho_diag.reshape(-1,1)
    C1_C2_1 = torch.cat((C1_rho, torch.zeros((n1, n2))), 1)
    C1_C2_2 = torch.cat((torch.zeros((n2, n1)), torch.diag(C2_rho_diag)), 1)
    C_rho = torch.cat((C1_C2_1, C1_C2_2))


    E_g_mu = torch.zeros([n, 1], dtype=torch.float64)
    E_delta_mu = torch.zeros([n, 1], dtype=torch.float64)
    E_g_C1 = torch.ones([n1, n1], dtype=torch.float64) * 0.01
    E_delta_C1 = torch.ones([n1, n1], dtype=torch.float64) * 0.01
    E_g_C2 = torch.ones([n2, 1], dtype=torch.float64) * 0.01
    E_delta_C2 = torch.ones([n2, 1], dtype=torch.float64) * 0.01


    rho_history = np.zeros((n, K1))
    mu_rho_history = np.zeros((n, K1))
    C_history = np.zeros((n,K1))
    grad_history = np.zeros((n, K1))
    M_A_history = np.zeros((n2, 2, K1))


    def delta(C):
        D = torch.diag(1 / C.diag())
        return D


    start_time = time.time()
    for t in tqdm(range(K1)):
        while True:
            z = np.random.multivariate_normal(np.zeros([n]), np.eye(n))
            z = torch.tensor(z).reshape((-1, 1))
            rho = torch.tensor(torch.mm(C_rho, z) + mu_rho, requires_grad=True)
            M_A = draw_M_A(rho)
            gradient = log_g_1(rho, M_A, G)
            if gradient is not None:
                break
        gradient.backward()

        g_mu = rho.grad
        E_g_mu = rate * E_g_mu + (1 - rate) * torch.mul(g_mu, g_mu)
        delta_mu = (torch.sqrt(E_delta_mu + epsilon) / torch.sqrt(E_g_mu + epsilon)) * g_mu
        E_delta_mu = rate * E_delta_mu + (1 - rate) * torch.mul(delta_mu, delta_mu)
        mu_rho = mu_rho + delta_mu

        g_C1 = torch.mm(rho.grad[:n1,:], z[:n1].transpose(0, 1)) + delta(C1_rho)
        E_g_C1 = rate * E_g_C1 + (1 - rate) * torch.mul(g_C1, g_C1)
        delta_C1 = (torch.sqrt(E_delta_C1 + epsilon) / torch.sqrt(E_g_C1 + epsilon)) * g_C1
        delta_C1 = torch.tril(delta_C1)
        E_delta_C1 = rate * E_delta_C1 + (1 - rate) * torch.mul(delta_C1, delta_C1)
        C1_rho = C1_rho + delta_C1

        g_C2 = torch.mul(rho.grad[n1:,:], z[n1:]) + 1/(C2_rho.reshape(-1,1))
        E_g_C2 = rate * E_g_C2 + (1 - rate) * torch.mul(g_C2, g_C2)
        delta_C2 = (torch.sqrt(E_delta_C2 + epsilon) / torch.sqrt(E_g_C2 + epsilon)) * g_C2
        E_delta_C2 = rate * E_delta_C2 + (1 - rate) * torch.mul(delta_C2, delta_C2)
        C2_rho = C2_rho + delta_C2

        C1_C2_1 = torch.cat((C1_rho, torch.zeros((n1, n2))), 1)
        C1_C2_2 = torch.cat((torch.zeros((n2, n1)), torch.diag(C2_rho.squeeze())), 1)
        C_rho = torch.cat((C1_C2_1, C1_C2_2))

        rho_history[:, t] = rho.detach().numpy().squeeze()
        mu_rho_history[:, t] = mu_rho.squeeze()
        C_history[:, t] = C_rho.diag().squeeze()
        grad_history[:, t] = rho.grad.detach().numpy().squeeze()
        M_A_history[:, :, t] = M_A.detach().numpy().squeeze()

    print("--- %s seconds ---" % (time.time() - start_time))
    torch.save(mu_rho, 'mu_rho_23.pt')
    torch.save(C_rho, 'C_rho_23.pt')
    torch.save(M_A_history, 'M_A_history_23.pt')
    torch.save(rho_history, 'rho_history_23.pt')

    fig, axes = plt.subplots(6, 3)
    axes[0, 0].plot(rho_history[0, :], label='rho_beta')
    axes[0, 0].legend()
    axes[0, 1].plot(mu_rho_history[0, :], label='mu_beta')
    axes[0, 1].legend()
    axes[0, 2].plot(C_history[0, :], label='C_beta')
    axes[0, 2].legend()
    axes[1, 0].plot(rho_history[4, :], label='rho_sigma')
    axes[1, 0].legend()
    axes[1, 1].plot(mu_rho_history[4, :], label='mu_sigma')
    axes[1, 1].legend()
    axes[1, 2].plot(C_history[4, :], label='C_sigma')
    axes[1, 2].legend()
    axes[2, 0].plot(rho_history[5, :], label='rho_sigmazeta')
    axes[2, 0].legend()
    axes[2, 1].plot(mu_rho_history[5, :], label='mu_sigmazeta')
    axes[2, 1].legend()
    axes[2, 2].plot(C_history[5, :], label='C_sigmazeta')
    axes[2, 2].legend()
    axes[3, 0].plot(rho_history[6, :], label='rho_v')
    axes[3, 0].legend()
    axes[3, 1].plot(mu_rho_history[6, :], label='mu_v')
    axes[3, 1].legend()
    axes[3, 2].plot(C_history[6, :], label='C_v')
    axes[3, 2].legend()
    axes[4, 0].plot(rho_history[7, :], label='rho_R')
    axes[4, 0].legend()
    axes[4, 1].plot(mu_rho_history[7, :], label='mu_R')
    axes[4, 1].legend()
    axes[4, 2].plot(C_history[7, :], label='C_R')
    axes[4, 2].legend()
    axes[5, 0].plot(M_A_history[1,0, :], label='M_A')
    axes[5, 0].legend()
    axes[5, 1].plot(M_A_history[10,0, :], label='M_A')
    axes[5, 1].legend()
    axes[5, 2].plot(M_A_history[150,0, :], label='M_A')
    axes[5, 2].legend()
    plt.savefig('rho_plot.pdf')
