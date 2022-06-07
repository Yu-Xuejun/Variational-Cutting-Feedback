#2h/100min
import pandas as pd
from sklearn.preprocessing import label_binarize
import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import scipy.stats as stats
import seaborn as sns

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
    temp = torch.diagonal(Sigma) + 1e-15
    Sigma = torch.diag_embed(temp)
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
    p_matrix = torch.cat((torch.log(p_low), torch.log(1e-15 + p_med - p_low), torch.log(1e-15 + 1 - p_med)), 1)
    p_matrix = p_matrix.unsqueeze(dim=1)
    #print(p_matrix)
    p_M = torch.bmm(p_matrix, ind_M_A)
    p_M = p_M.squeeze(dim = 2)
    return p_M

def log_p_M(M_A, Phi):
    return torch.sum(log_p_M_all(M_A, Phi))

def log_g(rho, Phi, M_A, G):
    V = torch.diag(Category_matrix(Category, torch.exp(rho[6, :])))
    Sigma_zeta = (5 * torch.exp(rho[5, :]) + 0.1) / (1 + torch.exp(rho[5, :])) * torch.eye(24)
    sigma_1 = (5 * torch.exp(rho[4, :]) + 0.1) / (1 + torch.exp(rho[4, :])) * V \
              + torch.mm(torch.mm(X2, Sigma_zeta),torch.transpose(X2, 0, 1))
    X1 = design_matrix(M_A, rho[7:,:])
    A1 = log_multinormal(Z, mu=torch.mm(X1, rho[0:4, :]), Sigma=sigma_1)
    A2 = log_multinormal(rho[0:4, :], mu=0, Sigma=G * torch.eye(4))  # beta
    A3 = - torch.log(torch.tensor(4.9)) - 2 * torch.log(1 + torch.exp(rho[4, :])) \
         + torch.log(5 * torch.exp(rho[4, :]) * (1 + torch.exp(rho[4, :]))
                     - (5 * torch.exp(rho[4, :]) + 0.1) * torch.exp(rho[4, :]))  # sigma_tuta
    try:
        assert not torch.isinf(A3)
    except:
        print('error3')
        return None
    A4 = - 2 * torch.log(1 + torch.exp(rho[5,:])) \
        + torch.log(5 * torch.exp(rho[5,:]) * (1 + torch.exp(rho[5,:])) - (5 * torch.exp(rho[5,:]) + 0.1) * torch.exp(rho[5,:])) # sigma_zeta_tuta
    try:
        assert not torch.isinf(A4)
    except:
        print('error4')
        return None
    A5 = log_normal(rho[6, :], mu=0, sigmasq=G)  # v
    A6 = log_multinormal(rho[7:, :], mu=mean_Rainfall, Sigma=torch.diag_embed(var_Rainfall))  # log_R_A
    A = A1 + A2 + A3 + A4 + A5 + A6
    B1 = log_p_M(M_A, Phi)
    B2 = log_normal(Phi[0, :], mu=0, sigmasq=4) #gamma
    B31 = log_normal(Phi[1, :], mu=0, sigmasq=1.5)  # alpha1
    B32 = log_normal(Phi[2, :], mu=-5, sigmasq=7)  # log(alpha2-alpha1)
    B4 = log_multinormal(Phi[3:8, :], mu=0, Sigma=(3.5 / (1 + torch.exp(-Phi[8, :]))) ** 2 * torch.eye(5))
    B5 = - Phi[8, :] - 2 * torch.log(1 + torch.exp(-Phi[8, :]))
    B = B1 + B2 + B31 + B32 + B4 + B5
    return A + B


M_A_low = torch.zeros((269, 2))
M_A_med = torch.cat((torch.ones((269, 1), dtype=torch.float64), torch.zeros((269, 1), dtype=torch.float64)), 1)
M_A_high = torch.cat((torch.zeros((269, 1), dtype=torch.float64), torch.ones((269, 1))), 1)

def cond_prob_M_A(M_A, rho, Phi, Z1):
    V = torch.diag(Category_matrix(Category, torch.exp(rho[6, :])))
    Sigma_zeta = (5 * torch.exp(rho[5, :]) + 0.1) / (1 + torch.exp(rho[5, :])) * torch.eye(24)
    sigma_1 = (5 * torch.exp(rho[4, :]) + 0.1) / (1 + torch.exp(rho[4, :])) * V \
              + torch.mm(torch.mm(X2, Sigma_zeta),torch.transpose(X2, 0, 1))
    Sigma = torch.diag(sigma_1[:269,:269]).reshape(-1,1)
    X1 = design_matrix(M_A, rho[7:,:])[:269,:]
    Z1 = Z1.reshape(-1, 1)
    A = - 0.5 * (Z1 - torch.mm(X1, rho[0:4, :])) ** 2 / Sigma
    B = log_p_M_all(M_A, Phi)
    return A + B

def draw_M_A(rho, Phi):
    log_p1 = cond_prob_M_A(M_A_low, rho, Phi, Z1)
    log_p2 = cond_prob_M_A(M_A_med, rho, Phi, Z1)
    log_p3 = cond_prob_M_A(M_A_high, rho, Phi, Z1)
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


n1 = 7
n2 = 269
n3 = 9
n = n1+n2+n3
G = 1000
K1 = 100000
mu_rho = torch.ones((n1+n2,1))
mu_Phi = torch.tensor(0.5 * np.ones((n3, 1))) + 2
mu_Phi[0,:] = -2
mu_Phi[1,:] = 1
mu_Phi[2,:] = 0.01
mu = torch.cat((mu_Phi, mu_rho))
# initial value of C
C1_rho_diag = torch.tensor(1 * np.ones(n1))
C1_rho = torch.diag(C1_rho_diag)
C2_rho_diag = torch.tensor(1 * np.ones(n2))
C2_rho = C2_rho_diag.reshape(-1,1)
C1_C2_1 = torch.cat((C1_rho, torch.zeros((n1, n2))), 1)
C1_C2_2 = torch.cat((torch.zeros((n2, n1)), torch.diag(C2_rho_diag)), 1)
C_rho = torch.cat((C1_C2_1, C1_C2_2))
C_Phi_diag = torch.tensor(0.99 * np.ones((9)))
C_Phi_diag[2] = 0.01
C_Phi = torch.diag(C_Phi_diag)
C = torch.zeros([n,n])
C[:n3, :n3] = C_Phi
C[n3:, n3:] = C_rho
C1 = C[:(n1+n3), :(n1+n3)]


theta_history = np.zeros((n, K1))
mu_history = np.zeros((n, K1))
C_history = np.zeros((n, K1))
M_A_history = np.zeros((269, 2, K1))

# parameter in Adam
beta1 = 0.9
beta2 = 0.999
alpha = 0.01
epsilon = 1e-8
m_mu = torch.zeros([n,1], dtype=torch.float64)
v_mu = torch.zeros([n,1], dtype=torch.float64)
m_C = torch.zeros([n1+n3, n1+n3], dtype=torch.float64)
v_C = torch.zeros([n1+n3, n1+n3], dtype=torch.float64)
m_C2 = torch.zeros([n2, 1], dtype=torch.float64)
v_C2 = torch.zeros([n2, 1], dtype=torch.float64)

def delta(C):
    D = torch.diag(1 / C.diag())
    return D

start_time = time.time()
for t in tqdm(range(1,K1)):
    while True:
        z = np.random.multivariate_normal(np.zeros([n]), np.eye(n))
        z = torch.tensor(z).reshape((-1, 1))
        theta = (torch.mm(C, z) + mu).clone().detach().requires_grad_(True)
        M_A = draw_M_A(theta[n3:,:], theta[:n3,:])
        gradient = log_g(theta[n3:,:], theta[:n3,:], M_A, G)
        if gradient is not None:
            try:
                gradient.backward()
                break
            except:
                print("error-cpu")

    g_mu = theta.grad
    m_mu = beta1 * m_mu + (1-beta1) * g_mu
    v_mu = beta2 * v_mu + (1-beta2) * torch.mul(g_mu, g_mu)
    m_mu_hat = m_mu / (1 - beta1 ** t)
    v_mu_hat = v_mu / (1 - beta2 ** t)
    delta_mu = alpha * m_mu_hat / (torch.sqrt(v_mu_hat) + epsilon)
    mu = mu + delta_mu

    g_C = torch.mm(theta.grad[:(n3+n1),:], z[:(n3+n1)].transpose(0, 1)) + delta(C1)
    m_C = beta1 * m_C + (1-beta1) * g_C
    v_C = beta2 * v_C + (1-beta2) * torch.mul(g_C, g_C)
    m_C_hat = m_C / (1 - beta1 ** t)
    v_C_hat = v_C / (1 - beta2 ** t)
    delta_C = alpha * m_C_hat / (torch.sqrt(v_C_hat) + epsilon)
    delta_C = torch.tril(delta_C)
    C1 = C1 + delta_C

    g_C2 = torch.mul(theta.grad[(n3+n1):,:], z[(n3+n1):]) + 1/(C2_rho.reshape(-1,1))
    m_C2 = beta1 * m_C2 + (1-beta1) * g_C2
    v_C2 = beta2 * v_C2 + (1-beta2) * torch.mul(g_C2, g_C2)
    m_C_hat2 = m_C2 / (1 - beta1 ** t)
    v_C_hat2 = v_C2 / (1 - beta2 ** t)
    delta_C2 = alpha * m_C_hat2 / (torch.sqrt(v_C_hat2) + epsilon)
    C2_rho = C2_rho + delta_C2

    C1_C2_1 = torch.cat((C1, torch.zeros((n1+n3, n2))), 1)
    C1_C2_2 = torch.cat((torch.zeros((n2, n1+n3)), torch.diag(C2_rho.squeeze())), 1)
    C = torch.cat((C1_C2_1, C1_C2_2))

    theta_history[:, t] = theta.detach().numpy().squeeze()
    mu_history[:, t] = mu.squeeze()
    C_history[:, t] = C.diag().squeeze()
    M_A_history[:, :, t] = M_A.detach().numpy().squeeze()

print("--- %s seconds ---" % (time.time() - start_time))

fig, axes = plt.subplots(7, 3)
axes[0, 0].plot(theta_history[0+n3, :], label='beta')
axes[0, 0].legend()
axes[0, 1].plot(mu_history[0+n3, :])
axes[0, 2].plot(C_history[0+n3, :])
axes[1, 0].plot(theta_history[4+n3, :], label='sigma')
axes[1, 0].legend()
axes[1, 1].plot(mu_history[4+n3, :])
axes[1, 2].plot(C_history[4+n3, :])
axes[2, 0].plot(theta_history[5+n3, :], label='sigmazeta')
axes[2, 0].legend()
axes[2, 1].plot(mu_history[5+n3, :])
axes[2, 2].plot(C_history[5+n3, :])
axes[3, 0].plot(theta_history[6+n3, :], label='v')
axes[3, 0].legend()
axes[3, 1].plot(mu_history[6+n3, :])
axes[3, 2].plot(C_history[6+n3, :])
axes[4, 0].plot(theta_history[1+n3, :], label='R')
axes[4, 0].legend()
axes[4, 1].plot(mu_history[1+n3, :])
axes[4, 2].plot(C_history[1+n3, :])
axes[5, 0].plot(theta_history[2+n3, :], label='R')
axes[5, 0].legend()
axes[5, 1].plot(mu_history[2+n3, :])
axes[5, 2].plot(C_history[2+n3, :])
axes[6, 0].plot(theta_history[3+n3, :], label='R')
axes[6, 0].legend()
axes[6, 1].plot(mu_history[3+n3, :])
axes[6, 2].plot(C_history[3+n3, :])
plt.savefig('agriculture_full_adam1.pdf')

plt.clf()
fig, axes = plt.subplots(5, 3)
axes[0, 0].plot(theta_history[0, :], label='gamma')
axes[0, 0].legend()
axes[0, 1].plot(mu_history[0, :])
axes[0, 2].plot(C_history[0, :])
axes[1, 0].plot(theta_history[1, :], label='alpha1')
axes[1, 0].legend()
axes[1, 1].plot(mu_history[1, :])
axes[1, 2].plot(C_history[1, :])
axes[2, 0].plot(theta_history[2, :], label='alpha2_tilde')
axes[2, 0].legend()
axes[2, 1].plot(mu_history[2, :])
axes[2, 2].plot(C_history[2, :])
axes[3, 0].plot(theta_history[3, :], label='xi')
axes[3, 0].legend()
axes[3, 1].plot(mu_history[3, :])
axes[3, 2].plot(C_history[3, :])
axes[4, 0].plot(theta_history[8, :], label='sigma_xi')
axes[4, 0].legend()
axes[4, 1].plot(mu_history[8, :])
axes[4, 2].plot(C_history[8, :])
plt.savefig('agriculture_full_adam2.pdf')

aver_mu = np.mean(mu_history[:, -10000:],1)
aver_C = np.mean(C_history[:, -10000:],1)
torch.save(aver_mu,"aver_mu_full.pt")
torch.save(aver_C,"aver_C_full.pt")
torch.save(aver_mu[0], 'full_gamma_mu_adam.pt')
torch.save(np.abs(aver_C[0]), 'full_gamma_sd_adam.pt')
torch.save(M_A_history, 'M_A_history_full.pt')

