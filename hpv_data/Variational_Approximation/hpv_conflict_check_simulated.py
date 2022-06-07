import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

np.random.seed(3)
torch.manual_seed(123456)
torch.set_default_tensor_type(torch.DoubleTensor)

Y11_np = np.array([7, 6, 10, 10, 1, 1, 10, 4, 35, 0, 10, 8, 4]).reshape((-1, 1))
Y12_np = np.array([111, 71, 162, 188, 145, 215, 166, 37, 173, 143, 229, 696, 93]).reshape((-1, 1))
Y21_np = np.array([16, 215, 362, 97, 76, 62, 710, 56, 133, 28, 62, 413, 194]).reshape((-1, 1))
Y22_np = np.array([26983, 250930, 829348, 157775, 150467, 352445, 553066, 26751, 75815, 150302, 354993, 3683043, 507218]).reshape((-1, 1))
Y22_np = Y22_np/1000
Y22 = torch.tensor(Y22_np, dtype=torch.float64)
Y12 = torch.tensor(Y12_np, dtype=torch.float64)


# 6h -> 446s
S = 100
K1 = 10000
rho = 0.95
epsilon = 1e-6
K2 = 100000
d = 15



time_start=time.time()
#generate W_i
sample_theta1 = np.random.beta(1,1,13)
sample_theta2 = np.random.normal(0,np.sqrt(1000),2)
sample_Y2 = np.zeros([S, 13, 1])
for i in range(S):
    for j in range(13):
        poi_lambda = np.exp(sample_theta2[0] + sample_theta2[1] * sample_theta1[j] + np.log(Y22_np[j]))
        sample_Y2[i,j,:] = np.random.poisson(poi_lambda,1)
sample_Y2 = torch.tensor(sample_Y2)

# Simulate data
sml_gamma = np.random.uniform(0,1,13)
sml_eta = np.random.normal(0, np.sqrt(1000), 2)
sml_poi_mu = np.multiply(Y22_np,np.exp(sml_eta[0] + sml_eta[1] * sml_gamma).reshape(-1,1))
sml_poi_mu_test = np.exp(sml_eta[0] + sml_eta[1] * sml_gamma.reshape(-1,1) + np.log(Y22_np))
sml_z = np.random.binomial(n=Y12_np, p=sml_gamma.reshape(-1,1))
sml_w = np.zeros((13,1))
sml_w_test = np.zeros((13,1))
for i in range(13):
    sml_w[i,0] = np.random.poisson(lam=sml_poi_mu[i,0])
    sml_w_test[i,0] = np.random.poisson(lam=sml_poi_mu_test[i,0])

#Y11 = torch.tensor(sml_z, dtype=torch.float64)
Y11 = torch.tensor(Y11_np, dtype=torch.float64)
Y21 = torch.tensor(sml_w, dtype=torch.float64)


#calculate q(\phi|z) -- cut model
def log_g_1(S, theta, Y11, Y12, Y21, Y22):
    Y11 = Y11.unsqueeze(dim=0)
    Y11 = Y11.expand((S,13,1))
    Y12 = Y12.unsqueeze(dim=0)
    Y12 = Y12.expand((S,13,1))
    A = torch.sum((- Y12 - 2) * torch.log(1 + torch.exp(- theta[:,:13,:])) - (Y12 - Y11 + 1) * theta[:,:13,:],1)
    return A

#calculate q(\phi|z, w_i) -- full model
def log_g_new(S, theta, Y11, Y12, sample_Y2, Y22):
    # for theta with size [S, d, 1]
    Y11 = Y11.unsqueeze(dim=0)
    Y11 = Y11.expand((S,13,1))
    Y12 = Y12.unsqueeze(dim=0)
    Y12 = Y12.expand((S,13,1))
    Y22 = Y22.unsqueeze(dim=0)
    Y22 = Y22.expand((S,13,1))
    Y21 = sample_Y2
    theta21 = theta[:,13,:].unsqueeze(dim=1)
    theta21 = theta21.expand((S,13,1))
    theta22 = theta[:,14,:].unsqueeze(dim=1)
    theta22 = theta22.expand((S,13,1))
    A = torch.sum((- Y12 - 2) * torch.log(1 + torch.exp(- theta[:,:13,:])) - (Y12 - Y11 + 1) * theta[:,:13,:],1)
    B = torch.sum(- torch.exp(theta21 + theta22/(1 + torch.exp(-theta[:,:13,:])) + torch.log(Y22)) +
                  Y21 * (theta21 + theta22/(1 + torch.exp(-theta[:,:13,:])) + torch.log(Y22)),1)
    C1 = (- theta[:,13,:] ** 2 - theta[:,14,:] ** 2)/2000
    return torch.sum(A + B + C1)


def q_phi_y(K2, S, d, sample_Y2, Y11, Y12, Y22, log_g_new, rho, epsilon):
    mu = torch.ones((S, d, 1)) * (-10)
    C = torch.eye(d)
    C = C.unsqueeze(dim=0)
    C = C.repeat((S, 1, 1))
    E_g_mu = torch.zeros([S, d, 1], dtype=torch.float64)
    E_delta_mu = torch.zeros([S, d, 1], dtype=torch.float64)
    E_g_C = torch.ones([S, d, d], dtype=torch.float64) * 0.01
    E_delta_C = torch.ones([S, d, d], dtype=torch.float64) * 0.01

    def delta_new(C):
        diag = torch.diagonal(C, offset=0, dim1=1, dim2=2)
        D = torch.diag_embed(1 / diag)
        return D

    for t in tqdm(range(K2)):
        z = np.random.multivariate_normal(np.zeros([d]), np.eye(d), S)
        z = torch.tensor(z).reshape((S, d, 1))
        theta = (torch.bmm(C, z) + mu).clone().detach().requires_grad_(True)
        gradient = log_g_new(S, theta, Y11, Y12, sample_Y2, Y22)
        gradient.backward()

        g_mu = theta.grad
        E_g_mu = rho * E_g_mu + (1 - rho) * torch.mul(g_mu, g_mu)
        delta_mu = (torch.sqrt(E_delta_mu + epsilon) / torch.sqrt(E_g_mu + epsilon)) * g_mu
        E_delta_mu = rho * E_delta_mu + (1 - rho) * torch.mul(delta_mu, delta_mu)
        mu = mu + delta_mu

        g_C = torch.bmm(theta.grad, z.transpose(1, 2)) + delta_new(C)
        E_g_C = rho * E_g_C + (1 - rho) * torch.mul(g_C, g_C)
        delta_C = (torch.sqrt(E_delta_C + epsilon) / torch.sqrt(E_g_C + epsilon)) * g_C
        delta_C = torch.tril(delta_C, 0)
        E_delta_C = rho * E_delta_C + (1 - rho) * torch.mul(delta_C, delta_C)
        C = C + delta_C

    mu_y = mu[:, :13, :].detach().numpy().squeeze()
    Sigma_y = (torch.bmm(C, C.transpose(1, 2))[:, :13, :13]).detach().numpy().squeeze()
    return mu_y, Sigma_y


mu_y_sample, Sigma_y_sample =q_phi_y(K2, S, d, sample_Y2, Y11, Y12, Y22, log_g_new, rho, epsilon)



mu_z, Sigma_z = q_phi_y(K1, 1, 13, Y21, Y11, Y12, Y22, log_g_1, rho, epsilon)
mu_y_data, Sigma_y_data =q_phi_y(K2, 1, d, Y21, Y11, Y12, Y22, log_g_new, rho, epsilon)


#calculate T(w|z)
def T_w_z_sample(mu_y, Sigma_y, mu_z, Sigma_z):
    S = mu_y.shape[0]
    T = []
    for i in range(S):
        Sigma_y_i = np.squeeze(Sigma_y[i,:,:])
        mu_y_i = mu_y[i,]
        T_i = 0.5*(np.log(np.linalg.det(Sigma_z)/np.linalg.det(Sigma_y_i)) +
                   np.trace(np.linalg.solve(Sigma_z, Sigma_y_i)) - 13 +
                   (mu_z - mu_y_i).T.dot(np.linalg.solve(Sigma_z, mu_z-mu_y_i)))
        T.append(T_i)
    return T

def T_w_z_data(mu_y, Sigma_y, mu_z, Sigma_z):
    T = 0.5 * (np.log(np.linalg.det(Sigma_z) / np.linalg.det(Sigma_y)) +
                 np.trace(np.linalg.solve(Sigma_z, Sigma_y)) - 13 +
                 (mu_z - mu_y).T.dot(np.linalg.solve(Sigma_z, mu_z - mu_y)))
    return T

T_sample = T_w_z_sample(mu_y_sample, Sigma_y_sample, mu_z, Sigma_z)
T_data = T_w_z_data(mu_y_data, Sigma_y_data, mu_z, Sigma_z)

def p_value(T_sample, T_data):
    count = 0
    S = len(T_sample)
    for i in range(S):
        if T_sample[i] >= T_data:
            count = count + 1
    count = count/S
    return count

p_value = p_value(T_sample, T_data)
print(p_value)

time_end=time.time()
print('time cost',time_end-time_start,'s')

np.save('hpv-T-sample.npy',T_sample)
np.save('hpv-T-data.npy',T_data)


plt.clf()
sns.distplot(T_sample, hist = False, kde = True,
            kde_kws = {'linewidth': 3})
plt.axvline(T_data, color="black",alpha=1, linewidth=2)


plt.xlabel("Test Statistic",fontsize=20)
plt.ylabel("Density",fontsize=20)

plt.savefig('HPV-conflict-check-simulated.pdf')


