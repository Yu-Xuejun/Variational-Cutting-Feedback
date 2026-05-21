# NOTE: Set n_jobs (below) to the number of parallel workers suitable for your machine.
# A safe default is half the number of physical CPU cores (e.g. 4 on an 8-core laptop).

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

S = 100
K1 = 10000
rho = 0.95
epsilon = 1e-6
K2 = 200000
d = 15

# --- optimizer / annealing selection (combinable) ---
USE_NATURAL_GRAD = True   # True = Algorithm 1N (natural gradient + momentum)
                           # False = Algorithm 1E (Euclidean gradient + ADADELTA, original)
USE_TEMPERED     = True   # True  = likelihood tempering (anneals B-term 0→1 over K_temp stages)
                           # False = direct optimisation with K2 steps (no annealing)

# natural gradient hyperparams (only used when USE_NATURAL_GRAD=True)
NG_LR   = 0.001   # fixed learning rate
NG_BETA = 0.9    # momentum decay
# likelihood tempering hyperparams
K_temp = 20       # number of temperature stages (0 .. K_temp inclusive)
K2_per_step = 10000  # iterations per temperature stage

np.random.seed(3)
torch.manual_seed(123456)
torch.set_default_tensor_type(torch.DoubleTensor)

Y11_np = np.array([7, 6, 10, 10, 1, 1, 10, 4, 35, 0, 10, 8, 4]).reshape((-1, 1))
Y11 = torch.tensor(Y11_np, dtype=torch.float64)
Y12_np = np.array([111, 71, 162, 188, 145, 215, 166, 37, 173, 143, 229, 696, 93]).reshape((-1, 1))
Y12 = torch.tensor(Y12_np, dtype=torch.float64)
Y22_np = np.array([26983, 250930, 829348, 157775, 150467, 352445, 553066, 26751, 75815, 150302, 354993, 3683043, 507218]).reshape((-1, 1))
Y22 = torch.tensor(Y22_np, dtype=torch.float64)
Y22 = Y22 / 1000

eta_prior_var = 100

# --- simulate observed data: Y21 = sml_w (this is the only difference from the real-data script) ---
sml_gamma = np.random.beta(1, 1, 13)
sml_eta   = np.random.normal(0, np.sqrt(eta_prior_var), 2)

# simulate z (HPV infection counts)
Y11_np_sml = np.random.binomial(Y12_np.flatten(), sml_gamma).reshape(-1, 1)
Y11 = torch.tensor(Y11_np_sml, dtype=torch.float64)  # overwrite with simulated z for all inference

# simulate w (cervical cancer counts)
sml_w = np.zeros((13, 1))
for _j in range(13):
    _lam = float(np.exp(sml_eta[0] + sml_eta[1] * sml_gamma[_j]
                        + np.log(Y22_np[_j, 0] / 1000.0)))
    sml_w[_j, 0] = np.random.poisson(lam=_lam)
Y21 = torch.tensor(sml_w, dtype=torch.float64)

time_start = time.time()
# generate W_i (replicated datasets from posterior predictive of simulated z)
sample_Y2 = np.zeros([S, 13, 1])
alpha_post = 1.0 + Y11_np_sml.flatten()
beta_post  = 1.0 + Y12_np.flatten() - Y11_np_sml.flatten()
_n_total = S * 13
_n_capped = 0
for i in range(S):
    gamma_i = np.random.beta(alpha_post, beta_post)
    eta_i   = np.random.normal(0.0, np.sqrt(eta_prior_var), 2)
    _w_i = np.zeros(13)
    _capped_i = 0
    for j in range(13):
        log_offset = float(np.log(Y22_np[j, 0] / 1000.0))  # matches inference
        log_lam = eta_i[0] + eta_i[1] * gamma_i[j] + log_offset
        if log_lam <= 43.6:                            # exact Poisson (numpy limit ≈ 9.2e18)
            _w_i[j] = np.random.poisson(np.exp(log_lam))
        else:
            _capped_i += 1
            lam = np.exp(min(log_lam, 350.0))         # float64 safe range
            _w_i[j] = max(0.0, np.random.normal(lam, np.sqrt(lam)))
    sample_Y2[i, :, 0] = _w_i
    _n_capped += _capped_i
print(f"[cap report] total={_n_total}, "
      f"exact Poisson={_n_total-_n_capped} ({100*(_n_total-_n_capped)/_n_total:.2f}%), "
      f"normal approx (log_lam>43.6)={_n_capped} ({100*_n_capped/_n_total:.2f}%)")
sample_Y2 = torch.tensor(sample_Y2)

#calculate q(\phi|z) -- cut model
def log_g_1(S, theta, Y11, Y12, Y21, Y22):
    Y11 = Y11.unsqueeze(dim=0)
    Y11 = Y11.expand((S,13,1))
    Y12 = Y12.unsqueeze(dim=0)
    Y12 = Y12.expand((S,13,1))
    A = torch.sum((- Y12 - 2) * torch.log(1 + torch.exp(- theta[:,:13,:])) - (Y12 - Y11 + 1) * theta[:,:13,:],1)
    return A

#calculate q(\phi|z, w_i) -- full model
def log_g_new(S, theta, Y11, Y12, sample_Y2, Y22, temperature=1.0):
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
    C1 = (- theta[:,13,:] ** 2 - theta[:,14,:] ** 2)/(2*eta_prior_var)
    return torch.sum(A + temperature * B + C1)


def q_phi_y(K2, S, d, sample_Y2, Y11, Y12, Y22, log_g_new, rho, epsilon, log_interval=1000, verbose=True, grad_clip=10.0, mu_init=None,
            use_natural_grad=False, ng_lr=0.01, ng_beta=0.9):
    mu = torch.ones((S, d, 1)) * (-10)
    if mu_init is not None:
        _mu_init_t = torch.tensor(mu_init, dtype=torch.float64).reshape(-1, 1)
        mu[:, :_mu_init_t.shape[0], :] = _mu_init_t.unsqueeze(0).expand(S, -1, -1)
    C = torch.eye(d)
    C = C.unsqueeze(dim=0)
    C = C.repeat((S, 1, 1))
    E_g_mu = torch.zeros([S, d, 1], dtype=torch.float64)
    E_delta_mu = torch.zeros([S, d, 1], dtype=torch.float64)
    E_g_C = torch.ones([S, d, d], dtype=torch.float64) * 0.01
    E_delta_C = torch.ones([S, d, d], dtype=torch.float64) * 0.01
    if use_natural_grad:
        m_mu = torch.zeros([S, d, 1], dtype=torch.float64)
        m_C  = torch.zeros([S, d, d], dtype=torch.float64)
    elbo_history = []

    def delta_new(C):
        diag = torch.diagonal(C, offset=0, dim1=1, dim2=2)
        D = torch.diag_embed(1 / diag)
        return D

    for t in tqdm(range(K2), disable=not verbose):
        z = np.random.multivariate_normal(np.zeros([d]), np.eye(d), S)
        z = torch.tensor(z).reshape((S, d, 1))
        theta = (torch.bmm(C, z) + mu).clone().detach().requires_grad_(True)
        gradient = log_g_new(S, theta, Y11, Y12, sample_Y2, Y22)
        gradient.backward()

        if t % log_interval == 0:
            elbo_history.append((t, gradient.item()))

        # skip update if ELBO is NaN/Inf (numerically blown-up sample)
        if not torch.isfinite(gradient):
            continue

        # gradient clipping by global L2 norm — suppresses the deep ELBO spikes
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([theta], max_norm=grad_clip)

        if use_natural_grad:
            # --- Algorithm 1N: natural gradient + momentum ---
            g_mu    = theta.grad                                        # [S, d, 1]
            G       = torch.bmm(theta.grad, z.transpose(1, 2))         # [S, d, d]
            G_lower = torch.tril(G)
            H       = torch.bmm(C.transpose(1, 2), G_lower)            # [S, d, d]
            H_bar   = (torch.tril(H, -1) +
                       0.5 * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2)))

            Sigma    = torch.bmm(C, C.transpose(1, 2))
            nat_g_mu = torch.bmm(Sigma, g_mu)
            nat_g_C  = torch.tril(torch.bmm(C, H_bar))

            m_mu = ng_beta * m_mu + (1 - ng_beta) * nat_g_mu
            m_C  = ng_beta * m_C  + (1 - ng_beta) * nat_g_C

            mu = mu + ng_lr * m_mu
            C  = torch.tril(C + ng_lr * m_C)
        else:
            # --- Algorithm 1E: Euclidean gradient + ADADELTA (original) ---
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
    return mu_y, Sigma_y, elbo_history


def q_phi_y_tempered(K2_per_step, K_temp, S, d, sample_Y2, Y11, Y12, Y22,
                     log_g_new_fn, rho, epsilon, mu_z_init,
                     verbose=False, grad_clip=10.0, log_interval=1000,
                     use_natural_grad=False, ng_lr=0.001, ng_beta=0.9):
    """Likelihood-tempered BBVI for q(phi | W^(i), z).

    Anneals the Poisson B-term from weight 0 (cut posterior) to 1 (full
    posterior) over K_temp+1 stages, running K2_per_step steps each.
    Accumulators (ADADELTA or momentum) are shared across stages.
    """
    mu = torch.ones((S, d, 1), dtype=torch.float64) * (-10)
    _mu_init_t = torch.tensor(mu_z_init, dtype=torch.float64).reshape(-1, 1)
    mu[:, :_mu_init_t.shape[0], :] = _mu_init_t.unsqueeze(0).expand(S, -1, -1)

    C = torch.eye(d, dtype=torch.float64).unsqueeze(0).repeat(S, 1, 1)

    E_g_mu = torch.zeros([S, d, 1], dtype=torch.float64)
    E_delta_mu = torch.zeros([S, d, 1], dtype=torch.float64)
    E_g_C = torch.ones([S, d, d], dtype=torch.float64) * 0.01
    E_delta_C = torch.ones([S, d, d], dtype=torch.float64) * 0.01
    if use_natural_grad:
        m_mu = torch.zeros([S, d, 1], dtype=torch.float64)
        m_C  = torch.zeros([S, d, d], dtype=torch.float64)

    elbo_history = []

    def delta_new(C):
        diag = torch.diagonal(C, offset=0, dim1=1, dim2=2)
        return torch.diag_embed(1 / diag)

    for k in range(K_temp + 1):
        temperature = k / K_temp  # 0, 1/K_temp, ..., 1

        global_offset = k * K2_per_step

        for t in tqdm(range(K2_per_step),
                      desc=f'temp={temperature:.2f}', disable=not verbose):
            z = np.random.multivariate_normal(np.zeros([d]), np.eye(d), S)
            z = torch.tensor(z).reshape((S, d, 1))
            theta = (torch.bmm(C, z) + mu).clone().detach().requires_grad_(True)
            gradient = log_g_new_fn(S, theta, Y11, Y12, sample_Y2, Y22,
                                    temperature=temperature)
            gradient.backward()

            global_t = global_offset + t
            if global_t % log_interval == 0:
                elbo_history.append((global_t, gradient.item()))

            if not torch.isfinite(gradient):
                continue

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([theta], max_norm=grad_clip)

            if use_natural_grad:
                # --- Algorithm 1N: natural gradient + momentum ---
                g_mu    = theta.grad
                G       = torch.bmm(theta.grad, z.transpose(1, 2))
                G_lower = torch.tril(G)
                H       = torch.bmm(C.transpose(1, 2), G_lower)
                H_bar   = (torch.tril(H, -1) +
                           0.5 * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2)))

                Sigma    = torch.bmm(C, C.transpose(1, 2))
                nat_g_mu = torch.bmm(Sigma, g_mu)
                nat_g_C  = torch.tril(torch.bmm(C, H_bar))

                m_mu = ng_beta * m_mu + (1 - ng_beta) * nat_g_mu
                m_C  = ng_beta * m_C  + (1 - ng_beta) * nat_g_C

                mu = mu + ng_lr * m_mu
                C  = torch.tril(C + ng_lr * m_C)
            else:
                # --- Algorithm 1E: Euclidean gradient + ADADELTA ---
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
    return mu_y, Sigma_y, elbo_history


def _fit_one_sample(i):
    # forked processes inherit PyTorch's multi-threaded BLAS lock; reset to 1 to avoid deadlock
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        np.random.seed(i + 200)
        torch.manual_seed(i + 200)
        sample_Y2_i = sample_Y2[i:i+1]  # shape [1, 13, 1]
        if USE_TEMPERED:
            mu_i, Sigma_i, elbo_i = q_phi_y_tempered(
                K2_per_step, K_temp, 1, d, sample_Y2_i, Y11, Y12, Y22,
                log_g_new, rho, epsilon, mu_z, verbose=(i == 0),
                use_natural_grad=USE_NATURAL_GRAD, ng_lr=NG_LR, ng_beta=NG_BETA
            )
        else:
            mu_i, Sigma_i, elbo_i = q_phi_y(
                K2, 1, d, sample_Y2_i, Y11, Y12, Y22, log_g_new, rho, epsilon,
                verbose=(i == 0), mu_init=mu_z,
                use_natural_grad=USE_NATURAL_GRAD, ng_lr=NG_LR, ng_beta=NG_BETA
            )
    return mu_i, Sigma_i, elbo_i

mu_z, Sigma_z, elbo_cut = q_phi_y(K1, 1, 13, Y21, Y11, Y12, Y22, log_g_1, rho, epsilon)
# Adjust n_jobs to suit your machine (e.g. half your physical core count).
n_jobs = 50
with Pool(processes=n_jobs) as pool:
    results = list(tqdm(pool.imap(_fit_one_sample, range(S)), total=S, desc='sample loop'))
mu_y_list, Sigma_y_list, elbo_sample_list = zip(*results)
mu_y_sample = np.stack(mu_y_list)       # [S, 13]
Sigma_y_sample = np.stack(Sigma_y_list) # [S, 13, 13]
mu_y_data, Sigma_y_data, elbo_data = q_phi_y(
    K2, 1, d, Y21, Y11, Y12, Y22, log_g_new, rho, epsilon,
    use_natural_grad=USE_NATURAL_GRAD, ng_lr=NG_LR, ng_beta=NG_BETA
)

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

p_value = np.mean(np.array(T_sample) >= T_data)
print("p-value = ", p_value)

time_end = time.time()
print('time cost', time_end - time_start, 's')

np.save('hpv-T-sample-simulated.npy', T_sample)
np.save('hpv-T-data-simulated.npy', T_data)

plt.clf()
sns.distplot(T_sample, hist=False, kde=True, kde_kws={'linewidth': 3})
plt.axvline(T_data, color="black", alpha=1, linewidth=2)
plt.xlabel("Test Statistic", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.savefig('HPV-conflict-check-simulated.pdf')
