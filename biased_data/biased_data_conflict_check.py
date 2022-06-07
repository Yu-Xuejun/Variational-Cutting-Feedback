import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab

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

#generate W_i
S = 100
sample_phi = np.random.normal(n1*np.mean(y1)/(n1+lambda1), scale=1/(n1+lambda1), size=1)
sample_eta = np.random.normal(0,1/lambda2,1)
sample_W = np.zeros([S, n2])
for i in range(S):
    sample_W[i,] = np.random.normal(sample_phi+sample_eta,1,n2)

#calculate q(\phi|z)
sigmasq_z = 1 / (lambda1 + n1)
mu_z = sigmasq_z * np.sum(y1)

#calculate q(\phi|y)
def q_phi_y_sample(sample_W, z, n1, n2, lambda1, lambda2, E):
    S = sample_W.shape[0]
    mu_y = []
    sigmasq_y = []
    for i in range(S):
        mu_q_theta1 = 0
        sigmasq_q_theta1 = 1
        y2 = sample_W[i,]
        y1 = z
        for e in range(E):
            sigmasq_q_theta2 = 1 / (lambda2 + n2)
            mu_q_theta2 = sigmasq_q_theta2 * n2 * (np.average(y2) - mu_q_theta1)
            sigmasq_q_theta1 = 1 / (lambda1 + n1 + n2)
            mu_q_theta1 = sigmasq_q_theta1 * (np.sum(y1) + np.sum(y2) - n2 * mu_q_theta2)
        mu_y.append(mu_q_theta1)
        sigmasq_y.append(sigmasq_q_theta1)
    return mu_y, sigmasq_y


def q_phi_y_data(y2, z, n1, n2, lambda1, lambda2, E):
    mu_q_theta1 = 0
    sigmasq_q_theta1 = 1
    for e in range(E):
        sigmasq_q_theta2 = 1 / (lambda2 + n2)
        mu_q_theta2 = sigmasq_q_theta2 * n2 * (np.average(y2) - mu_q_theta1)
        sigmasq_q_theta1 = 1 / (lambda1 + n1 + n2)
        mu_q_theta1 = sigmasq_q_theta1 * (np.sum(y1) + np.sum(y2) - n2 * mu_q_theta2)
    mu_y = mu_q_theta1
    sigmasq_y = sigmasq_q_theta1
    return mu_y, sigmasq_y


mu_y_sample, sigmasq_y_sample = q_phi_y_sample(sample_W, y1, n1, n2, lambda1, lambda2, E)
mu_y_data, sigmasq_y_data = q_phi_y_data(y2, y1, n1, n2, lambda1, lambda2, E)



#calculate T(w|z)
def T_w_z_sample(mu_y, sigmasq_y, mu_z, sigmasq_z):
    S = len(mu_y)
    T = []
    for i in range(S):
        T_i = 0.5*(np.log(sigmasq_z/sigmasq_y[i]) + (mu_y[i]-mu_z)**2/sigmasq_z)
        T.append(T_i)
    return T

def T_w_z_data(mu_y, sigmasq_y, mu_z, sigmasq_z):
    T = 0.5*(np.log(sigmasq_z/sigmasq_y) + (mu_y-mu_z)**2/sigmasq_z)
    return T

T_sample = T_w_z_sample(mu_y_sample, sigmasq_y_sample, mu_z, sigmasq_z)
T_data = T_w_z_data(mu_y_data, sigmasq_y_data, mu_z, sigmasq_z)

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


#plot
fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
#fig.subplots_adjust(wspace=0.05)  # adjust space between axes
sns.distplot(T_sample,ax=ax1, hist = False, kde = True,
            kde_kws = {'linewidth': 3})
ax2.axvline(T_data, color="black",linewidth=2)

ax1.set_xlim(1.15, 1.35)
ax2.set_xlim(13, 17)
ax1.set_ylim(0, 30)
ax2.set_ylim(0, 30)
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.axes.yaxis.set_ticks([])
d = .015 # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d,1+d), (-d,+d), **kwargs)
ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-3*d,d), (1-d,1+d), **kwargs)
ax2.plot((-3*d,d), (-d,+d), **kwargs)
ax1.set_xlabel("Test Statistic",fontsize=20)
ax1.xaxis.set_label_coords(0.83, -0.07)
ax1.set_ylabel("Density",fontsize=20)
#plt.show()
plt.savefig('biased-data-conflict-check.pdf')
