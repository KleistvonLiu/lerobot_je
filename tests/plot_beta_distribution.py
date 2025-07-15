import jax
import torch
import matplotlib.pyplot as plt

def sample_beta(alpha, beta, bsize):
    gamma1 = torch.empty((bsize,)).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,)).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)

def sample_time(bsize):
    time_beta = sample_beta(1.5, 1.0, bsize)
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=torch.float32)

batch_size = 10000
##### lerobot 实现
# time = sample_time(batch_size)
# time_samples = sample_time(batch_size).numpy()

##### openpi 实现
seed = 42                          # 任何整数都行
rng  = jax.random.PRNGKey(seed)
rng, time_rng = jax.random.split(rng)
# time = jax.random.beta(time_rng, 1.5, 1, batch_size) * 0.999 + 0.001
# time_samples = time

##### 我自己测试
beta_dist  = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
time_samples = beta_dist.sample((batch_size,)).numpy()      # 注意多了一个圆括号

#####
# def true_beta(alpha, beta, size, device=None):
#     gamma1 = torch.distributions.Gamma(alpha, 1.).sample((size,)).to(device)
#     gamma2 = torch.distributions.Gamma(beta, 1.).sample((size,)).to(device)
#     return gamma1 / (gamma1 + gamma2)
# time = true_beta(1.5, 1.0, batch_size)
# time_samples = sample_time(batch_size).numpy()
########
# ---------- PyTorch ----------
# batch_shape = (batch_size,)                 # 必须是 tuple
# time_torch = torch.distributions.Beta(1.5, 1.0).sample(batch_shape)
# time_samples = time_torch.numpy()
# ---------- JAX --------------
# time_jax = jax.random.beta(time_rng, 1.5, 1.0, batch_shape)


# Plot histogram
plt.figure(figsize=(8, 4))
plt.hist(time_samples, bins=500, density=True, edgecolor='black', alpha=0.7)
plt.title("Histogram of sampled time values (Beta(1.5, 1.0) scaled to 0.001–1.0)")
plt.xlabel("t")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
