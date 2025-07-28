import jax
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sample_beta(alpha, beta, bsize):
    gamma1 = torch.empty((bsize,)).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,)).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)

def sample_time(bsize):
    time_beta = sample_beta(1.5, 1.0, bsize)
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=torch.float32)

batch_size = 10_000

# ① previous implementation from lerobot
time_samples1 = sample_time(batch_size).cpu().numpy()

# ② original openpi from pi0 (JAX)
seed = 42
rng  = jax.random.PRNGKey(seed)
rng, time_rng = jax.random.split(rng)
time2 = jax.random.beta(time_rng, 1.5, 1.0, (batch_size,)) * 0.999 + 0.001
time_samples2 = np.array(time2)  # JAX → NumPy

# ③ fixed implementation from lerobot (PyTorch Beta)
beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
time_samples3 = (beta_dist.sample((batch_size,)) * 0.999 + 0.001).cpu().numpy()

# ---- Plot three subplots ----
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, sharey=True)

bins = 200
kwargs = dict(bins=bins, density=True, edgecolor="black", alpha=0.7)

axes[0].hist(time_samples1, **kwargs)
axes[0].set_title("Lerobot previous: Beta(1.5,1.0) scaled to (0.001,1.0)")

axes[1].hist(time_samples2, **kwargs)
axes[1].set_title("OpenPI (JAX): Beta(1.5,1.0) scaled to (0.001,1.0)")

axes[2].hist(time_samples3, **kwargs)
axes[2].set_title("Fixed PyTorch: Beta(1.5,1.0) scaled to (0.001,1.0)")

for ax in axes:
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("t")
for ax in axes:
    ax.set_ylabel("Density")

fig.tight_layout()
fig.savefig("beta_hist_3subplots.png", dpi=200)