import torch
import torch.nn as nn

from dataset import data_preprocessing
from config import (NOISE_STD, DROP_P, LAMBDA_STAB, LAMBDA_MMD, LAMBDA_AC, LEARNING_RATE, WEIGHT_DECAY)

# Dimensions
CAT_EMB_DIM  = 8
THER_EMB_DIM = 16
S_DIM, C_DIM = 32, 32
STEP_EMB_DIM = 32     # diffusion-step embedding dim
T_STEPS      = 100    # number of diffusion steps

class InputProcessing(nn.Module):
    """
    Input processing
    Concatenate numerical and categorical features (Dim 8)
    Output Vector x
    """
    def __init__(self, num_dim: int, cat_dim: list[int]):
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(c, CAT_EMB_DIM) for c in cat_dim])
        self.din = num_dim + len(cat_dim) * CAT_EMB_DIM     # Din for Encoder
        # print dimensions for debugging
        # print(f"InputProcessing: num_dim={num_dim}, cat_dim={cat_dim}, din={self.din}")

    def forward(self, num, cat):
        cat_vecs = [self.cat_embs[i](cat[:, i]) for i in range(len(self.cat_embs))]
        return torch.cat([num] + cat_vecs, dim=1)   # (B, din)


class Encoder(nn.Module):
    """
    2-Layer MLP(256, 128) + BatchNorm + ReLU + dropout 0.1
    Output: [S, C]
    """
    def __init__(self, din: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1), # Hidden Layer 1, 256
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1), # Hidden Layer 2, 128
            nn.Linear(128, S_DIM + C_DIM), # Output
        )

    def forward(self, x):
        h = self.net(x)
        return h[:, :S_DIM], h[:, S_DIM:]   # [S, C]

class PropensityHead(nn.Module):
    """
    Output: Logits over K
    
    1 Layer MLP (64) + ReLU
    """
    def __init__(self, K: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, K),
        )

    def forward(self, C):
        return self.net(C)  # (B, K)

class OutcomeModel(nn.Module):
    """
    f_theta(S, C, t) -> y, next_visit_HAMD

    2-Layer MLP(128, 64) + BatchNorm + ReLU + dropout 0.1
    """
    def __init__(self, K: int):
        super().__init__()
        self.treatment_embedding = nn.Embedding(K, THER_EMB_DIM) # DIM (K, 16)
        din = S_DIM + C_DIM + THER_EMB_DIM
        self.net = nn.Sequential(
            nn.Linear(din, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, S, C, t):
        te = self.treatment_embedding(t)    # (B, 16)
        return self.net(torch.cat([S, C, te], dim=1))   # (B, 1)


class ITEHead(nn.Module):
    """
    g(S, C) -> K-dim vector
    """
    def __init__(self, K: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(S_DIM + C_DIM, 64), 
            nn.ReLU(), 
            nn.Linear(64, K),
        )

    def forward(self, S, C):
        return self.net(torch.cat([S, C], dim=1))      # (B, K)


class Denoiser(nn.Module):
    """
    Denoiser(C_t, step, S, t) -> eps_hat
    Predict the noise that was added. 
    """
    def __init__(self, K: int, T: int = T_STEPS):
        super().__init__()
        self.step_emb = nn.Embedding(T, STEP_EMB_DIM)   # step index -> vector[32]
        self.t_emb    = nn.Embedding(K, THER_EMB_DIM)   # treatment -> vector[16]
        din = C_DIM + STEP_EMB_DIM + S_DIM + THER_EMB_DIM
        self.net = nn.Sequential(
            nn.Linear(din, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, C_DIM),  # predicted noise
        )

    def forward(self, C_t, step, S, t):
        h = torch.cat([C_t, self.step_emb(step), S, self.t_emb(t)], dim=1)
        return self.net(h)


def augment(x, noise_std=NOISE_STD, drop_p=DROP_P, invert_dropout=True):
    """
    Data augmentation for stable representation learning
    Guassian Noise + Feature Dropout
    """
    gauss = torch.randn_like(x) * noise_std
    mask = (torch.rand_like(x) > drop_p).float()
    return (x + gauss) * mask / (1 - drop_p*invert_dropout)        # inverted dropout


def anti_collapse(z1, z2, gamma=1.0, eps=1e-4):
    """
    z1:(B,d), z2:(B,d) -> average(var_loss + cov_loss)
    Loss Term for Anti-Collapse:
    Decorrelation & Variance Control
    """
    def ac(z):
        # Centered representations
        zc  = z - z.mean(dim=0)
        cov = (zc.T @ zc) / (z.shape[0] - 1)        # Matrix (d,d)

        # Variance
        diag = torch.diagonal(cov) 
        std  = torch.sqrt(diag + eps)
        var_loss = torch.relu(gamma - std).mean()

        # Covariance
        off = cov - torch.diag(diag)
        cov_loss = (off ** 2).sum() / z.shape[1]

        return var_loss + cov_loss
    return (ac(z1) + ac(z2)) / 2


def MMD(C, t, K):
    """
    Average MMD of each treatment group's C vs the rest. 
    C:(B,d) t:(B,) -> scalar
    """

    def rbf_mmd2(a, b, sigma=1.0):
        """RBF-kernel MMD^2: a:(n,d) b:(m,d) -> scalar"""
        def k(x, y):
            return torch.exp(-torch.cdist(x, y) ** 2 / (2 * sigma ** 2))
        return k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean()

    total = C.new_zeros(())
    count = 0
    for k in range(K):
        mask = (t == k)
        if mask.sum() < 2 or (~mask).sum() < 2:
            continue
        total = total + rbf_mmd2(C[mask], C[~mask])
        count += 1
    return total / max(count, 1)


def make_schedule(T=T_STEPS):
    """
    Linear noise schedule. 
    Returns (betas, alphas, alpha_bars).
    """
    betas = torch.linspace(1e-4, 0.02, T)       # Linear Noise Schedule
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)   # cumulative product
    return betas, alphas, alpha_bars


@torch.no_grad()
def sample_C(denoiser, S, t, schedule):
    """
    sample_C(denoiser, S, t, schedule) -> C_cf
    Reverse diffusion (DDPM ancestral sampling): 
    generate a counterfactual C from pure noise,
    conditioned on (S, t). 
    """
    betas, alphas, alpha_bars = schedule
    T = len(betas)                                          # number of diffusion steps
    B = S.shape[0]                                          # batch size
    C_cf = torch.randn(B, C_DIM)                            # start from pure noise (x_T)
    for step in reversed(range(T)):
        step_b = torch.full((B,), step, dtype=torch.long)   # current step index
        eps = denoiser(C_cf, step_b, S, t)                  # predicted noise at this step
        a, ab = alphas[step], alpha_bars[step]
        # posterior mean: 1/sqrt(a) * (C - (1-a)/sqrt(1-ab) * eps) - DDPM eqn
        C_cf = (C_cf - (1 - a) / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)

        if step > 0:                                        # add noise except at the last step
            C_cf = C_cf + torch.sqrt(betas[step]) * torch.randn_like(C_cf)  # ancestral sampling
    return C_cf
