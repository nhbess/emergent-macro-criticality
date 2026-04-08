import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import warnings
import numpy as np
import powerlaw as pl
from scipy.special import zeta


def single_cascade_size_duration(states: torch.Tensor):
    A = states.sum(dim=2).to(torch.int32)   # (B,T)
    B, T = A.shape

    active = A > 0
    silent = ~active

    has_silent = silent.any(dim=1)                      # (B,)
    first_silent = silent.to(torch.int32).argmax(dim=1) # (B,) returns 0 if none
    duration = torch.where(
        has_silent,
        first_silent,                                   # first t with A(t)=0
        torch.full((B,), T, device=A.device)            # didn’t die within window
    ).to(torch.int32)

    t = torch.arange(T, device=A.device).unsqueeze(0)   # (1,T)
    
    return A, duration


def mean_activity_per_run(states: torch.Tensor) -> torch.Tensor:
    # mean fraction of active neurons over time per run
    A = states.sum(dim=2).float()  # (B,T)
    N = states.shape[2]
    return (A / N).mean(dim=1)     # (B,)

def activity_entropy_per_run(states: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Per-run mean binary entropy of the population activity fraction p(t)=A(t)/N.

    states: (B,T,N) in {0,1}
    returns: (B,) in bits
    """
    A = states.sum(dim=2).float()          # (B,T)
    N = states.shape[2]
    p = (A / N).clamp(eps, 1.0 - eps)      # (B,T)

    H_t = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))  # (B,T), bits
    return H_t.mean(dim=1)                 # (B,)

def activity_level_entropy(states: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Entropy of the distribution of activity levels A(t) across all (B,T) samples.

    states: (B,T,N) in {0,1}
    returns: scalar in bits
    """
    A = states.sum(dim=2).to(torch.int64)  # (B,T)
    flat = A.reshape(-1)

    counts = torch.bincount(flat)          # exact over 0..max(A)
    p = counts.float() / counts.sum().clamp_min(1.0)
    p = p.clamp_min(eps)

    H = -(p * torch.log2(p)).sum()         # scalar, bits
    return H


def activity_level_entropy_per_run(states: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Per-run entropy of the distribution of activity levels A(t) over time.
    For each run b, entropy of the empirical distribution of A(b,t).
    states: (B,T,N) in {0,1}
    returns: (B,) in bits
    """
    A = states.sum(dim=2).to(torch.int64)   # (B,T)
    B, T = A.shape
    N_max = states.shape[2] + 1  # activity in [0, N], avoid .item() sync
    device = A.device
    # Vectorized: offset rows so one bincount over flat gives per-row counts
    offset = torch.arange(B, device=device, dtype=torch.int64).unsqueeze(1) * N_max
    flat = (A + offset).flatten()
    counts = torch.bincount(flat, minlength=B * N_max).view(B, N_max).float()
    p = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    p = p.clamp_min(eps)
    H_b = -(p * torch.log2(p)).sum(dim=1)
    return H_b


def total_activity(states: torch.Tensor) -> torch.Tensor:
    # states: (B,T,N) in {0,1}
    return states.sum(dim=2).float()  # A: (B,T)

def propagation_prob(states: torch.Tensor) -> torch.Tensor:
    # “produced at least one internal parent->child step” in the RECORDED series
    # i.e. there exists t with A(t)>0 and A(t+1)>0
    A = total_activity(states)              # (B,T)
    if A.shape[1] < 2:
        return torch.tensor(float("nan"), device=A.device)
    propagated = (A[:, :-1] > 0) & (A[:, 1:] > 0)
    return propagated.any(dim=1).float().mean()  # scalar

def susceptibility(states: torch.Tensor) -> torch.Tensor:
    # Var of activity across time and batch
    A = total_activity(states)              # (B,T)
    return A.reshape(-1).var(unbiased=False)  # scalar


def susceptibility_per_run(states: torch.Tensor) -> torch.Tensor:
    # Var of activity over time for each batch element -> (B,) values per run
    A = total_activity(states)              # (B,T)
    return A.var(dim=1, unbiased=False)     # (B,)

def sigma_eff_allruns(states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # per-run effective branching ratio over all recorded parent steps
    A = total_activity(states)              # (B,T)
    B, T = A.shape
    if T < 2:
        return torch.full((B,), float("nan"), device=A.device)

    num = A[:, 1:].sum(dim=1)
    den = A[:, :-1].sum(dim=1).clamp_min(eps)
    return num / den  # (B,)

def summarize_metrics(states: torch.Tensor) -> dict:
    A = total_activity(states)  # (B,T)

    # per-run propagation (0 or 1 per batch element)
    prop_runs = ((A[:, :-1] > 0) & (A[:, 1:] > 0)).any(dim=1).float()

    # per-run sigma_eff
    sigma_eff = sigma_eff_allruns(states)
    # per-run susceptibility (var of activity over time per batch element)
    chi_runs = susceptibility_per_run(states)
    # per-run activity entropy (mean binary entropy of p(t)=A(t)/N over time)
    act_ent_runs = activity_entropy_per_run(states)
    # per-run activity-level entropy (entropy of distribution of A(t) over time)
    level_ent_runs = activity_level_entropy_per_run(states)
    # per-run mean activity fraction over time
    mean_act_runs = mean_activity_per_run(states)

    # Return scalar tensors (no .item()) so caller can batch and sync once
    return {
        "propagation_prob_mean": prop_runs.mean(),
        "propagation_prob_std": prop_runs.std(unbiased=False),

        "susceptibility_mean": chi_runs.mean(),
        "susceptibility_std": chi_runs.std(unbiased=False),

        "sigma_eff_mean": sigma_eff.mean(),
        "sigma_eff_std": sigma_eff.std(unbiased=False),
        "sigma_eff_median": sigma_eff.median(),

        "activity_entropy_mean": act_ent_runs.mean(),
        "activity_entropy_std": act_ent_runs.std(unbiased=False),

        "activity_level_entropy_mean": level_ent_runs.mean(),
        "activity_level_entropy_std": level_ent_runs.std(unbiased=False),

        "mean_activity_mean": mean_act_runs.mean(),
        "mean_activity_std": mean_act_runs.std(unbiased=False),

        "sigma_eff": sigma_eff,  # keep tensor if needed
    }


def all_avalanche_durations_sizes(states_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract all complete avalanches from a (B, T, N) binary state tensor."""
    A = states_np.sum(axis=2)
    B, T = A.shape
    all_durations: list[int] = []
    all_sizes: list[int] = []

    for b in range(B):
        a = A[b]
        active = a > 0
        padded = np.empty(T + 2, dtype=np.int8)
        padded[0] = 0
        padded[1:-1] = active.astype(np.int8)
        padded[-1] = 0
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            if e == T:
                continue
            all_durations.append(int(e - s))
            all_sizes.append(int(a[s:e].sum()))

    return np.array(all_durations, dtype=np.int32), np.array(all_sizes, dtype=np.int32)


def fit_discrete_power_law_mle(data: np.ndarray, min_points: int = 20) -> dict:
    """Fit a discrete power law p(x) ∝ x^{-alpha} with KS-selected xmin."""

    def _failed_fit():
        return {
            "alpha": np.nan,
            "ks": np.nan,
            "sigma": np.nan,
            "LR_vs_exp": np.nan,
            "LR_pvalue": np.nan,
            "n_fit": 0,
            "xmin": None,
            "t_vals": None,
            "pdf": None,
            "fitted_pdf": None,
        }

    data = np.asarray(data)
    data = data[np.isfinite(data) & (data > 0)].astype(int)
    if data.size < min_points:
        return _failed_fit()

    t_vals, counts = np.unique(data, return_counts=True)
    if t_vals.size < 2:
        return _failed_fit()
    pdf = counts / counts.sum()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = pl.Fit(data, discrete=True, verbose=False)

        alpha = float(fit.power_law.alpha)
        xmin = int(fit.power_law.xmin)
        ks = float(fit.power_law.D)
        sigma = float(fit.power_law.sigma)
        n_fit = int(fit.power_law.n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            LR_vs_exp, LR_pvalue = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
        LR_vs_exp = float(LR_vs_exp)
        LR_pvalue = float(LR_pvalue)
    except (ValueError, RuntimeError):
        return _failed_fit()

    tail_mass = n_fit / len(data)
    fitted_pdf = np.full_like(t_vals, np.nan, dtype=float)
    tail_mask = t_vals >= xmin
    fitted_pdf[tail_mask] = tail_mass * (t_vals[tail_mask].astype(float) ** (-alpha)) / zeta(alpha, xmin)

    return {
        "alpha": alpha,
        "ks": ks,
        "sigma": sigma,
        "LR_vs_exp": LR_vs_exp,
        "LR_pvalue": LR_pvalue,
        "t_vals": t_vals.astype(float),
        "pdf": pdf,
        "fitted_pdf": fitted_pdf,
        "xmin": xmin,
        "n_fit": n_fit,
    }