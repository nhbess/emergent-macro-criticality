import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

N = 256
L = 565.0
VR_DATA = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
                    75, 80, 85, 90, 95, 100, 150, 200, 250])
_VR_CONN = np.sqrt(np.log(N) / ((N - 1) * np.pi / L ** 2))  # ≈ 47


def _root_slug(root_path: str) -> str:
    return os.path.basename(os.path.normpath(root_path)) or "analysis"


def _out_dir(root_path: str) -> str:
    key = _root_slug(root_path)
    d = os.path.join("media", key)
    os.makedirs(d, exist_ok=True)
    return d


def _connectivity_output_path(root_path: str) -> str:
    key = _root_slug(root_path)
    return os.path.join(_out_dir(root_path), f"{key}_theory_connectivity.csv")


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def _connected_components(adj: np.ndarray) -> np.ndarray:
    """Union-Find on a boolean adjacency matrix -> component sizes."""
    n = adj.shape[0]
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r, c in zip(*np.where(adj)):
        pr, pc = find(int(r)), find(int(c))
        if pr != pc:
            parent[pr] = pc

    roots = [find(i) for i in range(n)]
    _, sizes = np.unique(roots, return_counts=True)
    return sizes


def make_data_connectivity(root_path: str, n_trials: int = 2000, seed: int = 42) -> str:
    """Monte Carlo estimate of G(vr) and g(vr), saved as theory_connectivity.csv."""
    rng = np.random.default_rng(seed)
    G_runs = np.zeros((len(VR_DATA), n_trials))
    g_runs = np.zeros((len(VR_DATA), n_trials))

    for t in tqdm(range(n_trials), desc="MC connectivity"):
        pos = rng.uniform(0, L, size=(N, 2))
        dx = np.abs(pos[:, 0:1] - pos[:, 0])
        dx = np.minimum(dx, L - dx)
        dy = np.abs(pos[:, 1:2] - pos[:, 1])
        dy = np.minimum(dy, L - dy)
        dist2 = dx ** 2 + dy ** 2
        np.fill_diagonal(dist2, L ** 2 + 1)

        for vi, vr in enumerate(VR_DATA):
            sizes = _connected_components(dist2 <= float(vr) ** 2)
            G_runs[vi, t] = (sizes ** 2).sum() / N ** 2
            g_runs[vi, t] = sizes.max() / N

    result = pd.DataFrame({
        "vr": VR_DATA,
        "G_mean": G_runs.mean(axis=1),
        "G_std": G_runs.std(axis=1),
        "g_mean": g_runs.mean(axis=1),
        "g_std": g_runs.std(axis=1),
    })
    out = _connectivity_output_path(root_path)
    result.to_csv(out, index=False)
    print(f"Saved {len(result)} rows → {out}")
    return out


def _load_connectivity(root_path: str) -> pd.DataFrame:
    key = _root_slug(root_path)
    candidates = [
        _connectivity_output_path(root_path),                 # new artifact location
        os.path.join("media", key, "theory_connectivity.csv"),  # transitional location
        os.path.join(root_path, "theory_connectivity.csv"),   # legacy location
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    print("theory_connectivity.csv not found, generating it now...")
    make_data_connectivity(root_path)
    return pd.read_csv(_connectivity_output_path(root_path))

def plot_connectivity(root_path: str) -> None:
    """Plot g(vr) (giant connected fraction) from theory_connectivity.csv."""
    mc   = _load_connectivity(root_path)
    mask = mc["vr"].values <= 100
    vr   = mc["vr"].values[mask]

    GOLDEN_RATIO = 1.618
    H = 2
    W = H * GOLDEN_RATIO
    fig, ax = plt.subplots(figsize=(W, H))

    ax.plot(vr, mc["g_mean"].values[mask], color="black", lw=2,
            label=r"$g(v_r)$")
    ax.fill_between(vr,
                    mc["g_mean"].values[mask] - mc["g_std"].values[mask],
                    mc["g_mean"].values[mask] + mc["g_std"].values[mask],
                    alpha=0.20, color="black")
    ax.axvline(_VR_CONN, color="black", ls="--", lw=1.3,
               label=rf"$v_r^{{\rm conn}} \approx {_VR_CONN:.0f}$")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlim(vr.min(), 80)
    ax.set_xlabel(r"$v_r$")
    ax.set_ylabel(r"$g(v_r)$")
    ax.legend(loc="best")

    _save(fig, os.path.join(_out_dir(root_path), "connectivity_vs_vr.png"))

if __name__ == "__main__":
    root_path = "experiments_data/vision_radii"
    #make_data_connectivity(root_path)
    plot_connectivity(root_path)