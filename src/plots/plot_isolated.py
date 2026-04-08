import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from run_isolated import _make_reservoir
import glob

def make_exploration_plots(data_dir: str, plots_dir: str):
    CMAP = plt.colormaps["viridis"]
    W = 5
    GOLDEN_RATIO = 1.618
    os.makedirs(plots_dir, exist_ok=True)

    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not npz_files:
        print("No .npz files in", data_dir)
        return

    datasets = []
    for path in npz_files:
        data = np.load(path, allow_pickle=False)
        n = int(data["n_neurons"])
        datasets.append({"n_neurons": n, "path": path, **{k: data[k] for k in data.files}})
        data.close()
    datasets.sort(key=lambda d: d["n_neurons"])

    metrics_to_plot = [
        ("susceptibility_mean", "susceptibility_std", "Susceptibility"),
        ("sigma_eff_mean", "sigma_eff_std", r"Effective $\sigma$"),
        ("activity_level_entropy_mean", "activity_level_entropy_std", "Activity-level entropy"),
    ]

    for mean_key, std_key, title in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(W, W / GOLDEN_RATIO))
        for i, d in enumerate(datasets):
            p_vals = d["p"]
            mean = d[mean_key]
            std = d[std_key]
            label = f"N = {d['n_neurons']}"
            color = CMAP(i / max(len(datasets) - 1, 1))
            ax.plot(p_vals, mean, label=label, color=color)
            ax.fill_between(p_vals, mean - std, mean + std, alpha=0.25, color=color)
        ax.set_xlabel(r"$p$ (edge probability)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"plot_{mean_key}.png"), dpi=150)
        plt.close(fig)

    print("Plots saved in", plots_dir)



def _plot_state_history_regimes(regimes_dict: dict, save_path: str):
    order_keys = ["sub", "crit", "super"]
    regimes = [regimes_dict[k] for k in order_keys if k in regimes_dict]
    if not regimes:
        raise ValueError("regimes_dict must contain at least one of 'sub', 'crit', 'super'")

    def _to_np(x):
        if hasattr(x, "cpu"):
            return x.cpu().numpy()
        return np.asarray(x)

    def _pick_run(state: np.ndarray):
        if state.ndim == 2:
            return state
        B, T, _N = state.shape
        duration = np.zeros(B, dtype=int)
        for b in range(B):
            A = state[b].sum(axis=1)
            silent = np.where(A == 0)[0]
            duration[b] = int(silent[0]) if len(silent) > 0 else T
        run_idx = int(np.argmax(duration))
        return state[run_idx]

    GOLDEN_RATIO = 1.618
    HEIGHT = 4
    WIDTH = HEIGHT * GOLDEN_RATIO
    PLOT_ACTIVITIES = True

    n_cols = 2 if PLOT_ACTIVITIES else 1
    gridspec_kw = {"width_ratios": [1, 0.35]} if PLOT_ACTIVITIES else {}
    fig, axes_grid = plt.subplots(len(regimes), n_cols, figsize=(WIDTH, HEIGHT), gridspec_kw=gridspec_kw)
    if len(regimes) == 1:
        axes_grid = axes_grid[np.newaxis, :]
    if not PLOT_ACTIVITIES:
        axes_grid = axes_grid[:, np.newaxis]

    states_2d = []
    A_list = []
    for r in regimes:
        state = _to_np(r["state"])
        state = _pick_run(state)
        states_2d.append(state)
        A_list.append(state.sum(axis=1))
    N_neurons = states_2d[0].shape[1]
    A_max = max(np.max(A) for A in A_list)

    for i, r in enumerate(regimes):
        state = states_2d[i]
        A_t = A_list[i]
        label = r.get("label", order_keys[i] if i < len(order_keys) else f"Regime {i}")
        state_raster = state.T

        ax_raster = axes_grid[i, 0]
        ax_raster.imshow(state_raster, aspect="auto", cmap="binary_r", interpolation="nearest", origin="lower")
        ax_raster.set_ylim(0, N_neurons)
        ax_raster.set_yticks([])
        ax_raster.set_ylabel("Neuron index")
        ax_raster.text(
            0.02,
            0.97,
            label,
            transform=ax_raster.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=1, edgecolor="none"),
        )
        if PLOT_ACTIVITIES:
            ax_act = axes_grid[i, 1]
            ax_act.fill_between(np.arange(len(A_t)), A_t, alpha=0.7, color="grey")
            ax_act.plot(A_t, color="grey", linewidth=1)
            ax_act.set_ylabel("A(t)")
            ax_act.set_ylim(0, A_max)
            ax_act.set_yticks([])
        if i < len(regimes) - 1:
            ax_raster.tick_params(bottom=False, labelbottom=False)
            if PLOT_ACTIVITIES:
                axes_grid[i, 1].tick_params(bottom=False, labelbottom=False)
        else:
            ax_raster.set_xlabel("Time")
            if PLOT_ACTIVITIES:
                axes_grid[i, 1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.close(fig)


def run_state_history_plot(plots_dir: str):
    REGIME_PLOT_PARAMS = {
        "sub": {"p_val": 0.003, "seed": 387329, "label": "Subcritical"},
        "crit": {"p_val": 0.0043, "seed": 227735, "label": "Near critical"},
        "super": {"p_val": 0.01, "seed": 99298, "label": "Supercritical"},
    }

    n_input = 1
    homogeneous = True
    allow_self = False
    batch_size = 1
    device = "cpu"
    n_neurons = 256
    steps = 1000

    os.makedirs(plots_dir, exist_ok=True)
    regimes_state = {}
    for key in ["sub", "crit", "super"]:
        r = REGIME_PLOT_PARAMS[key]
        p_val, seed, label = r["p_val"], r["seed"], r["label"]
        torch.manual_seed(seed)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        reservoir = _make_reservoir(n_neurons, p_val, homogeneous, device, batch_size, allow_self, seed)
        reservoir.state.zero_()
        reservoir.state[:, :n_input] = 1.0

        states_list = []
        for _ in range(steps):
            states_list.append(reservoir.step(external_input=None).clone())
        states = torch.stack(states_list, dim=1)
        regimes_state[key] = {"state": states[0].detach().cpu(), "label": label}

    _plot_state_history_regimes(regimes_state, save_path=os.path.join(plots_dir, "example_history.png"))


if __name__ == "__main__":
    metrics_data_dir = "experiments_data/isolated"
    plots_dir = "media/isolated"

    make_exploration_plots(metrics_data_dir, plots_dir)
    run_state_history_plot(plots_dir)
