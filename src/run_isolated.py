import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.reservoir_beggs import BeggsProbabilisticNetwork
from metrics import summarize_metrics


def _make_reservoir(
    n_neurons: int,
    p: float,
    homogeneous: bool,
    device: str,
    batch_size: int,
    allow_self: bool,
    seed: int,
):
    if homogeneous:
        W = torch.ones(batch_size, n_neurons, n_neurons, device=device) * p
    else:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        std = min(p, 1.0 - p) * 0.5
        W = (p + std * torch.randn(batch_size, n_neurons, n_neurons, device=device, generator=g)).clamp(0.0, 1.0)
    if not allow_self:
        idx = torch.arange(n_neurons, device=device)
        W[:, idx, idx] = 0.0
    return BeggsProbabilisticNetwork(W, device=device, allow_self=allow_self)


def run_experiment_metrics(data_dir: str):
    P_VALUES = np.linspace(0, 1, 200)
    N_NEURONS = [16, 32, 64, 128, 256, 512, 1024]

    batch_size = 100
    steps = 100
    homogeneous = True
    allow_self = False
    seed = 42
    device = "cuda"
    n_input = 1

    os.makedirs(data_dir, exist_ok=True)

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    print(f"device: {device}")

    p_arr = np.asarray(P_VALUES, dtype=np.float64)

    for n_neurons in N_NEURONS:
        metrics_lists: dict[str, list] = {}

        for p_val in tqdm(P_VALUES, desc=f"N={n_neurons}", leave=True):
            reservoir = _make_reservoir(n_neurons, float(p_val), homogeneous, device, batch_size, allow_self, seed)
            reservoir.state.zero_()
            reservoir.state[:, :n_input] = 1.0

            states = []
            for _ in range(steps):
                states.append(reservoir.step(external_input=None).clone())
            states = torch.stack(states, dim=1)

            m = summarize_metrics(states)
            for k, v in m.items():
                metrics_lists.setdefault(k, []).append(v)

        def to_np(tlist):
            return torch.stack(tlist).cpu().numpy()

        out = {
            "n_neurons": np.int32(n_neurons),
            "p": p_arr,
            **{k: to_np(v) for k, v in metrics_lists.items()},
            "steps": np.int32(steps),
            "batch_size": np.int32(batch_size),
            "seed": np.int32(seed),
        }
        out_path = os.path.join(data_dir, f"data_{n_neurons}.npz")
        np.savez(out_path, **out)
        print("Saved", out_path)


if __name__ == "__main__":
    metrics_data_dir = "experiments_data/isolated"
    run_experiment_metrics(metrics_data_dir)
