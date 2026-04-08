import os
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from tqdm import tqdm

from run_base import run_environment_p
from env_collective import BatchFirefliesEnvironment
from metrics import all_avalanche_durations_sizes, fit_discrete_power_law_mle

import yaml

# General Functions -----------------------------

def save_metadata(data: dict, root: str):
    with open(os.path.join(root, 'metadata.yaml'), 'w') as f:
        yaml.dump(data, f)



def make_environment_states(
    p_values: list[float],
    vision_distance: int = 65,
    n_neurons: int = 50,
    n_agents: int = 256,
    batch_size: int = 20,
    macro_steps: int = 500,
    micro_steps: int = 5,
    n_exteroceptor: int = 8,
    agent_radius: int = 5,
    observation_noise_std: float = 0.0,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    root: str = 'power_laws/environment',
    n_reps: int = 10,
    initialization_mode: int = 4,
):
    density  = 0.0008
    map_side = int(np.sqrt(n_agents / density))
    map_size = [map_side, map_side]

    states_dir = os.path.join(root, 'data_states')
    os.makedirs(states_dir, exist_ok=True)

    Ps = np.sort(np.unique(np.round(p_values, 8)))

    config = {
        'n_neurons':             n_neurons,
        'allow_self':            False,
        'batch_size':            batch_size,
        'macro_steps':           macro_steps,
        'micro_steps':           micro_steps,
        'n_agents':              n_agents,
        'n_exteroceptor':        n_exteroceptor,
        'map_size':              map_size,
        'vision_distance':       vision_distance,
        'agent_radius':          agent_radius,
        'save_frequency':        1,
        'observation_noise_std': observation_noise_std,
        'collisions':            False,
        'binary_observations':   True,
        'seed':                  seed,
        'device':                device,
        'save_micro_states':     False,
    }

    metadata = {
        'p_values':              Ps.tolist(),
        'n_neurons':             int(n_neurons),
        'n_agents':              int(n_agents),
        'n_reps':                int(n_reps),
        'batch_size':            int(batch_size),
        'macro_steps':           int(macro_steps),
        'micro_steps':           int(micro_steps),
        'n_exteroceptor':        int(n_exteroceptor),
        'vision_distance':       int(vision_distance),
        'agent_radius':          int(agent_radius),
        'observation_noise_std': float(observation_noise_std),
        'initialization_mode':   int(initialization_mode),
        'seed':                  int(seed),
        'device':                str(device),
        'density':               float(density),
        'map_size':              [int(map_side), int(map_side)],
    }
    save_metadata(metadata, root)

    env = BatchFirefliesEnvironment(
        batch_size=batch_size,
        n_agents=n_agents,
        map_size=map_size,
        vision_distance=vision_distance,
        n_exteroceptor=n_exteroceptor,
        agent_radius=agent_radius,
        observation_noise_std=observation_noise_std,
        collisions=False,
        save_frequency=1,
        binary_observations=True,
        seed=seed,
    )

    for i, p_val in enumerate(tqdm(Ps, desc=f"environment states N={n_neurons}")):
        for rep in tqdm(range(n_reps), desc=f"  p={p_val:.5f} reps", leave=False):
            rep_seed = seed + i * 1000 + rep

            out_path = os.path.join(states_dir, f"p_{p_val:.6f}_rep_{rep:02d}.npz")
            if os.path.exists(out_path):
                continue

            rollout = run_environment_p(p_val, env, config, seed=rep_seed,
                                         initialization_mode=initialization_mode)

            # transpose (T, B, N) → (B, T, N) to match isolated states layout
            lights_btn = rollout['lights'].transpose(1, 0, 2)

            np.savez(out_path,
                     states=lights_btn.astype(np.int8),
                     p=np.float64(p_val),
                     n_neurons=np.int32(n_neurons),
                     batch_size=np.int32(batch_size),
                     steps=np.int32(macro_steps),
                     seed=np.int32(seed),
                     rep=np.int32(rep),
                     n_reps=np.int32(n_reps))

    print(f"Saved {len(Ps) * n_reps} state files to {states_dir}")
   

# CASCADE ANALYSIS -----------------------------


def compute_avalanches_from_states_dir(
    root: str = 'power_laws/isolated',
):
    """Extract ALL population-level avalanches from each simulation run.

    This is the preferred method for power-law fitting.  It scans the entire
    time series of each run and treats every contiguous block of timesteps
    where the total population activity A(t) > 0 as an independent avalanche
    event.  Consecutive avalanches are separated by at least one silent
    timestep (A(t) = 0).

    For example, a run with activity [0, 3, 5, 2, 0, 0, 4, 1, 0, 2, 3, 0]
    yields three avalanches (durations 3, 2, 2) rather than one.

    Contrast with compute_cascades (legacy), which only captures the very
    first burst per run and discards everything after the first silence.
    That single-cascade approach is appropriate when each run is seeded by
    a deliberate single-neuron perturbation and only the propagation of that
    specific perturbation is of interest.  When runs are not individually
    seeded, discarding later bursts wastes data and can bias the sample.

    Advantages over compute_cascades:
    - Orders-of-magnitude more events from the same simulations, which is
      critical for resolving the power-law tail reliably.
    - Matches the standard avalanche-detection method in the neuroscience
      literature (Beggs & Plenz 2003-style: record continuously, split on
      silence gaps).

    Incomplete bursts that are still active at the last timestep are discarded
    (censored), consistent with the filter applied in compute_cascades.

    Reads from <root>/data_states/, writes to <root>/data_avalanches/.
    The output schema is identical to compute_cascades so that
    compute_power_law_fits_from_avalanches_dir can be pointed at either via cascade_subdir.
    """
    states_dir = os.path.join(root, 'data_states')
    data_dir   = os.path.join(root, 'data_avalanches')
    os.makedirs(data_dir, exist_ok=True)
    all_files = sorted(glob.glob(os.path.join(states_dir, "*.npz")))
    if not all_files:
        print("No state files found in", states_dir)
        return

    groups: dict[float, list[str]] = {}
    for path in all_files:
        d = np.load(path, allow_pickle=False)
        p = float(d["p"])
        d.close()
        groups.setdefault(p, []).append(path)

    for p_val, rep_files in tqdm(sorted(groups.items()), desc="avalanche analysis"):
        all_durations: list[np.ndarray] = []
        all_sizes:     list[np.ndarray] = []

        meta = np.load(rep_files[0], allow_pickle=False)
        n_neurons  = int(meta["n_neurons"])
        batch_size = int(meta["batch_size"])
        steps      = int(meta["steps"])
        seed       = int(meta["seed"])
        n_reps     = int(meta["n_reps"])
        meta.close()

        for rep_path in rep_files:
            d = np.load(rep_path, allow_pickle=False)
            states_np = d["states"].astype(np.int32)
            d.close()

            durs, szs = all_avalanche_durations_sizes(states_np)
            all_durations.append(durs)
            all_sizes.append(szs)

        out_path = os.path.join(data_dir, f"p_{p_val:.6f}.npz")
        np.savez(out_path,
                 p=np.float64(p_val),
                 durations=np.concatenate(all_durations) if all_durations else np.array([], dtype=np.int32),
                 activities=np.concatenate(all_sizes)    if all_sizes    else np.array([], dtype=np.int32),
                 n_neurons=np.int32(n_neurons),
                 batch_size=np.int32(batch_size),
                 steps=np.int32(steps),
                 seed=np.int32(seed),
                 n_reps=np.int32(n_reps),
                 total_runs=np.int32(batch_size * n_reps))

    print(f"Avalanche files saved to {data_dir}")




# POWER LAW FITS -----------------------------


def compute_power_law_fits_from_avalanches_dir(
    root: str = 'power_laws/isolated',
    cascade_subdir: str = 'data_cascade',
    fits_subdir: str = 'data_fits',
):
    """Fit power laws to all cascade npz files and save results.

    Reads from <root>/<cascade_subdir>/, writes to <root>/<fits_subdir>/.
    Also writes a summary.csv with the scalar series.

    Defaults (cascade_subdir='data_cascade', fits_subdir='data_fits') preserve
    the original behaviour.  Pass cascade_subdir='data_avalanches' and
    fits_subdir='data_fits' to run on burst-detection output.

    Separating fitting from plotting means plot styling can be iterated
    instantly without re-running the expensive grid search.
    """
    data_dir  = os.path.join(root, cascade_subdir)
    fits_dir  = os.path.join(root, fits_subdir)
    os.makedirs(fits_dir, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    #npz_files = [f for f in all_files if "durations" in np.load(f, allow_pickle=False).files]
    #if not npz_files:
    #    print("No cascade npz files (with 'durations' key) found in", data_dir)
    #    return

    records = []
    for path in tqdm(npz_files, desc="fitting power laws"):
        d = np.load(path, allow_pickle=False)
        p_val     = float(d["p"])
        durations = d["durations"]
        sizes = d["activities"]
        n_neurons = int(d["n_neurons"])
        steps     = int(d["steps"])
        batch_size = int(d["batch_size"])

        dur = fit_discrete_power_law_mle(durations)
        sz = fit_discrete_power_law_mle(sizes)
        
        # Relation: mean activity during burst = size/duration (no exclusion)
        dur_f = np.maximum(durations.astype(np.float64), 1)
        ratio = sizes.astype(np.float64) / dur_f
        mean_activity_ratio = float(np.mean(ratio))
        std_activity_ratio  = float(np.std(ratio))

        # save per-p fit file (keys use _duration / _size to distinguish fit type)
        fit_path = os.path.join(fits_dir, f"fit_p_{p_val:.6f}.npz")
        save_dict = {
            "p": np.float64(p_val),
            "n_neurons": np.int32(n_neurons),
            "steps": np.int32(steps),
            "batch_size": np.int32(batch_size),
            "n_complete": np.int32(len(durations)),
            "alpha_duration": np.float64(dur["alpha"]),
            "alpha_sigma_duration": np.float64(dur["sigma"]),
            "ks_duration": np.float64(dur["ks"]),
            "LR_vs_exp_duration": np.float64(dur["LR_vs_exp"]),
            "LR_pvalue_duration": np.float64(dur["LR_pvalue"]),
            "n_fit_duration": np.int32(dur["n_fit"]),
            "Tmin_duration": np.int32(dur["xmin"] if dur["xmin"] is not None else -1),
            "t_vals_duration": dur["t_vals"] if dur["t_vals"] is not None else np.array([]),
            "pdf_duration": dur["pdf"] if dur["pdf"] is not None else np.array([]),
            "fitted_pdf_duration": dur["fitted_pdf"] if dur["fitted_pdf"] is not None else np.array([]),
            "mean_activity_ratio": np.float64(mean_activity_ratio),
            "std_activity_ratio": np.float64(std_activity_ratio),
            "alpha_size": np.float64(sz["alpha"]),
            "alpha_sigma_size": np.float64(sz["sigma"]),
            "ks_size": np.float64(sz["ks"]),
            "LR_vs_exp_size": np.float64(sz["LR_vs_exp"]),
            "LR_pvalue_size": np.float64(sz["LR_pvalue"]),
            "n_fit_size": np.int32(sz["n_fit"]),
            "xmin_size": np.int32(sz["xmin"] if sz["xmin"] is not None else -1),
            "t_vals_size": sz["t_vals"] if sz["t_vals"] is not None else np.array([]),
            "pdf_size": sz["pdf"] if sz["pdf"] is not None else np.array([]),
            "fitted_pdf_size": sz["fitted_pdf"] if sz["fitted_pdf"] is not None else np.array([]),
        }
        np.savez(fit_path, **save_dict)

        rec = {
            "p": p_val, "n_neurons": n_neurons, "steps": steps,
            "batch_size": batch_size, "n_complete": len(durations),
            "alpha_duration": dur["alpha"], "alpha_sigma_duration": dur["sigma"], "ks_duration": dur["ks"],
            "LR_vs_exp_duration": dur["LR_vs_exp"], "LR_pvalue_duration": dur["LR_pvalue"],
            "n_fit_duration": dur["n_fit"], "Tmin_duration": dur["xmin"],
            "mean_activity_ratio": mean_activity_ratio,
            "std_activity_ratio": std_activity_ratio,
            "alpha_size": sz["alpha"], "alpha_sigma_size": sz["sigma"], "ks_size": sz["ks"],
            "LR_vs_exp_size": sz["LR_vs_exp"], "LR_pvalue_size": sz["LR_pvalue"],
            "n_fit_size": sz["n_fit"], "xmin_size": sz["xmin"],
        }
        records.append(rec)

    records.sort(key=lambda r: r["p"])

    # CSV: duration columns + size columns + relation columns (same _duration / _size naming)
    csv_path = os.path.join(fits_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("p,n_complete,alpha_duration,alpha_sigma_duration,ks_duration,LR_vs_exp_duration,LR_pvalue_duration,n_fit_duration,Tmin_duration,")
        f.write("alpha_size,alpha_sigma_size,ks_size,LR_vs_exp_size,LR_pvalue_size,n_fit_size,xmin_size,")
        f.write("mean_activity_ratio,std_activity_ratio\n")
        for r in records:
            def _fmt(v): return f"{v:.6f}" if (v is not None and not np.isnan(v)) else ""
            tmin_str = str(r["Tmin_duration"]) if r["Tmin_duration"] is not None else ""
            xmin_s_str = str(r.get("xmin_size", "")) if r.get("xmin_size") is not None else ""
            f.write(
                f"{r['p']:.6f},{r['n_complete']},"
                f"{_fmt(r['alpha_duration'])},{_fmt(r['alpha_sigma_duration'])},{_fmt(r['ks_duration'])},"
                f"{_fmt(r['LR_vs_exp_duration'])},{_fmt(r['LR_pvalue_duration'])},"
                f"{r['n_fit_duration']},{tmin_str},"
                f"{_fmt(r.get('alpha_size'))},{_fmt(r.get('alpha_sigma_size'))},{_fmt(r.get('ks_size'))},"
                f"{_fmt(r.get('LR_vs_exp_size'))},{_fmt(r.get('LR_pvalue_size'))},"
                f"{r.get('n_fit_size', '')},{xmin_s_str},"
                f"{_fmt(r.get('mean_activity_ratio'))},{_fmt(r.get('std_activity_ratio'))}\n"
            )
    print(f"Saved summary CSV -> {csv_path}")


def _run_avalanche_pipeline(root: str):
    compute_avalanches_from_states_dir(root=root)
    compute_power_law_fits_from_avalanches_dir(
        root=root,
        cascade_subdir='data_avalanches',
        fits_subdir='data_fits',
    )


def experiment_vision_radius():
    VISION_DISTANCES = np.arange(5, 281, 5).tolist()
    for n_agents in [256]:
        for n_neurons in [64]:
            for vision_distance in VISION_DISTANCES:
                root = f'experiments_data/vision_radii/{n_agents}agents_{n_neurons}neurons_{vision_distance}vision_distance'
                p_c = 1.0 / (n_neurons - 1)
                Ps = np.linspace(0.5 * p_c, 1.5 * p_c, 20)
                delta_p = Ps[1] - Ps[0]
                extend_ps = 5
                for p in range(extend_ps):
                    Ps = np.append(Ps, Ps[-1] + delta_p)
                
                print(Ps)
                make_environment_states(
                    n_agents=n_agents,
                    n_neurons=n_neurons,
                    p_values=Ps.tolist(),
                    vision_distance=vision_distance,
                    macro_steps=1000,
                    batch_size=10,
                    n_reps=20, # changed from 10
                    root=root,
                )
                _run_avalanche_pipeline(root)


def experiment_number_sensors():
    VISION_DISTANCES = [30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250]
    N_SENSORS = [2, 4, 8, 16, 32]

    import time
    import datetime

    for n_agents in [256]:
        for n_neurons in [64]:
            for vision_distance in VISION_DISTANCES:
                for n_sensors in N_SENSORS: 
                    root = f'experiments_data/n_sensors/{n_agents}agents_{n_neurons}neurons_{vision_distance}vision_distance_{n_sensors}sensors'
                    p_c = 1.0 / (n_neurons - 1)
                    Ps = np.linspace(0.5 * p_c, 1.5 * p_c, 20)

                    t0 = time.time()
                    make_environment_states(
                        n_agents=n_agents,
                        n_neurons=n_neurons,
                        n_exteroceptor=n_sensors,
                        p_values=Ps.tolist(),
                        vision_distance=vision_distance,
                        macro_steps=1000,
                        batch_size=20,
                        n_reps=5, # changed from 10
                        root=root,
                    )
                    _run_avalanche_pipeline(root)
                    t1 = time.time()
                    print(f"Time per run: {datetime.timedelta(seconds=t1 - t0)}")




if __name__ == "__main__":
    # WARNING: This will take a lot of time to run.
    #experiment_vision_radius()
    #experiment_number_sensors()
    pass