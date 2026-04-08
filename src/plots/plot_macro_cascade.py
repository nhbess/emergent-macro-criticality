import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _select_frame(data: dict, key: str, step: int, batch: int) -> np.ndarray:
    """
    Select the per-step array for a single environment.
    """
    arr = np.asarray(data[key])

    if key == "positions":
        if arr.ndim == 4:   # [T, B, N, 2]
            return arr[step, batch]
        if arr.ndim == 3:   # [T, N, 2]
            return arr[step]
        raise ValueError(f"Unsupported positions shape {arr.shape}")

    if key in ("speeds", "angles", "lights"):
        if arr.ndim == 3:   # [T, B, N]
            return arr[step, batch]
        if arr.ndim == 2:   # [T, N]
            return arr[step]
        raise ValueError(f"Unsupported {key} shape {arr.shape}")

    if key == "adjacency":
        if arr.ndim == 4:   # [T, B, N, N]
            return arr[step, batch]
        if arr.ndim == 3:   # [T, N, N]
            return arr[step]
        raise ValueError(f"Unsupported adjacency shape {arr.shape}")

    if key == "observations":
        if arr.ndim == 4:   # [T, B, N, K]
            return arr[step, batch]
        if arr.ndim == 3:   # [T, N, K]
            return arr[step]
        raise ValueError(f"Unsupported observations shape {arr.shape}")

    raise KeyError(f"Unsupported key '{key}'")


def _render_vision_field_on_ax(
    ax,
    agent_idx: int,
    positions: np.ndarray,
    angles: np.ndarray,
    observations: np.ndarray,
    vision_distance: float,
    n_exteroceptor: int,
    vision_cmap: str = "viridis",
):

    VISION_CMAP_ALPHA  = 0.5
    VISION_LINES_ALPHA = 0.9

    center_agent_pos     = positions[agent_idx]
    center_agent_heading = angles[agent_idx]

    ax.add_patch(plt.Circle(center_agent_pos, vision_distance,
                            fill=False, edgecolor='gray', linestyle='-',
                            linewidth=0.5, alpha=VISION_LINES_ALPHA, zorder=1))

    bin_size     = 2 * np.pi / n_exteroceptor
    n_arc_points = 16
    for i in range(n_exteroceptor):
        obs        = float(observations[agent_idx, i])
        bin_center = center_agent_heading + i * bin_size
        start      = bin_center - 0.5 * bin_size
        end        = bin_center + 0.5 * bin_size
        angles_arc = np.linspace(start, end, n_arc_points + 1)
        pts = [center_agent_pos] + [
            center_agent_pos + vision_distance * np.array([np.cos(a), np.sin(a)])
            for a in angles_arc
        ]
        color = plt.colormaps[vision_cmap](obs)
        ax.add_patch(plt.Polygon(np.array(pts), facecolor=color,
                                 alpha=VISION_CMAP_ALPHA, zorder=1))

    half_bin = 0.5 * bin_size
    for i in range(n_exteroceptor):
        boundary_angle = center_agent_heading + i * bin_size - half_bin
        end_pt = center_agent_pos + vision_distance * np.array(
            [np.cos(boundary_angle), np.sin(boundary_angle)]
        )
        ax.plot([center_agent_pos[0], end_pt[0]], [center_agent_pos[1], end_pt[1]],
                'k-', linewidth=0.5, alpha=VISION_LINES_ALPHA, zorder=1)
    # final boundary to close the circle
    boundary_angle = center_agent_heading + n_exteroceptor * bin_size - half_bin
    end_pt = center_agent_pos + vision_distance * np.array(
        [np.cos(boundary_angle), np.sin(boundary_angle)]
    )
    ax.plot([center_agent_pos[0], end_pt[0]], [center_agent_pos[1], end_pt[1]],
            'k-', linewidth=0.5, alpha=VISION_LINES_ALPHA, zorder=1)


def _adjacency_from_positions(pos: np.ndarray, vision_distance: float, map_size: list) -> np.ndarray:
    """Vectorized toroidal adjacency matrix. Returns (N, N) bool array."""
    W, H = float(map_size[0]), float(map_size[1])
    delta = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]   # (N, N, 2)
    delta[..., 0] = (delta[..., 0] + W / 2) % W - W / 2
    delta[..., 1] = (delta[..., 1] + H / 2) % H - H / 2
    dist2 = (delta ** 2).sum(axis=-1)                        # (N, N)
    adj = dist2 <= vision_distance ** 2
    np.fill_diagonal(adj, False)
    return adj


def _render_frame_on_ax(
    ax,
    positions: np.ndarray,
    lights: np.ndarray,
    adjacency: np.ndarray,
    map_size,
    agent_radius: float,
    adjacency_line: bool = True,
    light_cmap: str = None,
    bg_darkness: float = 0.18,
):

    W, H = float(map_size[0]), float(map_size[1])

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    v = 1.0 - float(np.clip(bg_darkness, 0.0, 1.0))
    ax.set_facecolor((v, v, v))

    if adjacency_line:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if adjacency[i, j]:
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dx = (dx + W / 2) % W - W / 2
                    dy = (dy + H / 2) % H - H / 2
                    ax.plot(
                        [positions[i, 0], positions[i, 0] + dx],
                        [positions[i, 1], positions[i, 1] + dy],
                        color='#444444', alpha=0.5, linewidth=0.5, zorder=2,
                    )

    cmap = plt.colormaps[light_cmap] if light_cmap else None
    for i, pos in enumerate(positions):
        lv  = float(np.clip(lights[i], 0.0, 1.0))
        lit = lv > 0.5
        if cmap is not None:
            fc = cmap(1.0) if lit else cmap(0.0)
            ec = tuple(np.clip(np.array(fc[:3]) * 0.7, 0, 1))
        else:
            fc = (1.0, 0.85, 0.0) if lit else (0.45, 0.45, 0.45)
            ec = '#bb8800' if lit else '#333333'
        circle = plt.Circle(
            pos, agent_radius,
            facecolor=fc, edgecolor=ec, linewidth=1.0, alpha=1, zorder=3,
        )
        ax.add_patch(circle)


def run_and_save_firefly_cascade(
    p: float,
    vision_distance: int = 60,
    n_agents: int = 256,
    n_neurons: int = 64,
    n_exteroceptor: int = 8,
    agent_radius: int = 5,
    macro_steps: int = 200,
    micro_steps: int = 5,
    seed: int = 42,
    save_path: str = "media/macro_cascade/macro_cascade.npz",
    device: str = None,
):
   
    import torch
    import jax.numpy as jnp

    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from env_collective import BatchFirefliesEnvironment   # noqa: E402
    from reservoir_beggs import BeggsProbabilisticNetwork  # noqa: E402

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    density  = 0.0008
    map_side = int(np.sqrt(n_agents / density))
    map_size = [map_side, map_side]

    env = BatchFirefliesEnvironment(
        batch_size            = 1,
        n_agents              = n_agents,
        map_size              = map_size,
        vision_distance       = vision_distance,
        n_exteroceptor        = n_exteroceptor,
        agent_radius          = agent_radius,
        observation_noise_std = 0.0,
        collisions            = False,
        save_frequency        = 1,
        binary_observations   = True,
        seed                  = seed,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    W = torch.full((n_agents, n_neurons, n_neurons), p, dtype=torch.float32, device=device)
    idx = torch.arange(n_neurons, device=device)
    W[:, idx, idx] = 0.0
    reservoir = BeggsProbabilisticNetwork(W=W, device=device, allow_self=False)
    reservoir.state.zero_()

    env.reset(seed=seed)
    # initialization mode 4: seed agent 0's light, advance one step
    seed_lights = jnp.zeros((1, n_agents), dtype=jnp.float32).at[:, 0].set(1.0)
    env.state = env.state._replace(lights=seed_lights)
    env.step(dAction=None)

    positions = np.array(env.state.positions)[0]   # (N, 2) — static
    angles    = np.array(env.state.angles)[0]      # (N,)   — static

    all_lights = np.zeros((macro_steps, n_agents), dtype=np.float32)
    all_obs    = np.zeros((macro_steps, n_agents, n_exteroceptor), dtype=np.float32)
    all_lights[0, 0] = 1.0   # record the seed

    reservoir_state = None
    for t in range(1, macro_steps):
        obs   = np.array(env.state.observations)                      # (1, N, K)
        obs_t = torch.from_numpy(obs).float().to(device).reshape(n_agents, n_exteroceptor)

        for mi in range(micro_steps):
            reservoir_state = reservoir.step(external_input=obs_t if mi == 0 else None)

        last    = reservoir_state[:, -1].float()                      # (N,)
        actions = last.reshape(1, n_agents, 1).cpu().numpy()
        env.step(dAction=actions)
        all_lights[t] = np.array(env.state.lights)[0]
        # save observations AFTER the step so obs[t] matches lights[t]
        all_obs[t] = np.array(env.state.observations)[0]

    adj = _adjacency_from_positions(positions, vision_distance, map_size)  # (N, N)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        # scalars / config
        map_size        = np.array(map_size),
        agent_radius    = np.float32(agent_radius),
        vision_distance = np.float32(vision_distance),
        n_exteroceptor  = np.int32(n_exteroceptor),
        p               = np.float64(p),
        seed            = np.int32(seed),
        # static spatial data  — shape [T, N, ...] for _select_frame compatibility
        positions       = np.tile(positions[None],  (macro_steps, 1, 1)),   # (T, N, 2)
        angles          = np.tile(angles[None],     (macro_steps, 1)),       # (T, N)
        adjacency       = np.tile(adj[None],        (macro_steps, 1, 1)),    # (T, N, N)
        # dynamic data
        lights          = all_lights,   # (T, N)
        observations    = all_obs,      # (T, N, K)
        step_counters   = np.arange(macro_steps, dtype=np.int32),
    )
    print(f"Cascade saved → {save_path}  "
          f"({macro_steps} steps, p={p:.5f}, seed={seed})")
    return save_path


def plot_firefly_multi_frame(
    data_path: str,
    steps: list,
    adjacency_line: bool = True,
    vision_field_frame: int = None,
    vision_field_agent: int = None,
    vision_cmap: str = "viridis",
    light_cmap: str = None,
    bg_darkness: float = 0.18,
    filename: str = "firefly_multi_frame",
    folder: str = "media",
):

    data     = np.load(data_path)
    map_size = data["map_size"].tolist()

    n    = len(steps)
    SIZE = 4

    fig = plt.figure(figsize=(n * SIZE, SIZE))
    gs  = fig.add_gridspec(1, n, left=0, right=1, top=1, bottom=0,
                           wspace=0.02, hspace=0)
    axes = [fig.add_subplot(gs[0, col]) for col in range(n)]

    for col, t in enumerate(steps):
        ax        = axes[col]
        positions = _select_frame(data, "positions",    step=t, batch=0)
        lights    = _select_frame(data, "lights",       step=t, batch=0)
        adjacency = _select_frame(data, "adjacency",    step=t, batch=0)

        _render_frame_on_ax(ax, positions, lights, adjacency, map_size,
                            float(data["agent_radius"]),
                            adjacency_line=adjacency_line,
                            light_cmap=light_cmap,
                            bg_darkness=bg_darkness)

        if vision_field_frame is not None and col == vision_field_frame:
            angles   = _select_frame(data, "angles",       step=t, batch=0)
            obs      = _select_frame(data, "observations", step=t, batch=0)
            vd       = float(data["vision_distance"])
            n_extero = int(data["n_exteroceptor"])

            if vision_field_agent is None:
                agent_idx = int(np.argmin(
                    np.linalg.norm(positions - np.array(map_size) / 2, axis=1)
                ))
            else:
                agent_idx = vision_field_agent

            _render_vision_field_on_ax(ax, agent_idx, positions, angles, obs,
                                       vd, n_extero, vision_cmap)

        ax.text(0.02, 0.98, f"t = {t}", transform=ax.transAxes,
                fontsize=15, va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=1, pad=2))

    out = Path(folder) / f"{filename}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
    return out


if __name__ == "__main__":
    FOLDER_PATH = "media/macro_cascade"
    CACHE = os.path.join(FOLDER_PATH, "macro_cascade.npz")
    CMAP = 'inferno'
    

    run_and_save_firefly_cascade(
        p=0.017962, vision_distance=60, seed=123,
        macro_steps=200, save_path=CACHE,
    )

    
    plot_firefly_multi_frame(CACHE, steps=[0, 25, 50, 100],
                             filename="macro_cascade_frames",
                             folder=FOLDER_PATH,
                             vision_field_frame=3,
                             vision_cmap=CMAP,
                             light_cmap=CMAP,
                             bg_darkness=0.5)
