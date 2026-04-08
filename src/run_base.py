import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp
import torch

from env_collective import BatchFirefliesEnvironment
from src.reservoir_beggs import BeggsProbabilisticNetwork


def run_environment_p(
    p: float,
    env: BatchFirefliesEnvironment,
    config: dict,
    seed: int = None,
    initialization_mode: int = None, 
) -> dict:
    """
    Run one rollout with a homogeneous W(p) shared by all agents.

    Light state of each agent = last neuron of its reservoir (binary, no readout).
    External input to each reservoir = egocentric observations from the environment
    (which encode which visible neighbours have their light ON).

    Parameters
    ----------
    p      : edge probability — all W entries are set to this value (diagonal 0)
    env    : BatchFirefliesEnvironment, already created with the right batch_size
    config : experiment config dict
    seed   : optional RNG seed for env.reset()
    initialization_mode:
            mode 0: neuron 0 of ALL reservoirs (every batch × every agent) is set to 1
            mode 1: neuron 0 of ONE reservoir (batch 0, agent 0) is set to 1
            mode 2: agent 0's light is set to 1 for every batch element; seed is not recorded in lights[0]
                    because the macro loop starts at step 0 and lights[0] holds the result of the first step
            mode 3: last neuron of agent 0's reservoir is set to 1 for every batch element; it fires on the
                    first micro-step so agent 0's light is ON at lights[0], giving the cascade a proper seed
            mode 4: agent 0's light is set to 1 for every batch element; seed is recorded in lights[0] and
                    the macro loop starts at step 1; neighbours see the light at t=1 and can begin propagating
    Returns
    -------
    dict with:
        micro_states : torch.Tensor  (batch_size, n_agents, total_steps, n_neurons)  int32
                       or None if config['save_micro_states'] is False
        lights       : np.ndarray    (macro_steps, batch_size, n_agents)
    """
    batch_size       = config['batch_size']
    n_agents         = env.N_AGENTS
    n_neurons        = config['n_neurons']
    device           = config['device']
    total_ins        = batch_size * n_agents  # total reservoir instances
    save_micro       = config['save_micro_states']

    # Homogeneous W(p): all connections get probability p, no self-loops
    W = torch.full((total_ins, n_neurons, n_neurons), p, dtype=torch.float32, device=device)
    if not config['allow_self']:
        idx = torch.arange(n_neurons, device=device)
        W[:, idx, idx] = 0.0

    reservoir = BeggsProbabilisticNetwork(W=W, device=device, allow_self=config['allow_self'])
    reservoir.state.zero_()
    
    env.reset(seed=seed if seed is not None else int(np.random.randint(0, 1_000_000)))

    if initialization_mode == 0:
        reservoir.state[:, 0] = 1.0
    elif initialization_mode == 1:
        reservoir.state[0, 0] = 1.0
    elif initialization_mode == 2:
        new_lights = jnp.zeros((batch_size, n_agents), dtype=jnp.float32).at[:, 0].set(1.0)
        env.state = env.state._replace(lights=new_lights)
        env.step(dAction=None)
    elif initialization_mode == 3:
        agent0_indices = torch.arange(batch_size, device=device) * n_agents
        reservoir.state[agent0_indices, -1] = 1.0
    elif initialization_mode == 4:
        seed_lights = jnp.zeros((batch_size, n_agents), dtype=jnp.float32).at[:, 0].set(1.0)
        env.state = env.state._replace(lights=seed_lights)
        env.step(dAction=None)
    else:
        raise ValueError(f"Invalid initialization mode: {initialization_mode}")
    positions = np.array(env.state.positions)  # (B, N, 2)

    total_steps = config['macro_steps'] * config['micro_steps']
    if save_micro:
        micro_states = torch.zeros((total_steps, total_ins, n_neurons), device=device, dtype=torch.int32)
    lights = np.zeros((config['macro_steps'], batch_size, n_agents), dtype=np.float32)

    start_macro_step = 0
    if initialization_mode == 4:
        lights[0, :, 0] = 1.0
        start_macro_step = 1

    step_idx = 0
    reservoir_state = None  # will be set on first micro step

    for macro_step in range(start_macro_step, config['macro_steps']):
        # --- Sense: read JAX arrays directly, convert once ---
        obs = np.array(env.state.observations)                 # (B, N, K) — writable copy required by torch
        obs_t = torch.from_numpy(obs).float().to(device).reshape(total_ins, config['n_exteroceptor'])

        # --- Micro steps ---
        for micro_idx in range(config['micro_steps']):
            ext = obs_t if micro_idx == 0 else None
            reservoir_state = reservoir.step(external_input=ext)  # (B*N, n_neurons)
            if save_micro:
                micro_states[step_idx] = reservoir_state.to(torch.int32)
            step_idx += 1

        # --- Act: last neuron → light (no readout weights) ---
        # last_neuron is binary {0, 1}; env threshold is > 0, so 1 → ON, 0 → OFF
        last_neuron = reservoir_state[:, -1].float()          # (B*N,)
        actions = last_neuron.reshape(batch_size, n_agents, 1).cpu().numpy()
        env.step(dAction=actions)

        # --- Collect macro state: read JAX array directly ---
        lights[macro_step] = np.array(env.state.lights)

    if save_micro:
        # (total_steps, B*N, n_neurons) → (B, N, total_steps, n_neurons)
        micro_states = micro_states.transpose(0, 1).reshape(batch_size, n_agents, total_steps, n_neurons)
    else:
        micro_states = None

    return {
        'micro_states': micro_states,  # (batch_size, n_agents, total_steps, n_neurons) or None
        'lights':       lights,         # (macro_steps, batch_size, n_agents)
        'positions':    positions,      # (batch_size, n_agents, 2)
    }