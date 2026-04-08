import os
import jax
import jax.numpy as jnp
from jax import jit, random
from typing import Tuple, NamedTuple, Optional
import numpy as np
import platform
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Platform detection and JAX setup
def setup_jax():
    """Setup JAX based on platform and available backends"""
    system = platform.system().lower()
    
    # Check available backends
    available_backends = jax.devices()    
    # Use GPU if available, otherwise CPU
    if any(d.platform == 'gpu' for d in available_backends):
        device = jax.devices('gpu')[0]
    else:
        device = jax.devices('cpu')[0]
    return device

# Initialize JAX
device = setup_jax()

class FireflySimulationState(NamedTuple):
    """Immutable state container for JAX compatibility - Firefly version with lights"""
    positions: jnp.ndarray      # [B, N_agents, 2]
    speeds: jnp.ndarray         # [B, N_agents] - always 0 for fireflies (static)
    angles: jnp.ndarray         # [B, N_agents] - random orientation (for egocentric observations)
    lights: jnp.ndarray         # [B, N_agents] - light state (0.0 = OFF, 1.0 = ON)
    adjacency: jnp.ndarray      # [B, N_agents, N_agents]
    observations: jnp.ndarray   # [B, N_agents, n_exteroceptor] - presence × light state
    key: jnp.ndarray            # [2] PRNGKey (shared across the batch)
    step_counter: jnp.ndarray   # JAX int32 scalar (shared across the batch)

class BatchFirefliesEnvironment:
    def __init__(self,
                batch_size: int,
                n_agents: int, 
                map_size: list[float], 
                vision_distance: float, 
                n_exteroceptor: int, 
                agent_radius: float, 
                
                observation_noise_std: float, 
                
                collisions: bool,
                save_frequency: int, 
                seed: int,
                binary_observations: bool = True,  # Always binary for fireflies
            ):

        self.N_AGENTS = n_agents
        self.BATCH_SIZE = int(batch_size)
        self.MAP_SIZE = jnp.array(map_size, dtype=jnp.float32)
        self.VISION_DISTANCE = float(vision_distance)
        self.N_EXTEROCEPTOR = n_exteroceptor
        self.AGENT_RADIUS = float(agent_radius)
        
        self.OBSERVATION_NOISE_STD = observation_noise_std
        self.BINARY_OBSERVATIONS = binary_observations
        
        # Per request: ignore collisions (keep the arg for API compatibility)
        self.COLLISIONS = False
        self.SAVE_FREQUENCY = save_frequency
        self.key = random.PRNGKey(seed)
        
        # Initialize state with proper dtypes
        self.key, pos_key, angle_key, light_key = random.split(self.key, 4)
        positions = (
            random.uniform(pos_key, (self.BATCH_SIZE, self.N_AGENTS, 2), dtype=jnp.float32)
            * self.MAP_SIZE
        )  # [B, N, 2]
        angles = (
            random.uniform(angle_key, (self.BATCH_SIZE, self.N_AGENTS), dtype=jnp.float32)
            * (2.0 * jnp.pi)
        )  # [B, N] - random orientation for egocentric observations
        speeds = jnp.zeros((self.BATCH_SIZE, self.N_AGENTS), dtype=jnp.float32)  # Always 0 - fireflies are static
        lights = jnp.zeros((self.BATCH_SIZE, self.N_AGENTS), dtype=jnp.float32)  # [B, N] - all OFF at start
        adjacency = jnp.zeros((self.BATCH_SIZE, self.N_AGENTS, self.N_AGENTS), dtype=bool)  # [B, N, N]
        
        # Initialize observations
        observations = jnp.zeros(
            (self.BATCH_SIZE, self.N_AGENTS, self.N_EXTEROCEPTOR), dtype=jnp.float32
        )  # [B, N, K]
        
        # Pre-compute constants - use endpoint=False to avoid duplicate 0 and 2π
        self._sensor_bins_base = jnp.linspace(0, 2 * jnp.pi, self.N_EXTEROCEPTOR, endpoint=False, dtype=jnp.float32)
        self._half_bin = jnp.array(jnp.pi / self.N_EXTEROCEPTOR, dtype=jnp.float32)
        self._vision_distance_minus_radius = self.VISION_DISTANCE - self.AGENT_RADIUS
        self._vision_distance2 = jnp.float32(self.VISION_DISTANCE * self.VISION_DISTANCE)
        self._no_self_mask = (~jnp.eye(self.N_AGENTS, dtype=bool))[None, :, :]
        
        # Create initial state with PRNGKey (single key shared across the batch)
        self.state = FireflySimulationState(
            positions=positions,
            speeds=speeds,
            angles=angles,
            lights=lights,
            adjacency=adjacency,
            observations=observations,
            key=self.key,
            step_counter=jnp.int32(0)
        )
        
        # JIT compile the step (avoid nested jit boundaries inside the step)
        self._jit_step = jit(self._step_impl)

        # Initialize adjacency and observations (computed together to avoid duplicate geometry)
        init_obs, init_adj = self._update_observations_binary_impl(self.state)
        self.state = self.state._replace(adjacency=init_adj, observations=init_obs)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset environment to initial state. Much faster than recreating the environment."""
        if seed is not None:
            self.key = random.PRNGKey(seed)
        else:
            # Advance key deterministically
            self.key, _ = random.split(self.key, 2)
        
        # Split key for initialization
        self.key, pos_key, angle_key = random.split(self.key, 3)
        positions = (
            random.uniform(pos_key, (self.BATCH_SIZE, self.N_AGENTS, 2), dtype=jnp.float32)
            * self.MAP_SIZE
        )
        angles = (
            random.uniform(angle_key, (self.BATCH_SIZE, self.N_AGENTS), dtype=jnp.float32)
            * (2.0 * jnp.pi)
        )
        speeds = jnp.zeros((self.BATCH_SIZE, self.N_AGENTS), dtype=jnp.float32)  # Always 0
        lights = jnp.zeros((self.BATCH_SIZE, self.N_AGENTS), dtype=jnp.float32)  # all OFF at reset

        self.state = self.state._replace(
            positions=positions,
            speeds=speeds,
            angles=angles,
            lights=lights,
            step_counter=jnp.int32(0)
        )
        
        # Recompute observations and adjacency
        init_obs, init_adj = self._update_observations_binary_impl(self.state)
        self.state = self.state._replace(adjacency=init_adj, observations=init_obs)

    def step(self, dAction: Optional[jnp.ndarray] = None) -> None:
        """Step the environment forward one timestep."""
        if dAction is not None:
            dAction = jnp.array(dAction, dtype=jnp.float32)
            if dAction.shape != (self.BATCH_SIZE, self.N_AGENTS, 1):
                raise ValueError(f"dAction must have shape ({self.BATCH_SIZE}, {self.N_AGENTS}, 1), got {dAction.shape}")
        self.state = self._jit_step(self.state, dAction)

    def get_state(self, keys: Optional[list] = None) -> dict:
        """Get current state as a dictionary."""
        state_dict = {
            'positions': np.array(self.state.positions),
            'speeds': np.array(self.state.speeds),
            'angles': np.array(self.state.angles),
            'lights': np.array(self.state.lights),
            'adjacency': np.array(self.state.adjacency),
            'observations': np.array(self.state.observations),
            'step_counter': np.array(self.state.step_counter),
        }
        if keys is not None:
            return {k: state_dict[k] for k in keys}
        return state_dict

    def _compute_toroidal_delta_pos_impl(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Compute pairwise delta positions with toroidal wrapping."""
        # positions: [B, N, 2]
        # Return: [B, N, N, 2] where delta[b, i, j] = pos[b, j] - pos[b, i] (toroidal)
        B, N, _ = positions.shape
        pos_i = positions[:, :, None, :]  # [B, N, 1, 2]
        pos_j = positions[:, None, :, :]  # [B, 1, N, 2]
        delta = pos_j - pos_i  # [B, N, N, 2]
        
        # Toroidal wrapping: wrap each component independently
        half_map = self.MAP_SIZE / 2.0
        delta = delta + half_map
        delta = delta % self.MAP_SIZE
        delta = delta - half_map
        
        return delta

    def _update_observations_binary_impl(self, state: FireflySimulationState) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute binary observations (0/1) with light state multiplication.
        
        Observations = presence × light_state
        - presence: 1 if neighbor exists in bin, 0 otherwise (same as boids)
        - light_state: 1 if neighbor's light is ON, 0 if OFF
        - Final observation: 1 only if neighbor exists AND light is ON
        """
        # pos[j] - pos[i] (direction from agent i to agent j)
        delta_pos = self._compute_toroidal_delta_pos_impl(state.positions)
        
        # Compute distances
        distances = jnp.linalg.norm(delta_pos, axis=3)
        
        # Create adjacency matrix for observations
        obs_adjacency = (distances <= self.VISION_DISTANCE) & self._no_self_mask
        
        # Use obs_adjacency instead of state.adjacency for consistency
        distances = jnp.where(obs_adjacency, distances, jnp.inf)
        
        # Compute relative angles
        rel_angles = (jnp.arctan2(delta_pos[:, :, :, 1], delta_pos[:, :, :, 0]) + 2 * jnp.pi) % (2 * jnp.pi)
        rel_angles = jnp.where(obs_adjacency, rel_angles, 0.0)
        
        # Compute angular arcs
        radii_j = jnp.full((1, 1, self.N_AGENTS), self.AGENT_RADIUS, dtype=jnp.float32)
        clamp_value = jnp.clip(radii_j / distances, -1.0 + 1e-7, 1.0 - 1e-7)  # pyright: ignore[reportOperatorIssue]
        arcs = 2.0 * jnp.arcsin(clamp_value)  # pyright: ignore[reportOperatorIssue]
        arcs = jnp.where(distances < radii_j, 2 * jnp.pi, arcs)  # pyright: ignore[reportOperatorIssue]

        # ---- Interval marking in bin-index space (agent frame) ----
        # Bin centers in agent frame are fixed: theta_k = k * (2π/K).
        # Neighbor direction in agent frame:
        #   phi = wrap2pi(rel_angles - agent_angle)
        two_pi = jnp.float32(2.0 * jnp.pi)
        K = int(self.N_EXTEROCEPTOR)
        bin_w = two_pi / jnp.float32(K)

        agent_angles = state.angles[:, :, None]  # [B, N, 1]
        phi = (rel_angles - agent_angles + two_pi) % two_pi  # [B, N, N] in [0, 2π)

        # Half-width in radians used by the original dense overlap test.
        w = (arcs * 0.5) + self._half_bin  # [B, N, N]

        valid = obs_adjacency  # [B, N, N]
        
        # Filter by light state: only neighbors with lights ON contribute to observations
        lights_j = state.lights[:, None, :]  # [B, 1, N] - light states of all neighbors
        valid = valid & (lights_j > 0.5)  # [B, N, N] - neighbor exists AND light is ON
        
        covers_all = w >= jnp.pi  # if true, then all bins are covered (dense test would be all-true)

        # Convert [phi-w, phi+w] to inclusive bin-index bounds.
        # For w < π this is at most one wrapped interval on the ring.
        lo = (jnp.ceil((phi - w) / bin_w).astype(jnp.int32)) % K
        hi = (jnp.floor((phi + w) / bin_w).astype(jnp.int32)) % K
        hi1 = hi + jnp.int32(1)  # may equal K (allowed; diff has length K+1)

        wraps = lo > hi  # wrapped interval in modulo-K indices

        # Difference array per (batch, agent): diff[..., k] accumulates interval endpoints.
        diff = jnp.zeros((self.BATCH_SIZE, self.N_AGENTS, K + 1), dtype=jnp.int32)

        b_idx = jnp.arange(self.BATCH_SIZE, dtype=jnp.int32)[:, None, None]  # [B,1,1]
        i_idx = jnp.arange(self.N_AGENTS, dtype=jnp.int32)[None, :, None]    # [1,N,1]

        # Full coverage: add +1 on [0, K-1] => diff[0]+=1, diff[K]-=1
        # NOTE: endpoints 0 and K are constant indices, so sum over neighbors (axis=2) to get [B,N].
        full = (valid & covers_all).astype(jnp.int32)  # [B,N,N]
        full_sum = jnp.sum(full, axis=2)  # [B,N]
        diff = diff.at[:, :, 0].add(full_sum)
        diff = diff.at[:, :, K].add(-full_sum)

        # Non-wrapping intervals: [lo, hi]
        nonwrap = (valid & (~covers_all) & (~wraps)).astype(jnp.int32)
        diff = diff.at[b_idx, i_idx, lo].add(nonwrap)
        diff = diff.at[b_idx, i_idx, hi1].add(-nonwrap)

        # Wrapping intervals: [lo, K-1] and [0, hi]
        wrap = (valid & (~covers_all) & wraps).astype(jnp.int32)  # [B,N,N]
        diff = diff.at[b_idx, i_idx, lo].add(wrap)
        # Endpoints K and 0 are constant indices, so sum over neighbors for those updates.
        wrap_sum = jnp.sum(wrap, axis=2)  # [B,N]
        diff = diff.at[:, :, K].add(-wrap_sum)
        diff = diff.at[:, :, 0].add(wrap_sum)
        diff = diff.at[b_idx, i_idx, hi1].add(-wrap)

        counts = jnp.cumsum(diff[:, :, :K], axis=2)
        binary_obs = (counts > 0).astype(jnp.float32)  # [B, N, K] - presence AND light ON

        return binary_obs, obs_adjacency

    def _step_impl(
        self,
        state: FireflySimulationState,
        dAction: Optional[jnp.ndarray],
    ) -> FireflySimulationState:
        """Single step: update light states based on binary action (0 = OFF, 1 = ON).
        
        dAction: [B, N, 1] where dAction[b, i, 0] in [-1, 1] from tanh output
        - dAction > 0 → light ON (1.0)
        - dAction <= 0 → light OFF (0.0)
        """
        if dAction is None:
            new_lights = state.lights  # No change
        else:
            # Convert action to binary: > 0 → ON (1.0), <= 0 → OFF (0.0)
            light_actions = dAction[:, :, 0]  # [B, N]
            new_lights = (light_actions > 0.0).astype(jnp.float32)  # Binary: 0.0 or 1.0
        
        # Fireflies are static - no position/angle/speed updates
        # Just update lights and recompute observations
        
        # Generate noise key (for future use if needed)
        base = random.fold_in(state.key, state.step_counter)
        observation_key, key = random.split(base, 2)
        
        # Create new state with updated lights
        new_state = state._replace(
            lights=new_lights,
            key=key,
            step_counter=state.step_counter + 1
        )

        # Compute observations and adjacency (observations depend on light states)
        new_obs, new_adj = self._update_observations_binary_impl(new_state)
        
        # Add observation noise if needed (though binary observations typically don't need noise)
        if self.OBSERVATION_NOISE_STD > 0.0:
            observation_noise = (
                random.normal(
                    observation_key,
                    (self.BATCH_SIZE, self.N_AGENTS, self.N_EXTEROCEPTOR),
                    dtype=jnp.float32,
                )
                * self.OBSERVATION_NOISE_STD
            )
            new_obs = new_obs + observation_noise
            new_obs = jnp.clip(new_obs, 0.0, 1.0)
            # Re-binarize after noise
            new_obs = (new_obs > 0.5).astype(jnp.float32)
        
        return new_state._replace(
            adjacency=new_adj,
            observations=new_obs
        )

    def save_simulation_data(self, states: list, filename: str = "firefly_simulation_data", folder: str = "."):
        """Save simulation data to npz file."""
        os.makedirs(folder, exist_ok=True)
        
        # Extract arrays from state dictionaries
        positions = np.array([s['positions'] for s in states])
        speeds = np.array([s['speeds'] for s in states])
        angles = np.array([s['angles'] for s in states])
        lights = np.array([s['lights'] for s in states])
        adjacency = np.array([s['adjacency'] for s in states])
        observations = np.array([s['observations'] for s in states])
        step_counters = np.array([s.get('step_counter', i) for i, s in enumerate(states)])
        
        np.savez(
            os.path.join(folder, f"{filename}.npz"),
            positions=positions,
            speeds=speeds,
            angles=angles,
            lights=lights,
            adjacency=adjacency,
            observations=observations,
            step_counters=step_counters,
            # Parameters
            map_size=self.MAP_SIZE,
            vision_distance=self.VISION_DISTANCE,
            agent_radius=self.AGENT_RADIUS,
            n_exteroceptor=self.N_EXTEROCEPTOR,
            n_agents=self.N_AGENTS,
            batch_size=self.BATCH_SIZE,
        )
