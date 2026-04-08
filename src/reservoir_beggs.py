import torch

class BeggsProbabilisticNetwork(torch.nn.Module):
    def __init__(self, W, device="cpu", allow_self=False):
        """
        Beggs-style probabilistic propagation where W_ij is a transmission probability.

        Interpretation:
          W[b, i, j] in [0,1] is the probability that an active neuron i at time t
          activates neuron j at time t+1.

        Update rule:
          Sample X_ij(t) ~ Bernoulli(W_ij)
          S_{t+1}(j) = 1  iff  exists i with S_t(i)=1 and X_ij(t)=1

        Args:
            W: (batch_size, N, N) transmission-probability matrix in [0,1]
            device: torch device
            allow_self: if False, diagonal is forced to 0
        """
        super().__init__()
        self.device = device
        self.batch_size = W.shape[0]
        self.size = W.shape[-1]

        self.W = W.to(self.device).clamp(0.0, 1.0)

        if not allow_self:
            eye = torch.eye(self.size, device=self.device, dtype=torch.bool).unsqueeze(0)
            self.W = self.W.masked_fill(eye, 0.0)

        self.state = torch.zeros(self.batch_size, self.size, device=self.device, dtype=self.W.dtype)
        self.eval()

    @torch.no_grad()
    def step(self, external_input=None):
        """
        Update network one timestep.
        Sparse: only samples Bernoulli for active presynaptic neurons (much faster when activity is low).

        Args:
            external_input:
              If provided, forces some neurons active at t+1.
              Accepted shapes:
                (batch_size, N_external): mapped to first N_external neurons
                (batch_size, N): mapped to all neurons
              Values > 0.5 are treated as "force active".

        Returns:
            state: (batch_size, N) binary state in {0,1} (dtype matches W)
        """
        B, N = self.batch_size, self.size
        s = (self.state > 0.5)  # (B, N) bool, presynaptic active

        # Sparse: only sample transmission from active presynaptic neurons
        # Use nonzero with explicit device to ensure GPU execution
        active_b, active_i = s.nonzero(as_tuple=True)  # each (num_active,)
        if active_b.numel() == 0:
            next_state = torch.zeros(B, N, device=self.device, dtype=self.W.dtype)
        else:
            # Sample only rows W[b,i,:] for active (b,i) -> (num_active, N)
            weights_to_sample = self.W[active_b, active_i, :]  # (num_active, N)
            samples = torch.bernoulli(weights_to_sample)  # (num_active, N)
            next_state = torch.zeros(B, N, device=self.device, dtype=torch.float32)
            idx = active_b.unsqueeze(1).expand(-1, N)
            next_state.scatter_add_(0, idx, samples.float())
            next_state = (next_state >= 1).to(dtype=self.W.dtype)

        # Optional forcing input - replace sensory neurons entirely (clamp to environment)
        if external_input is not None:
            ext = external_input.to(device=self.device, dtype=self.W.dtype)
            N_ext = ext.shape[-1]
            next_state[:, :N_ext] = (ext > 0.5).to(dtype=self.W.dtype)

        self.state = next_state
        return self.state

if __name__ == "__main__":
    #run some steps and plot the state history
    N = 500
    BATCH_SIZE = 1
    # Initialize weights as probabilities in [0, 1] (not [-1, 1])
    W = torch.rand(BATCH_SIZE, N, N) * 0.1  # Use 0.3 to keep probabilities moderate
    reservoir = BeggsProbabilisticNetwork(W)
    states = []
    
    # Add initial external input to kickstart the network
    initial_input = torch.zeros(BATCH_SIZE, 5)  # Activate first 5 neurons
    initial_input[:, :] = 1.0
    
    for step in range(500):
        # Provide input only at the first step
        external_input = initial_input if step == 0 else None
        state = reservoir.step(external_input=external_input)
        states.append(state.clone())
    states = torch.stack(states, dim=1)
    # Visualization can be done using experiment_beggs.py or external plotting
    print(f"Generated {len(states)} timesteps of state history")
    print(f"State shape: {states.shape}")