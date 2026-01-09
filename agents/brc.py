"""
BRC (Bigger, Regularized, Categorical) Agent for Multi-Task RL

Implementation based on "Bigger, Regularized, Categorical: High-Capacity Value Functions
are Efficient Multi-Task Learners"

Key innovations:
1. BroNet: Large residual network with task embeddings
2. Categorical DQN: Distributional RL with cross-entropy loss (C51-style)
3. Regularization: Weight decay + LayerNorm for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """
    Single residual block with LayerNorm.

    Structure: x -> LayerNorm -> Linear -> ReLU -> Linear -> (+x) -> output

    Uses LayerNorm (NOT BatchNorm) - BatchNorm doesn't work well in RL
    due to non-stationary data distributions.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: (batch, hidden_dim)

        Returns:
            output: (batch, hidden_dim)
        """
        residual = x
        x = self.ln(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x + residual  # Residual connection


class BroNet(nn.Module):
    """
    BRC's residual network architecture (BroNet).

    Architecture:
        [state + task_embedding] -> Linear -> [ResBlock x N] -> LayerNorm -> Linear -> [num_actions * num_atoms]

    Args:
        state_dim: Dimension of state (8 for LunarLander)
        num_actions: Number of discrete actions (4 for LunarLander)
        num_tasks: Number of tasks (3 for our setup)
        embed_dim: Task embedding dimension (default: 32)
        hidden_dim: Hidden layer dimension (default: 256)
        num_blocks: Number of residual blocks (default: 3)
        num_atoms: Number of atoms for categorical distribution (default: 51)
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        num_tasks: int,
        embed_dim: int = 32,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        num_atoms: int = 51
    ):
        super().__init__()

        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.hidden_dim = hidden_dim

        # Task embeddings - learnable vector per task
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)
        nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.1)

        # Input projection: state + task_embedding -> hidden
        self.input_layer = nn.Linear(state_dim + embed_dim, hidden_dim)
        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.input_layer.bias)

        # Stack of residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # Final layer norm before output
        self.final_ln = nn.LayerNorm(hidden_dim)

        # Output head: predicts distribution over returns for each action
        self.output_layer = nn.Linear(hidden_dim, num_actions * num_atoms)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: (batch, state_dim)
            task_id: (batch,) integer task indices

        Returns:
            logits: (batch, num_actions, num_atoms) - raw logits for categorical distribution
        """
        # Get task embedding
        task_emb = self.task_embedding(task_id)  # (batch, embed_dim)

        # Concatenate state and task embedding
        x = torch.cat([state, task_emb], dim=-1)  # (batch, state_dim + embed_dim)

        # Input projection
        x = F.relu(self.input_layer(x))

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Final norm and output
        x = self.final_ln(x)
        logits = self.output_layer(x)  # (batch, num_actions * num_atoms)

        # Reshape to (batch, num_actions, num_atoms)
        logits = logits.view(-1, self.num_actions, self.num_atoms)

        return logits

    def get_q_values(self, state: torch.Tensor, task_id: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values by computing expected value of the distribution.

        Q(s,a) = Σ_i z_i * p_i

        Args:
            state: (batch, state_dim)
            task_id: (batch,)
            support: (num_atoms,) - the atom values z_i

        Returns:
            q_values: (batch, num_actions)
        """
        logits = self.forward(state, task_id)
        probs = F.softmax(logits, dim=-1)  # (batch, num_actions, num_atoms)

        # Expected value: sum over atoms weighted by probabilities
        q_values = (probs * support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return q_values

    def get_probs(self, state: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over atoms for each action."""
        logits = self.forward(state, task_id)
        return F.softmax(logits, dim=-1)


class BRCAgent:
    """
    BRC (Bigger, Regularized, Categorical) Agent for Multi-Task RL.

    Combines:
    1. BroNet: Large residual network with task embeddings
    2. Categorical DQN: Distributional RL with cross-entropy loss
    3. Regularization: Weight decay + LayerNorm
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        num_tasks: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        embed_dim: int = 32,
        num_atoms: int = 51,
        v_min: float = -100.0,
        v_max: float = 300.0,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        device: str = "cpu"
    ):
        """
        Initialize BRC agent.

        Args:
            state_dim: State dimension
            num_actions: Number of actions
            num_tasks: Number of tasks
            hidden_dim: Hidden layer size (256 or 512)
            num_blocks: Number of residual blocks (2-4)
            embed_dim: Task embedding dimension
            num_atoms: Number of categorical atoms (51 standard)
            v_min: Minimum return value (-100 for LunarLander)
            v_max: Maximum return value (300 for LunarLander)
            gamma: Discount factor
            learning_rate: Learning rate
            weight_decay: L2 regularization strength
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.num_tasks = num_tasks
        self.gamma = gamma

        # Categorical DQN parameters
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Networks
        self.q_network = BroNet(
            state_dim=state_dim,
            num_actions=num_actions,
            num_tasks=num_tasks,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_atoms=num_atoms
        ).to(self.device)

        self.target_network = BroNet(
            state_dim=state_dim,
            num_actions=num_actions,
            num_tasks=num_tasks,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_atoms=num_atoms
        ).to(self.device)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer with weight decay (AdamW for regularization)
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        print(f"\n🤖 BRC Agent initialized:")
        print(f"   Network: BroNet (hidden_dim={hidden_dim}, blocks={num_blocks})")
        print(f"   Parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        print(f"   Categorical: {num_atoms} atoms, range=[{v_min}, {v_max}]")
        print(f"   Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")

    def select_action(self, state: np.ndarray, task_id: int, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: Current state
            task_id: Task index (0, 1, 2)
            epsilon: Exploration probability

        Returns:
            action: Selected action index
        """
        if random.random() < epsilon:
            return random.randrange(self.num_actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            task_t = torch.LongTensor([task_id]).to(self.device)
            q_values = self.q_network.get_q_values(state_t, task_t, self.support)
            return q_values.argmax(dim=1).item()

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        task_ids: torch.Tensor
    ) -> float:
        """
        Update networks using categorical DQN loss.

        Args:
            states: (batch, state_dim)
            actions: (batch,)
            rewards: (batch,)
            next_states: (batch, state_dim)
            dones: (batch,)
            task_ids: (batch,)

        Returns:
            loss: Cross-entropy loss value
        """
        # Get current distribution logits
        current_logits = self.q_network(states, task_ids)  # (batch, actions, atoms)

        # Get next state distributions from target network (Double DQN)
        with torch.no_grad():
            # Use online network to select best action
            next_q = self.q_network.get_q_values(next_states, task_ids, self.support)
            next_actions = next_q.argmax(dim=1)  # (batch,)

            # Use target network to get distribution of selected action
            next_probs = self.target_network.get_probs(next_states, task_ids)  # (batch, actions, atoms)

            # Select distribution for best action
            next_action_indices = next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.num_atoms)
            next_probs_selected = next_probs.gather(1, next_action_indices).squeeze(1)  # (batch, atoms)

            # Project Bellman update onto support
            target_probs = self._project_distribution(next_probs_selected, rewards, dones)

        # Select current distribution for taken action
        action_indices = actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.num_atoms)
        current_logits_selected = current_logits.gather(1, action_indices).squeeze(1)  # (batch, atoms)

        # Cross-entropy loss: -Σ target * log(softmax(logits))
        log_probs = F.log_softmax(current_logits_selected, dim=-1)
        loss = -(target_probs * log_probs).sum(dim=-1).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def _project_distribution(
        self,
        next_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Project Bellman-updated distribution onto categorical support.

        This is the key distributional RL operation:
        1. Compute Tz_j = r + γ * z_j for each atom (clipped to [v_min, v_max])
        2. Distribute probability mass to neighboring atoms

        Args:
            next_probs: (batch, num_atoms) - next state distribution
            rewards: (batch,)
            dones: (batch,)

        Returns:
            target_probs: (batch, num_atoms) - projected target distribution
        """
        batch_size = rewards.shape[0]

        # Compute Tz = r + γ * z (γ=0 if done)
        # Shape: (batch, num_atoms)
        Tz = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * self.support.unsqueeze(0)

        # Clip to support range
        Tz = Tz.clamp(self.v_min, self.v_max)

        # Compute projection indices
        b = (Tz - self.v_min) / self.delta_z  # (batch, num_atoms)
        l = b.floor().long().clamp(0, self.num_atoms - 1)  # lower index
        u = b.ceil().long().clamp(0, self.num_atoms - 1)   # upper index

        # Distribute probability mass
        target_probs = torch.zeros(batch_size, self.num_atoms, device=self.device)

        # Efficient batched projection using scatter_add
        for i in range(batch_size):
            for j in range(self.num_atoms):
                # Lower projection: p_l += p_j * (u - b)
                target_probs[i, l[i, j]] += next_probs[i, j] * (u[i, j].float() - b[i, j])
                # Upper projection: p_u += p_j * (b - l)
                target_probs[i, u[i, j]] += next_probs[i, j] * (b[i, j] - l[i, j].float())

        return target_probs

    def update_target_network(self):
        """Hard update: copy weights from q_network to target_network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filepath: str):
        """Save agent checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class MultiTaskReplayBuffer:
    """
    Replay buffer for multi-task learning.
    Stores transitions with task IDs.
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        task_id: int
    ):
        """Store transition."""
        self.buffer.append((state, action, reward, next_state, done, task_id))

    def sample(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, ...]:
        """
        Sample batch from buffer.

        Returns:
            states, actions, rewards, next_states, dones, task_ids (all as tensors)
        """
        batch = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor([t[0] for t in batch]).to(device)
        actions = torch.LongTensor([t[1] for t in batch]).to(device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(device)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(device)
        dones = torch.FloatTensor([t[4] for t in batch]).to(device)
        task_ids = torch.LongTensor([t[5] for t in batch]).to(device)

        return states, actions, rewards, next_states, dones, task_ids

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
