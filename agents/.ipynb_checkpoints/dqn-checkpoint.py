"""
Deep Q-Network (DQN) Agent

This module implements the classic DQN algorithm with the following key components:
1. Q-Network: Neural network that approximates the Q-function Q(s,a)
2. Target Network: Stabilizes training by providing fixed Q-targets
3. Epsilon-Greedy Exploration: Balances exploration vs exploitation
4. Experience Replay: Handled externally via ReplayBuffer

DQN Algorithm Overview:
=======================
Q-Learning is a value-based RL algorithm that learns the optimal action-value function:
    Q*(s,a) = Expected return starting from state s, taking action a, then following optimal policy

The Bellman equation for Q-learning:
    Q(s,a) = r + γ * max_a' Q(s',a')

DQN approximates Q using a neural network and uses two key innovations:
1. **Experience Replay**: Store transitions and sample randomly to break correlations
2. **Target Network**: Use a separate, slowly-updated network for computing targets

Training Update:
    Loss = E[(Q(s,a) - (r + γ * max_a' Q_target(s',a')))²]

Where Q_target is the target network, updated every N steps or via soft updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import os


class QNetwork(nn.Module):
    """
    Q-Network: Neural network that approximates Q(s,a) for all actions.

    Architecture:
        Input (state_dim) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(num_actions)

    This architecture is relatively simple but effective for Lunar Lander.
    For more complex tasks, you might use:
        - Deeper networks
        - Convolutional layers (for visual inputs)
        - Dueling architecture (separate value and advantage streams)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, int] = (256, 128)):
        """
        Initialize the Q-Network.

        Args:
            state_dim: Dimension of state space (8 for Lunar Lander)
            action_dim: Number of discrete actions (4 for Lunar Lander)
            hidden_dims: Tuple of hidden layer sizes
        """
        super(QNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Neural network layers
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], action_dim)

        # Initialize weights using He initialization (good for ReLU)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute Q-values for all actions.

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Q-values for all actions, shape (batch_size, action_dim) or (action_dim,)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)  # No activation on output (Q-values can be any real number)
        return q_values


class DQNAgent:
    """
    DQN Agent that learns to play a single task.

    This agent uses the Q-learning algorithm with function approximation (neural network)
    to learn an optimal policy.

    Key Hyperparameters:
    ====================
    - learning_rate: Step size for gradient descent (typically 1e-4 to 5e-4)
    - gamma: Discount factor (0.99 typical). How much to value future rewards.
    - epsilon_start/end/decay: Exploration schedule. Start high (1.0 = random),
      decay to low (0.01 = mostly greedy)
    - target_update_freq: How often to update target network (stability vs recency trade-off)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 10,
        device: str = 'cpu'
    ):
        """
        Initialize the DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Multiplicative decay factor for epsilon
            target_update_freq: Update target network every N episodes
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Create Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)

        # Initialize target network with same weights as Q-network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode

        # Optimizer (Adam is standard for DQN)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Loss function (Mean Squared Error for TD error)
        self.criterion = nn.MSELoss()

        # Training statistics
        self.steps = 0
        self.episodes = 0

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Select an action using epsilon-greedy policy.

        Epsilon-Greedy Exploration:
        ===========================
        With probability epsilon: choose random action (exploration)
        With probability (1-epsilon): choose argmax_a Q(s,a) (exploitation)

        This balances exploration (trying new actions to discover better strategies)
        and exploitation (using current knowledge to maximize reward).

        Args:
            state: Current state observation (numpy array)
            epsilon: Exploration rate. If None, use agent's current epsilon.

        Returns:
            Selected action (int)
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Exploration: random action
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)

        # Exploitation: greedy action
        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():  # Don't track gradients during inference
            q_values = self.q_network(state_tensor)

        # Return action with highest Q-value
        return q_values.argmax(dim=1).item()

    def update(self, batch: Tuple[np.ndarray, ...]) -> float:
        """
        Perform one gradient descent step on a batch of transitions.

        DQN Update Rule:
        ================
        1. Compute current Q-values: Q(s,a) using q_network
        2. Compute target Q-values: r + γ * max_a' Q_target(s',a')
        3. Compute TD error: δ = Q(s,a) - target
        4. Minimize loss: L = E[δ²]
        5. Update q_network weights using gradient descent

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
                   Each is a numpy array of shape (batch_size, ...)

        Returns:
            Loss value (float)
        """
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # --- Compute Current Q-Values ---
        # q_network outputs Q(s,a) for all actions
        # We only want Q(s,a) for the action that was actually taken
        current_q_values = self.q_network(states)  # Shape: (batch_size, action_dim)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # gather selects Q-values for taken actions, shape: (batch_size,)

        # --- Compute Target Q-Values ---
        with torch.no_grad():  # Don't track gradients for target computation
            # Use target network for stability
            next_q_values = self.target_network(next_states)  # Shape: (batch_size, action_dim)

            # Take maximum Q-value over actions (greedy policy)
            max_next_q_values = next_q_values.max(dim=1)[0]  # Shape: (batch_size,)

            # Bellman equation: Q(s,a) = r + γ * max_a' Q(s',a')
            # If episode terminated (done=True), there's no next state, so target = r
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # --- Compute Loss and Update ---
        loss = self.criterion(current_q_values, target_q_values)

        # Gradient descent step
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()             # Compute gradients
        # Gradient clipping helps prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()       # Update weights

        self.steps += 1

        return loss.item()

    def update_target_network(self) -> None:
        """
        Update target network by copying weights from Q-network.

        Why Target Network?
        ===================
        Without a target network, we'd compute targets using the same network
        we're updating, creating a "moving target" problem that destabilizes training.

        The target network provides a fixed target for several updates, then
        gets synchronized with the Q-network. This stabilizes learning.

        Two common strategies:
        1. Hard update (this implementation): Copy weights every N steps
        2. Soft update (alternative): θ_target = τ*θ + (1-τ)*θ_target
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """
        Decay epsilon according to epsilon_decay factor.

        Epsilon Decay Schedule:
        =======================
        Start with high exploration (epsilon=1.0), gradually decrease to
        mostly exploitation (epsilon=0.01).

        This schedule implements:
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        Example: epsilon_start=1.0, epsilon_decay=0.995, epsilon_end=0.01
            Episode 0: epsilon = 1.0 (100% random)
            Episode 100: epsilon ≈ 0.606
            Episode 200: epsilon ≈ 0.367
            Episode 500: epsilon ≈ 0.082
            Episode 920: epsilon = 0.01 (mostly greedy)
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """
        Save agent's Q-network to disk.

        Args:
            filepath: Path to save file (e.g., 'results/models/dqn_checkpoint.pth')
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, filepath)
        print(f"Model saved to: {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load agent's Q-network from disk.

        Args:
            filepath: Path to saved file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        print(f"Model loaded from: {filepath}")

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (f"DQNAgent(state_dim={self.state_dim}, action_dim={self.action_dim}, "
                f"epsilon={self.epsilon:.4f}, steps={self.steps})")


# Example usage and testing
if __name__ == "__main__":
    print("Testing DQN Agent")
    print("=" * 60)

    # Create agent for Lunar Lander
    agent = DQNAgent(
        state_dim=8,      # Lunar Lander state space
        action_dim=4,     # Lunar Lander action space
        learning_rate=5e-4,
        gamma=0.99,
        device='cpu'
    )

    print(f"Created: {agent}\n")

    # Test forward pass
    print("Testing forward pass...")
    dummy_state = np.random.randn(8)
    action = agent.select_action(dummy_state, epsilon=0.1)
    print(f"  State: {dummy_state[:3]}... (truncated)")
    print(f"  Selected action: {action}\n")

    # Test update
    print("Testing training update...")
    batch_size = 32
    dummy_batch = (
        np.random.randn(batch_size, 8),  # states
        np.random.randint(0, 4, batch_size),  # actions
        np.random.randn(batch_size),  # rewards
        np.random.randn(batch_size, 8),  # next_states
        np.random.randint(0, 2, batch_size).astype(float)  # dones
    )

    loss = agent.update(dummy_batch)
    print(f"  Loss: {loss:.4f}\n")

    # Test epsilon decay
    print("Testing epsilon decay...")
    initial_epsilon = agent.epsilon
    for _ in range(10):
        agent.decay_epsilon()
    print(f"  Initial epsilon: {initial_epsilon:.4f}")
    print(f"  After 10 decays: {agent.epsilon:.4f}\n")

    print("=" * 60)
    print("DQN Agent test complete!")
