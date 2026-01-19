"""
Shared DQN Agent for Multi-Task Learning

This module implements a DQN agent with a shared network across multiple tasks.
Key components:
1. SharedQNetwork: Q-network with learned task embeddings
2. MultiTaskReplayBuffer: Single buffer storing transitions from all tasks
3. SharedDQNAgent: Agent that trains on multiple tasks simultaneously

Gradient Conflict:
==================
Unlike Independent DQN where each task has its own network, Shared DQN uses
a single network for all tasks. This creates gradient conflicts when:
- Task A wants parameters to go in direction X
- Task B wants parameters to go in direction Y
- Result: Compromised performance on both tasks

This baseline demonstrates the need for methods like VarShare that resolve
gradient conflicts through adaptive parameter sharing.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Tuple, Optional


class SharedQNetwork(nn.Module):
    """
    Q-Network with task conditioning via learned embeddings.

    Architecture:
        Task Embedding: task_id → Embedding(num_tasks, embedding_dim)
        Input: [state, task_embedding] → Linear(256) → ReLU → Linear(128) → ReLU → Linear(4)

    The network learns task-specific representations through the embedding layer,
    allowing it to distinguish between different task contexts.
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4, num_tasks: int = 3,
                 embedding_dim: int = 8, hidden_dims: Tuple[int, int] = (256, 128),
                 use_task_embedding: bool = True):
        """
        Initialize Shared Q-Network.

        Args:
            state_dim: Dimension of state space (8 for Lunar Lander)
            action_dim: Number of discrete actions (4 for Lunar Lander)
            num_tasks: Number of tasks (3: standard, windy, heavy)
            embedding_dim: Size of learned task embeddings (8-16 for moderate capacity)
            hidden_dims: Tuple of hidden layer sizes
            use_task_embedding: If False, network is task-blind (no task conditioning)
        """
        super(SharedQNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim if use_task_embedding else 0
        self.use_task_embedding = use_task_embedding

        # Learned task embeddings (only if enabled)
        if use_task_embedding:
            self.task_embedding = nn.Embedding(num_tasks, embedding_dim)
        else:
            self.task_embedding = None

        # Shared network (same architecture as Independent DQN)
        input_dim = state_dim + self.embedding_dim  # state only if no embedding
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], action_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, state: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute Q-values conditioned on task.

        Args:
            state: State tensor of shape (batch_size, state_dim)
            task_id: Task ID tensor of shape (batch_size,) with integers [0, 1, 2]
                     (ignored if use_task_embedding=False)

        Returns:
            Q-values for all actions, shape (batch_size, action_dim)
        """
        if self.use_task_embedding:
            # Get task embedding and concatenate with state
            task_emb = self.task_embedding(task_id)  # (batch_size, embedding_dim)
            x = torch.cat([state, task_emb], dim=-1)  # (batch_size, state_dim + embedding_dim)
        else:
            # Task-blind: use state only
            x = state

        # Forward through shared network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class MultiTaskReplayBuffer:
    """
    Single shared replay buffer storing transitions from all tasks.

    VarShare-compatible design:
    - All tasks share same buffer (unified experience pool)
    - Task ID stored with each transition for proper conditioning
    - Random sampling creates naturally mixed batches → gradient conflicts

    This is intentionally simple to demonstrate the baseline gradient conflict.
    More sophisticated buffers could:
    - Balance sampling across tasks
    - Prioritize important transitions
    - Separate buffers with controlled mixing
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, task_id: int):
        """
        Store a transition in the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            task_id: Task identifier (0=standard, 1=windy, 2=heavy)
        """
        self.buffer.append((state, action, reward, next_state, done, task_id))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random batch of transitions (mixed across tasks).

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, task_ids)
            All as numpy arrays
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, task_ids = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(task_ids)
        )

    def __len__(self):
        return len(self.buffer)


class SharedDQNAgent:
    """
    DQN agent that learns multiple tasks with a single shared network.

    Key difference from Independent DQN:
    - Single Q-network for all tasks (gradient conflicts!)
    - Task conditioning via learned embeddings
    - Single optimizer updates shared parameters with mixed gradients

    Expected behavior:
    - Lower sample efficiency than Independent DQN
    - Performance degradation (~60%) due to gradient conflicts
    - Some tasks suffer more than others (differential impact)
    """

    def __init__(self, state_dim: int, action_dim: int, num_tasks: int = 3,
                 embedding_dim: int = 8, hidden_dims: Tuple[int, int] = (256, 128),
                 learning_rate: float = 5e-4, gamma: float = 0.99, device: str = 'cpu',
                 use_task_embedding: bool = True):
        """
        Initialize Shared DQN Agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            num_tasks: Number of tasks to learn
            embedding_dim: Size of task embeddings
            hidden_dims: Hidden layer sizes
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            device: Device for torch tensors ('cpu' or 'cuda')
            use_task_embedding: If False, network is task-blind (no task conditioning)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        self.gamma = gamma
        self.device = device
        self.use_task_embedding = use_task_embedding

        # Single Q-network shared across all tasks
        self.q_network = SharedQNetwork(
            state_dim, action_dim, num_tasks, embedding_dim, hidden_dims,
            use_task_embedding=use_task_embedding
        ).to(device)

        # Target network (for stable Q-learning)
        self.target_network = SharedQNetwork(
            state_dim, action_dim, num_tasks, embedding_dim, hidden_dims,
            use_task_embedding=use_task_embedding
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Single optimizer (gradient conflicts happen here!)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state: np.ndarray, task_id: int, epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            task_id: Task identifier (0, 1, or 2)
            epsilon: Exploration rate (uses self.epsilon if None)

        Returns:
            Selected action (0-3 for Lunar Lander)
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        # Greedy action selection
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            task_id_t = torch.LongTensor([task_id]).to(self.device)
            q_values = self.q_network(state_t, task_id_t)
            return q_values.argmax(1).item()

    def update(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
               next_states: np.ndarray, dones: np.ndarray, task_ids: np.ndarray) -> float:
        """
        Update Q-network with a batch of transitions (mixed across tasks).

        This is where gradient conflicts occur:
        - Batch contains transitions from different tasks
        - All tasks share same network parameters
        - Gradients from different tasks may point in conflicting directions
        - Single optimizer step compromises all tasks

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            task_ids: Batch of task identifiers

        Returns:
            Loss value
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        task_ids = torch.LongTensor(task_ids).to(self.device)

        # Current Q-values: Q(s, a)
        current_q = self.q_network(states, task_ids).gather(1, actions)

        # Target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states, task_ids).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # MSE loss (Bellman error)
        loss = F.mse_loss(current_q, target_q)

        # Backpropagation (gradient conflicts happen here!)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """
        Save agent state to file.

        Args:
            filepath: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)

    def load(self, filepath: str):
        """
        Load agent state from file.

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0.01)


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test Shared DQN components
    print("="*60)
    print("TESTING SHARED DQN COMPONENTS")
    print("="*60)

    # Test network
    print("\n1. Testing SharedQNetwork:")
    network = SharedQNetwork(state_dim=8, action_dim=4, num_tasks=3, embedding_dim=8)
    print(f"   Parameters: {count_parameters(network):,}")

    # Test forward pass
    state = torch.randn(2, 8)  # Batch of 2
    task_id = torch.LongTensor([0, 1])  # Tasks 0 and 1
    q_values = network(state, task_id)
    print(f"   Input: state {state.shape}, task_id {task_id.shape}")
    print(f"   Output: Q-values {q_values.shape}")
    assert q_values.shape == (2, 4), "Output shape mismatch!"
    print("   ✓ Forward pass successful")

    # Test replay buffer
    print("\n2. Testing MultiTaskReplayBuffer:")
    buffer = MultiTaskReplayBuffer(capacity=1000)
    for i in range(100):
        buffer.push(
            state=np.random.randn(8),
            action=np.random.randint(4),
            reward=np.random.randn(),
            next_state=np.random.randn(8),
            done=False,
            task_id=i % 3
        )
    print(f"   Buffer size: {len(buffer)}")

    batch = buffer.sample(32)
    print(f"   Sampled batch: {len(batch)} arrays")
    print(f"   Task IDs in batch: {set(batch[5])}")  # Should see mix of 0, 1, 2
    print("   ✓ Buffer sampling successful")

    # Test agent
    print("\n3. Testing SharedDQNAgent:")
    agent = SharedDQNAgent(state_dim=8, action_dim=4, num_tasks=3)
    print(f"   Total parameters: {count_parameters(agent.q_network):,}")

    # Test action selection
    state = np.random.randn(8)
    action = agent.select_action(state, task_id=1, epsilon=0.1)
    print(f"   Selected action: {action}")
    assert 0 <= action < 4, "Invalid action!"
    print("   ✓ Action selection successful")

    # Test update
    states, actions, rewards, next_states, dones, task_ids = buffer.sample(32)
    loss = agent.update(states, actions, rewards, next_states, dones, task_ids)
    print(f"   Update loss: {loss:.4f}")
    print("   ✓ Network update successful")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
