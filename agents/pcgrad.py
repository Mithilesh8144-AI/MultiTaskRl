"""
PCGrad (Projected Gradient) Multi-Task DQN Agent

Implements gradient surgery from "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020).
When task gradients conflict (dot product < 0), projects them onto the normal plane
to eliminate negative interference.

Key Components:
    - PCGradOptimizer: Wraps base optimizer with pc_backward for gradient projection
    - PerTaskReplayBuffer: Separate buffer per task for clean per-task gradient computation
    - PCGradDQNAgent: DQN agent using PCGrad optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

# Import shared network architecture
from agents.shared_dqn import SharedQNetwork


class PCGradOptimizer:
    """
    Wrapper around base optimizer that implements PCGrad algorithm.

    PCGrad projects conflicting gradients to eliminate negative interference:
    For each task gradient g_i conflicting with g_j (dot product < 0):
        g_i' = g_i - (g_i · g_j / ||g_j||²) * g_j

    This removes the component of g_i that conflicts with g_j.
    """

    def __init__(self, optimizer: optim.Optimizer, num_tasks: int = 3):
        """
        Initialize PCGrad optimizer wrapper.

        Args:
            optimizer: Base PyTorch optimizer (e.g., Adam)
            num_tasks: Number of tasks for multi-task learning
        """
        self.optimizer = optimizer
        self.num_tasks = num_tasks

        # Statistics tracking
        self.conflict_count = 0
        self.total_pairs = 0
        self.conflict_history = []

    def zero_grad(self):
        """Clear gradients in base optimizer."""
        self.optimizer.zero_grad()

    def step(self):
        """Take optimization step with base optimizer."""
        self.optimizer.step()

    def pc_backward(self, losses: List[torch.Tensor], parameters: List[torch.Tensor]) -> Dict:
        """
        Compute PCGrad-projected gradients and set them on parameters.

        Args:
            losses: List of per-task loss tensors
            parameters: List of model parameters to optimize

        Returns:
            Dictionary with conflict statistics
        """
        # Compute per-task gradients
        task_grads = []
        for i, loss in enumerate(losses):
            self.optimizer.zero_grad()
            loss.backward(retain_graph=(i < len(losses) - 1))

            # Collect gradients as flat vector
            grad_vec = []
            for param in parameters:
                if param.grad is not None:
                    grad_vec.append(param.grad.clone().flatten())
                else:
                    grad_vec.append(torch.zeros(param.numel(), device=param.device))
            task_grads.append(torch.cat(grad_vec))

        # Project conflicting gradients (PCGrad core algorithm)
        projected_grads = self._project_gradients(task_grads)

        # Average projected gradients
        final_grad = torch.stack(projected_grads).mean(dim=0)

        # Set gradients back to parameters
        self.optimizer.zero_grad()
        offset = 0
        for param in parameters:
            numel = param.numel()
            param.grad = final_grad[offset:offset + numel].view_as(param)
            offset += numel

        # Calculate conflict ratio for this update
        conflict_ratio = self.conflict_count / max(self.total_pairs, 1)
        self.conflict_history.append(conflict_ratio)

        # Reset counters for next update
        stats = {
            'conflict_count': self.conflict_count,
            'total_pairs': self.total_pairs,
            'conflict_ratio': conflict_ratio
        }
        self.conflict_count = 0
        self.total_pairs = 0

        return stats

    def _project_gradients(self, task_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Project conflicting gradients using PCGrad algorithm.

        For each pair (i, j) where g_i · g_j < 0:
            g_i' = g_i - (g_i · g_j / ||g_j||²) * g_j

        Args:
            task_grads: List of gradient vectors, one per task

        Returns:
            List of projected gradient vectors
        """
        num_tasks = len(task_grads)
        projected = [g.clone() for g in task_grads]

        # Random order for fairness (as in original paper)
        order = list(range(num_tasks))
        random.shuffle(order)

        for i in order:
            for j in order:
                if i == j:
                    continue

                self.total_pairs += 1

                # Check for conflict: dot product < 0
                dot = torch.dot(projected[i], task_grads[j])

                if dot < 0:
                    self.conflict_count += 1

                    # Project g_i onto plane normal to g_j
                    # g_i' = g_i - (g_i · g_j / ||g_j||²) * g_j
                    norm_sq = torch.dot(task_grads[j], task_grads[j])
                    if norm_sq > 1e-10:  # Avoid division by zero
                        projected[i] = projected[i] - (dot / norm_sq) * task_grads[j]

        return projected

    def get_average_conflict_ratio(self, window: int = 100) -> float:
        """
        Get average conflict ratio over recent updates.

        Args:
            window: Number of recent updates to average

        Returns:
            Average conflict ratio
        """
        if not self.conflict_history:
            return 0.0
        recent = self.conflict_history[-window:]
        return np.mean(recent)


class PerTaskReplayBuffer:
    """
    Separate replay buffer for each task.

    Unlike MultiTaskReplayBuffer (shared), this maintains separate buffers
    to enable clean per-task gradient computation for PCGrad.

    Benefits:
        - Clean per-task batches for accurate gradient computation
        - No cross-task contamination in batches
        - Balanced sampling across tasks
    """

    def __init__(self, num_tasks: int = 3, capacity_per_task: int = 33333):
        """
        Initialize per-task replay buffers.

        Args:
            num_tasks: Number of tasks
            capacity_per_task: Maximum transitions per task buffer
        """
        self.num_tasks = num_tasks
        self.capacity_per_task = capacity_per_task
        self.buffers = {i: deque(maxlen=capacity_per_task) for i in range(num_tasks)}

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, task_id: int):
        """
        Store a transition in the appropriate task buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            task_id: Task identifier (0=standard, 1=windy, 2=heavy)
        """
        self.buffers[task_id].append((state, action, reward, next_state, done))

    def sample(self, task_id: int, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch from a specific task's buffer.

        Args:
            task_id: Task identifier to sample from
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffers[task_id], batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def can_sample(self, task_id: int, batch_size: int) -> bool:
        """Check if task buffer has enough samples."""
        return len(self.buffers[task_id]) >= batch_size

    def can_sample_all(self, batch_size: int) -> bool:
        """Check if all task buffers have enough samples."""
        return all(self.can_sample(i, batch_size) for i in range(self.num_tasks))

    def __len__(self) -> int:
        """Total transitions across all buffers."""
        return sum(len(buf) for buf in self.buffers.values())

    def task_size(self, task_id: int) -> int:
        """Number of transitions in specific task buffer."""
        return len(self.buffers[task_id])


class PCGradDQNAgent:
    """
    DQN agent with PCGrad optimization for multi-task learning.

    Uses SharedQNetwork architecture with PCGrad optimizer to handle
    gradient conflicts between tasks.

    Key differences from SharedDQNAgent:
        - Uses PerTaskReplayBuffer instead of shared buffer
        - Computes per-task losses separately
        - Applies PCGrad projection before optimizer step
        - Tracks gradient conflict statistics
    """

    def __init__(self, state_dim: int, action_dim: int, num_tasks: int = 3,
                 embedding_dim: int = 8, hidden_dims: Tuple[int, int] = (256, 128),
                 learning_rate: float = 5e-4, gamma: float = 0.99, device: str = 'cpu',
                 use_task_embedding: bool = True):
        """
        Initialize PCGrad DQN Agent.

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

        # Use same SharedQNetwork architecture for fair comparison
        self.q_network = SharedQNetwork(
            state_dim, action_dim, num_tasks, embedding_dim, hidden_dims,
            use_task_embedding=use_task_embedding
        ).to(device)

        # Target network
        self.target_network = SharedQNetwork(
            state_dim, action_dim, num_tasks, embedding_dim, hidden_dims,
            use_task_embedding=use_task_embedding
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # PCGrad optimizer wrapper
        base_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.optimizer = PCGradOptimizer(base_optimizer, num_tasks)

        # List of parameters for gradient computation
        self.parameters = list(self.q_network.parameters())

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Conflict tracking
        self.conflict_stats_history = []

    def select_action(self, state: np.ndarray, task_id: int,
                      epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            task_id: Task identifier (0, 1, or 2)
            epsilon: Exploration rate (uses self.epsilon if None)

        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            task_id_t = torch.LongTensor([task_id]).to(self.device)
            q_values = self.q_network(state_t, task_id_t)
            return q_values.argmax(1).item()

    def compute_task_loss(self, states: np.ndarray, actions: np.ndarray,
                          rewards: np.ndarray, next_states: np.ndarray,
                          dones: np.ndarray, task_id: int) -> torch.Tensor:
        """
        Compute MSE loss for a single task batch.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            task_id: Task identifier

        Returns:
            MSE loss tensor
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Task IDs for this batch (all same)
        batch_size = states.shape[0]
        task_ids = torch.LongTensor([task_id] * batch_size).to(self.device)

        # Current Q-values
        current_q = self.q_network(states, task_ids).gather(1, actions)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states, task_ids).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # MSE loss
        loss = F.mse_loss(current_q, target_q)
        return loss

    def update(self, replay_buffer: PerTaskReplayBuffer, batch_size: int) -> Tuple[float, Dict]:
        """
        Update Q-network using PCGrad with per-task losses.

        Args:
            replay_buffer: PerTaskReplayBuffer with separate task buffers
            batch_size: Batch size per task

        Returns:
            Tuple of (average loss, conflict statistics)
        """
        # Compute per-task losses
        losses = []
        for task_id in range(self.num_tasks):
            batch = replay_buffer.sample(task_id, batch_size)
            loss = self.compute_task_loss(*batch, task_id)
            losses.append(loss)

        # Apply PCGrad and update
        conflict_stats = self.optimizer.pc_backward(losses, self.parameters)
        self.optimizer.step()

        # Track statistics
        self.conflict_stats_history.append(conflict_stats)

        # Return average loss and stats
        avg_loss = sum(l.item() for l in losses) / len(losses)
        return avg_loss, conflict_stats

    def update_target_network(self):
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_average_conflict_ratio(self, window: int = 100) -> float:
        """Get average conflict ratio over recent updates."""
        return self.optimizer.get_average_conflict_ratio(window)

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
            'optimizer': self.optimizer.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'conflict_history': self.optimizer.conflict_history,
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
        self.optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        if 'conflict_history' in checkpoint:
            self.optimizer.conflict_history = checkpoint['conflict_history']
