"""
GradNorm Multi-Task DQN Agent

Implements GradNorm from "GradNorm: Gradient Normalization for Adaptive Loss
Balancing in Deep Multitask Networks" (Chen et al., 2018).

Dynamically balances task losses via learnable weights, encouraging tasks
to train at similar rates.

Key Components:
    - GradNormLossWeighter: Manages learnable task weights and GradNorm updates
    - GradNormDQNAgent: DQN agent using GradNorm loss balancing
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

# Import shared network architecture
from agents.shared_dqn import SharedQNetwork


class GradNormLossWeighter:
    """
    Learnable task weights with GradNorm optimization.

    GradNorm Algorithm:
        1. Compute gradient norms: G_i = ||∇_W_shared (w_i * L_i)||
        2. Compute training rates: r_i = L_i(t) / L_i(0)
        3. Compute relative inverse training rates: r_tilde_i = r_i / E[r]
        4. Compute target norms: target_i = E[G] * (r_tilde_i)^α
        5. Compute GradNorm loss: L_grad = Σ |G_i - target_i|
        6. Update weights to minimize L_grad (while keeping them positive)

    The asymmetry parameter α controls how much to focus on balancing:
        - α = 0: No balancing (standard weighted sum)
        - α > 0: Harder tasks (higher loss) get higher weights
        - α = 1: Linear balancing
        - α > 1: Stronger emphasis on balancing (α = 1.5 recommended)
    """

    def __init__(self, num_tasks: int = 3, alpha: float = 1.5,
                 weight_lr: float = 0.01, device: str = 'cpu'):
        """
        Initialize GradNorm loss weighter.

        Args:
            num_tasks: Number of tasks
            alpha: Asymmetry parameter (controls focus on balancing)
            weight_lr: Learning rate for task weights
            device: Device for tensors
        """
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.device = device

        # Learnable task weights (initialized to 1.0)
        # Using raw values that get normalized during forward pass
        self.log_weights = nn.Parameter(torch.zeros(num_tasks, device=device))

        # Optimizer for weights only
        self.weight_optimizer = optim.Adam([self.log_weights], lr=weight_lr)

        # Initial losses for computing training rates (set on first update)
        self.initial_losses = None
        self.loss_history = {i: [] for i in range(num_tasks)}

        # Weight evolution tracking
        self.weight_history = []

    def get_weights(self) -> torch.Tensor:
        """
        Get normalized task weights.

        Weights are kept positive via exp and normalized to sum to num_tasks
        (so average weight is 1.0).

        Returns:
            Tensor of normalized task weights
        """
        raw_weights = torch.exp(self.log_weights)
        normalized_weights = raw_weights / raw_weights.sum() * self.num_tasks
        return normalized_weights

    def compute_weighted_loss(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted sum of task losses.

        Args:
            losses: List of per-task loss tensors

        Returns:
            Weighted total loss
        """
        weights = self.get_weights()
        weighted_loss = sum(w * L for w, L in zip(weights, losses))
        return weighted_loss

    def update_weights(self, losses: List[torch.Tensor],
                       shared_layer: nn.Module,
                       main_optimizer: optim.Optimizer) -> Dict:
        """
        Update task weights using GradNorm algorithm.

        Args:
            losses: List of per-task loss tensors
            shared_layer: The shared layer to compute gradient norms on (e.g., fc2)
            main_optimizer: Main network optimizer (to restore gradients)

        Returns:
            Dictionary with GradNorm statistics
        """
        weights = self.get_weights()

        # Record losses for training rate computation
        current_losses = [L.detach().item() for L in losses]
        for i, loss in enumerate(current_losses):
            self.loss_history[i].append(loss)

        # Initialize initial losses on first update
        if self.initial_losses is None:
            self.initial_losses = current_losses.copy()
            # Return dummy stats for first update (no weight change)
            self.weight_history.append(weights.detach().cpu().numpy().tolist())
            return {
                'weights': weights.detach().cpu().numpy().tolist(),
                'grad_norms': [0.0] * self.num_tasks,
                'target_norms': [0.0] * self.num_tasks,
                'gradnorm_loss': 0.0
            }

        # Step 1: Compute gradient norms for each task using autograd.grad
        # This preserves the computational graph for backprop through weights
        grad_norms = []
        for i, (w, L) in enumerate(zip(weights, losses)):
            # Compute gradient of weighted loss w.r.t. shared layer
            weighted_loss = w * L

            # Use autograd.grad to get gradients while preserving graph for weight updates
            grads = torch.autograd.grad(
                weighted_loss, shared_layer.weight,
                create_graph=True, retain_graph=True
            )

            # Compute gradient norm (L2 norm)
            grad_norm = grads[0].norm()
            grad_norms.append(grad_norm)

        # Step 2: Compute training rates (relative to initial loss)
        training_rates = []
        for i in range(self.num_tasks):
            if self.initial_losses[i] > 1e-8:
                r_i = current_losses[i] / self.initial_losses[i]
            else:
                r_i = 1.0
            training_rates.append(r_i)

        # Step 3: Compute relative inverse training rates
        avg_rate = np.mean(training_rates)
        if avg_rate > 1e-8:
            rel_rates = [r / avg_rate for r in training_rates]
        else:
            rel_rates = [1.0] * self.num_tasks

        # Step 4: Compute target gradient norms
        # Targets are computed from avg_grad_norm but treated as constants (detached)
        # This follows the GradNorm paper: gradients flow through G_i, not through targets
        grad_norms_tensor = torch.stack(grad_norms)
        avg_grad_norm = grad_norms_tensor.mean().detach()  # Detach for target computation
        target_norms = []
        for rel_rate in rel_rates:
            target = avg_grad_norm * (rel_rate ** self.alpha)
            target_norms.append(target)

        # Step 5: Compute GradNorm loss
        gradnorm_loss = sum(
            torch.abs(G - target)
            for G, target in zip(grad_norms, target_norms)
        )

        # Step 6: Update weights to minimize GradNorm loss
        self.weight_optimizer.zero_grad()
        gradnorm_loss.backward()
        self.weight_optimizer.step()

        # Track weight evolution
        new_weights = self.get_weights()
        self.weight_history.append(new_weights.detach().cpu().numpy().tolist())

        # Return statistics
        return {
            'weights': new_weights.detach().cpu().numpy().tolist(),
            'grad_norms': [g.item() for g in grad_norms],
            'target_norms': [t.item() for t in target_norms],
            'gradnorm_loss': gradnorm_loss.item(),
            'training_rates': training_rates
        }

    def get_weight_evolution(self) -> Dict[str, List[float]]:
        """
        Get weight evolution over training.

        Returns:
            Dict mapping task name to list of weights over time
        """
        if not self.weight_history:
            return {}

        evolution = {
            'standard': [w[0] for w in self.weight_history],
            'windy': [w[1] for w in self.weight_history],
            'heavy': [w[2] for w in self.weight_history]
        }
        return evolution


class GradNormDQNAgent:
    """
    DQN agent with GradNorm loss balancing for multi-task learning.

    Uses SharedQNetwork architecture with GradNorm to dynamically balance
    task losses during training.

    Key differences from SharedDQNAgent:
        - Learnable task weights via GradNormLossWeighter
        - Uses fc2 (last hidden layer) as shared layer for gradient norm computation
        - Tracks weight evolution over training
    """

    def __init__(self, state_dim: int, action_dim: int, num_tasks: int = 3,
                 embedding_dim: int = 8, hidden_dims: Tuple[int, int] = (256, 128),
                 learning_rate: float = 5e-4, gamma: float = 0.99,
                 gradnorm_alpha: float = 1.5, gradnorm_lr: float = 0.01,
                 use_task_embedding: bool = True, device: str = 'cpu'):
        """
        Initialize GradNorm DQN Agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            num_tasks: Number of tasks to learn
            embedding_dim: Size of task embeddings
            hidden_dims: Hidden layer sizes
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            gradnorm_alpha: GradNorm asymmetry parameter
            gradnorm_lr: Learning rate for task weights
            use_task_embedding: Whether to use task embeddings (False for task-blind)
            device: Device for torch tensors
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        self.gamma = gamma
        self.device = device
        self.use_task_embedding = use_task_embedding

        # Use same SharedQNetwork architecture
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

        # Main optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # GradNorm loss weighter
        self.loss_weighter = GradNormLossWeighter(
            num_tasks=num_tasks,
            alpha=gradnorm_alpha,
            weight_lr=gradnorm_lr,
            device=device
        )

        # Reference to shared layer for gradient norm computation
        # Using fc2 (last hidden layer before output)
        self.shared_layer = self.q_network.fc2

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Statistics tracking
        self.gradnorm_stats_history = []

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
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        batch_size = states.shape[0]
        task_ids = torch.LongTensor([task_id] * batch_size).to(self.device)

        current_q = self.q_network(states, task_ids).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_network(next_states, task_ids).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(current_q, target_q)
        return loss

    def update(self, task_batches: List[Tuple[np.ndarray, ...]]) -> Tuple[float, Dict]:
        """
        Update Q-network using GradNorm-weighted losses.

        Args:
            task_batches: List of (states, actions, rewards, next_states, dones)
                          tuples, one per task

        Returns:
            Tuple of (weighted loss value, GradNorm statistics)
        """
        # Step 1: Compute per-task losses for GradNorm weight update
        losses = []
        for task_id, batch in enumerate(task_batches):
            loss = self.compute_task_loss(*batch, task_id)
            losses.append(loss)

        # Step 2: Update GradNorm weights (this uses the computation graph)
        gradnorm_stats = self.loss_weighter.update_weights(
            losses, self.shared_layer, self.optimizer
        )

        # Step 3: Recompute losses for main network update (fresh forward pass)
        # This is necessary because the computation graph was freed during GradNorm update
        losses_for_update = []
        for task_id, batch in enumerate(task_batches):
            loss = self.compute_task_loss(*batch, task_id)
            losses_for_update.append(loss)

        # Step 4: Compute weighted loss and update main network
        self.optimizer.zero_grad()
        weighted_loss = self.loss_weighter.compute_weighted_loss(losses_for_update)
        weighted_loss.backward()
        self.optimizer.step()

        # Track statistics
        self.gradnorm_stats_history.append(gradnorm_stats)

        return weighted_loss.item(), gradnorm_stats

    def update_target_network(self):
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_current_weights(self) -> List[float]:
        """Get current task weights."""
        return self.loss_weighter.get_weights().detach().cpu().numpy().tolist()

    def get_weight_evolution(self) -> Dict[str, List[float]]:
        """Get weight evolution over training."""
        return self.loss_weighter.get_weight_evolution()

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
            'log_weights': self.loss_weighter.log_weights.data,
            'initial_losses': self.loss_weighter.initial_losses,
            'weight_history': self.loss_weighter.weight_history,
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
        if 'log_weights' in checkpoint:
            self.loss_weighter.log_weights.data = checkpoint['log_weights']
        if 'initial_losses' in checkpoint:
            self.loss_weighter.initial_losses = checkpoint['initial_losses']
        if 'weight_history' in checkpoint:
            self.loss_weighter.weight_history = checkpoint['weight_history']
        self.epsilon = checkpoint.get('epsilon', 0.01)
