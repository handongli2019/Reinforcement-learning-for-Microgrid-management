from typing import NamedTuple, Optional

import numpy as np
import torch


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class experience_pool:

    def __init__(
        self,
        buffer_size: int,
        feature_dim: int,
        action_dim: int,
        gae_lambda: float = 1,
        gamma: float = 0.99,
    ):

        self.buffer_size = buffer_size
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.reset()

    def reset(self) -> None:

        self.observations = torch.zeros((self.buffer_size, self.feature_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.buffer_size, self.action_dim), dtype=torch.long)
        self.rewards = torch.zeros((self.buffer_size,), dtype=torch.float32)
        self.returns = torch.zeros((self.buffer_size,), dtype=torch.float32)
        self.episode_starts = torch.zeros((self.buffer_size,), dtype=torch.float32)
        self.values = torch.zeros((self.buffer_size,), dtype=torch.float32)
        self.log_probs = torch.zeros((self.buffer_size,), dtype=torch.float32)
        self.advantages = torch.zeros((self.buffer_size,), dtype=torch.float32)
        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(self, last_values: torch.Tensor, done: bool) -> None:

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - done
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: torch.Tensor,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: bool,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ):

        self.observations[self.pos] = obs.clone()
        self.actions[self.pos] = torch.tensor(action)
        self.rewards[self.pos] = torch.tensor(reward)
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.clone()
        self.log_probs[self.pos] = log_prob.clone()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def to_torch(self, tensor: torch.Tensor, copy: bool = True) -> torch.Tensor:
        if copy:
            return tensor.clone()
        return tensor

    def get(self, batch_size: Optional[int] = None):

        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds],
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self.returns[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))