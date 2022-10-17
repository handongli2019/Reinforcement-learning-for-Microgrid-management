from typing import Union, List, Tuple
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticModule(nn.Module):

    def __init__(self, feature_dim):
        super(ActorCriticModule, self).__init__()
        policy_net = [nn.BatchNorm1d(feature_dim)]
        value_net = [nn.BatchNorm1d(feature_dim)]
        feature_dims = [feature_dim, 256, 256, 64]
        for dim in range(len(feature_dims)):
            if dim > 0:
                policy_net.append(nn.Linear(feature_dims[dim - 1], feature_dims[dim]))
                policy_net.append(nn.Tanh())
                value_net.append(nn.Linear(feature_dims[dim - 1], feature_dims[dim]))
                value_net.append(nn.Tanh())
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

    def forward(self, state_features: torch.Tensor,):
        policy_output = self.policy_net.forward(state_features)
        value_output = self.value_net.forward(state_features)
        return policy_output, value_output


class MultiCategoricalDistribution:

    def __init__(self, action_dims: List[int]):
        super().__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self, action_logits: torch.Tensor) -> "MultiCategoricalDistribution":
        self.distribution = [
            Categorical(logits=split) for split in torch.split(action_logits, self.action_dims, dim=1)]
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> torch.Tensor:
        return torch.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> torch.Tensor:
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class ActorCriticPolicy(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        action_dims: List[int],
        learning_rate: float,
    ):
        '''

        :param feature_dim:
        :param action_dims:
        :param learning_rate:
        :return:
        '''

        super(ActorCriticPolicy, self).__init__()

        self.model = ActorCriticModule(feature_dim=feature_dim,)
        # Action distribution
        self.action_distribution = MultiCategoricalDistribution(action_dims=action_dims)
        self.action_net = self.action_distribution.proba_distribution_net(latent_dim=64)
        self.value_net = nn.Linear(64, 1)

        module_gains = {
            self.model: np.sqrt(2),
            self.action_net: 0.01,
            self.value_net: 1,
        }
        for module, gain in module_gains.items():
            module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, state_features: torch.Tensor, actions: Union[torch.Tensor, None] = None,):
        """
        Forward pass in all the networks (actor and critic)

        :param state_features:
        :param actions:
        :return:
        """
        actor_output, critic_output = self.model.forward(state_features=state_features)
        action_logits = self.action_net(actor_output)
        values = self.value_net(critic_output)
        distributions = self.action_distribution.proba_distribution(action_logits=action_logits)
        if actions is None:
            return distributions, values, None
        else:
            log_probs = distributions.log_prob(actions)
            entropys = distributions.entropy()
            return values, log_probs, entropys

    def optimize(self, loss: torch.Tensor, max_grad_norm):

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()