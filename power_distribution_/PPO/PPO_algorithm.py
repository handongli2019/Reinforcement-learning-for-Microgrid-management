import sys
import os
from copy import deepcopy
from typing import Optional
import random
import time
import pickle

import numpy as np
import torch
from torch.nn import functional as F

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from PPO import buffers, policies


class PPO_pipline:

    def __init__(
            self,
            file_name,
            env,
            learning_rate: float = 0.0003,
            n_steps: int = 2048,
            batch_size: Optional[int] = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            clip_range_vf: float = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            seed: int = 400,
            device: str = 'cpu',
    ):
        '''

        :param file_name: The filename where the parameter log file is stored.
        :param learning_rate: learning rate
        :param n_steps: The number of timesteps each time the ppo interacts with the environment and samples
        :param batch_size: batch size for training the model
        :param n_epochs: number of epochs per training
        :param gamma: discount factor
        :param gae_lambda: One of the parameters to calculate GAE in PPO
        :param clip_range: Parameters for PPO clipping surrogate object update ratio
        :param clip_range_vf: Parameters for PPO clipping state value object update ratio
        :param normalize_advantage: whether to normalize estimation advantage value
        :param ent_coef: param for encouraging exploration
        :param vf_coef: param for state value object loss proportion
        :param max_grad_norm: gradient clipping strength
        :param seed: Number of random seeds
        :param device: Use 'cpu' or 'gpu' to train the model
        '''

        random.seed(seed)
        # Seed numpy RNG
        np.random.seed(seed)
        # seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(seed)

        self.file_name = file_name
        self.init_learning_rate = learning_rate
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage

        self.env = env
        feature_dim = self.env.feature_dim
        action_dim = 10
        action_dims = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.rollout_buffer = buffers.experience_pool(
            buffer_size=self.n_steps,
            feature_dim=feature_dim,
            action_dim=action_dim,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
        )
        self.policy = policies.ActorCriticPolicy(
            feature_dim=feature_dim,
            action_dims=action_dims,
            learning_rate=learning_rate,
        ).to(torch.float32).to(self.device)

        self.num_timesteps = 0
        self.num_episodes = 0
        self._last_obs = None
        self._last_episode_starts = None
        # log
        # training
        self.episode_reward = []
        self.training_cumulative_reward = []
        self.training_best_reward = -float('inf')
        # loss
        self.value_loss = []
        self.policy_loss = []
        self.entropy_loss = []
        # testing
        self.testing_cumulative_reward = [[], []]
        self.testing_init_reward = -float('inf')
        self.testing_best_reward = -float('inf')
        self.last_testing_time = time.strftime("%Y-%m-%d %H", time.localtime())
        # params
        self.params = self.get_params()

    def feature_engineering(self, state, device):
        '''
        Feature engineering
        :param state:
        :param device:
        :return:
        '''

        carport, car_energy_requirements, weather_info, past_time = state
        state_feature = np.hstack((carport, car_energy_requirements, weather_info.flatten(), past_time))[np.newaxis, :]
        state_feature = torch.from_numpy(state_feature).to(torch.float32).to(device)

        return state_feature

    def sample_from_self_play(self):
        '''
        Let the PPO interact with the environment to obtain (state, action, reward)
        tuples of several fixed timesteps pairs.
        :return:
        '''

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.eval()
        need_testing = False

        n_steps = 0
        self.rollout_buffer.reset()

        while n_steps < self.n_steps:

            with torch.no_grad():
                # Convert to pytorch tensor
                state_feature = self.feature_engineering(
                    state=self._last_obs,
                    device=self.device,
                )
                distribution, value, _ = self.policy.forward(state_features=state_feature)

            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            action = action.cpu().detach().numpy()[0]
            value = value.cpu()[0]
            log_prob = log_prob.cpu()[0]

            new_obs, reward, done = self.env.step(action=action)

            self.episode_reward.append(deepcopy(reward))

            if done:
                cumulative_reward = np.sum(self.episode_reward)
                self.training_cumulative_reward.append(deepcopy(cumulative_reward))
                if len(self.training_cumulative_reward) > 100:
                    mean_cumulative_reward = np.mean(self.training_cumulative_reward[-100:])
                    current_time = time.strftime("%Y-%m-%d %H", time.localtime())
                    if (mean_cumulative_reward > self.training_best_reward) or \
                        (current_time != self.last_testing_time):
                        if mean_cumulative_reward > self.training_best_reward:
                            self.training_best_reward = mean_cumulative_reward
                            self.params = self.get_params()
                        if current_time != self.last_testing_time:
                            self.last_testing_time = current_time
                        need_testing = True
                    print(f'''
                    *********************************************************************************************
                    During training session in {self.num_episodes} training episode, {current_time},
                    the current mean cumulative reward is {mean_cumulative_reward},
                    and the best mean cumulative reward is {self.training_best_reward}.
                    *********************************************************************************************
                    ''')
                self.num_episodes += 1
                new_obs = self.env.reset()
                self.episode_reward = []

            self.num_timesteps += 1
            n_steps += 1

            self.rollout_buffer.add(
                obs=state_feature,
                action=action,
                reward=reward,
                episode_start=self._last_episode_starts,
                value=value,
                log_prob=log_prob,
            )
            self._last_obs = new_obs
            self._last_episode_starts = done

        with torch.no_grad():
            state_feature = self.feature_engineering(
                state=self._last_obs,
                device=self.device,
            )
            _, value, _ = self.policy.forward(state_features=state_feature)
        value = value.cpu()[0]

        self.rollout_buffer.compute_returns_and_advantage(last_values=value, done=done)

        return need_testing

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """

        self.policy.train()
        # # Update optimizer learning rate
        # for param_group in self.policy.optimizer.param_groups:
        #     param_group['lr'] = learning_rate

        all_value_loss = []
        all_policy_loss = []
        all_entropy_loss = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                observations = rollout_data.observations.to(self.device)
                actions = rollout_data.actions.to(self.device)
                old_values = rollout_data.old_values.to(self.device)
                old_log_prob = rollout_data.old_log_prob.to(self.device)
                advantages = rollout_data.advantages.to(self.device)
                returns = rollout_data.returns.to(self.device)

                values, log_prob, entropy = self.policy.forward(
                    state_features=observations,
                    actions=actions,
                )

                # Normalize advantage
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # flatten data
                values = torch.flatten(values)
                log_prob = torch.flatten(log_prob)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    values_pred = old_values + torch.clamp(
                        values - old_values, - self.clip_range_vf, self.clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(returns, values_pred)

                entropy_loss = - torch.mean(entropy)

                all_value_loss.append(value_loss.item())
                all_policy_loss.append(policy_loss.item())
                all_entropy_loss.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimize(
                    loss=loss,
                    max_grad_norm=self.max_grad_norm,
                )

        return np.mean(all_value_loss), np.mean(all_policy_loss), np.mean(all_entropy_loss)

    def test(self, test_episode_times: int = 100, deterministic: bool = True):
        '''
        Test the performance of the current model.
        :param test_episode_times: Number of simulation tests
        :return:
        '''

        self.policy.eval()
        env = deepcopy(self.env)

        cumulative_rewards = []

        for _ in range(test_episode_times):

            new_obs = env.reset()
            rewards = []
            done = False

            while not done:

                with torch.no_grad():
                    # Convert to pytorch tensor
                    state_feature = self.feature_engineering(
                        state=new_obs,
                        device=self.device,
                    )
                    distribution, value, _ = self.policy.forward(state_features=state_feature)

                if not deterministic:
                    action = distribution.sample()
                else:
                    action = distribution.mode()
                action = action.cpu().detach().numpy()[0]

                new_obs, reward, done = env.step(action=action)

                rewards.append(deepcopy(reward))

            cumulative_reward = np.sum(rewards)
            cumulative_rewards.append(deepcopy(cumulative_reward))

        return np.mean(cumulative_rewards)

    def get_params(self):
        '''
        Get the parameters of the model
        :return:
        '''
        params = deepcopy(self.policy.cpu().state_dict())
        self.policy = self.policy.to(self.device)
        return params

    def saving(self):
        '''
        Store logs and model parameters
        :return:
        '''

        with open(project_path + f'/{self.file_name}.pickle', 'wb') as file:
            pickle.dump({
                'params': self.params,
                'training_cumulative_reward': self.training_cumulative_reward,
                'training_best_reward': self.training_best_reward,
                'value_loss': self.value_loss,
                'policy_loss': self.policy_loss,
                'entropy_loss': self.entropy_loss,
                'testing_cumulative_reward': self.testing_cumulative_reward,
                'testing_init_reward': self.testing_init_reward,
                'testing_best_reward': self.testing_best_reward,
            }, file)

    def load_params(self, params):
        '''
        Read model parameters
        :param params:
        :return:
        '''
        self.policy = self.policy.to('cpu')
        self.policy.load_state_dict(params)
        self.policy = self.policy.to(self.device)

    def test_and_update_test_log(self):
        mean_cumulative_reward = self.test()
        self.testing_cumulative_reward[0].append(deepcopy(self.num_episodes))
        self.testing_cumulative_reward[1].append(deepcopy(mean_cumulative_reward))
        if mean_cumulative_reward > self.testing_best_reward:
            self.testing_best_reward = mean_cumulative_reward
            self.params = self.get_params()
            self.saving()
        print(f'''
        *********************************************************************************************
        During testing session after {self.num_episodes} training episode, 
        the current mean cumulative reward is {mean_cumulative_reward}, 
        the initial mean cumulative reward is {self.testing_init_reward}, 
        and the best mean cumulative reward is {self.testing_best_reward}. 
        Under random policy, mean cumulative reward is -251.7958. 
        *********************************************************************************************
        ''')
        return mean_cumulative_reward

    def learn(self, total_timesteps: int, ):
        '''
        The main method for training the model
        :param total_timesteps: total training steps (timesteps)
        :return:
        '''

        self.num_timesteps = 0
        self.num_episodes = 0
        self._last_obs = self.env.reset()
        self._last_episode_starts = True
        # log
        # training
        self.episode_reward = []
        self.training_cumulative_reward = []
        self.training_best_reward = -float('inf')
        # loss
        self.value_loss = []
        self.policy_loss = []
        self.entropy_loss = []
        # testing
        self.testing_cumulative_reward = [[], []]
        self.testing_init_reward = -float('inf')
        self.testing_best_reward = -float('inf')
        self.last_testing_time = time.strftime("%Y-%m-%d %H", time.localtime())
        # params
        self.params = self.get_params()

        # first test before training
        mean_cumulative_reward = self.test_and_update_test_log()
        self.testing_init_performance = mean_cumulative_reward

        # mean_policy_losss = []
        # mean_entropy_losss = []
        last_time = time.strftime("%Y-%m-%d %H", time.localtime())
        while self.num_timesteps < total_timesteps:
            current_time = time.strftime("%Y-%m-%d %H", time.localtime())
            if current_time != last_time:
                last_time = current_time
                print(last_time)
                self.saving()
            need_testing = self.sample_from_self_play()
            # if need_testing:
            #     self.test_and_update_test_log()
            mean_value_loss, mean_policy_loss, mean_entropy_loss = self.train()
            # mean_policy_losss.append(deepcopy(mean_policy_loss))
            # mean_entropy_losss.append(deepcopy(mean_entropy_loss))
            # if len(mean_policy_losss) > 100:
            #     mean_policy_losss.pop(0)
            #     mean_entropy_losss.pop(0)
            #     print(np.mean(mean_policy_losss))
            #     print(np.mean(mean_entropy_losss))
            self.value_loss.append(deepcopy(mean_value_loss))
            self.policy_loss.append(deepcopy(mean_policy_loss))
            self.entropy_loss.append(deepcopy(mean_entropy_loss))

        self.load_params(self.params)
        self.test_and_update_test_log()
        self.saving()