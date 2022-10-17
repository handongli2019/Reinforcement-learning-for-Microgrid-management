import sys
import os
import pickle
import random

project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)

from simulation_environment.environment import rl_env
from PPO.PPO_algorithm import PPO_pipline


def testing():
    '''
    Let the PPO interact with the environment, train the PPO model.
    :return:
    '''

    env = rl_env()
    seed = random.sample(list(range(4000)), 1)[0]

    rl = PPO_pipline(
        file_name='_',
        env=env,
        seed=seed,
        device='cpu',
    )
    with open(project_path + '/common.pickle', 'rb') as file:
        log = pickle.load(file)
    params = log['params']
    rl.load_params(params)
    mean_cumulative_reward = rl.test()
    return mean_cumulative_reward


if __name__ == '__main__':

    # import numpy as np
    #
    # all_cumulative_rewards = []
    # for _ in range(100):
    #     mean_cumulative_reward = testing()
    #     all_cumulative_rewards.append(mean_cumulative_reward.copy())
    #
    # print(np.mean(all_cumulative_rewards))
    # print(np.std(all_cumulative_rewards))

    mean_cumulative_reward = testing()
    print(f'{mean_cumulative_reward}')