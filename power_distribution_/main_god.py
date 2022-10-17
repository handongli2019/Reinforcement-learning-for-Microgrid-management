import sys
import os

project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)

from simulation_environment.environment_god import rl_env
from PPO.PPO_algorithm import PPO_pipline


def training():
    '''
    Let the PPO interact with the environment, train the PPO model.
    :return:
    '''

    env = rl_env()

    ent_coef = 0
    file_name = 'god'
    rl = PPO_pipline(
        file_name=file_name,
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=None,
        n_epochs=10,
        ent_coef=ent_coef,
        seed=400,
        device='cuda',
    )
    rl.learn(total_timesteps=10000000)


if __name__ == '__main__':

    training()