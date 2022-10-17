'''
reinforcement learning simulation environment
'''

import sys
import os
import random
from copy import deepcopy
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rc('font', family='Times New Roman')
COLORS = ['#FFFFFF', '#000000', '#FF0000', '#FFFF00']
CMAP = mcolors.LinearSegmentedColormap.from_list("", COLORS)
NORM = mcolors.Normalize(vmax=3, vmin=0)

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)


class rl_env:
    '''
    Use for training RL
    '''

    def __init__(self,):

        self.past_time = 0  # Count how many timesteps has gone in the current state
        with open(project_path + '/simulation_environment/data/electric.pickle', 'rb') as file:
            # (3, 96, 4), ((chance of coming, parking_time, volume), timestep, (max, min, mean, std))
            self.electric = pickle.load(file)
        self.charging_speed = 20

        # It is used to indicate whether a certain parking space is occupied among the 10 parking spaces, and how many
        # timesteps to be occupied. If it is 0, it means it is not occupied, 1 means it will be occupied for 1 timestep,
        # and so on.
        self.carport = np.zeros((10, ), dtype=int)

        # Record the energy required by the parked car in the current timestep
        self.car_energy_requirements = np.zeros((10, ), dtype=np.float32)

        with open(project_path + '/simulation_environment/data/weather.pickle', 'rb') as file:
            # (('Temperature', 'Sunshine Duration', 'Shortwave Radiation', 'Cloud Cover Total',
            # 'Mean Sea Level Pressure', 'Wind Speed', 'Wind Direction'), timestep)
            # (7, 96, 3), (max, min, mean)
            # (7, 96, 96), (cov std, )
            self.weather = pickle.load(file)

        # The weather information: Temperature, Sunshine Duration, Shortwave Radiation, Cloud Cover Total,
        # Mean Sea Level Pressure, Wind Speed, Wind Direction
        self.weather_info = np.zeros((7, 192), dtype=np.float32)
        self.clean_energy_generated = np.zeros((96, ))  # clean energy generated
        self.clean_energy_stored = np.zeros((96, ))
        self.borrow_energy = np.zeros((96,))

        self.feature_dim = 23

    @staticmethod
    def cal_generated_clean_energy(weather_info, uc=4, ur=15, uf=25, eta=0.15, A=100, ):
        wind_speed, shortwave_radiation = weather_info[5, 96:], weather_info[2, 96:]
        wind_power = np.zeros_like(wind_speed)
        index_ = np.where((wind_speed > uc) & (wind_speed < ur))
        wind_power[index_] = 1.5e6 * ((uc ** 3 - wind_speed[index_] ** 3) / (uc ** 3 - ur ** 3)) * 900
        index_ = np.where((wind_speed >= ur) & (wind_speed < uf))
        wind_power[index_] = 1.5e6
        solar_power = shortwave_radiation * eta * A * 900
        power = wind_power + solar_power
        power /= 3600000
        return power

    def reset(self):
        '''
        Reset all environment variables to the initial state of the new episode.
        :return: state of env
        '''

        self.past_time = 0
        self.carport = np.random.choice([0, 1], 10, replace=True)
        notempty_carport = np.where(self.carport != 0)
        parking_time = np.clip(np.random.normal(
            loc=self.electric[1, self.past_time, 2],
            scale=self.electric[1, self.past_time, 3],
            size=len(notempty_carport[0]),
        ), a_min=self.electric[1, self.past_time, 1], a_max=self.electric[1, self.past_time, 0])
        self.carport[notempty_carport] = (parking_time / 15 + 0.5).astype(int)
        self.car_energy_requirements = np.array([0] * 10, dtype=np.float32)
        energy = np.clip(np.random.normal(
            loc=self.electric[2, self.past_time, 2],
            scale=self.electric[2, self.past_time, 3],
            size=len(notempty_carport[0]),
        ), a_min=self.electric[2, self.past_time, 1], a_max=self.electric[2, self.past_time, 0])
        self.car_energy_requirements[notempty_carport] = energy

        for i in range(7):
            self.weather_info[i] = np.random.multivariate_normal(
                mean=self.weather[0][i, :, 2],
                cov=self.weather[1][i],
            )
            self.weather_info[i] = np.clip(
                self.weather_info[i],
                a_min=self.weather[0][i, :, 1],
                a_max=self.weather[0][i, :, 0],
            )

        self.clean_energy_generated = self.cal_generated_clean_energy(weather_info=self.weather_info)
        self.clean_energy_stored[:] = 0
        self.borrow_energy[:] = 0

        return_weather_info = self.weather_info[[5, 2], self.past_time + 96]

        return (self.carport.copy(), self.car_energy_requirements.copy(),
                return_weather_info.copy(), deepcopy(self.past_time))

    def reward_func(self):
        return

    def step(self, action: np.array):
        '''
        Env receives the action, and make one timestep forward
        :param action: binary vector with 10 item like [1, 0, 0, 1, 1, 1, 0, 1, 0, 0]
        :return:
        '''

        # Charge, calculate clean energy consumption and borrow power
        new_car_energy_requirements = self.car_energy_requirements - action * self.charging_speed
        new_car_energy_requirements[np.where(new_car_energy_requirements < 0)] = 0
        energy_cost = self.car_energy_requirements.sum() - new_car_energy_requirements.sum()

        #
        reward = 0
        clean_energy_left = self.clean_energy_generated[self.past_time] - energy_cost
        if clean_energy_left > 0:
            self.clean_energy_stored[self.past_time] = clean_energy_left
        elif clean_energy_left < 0:
            self.borrow_energy[self.past_time] = -clean_energy_left
            reward = clean_energy_left

        # One step forward in the environment, judging whether the old car has left, randomly entering the new car,
        # and new car parking timestep and power demand
        self.carport -= 1
        empty_carport = np.where(self.carport <= 0)
        self.carport[empty_carport] = 0
        # Calculate the battery that is not fully charged
        not_fully_charged = new_car_energy_requirements[empty_carport].sum()
        reward -= not_fully_charged
        new_car_energy_requirements[empty_carport] = 0
        self.car_energy_requirements = new_car_energy_requirements
        # new car entry
        new_car_num = int(np.clip(np.random.normal(
            loc=self.electric[0, self.past_time, 2],
            scale=self.electric[0, self.past_time, 3],
        ), a_min=self.electric[0, self.past_time, 1], a_max=self.electric[0, self.past_time, 0]) + 0.5)
        parking_num = len(empty_carport[0])
        car_entry_num = min([new_car_num, parking_num])
        if car_entry_num > 0:
            parking_entry_id = random.sample(empty_carport[0].tolist(), car_entry_num)
            parking_time = np.clip(np.random.normal(
                loc=self.electric[1, self.past_time, 2],
                scale=self.electric[1, self.past_time, 3],
                size=car_entry_num,
            ), a_min=self.electric[1, self.past_time, 1], a_max=self.electric[1, self.past_time, 0])
            self.carport[parking_entry_id] = (parking_time / 15 + 0.5).astype(int)
            self.car_energy_requirements[parking_entry_id] = np.clip(np.random.normal(
                loc=self.electric[2, self.past_time, 2],
                scale=self.electric[2, self.past_time, 3],
                size=car_entry_num,
            ), a_min=self.electric[2, self.past_time, 1], a_max=self.electric[2, self.past_time, 0])

        self.past_time += 1
        done = False
        if self.past_time > 95:
            done = True

        if self.past_time != 96:
            return_weather_info = self.weather_info[[5, 2], self.past_time + 96]
        else:
            return_weather_info = self.weather_info[[5, 2], self.past_time - 96]

        return (self.carport.copy(), self.car_energy_requirements.copy(),
                return_weather_info.copy(), deepcopy(self.past_time)), reward, done

    def render(self):
        '''
        Render the parking situation.
        :return:
        '''

        grid_map = np.zeros((1, 10), dtype=int)
        grid_map[:, np.where(self.carport > 0)[0].tolist()] = 1

        plt.imshow(grid_map, cmap=CMAP, norm=NORM, interpolation='none', extent=(0, 10, 1, 0))
        plt.xticks(range(10), labels=[])
        plt.yticks(range(1), labels=[])
        plt.grid(linewidth=1, color='black')
        plt.show()


if __name__ == '__main__':

    env = rl_env()

    all_cumulative_rewards = []
    for _ in range(100):
        cumulative_rewards = []
        for _ in range(100):
            done = False
            env.reset()
            rewards = []
            while not done:
                action = np.random.randint(low=0, high=2, size=(10, ))
                state, reward, done = env.step(action=action)
                past_time = state[2]
                rewards.append(reward)
            cumulative_reward = np.sum(rewards)
            cumulative_rewards.append(cumulative_reward.copy())
        all_cumulative_rewards.append(np.mean(cumulative_rewards))

    print(np.mean(all_cumulative_rewards))
    print(np.std(all_cumulative_rewards))