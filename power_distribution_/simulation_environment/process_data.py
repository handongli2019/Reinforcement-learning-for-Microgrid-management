import sys
import os
import datetime
import random
from copy import deepcopy
import pickle

import numpy as np
import pandas as pd

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)


def process_weather():

    weather = pd.read_excel(project_path + '/simulation_environment/original_data/weather.xls')
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    weather = weather.loc[(weather['timestamp'] >= pd.to_datetime('2020-08-31 00:00:00')) & (
        weather['timestamp'] < pd.to_datetime('2022-04-02 00:00:00'))]
    weather.sort_values(by='timestamp', inplace=True, ignore_index=True)
    solar = pd.read_csv(project_path + '/simulation_environment/original_data/solar_power.csv')
    solar['timestamp'] = pd.to_datetime(solar['Time'])
    solar = solar.loc[(solar['timestamp'] >= pd.to_datetime('2020-08-31 00:00:00')) & (
        solar['timestamp'] < pd.to_datetime('2022-04-02 00:00:00'))]

    weather['date'] = weather['timestamp'].dt.date
    solar['date'] = solar['timestamp'].dt.date
    date_intersection = set(weather['date']) & set(solar['date'])
    weather = weather.loc[(weather['date'].isin(date_intersection))]
    solar = solar.loc[(solar['date'].isin(date_intersection))]

    weather.sort_values(by='timestamp', inplace=True, ignore_index=True)
    solar.sort_values(by='timestamp', inplace=True, ignore_index=True)

    weather['Wind Speed'] = weather['Wind Speed'] * 0.2778
    weather = weather[['Temperature', 'Sunshine Duration', 'Shortwave Radiation', 'Cloud Cover Total',
        'Mean Sea Level Pressure', 'Wind Speed', 'Wind Direction']].values
    weather = weather.repeat(repeats=4, axis=0).T
    solar = solar['System Production (W)'].values

    # (('Temperature', 'Sunshine Duration', 'Shortwave Radiation', 'Cloud Cover Total',
    # 'Mean Sea Level Pressure', 'Wind Speed', 'Wind Direction'), timestep)
    # (max, min, mean)
    weather_mmm = np.zeros((7, 192, 3))
    # (cov std, )
    weather_cs = np.zeros((7, 192, 192))

    for i, weather_name in enumerate(('Temperature', 'Sunshine Duration', 'Shortwave Radiation', 'Cloud Cover Total',
        'Mean Sea Level Pressure', 'Wind Speed', 'Wind Direction')):
        # if weather_name != 'Shortwave Radiation':
        #     weather_data = weather[i].reshape((96, -1))
        # else:
        #     weather_data = solar.reshape((96, -1))
        weather_data = weather[i].reshape((96, -1))
        last_weather_data = weather_data[:, : -1]
        today_weather_data = weather_data[:, 1:]
        weather_data = np.concatenate((last_weather_data, today_weather_data), axis=0)
        weather_mmm[i, :, 0] = weather_data.max(axis=1)
        weather_mmm[i, :, 1] = weather_data.min(axis=1)
        weather_mmm[i, :, 2] = weather_data.mean(axis=1)
        weather_cs[i, :, :] = np.cov(weather_data)

    with open(project_path + '/simulation_environment/data/weather.pickle', 'wb') as file:
        pickle.dump((weather_mmm, weather_cs), file)


def process_electric_data():

    electric = pd.read_excel(project_path + '/simulation_environment/original_data/electric.xls')
    electric['timestamp'] = pd.to_datetime(electric['timestamp'])
    electric['Device'] = electric['Device'].apply(lambda x: int(x[-4:]))
    electric = electric.loc[
        (electric['Device'].isin([1012, 1013, 1014, 1027, 1028, 1029, 1039, 1040, 1044, 1046])),
        ['timestamp', 'Time', 'Volume'],
    ]
    electric.sort_values(by='timestamp', inplace=True, ignore_index=True)

    length_statistics = {}
    start_time = pd.to_datetime('2017-01-03 17:45:00')
    while start_time < pd.to_datetime('2017-11-16 23:15:00'):
        end_time = start_time + datetime.timedelta(minutes=15)
        instances = electric.loc[(electric['timestamp'] >= start_time) & (electric['timestamp'] < end_time)]
        num = len(instances)
        start_hm = str(start_time)[11:]
        if start_hm not in length_statistics.keys():
            length_statistics[start_hm] = [deepcopy(num)]
        else:
            length_statistics[start_hm].append(deepcopy(num))
        start_time += datetime.timedelta(minutes=15)

    electric['time'] = electric['timestamp'].astype(str)
    electric['time'] = electric['time'].apply(lambda x: x[11:])
    start_time = pd.to_datetime('00:00:00')
    end_time = start_time + datetime.timedelta(minutes=15)
    start_str = str(start_time)[11:]
    end_str = str(end_time)[11:]
    go_iter = True
    # ((chance of coming, parking_time, volume), timestep, (max, min, mean, std))
    v_statistics = np.zeros((3, 96, 4))
    n = 0
    while go_iter:
        group = electric.loc[(electric['time'] >= start_str) & (electric['time'] < end_str)]
        if len(group) > 1:
            v_statistics[0, n, :] = [np.max(length_statistics[start_str]), np.min(length_statistics[start_str]), np.mean(length_statistics[start_str]), np.std(length_statistics[start_str])]
            v_statistics[1, n, :] = [group['Time'].max(), group['Time'].min(), group['Time'].mean(), group['Time'].std()]
            v_statistics[2, n, :] = [group['Volume'].max(), group['Volume'].min(), group['Volume'].mean(), group['Volume'].std()]
        n += 1
        start_time += datetime.timedelta(minutes=15)
        end_time = start_time + datetime.timedelta(minutes=15)
        start_str = str(start_time)[11:]
        end_str = str(end_time)[11:]
        if end_str == '00:00:00':
            end_str = '24:00:00'
        if start_str == '00:00:00':
            go_iter = False
    with open(project_path + '/simulation_environment/data/electric.pickle', 'wb') as file:
        pickle.dump(v_statistics, file)

if __name__ == '__main__':

    process_weather()
    process_electric_data()