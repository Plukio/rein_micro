# src/environment/environment_setup.py

import gym

def setup_environment():
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': './logs/dssat-pdi.log',
        'mode': 'all',
        'seed': 123456,
        'random_weather': True,
        'cultivar': 'rice'
    }
    env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    print('Observation:', env.observation)
    print(len(env.observation), len(env.observation['sw']))
    return env
