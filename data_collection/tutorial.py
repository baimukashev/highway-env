""" Example for dataset generation using highway-env and IDM model"""

# imports
import os
import sys
import numpy as np

import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation

from utils_video import record_videos
import warnings
warnings.filterwarnings("ignore")


# Defining main function
def main():
      
    # Create RL agent as an IDM Vehicle
    # https://github.com/Farama-Foundation/HighwayEnv/issues/295 
    # Change Action type
    # 1. Vehicle class -> IDM
    # 2. Act() -> none
    # Randomize IDM params vehicle.behavior.IDM
    # 1. ACC_MAX
    # 2. COMFORT_ACC_MAX
    # 3. COMFORT_ACC_MIN
    # 4. target_speeds

    # Create environment
    env_config = {
        'id': 'highway-v0',
        'config': {
            'action': {'type': 'DiscreteMetaAction'},
            "lanes_count": 1,
            "vehicles_count": 10,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": [
                    "presence",
                    "x",
                    "y",
                    "vx",
                    "vy",
                    'heading',
                    'cos_h',
                    'sin_h',
                    'cos_d',
                    'sin_d',
                    'long_off',
                    'lat_off',
                    'ang_off',
                ],
                "absolute": False
            },
            "policy_frequency": 1,
            "duration": 10,
            "controlled_vehicles": 1,
        }
    }

    env = gym.make("highway-v0", render_mode="rgb_array")
    # print(env.config)

    # Collect data
    save_video = False
    num_trajs = 5
    states = []
    
    # save recording
    model_path = f"temp/"
    if save_video:
        env = record_videos(env, video_folder=model_path)

    for ind in range(num_trajs):
        
        # print('Iteration: ', ind)
        
        # update default configs
        env.configure(env_config["config"])
        env = FlattenObservation(env)

        obs, _ = env.reset(seed=ind)
        done = False
        iter = 0
        
        while not done and iter < 500:
            
            # For data collection with IDM, action is ignored
            action = env.action_space.sample()
            obs, reward, done, _ , info = env.step(action)
            
            # add traj index as first element of the feature array
            features = np.insert(obs, 0, ind)
            states.append(features)
            iter += 1

    data = np.vstack(states).swapaxes(0,1)   # [num_states, num_features]

    # save
    save_name = 'sample_acc_n5'
    np.save(save_name, data)

    # print(data.shape)
    env.close()
    
    # load 
    # data = np.load(save_name + '.npy')
    
    print('Data generation finished..')
  


if __name__=="__main__":
    main()
    
    

