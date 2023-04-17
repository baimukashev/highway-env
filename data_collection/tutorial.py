""" Example for dataset generation using highway-env and IDM model"""

# imports
import os
import sys
import numpy as np
import pickle

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
            "lanes_count": 3,
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
            "simulation_frequency": 5,
        }
    }

    env = gym.make("highway-v0", render_mode="rgb_array")
    # print(env.config)

    # Collect data
    save_video = False
    num_trajs = 200
    trajs = []
    
    # save recording
    model_path = f"temp/"
    if save_video:
        env = record_videos(env, video_folder=model_path)

    for ind in range(num_trajs):
        
        traj = []
        
        print('Iteration: ', ind)
        
        # update default configs
        
        env.configure(env_config["config"])
        env.reset()
        env = FlattenObservation(env)

        obs, _ = env.reset(seed=ind)
        done = False
        iter = 0
        
        while not done and iter < 300:
            
            # For data collection with IDM, action is ignored
            action = env.action_space.sample()
            
            obs, reward, done, _ , info = env.step(action)
            
            steering = info["demo_action"]["steering"]
            acc = info["demo_action"]["acceleration"]
                        
            # add traj index as first element of the feature array
            # features = np.insert(obs, 0, ind)
            
            actions = np.array((acc, steering))
            
            state_action = np.concatenate((obs, actions))
            
            traj.append(state_action)
            iter += 1
            
        trajs.append(np.array(traj))

    # data = np.vstack(trajs)   # [num_states, num_features + actions]
    

    # save
    # save_name = 'sample_acc_n5'
    # np.save(save_name, data)
    
    with open("sample200.pkl", 'wb') as f:
        pickle.dump(trajs, f)

    # print(data.shape)
    env.close()
    
    # load 
    # data = np.load(save_name + '.npy')
    
    print('Data generation finished..')
    

  


if __name__=="__main__":
    main()
    
    

