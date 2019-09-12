import tensorflow as tf
import numpy as np
import random
import gym
import math
import pickle
import hyperopt.pyll.stochastic
from hyperopt import hp
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO

def run_random_search(env, num_configs, algorithm, space):
    samples = sample(space, num_configs) 
    config_evals = [] #used to extract resulting metric after training of each config
    
    for config_num in range(num_configs):
        # Fixed random state #Move to util script
        rand_state = np.random.RandomState(1).get_state()
        np.random.set_state(rand_state)
        seed = np.random.randint(1, 2**31 - 1)
        tf.set_random_seed(seed)

        random.seed(seed)

        model = get_model(algorithm, env, samples, config_num)

        result = model.learn(total_timesteps=10000)
        model.save("trpo_cartpole_" + str(config_num))
        
        result = evaluate(env, model, num_configs)
        config_evals.append([config_num, result])

    best = max(config_evals, key=lambda x: x[1])
    return best[1]

def get_model(algorithm, env, samples, config_num=0):
    """
        Initializes the model to be evaluated, with specified hyperparameter values from a given 
        configuration. In this case we want to run the cartpole environment using TRPO with a 
        multilayer perceptron as the function approximator.
        
        
        Parameters:
        -----------
            samples: nested lists containing all sampled hyperparameter configurations and the 
                     corresponding values of each hyperparameter.
            env: environment to evaluate model in
            config_num: number of the configuration sampled to evaluate
        
        Returns:
        --------
            model: initialized model with specified settings ready to be trained
        
        TODO: Add more algorithms, should it be possible to 
    """ 
    if algorithm == "trpo":
        model = TRPO(MlpPolicy, env, 
                verbose=1,
                timesteps_per_batch=samples[config_num]['timesteps_per_batch'],
                vf_stepsize=samples[config_num]['vf_stepsize'],
                max_kl=samples[config_num]['max_kl'],
                gamma=samples[config_num]['gamma'],
                lam=samples[config_num]['lam']
                )
    return model

def sample(space, num_configs):
    """
        Returns a specified number of configurations drawn randomly from the search space
        
        
        Parameters:
        -----------
            space: dictionary containing function expressions for all hyperparameters to 
                   optimize
            num_configs: number of sample configurations to draw from the search space
        
        Returns:
        --------
            samples: nested lists containing all sampled hyperparameter configurations and the 
                     corresponding values of each hyperparameter.
    """   
    samples = []
    for i in range(num_configs):
        samples.append(hyperopt.pyll.stochastic.sample(space))
    return samples

def evaluate(env, model, num_configs):
    """
        Computes evaluation metric. In this case, the metric chosen is the mean of 
        the sum of episodic rewards obtained during training
        
        
        Parameters:
        -----------
            env: environment to evaluate model in
            model: current model in a given state during training 
        
        Returns:
        --------
            mean of sum of episodic rewards for a full run of a given configuration
    """ 
    episode_rewards = []
    for _ in range(num_configs):
        reward_sum = 0
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            reward_sum += reward
        episode_rewards.append(reward_sum)
    return np.mean(episode_rewards)



