import tensorflow as tf
import numpy as np
import random
import gym
import math
import pickle
import hyperopt.pyll.stochastic

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO
from hyperopt import hp

def get_space():
    """
        Defines the search space to sample from for each hyperparameter for the hyperparameter 
        optimization. Define all parameters to tune in the given model here. 
        
        Returns:
        --------
            dict-like expression graph consisting of nested function expressions for all 
            hyperparameters to optimize.
    """   
    space = {
            'timesteps_per_batch': hp.choice('timesteps_per_batch', [512, 1024, 2048, 4096, 8192]),
            'vf_stepsize': hp.loguniform('vf_stepsize', -5, -2),
            'max_kl' : hp.loguniform('max_kl', -2.5, -0.5),
            'gamma': hp.uniform('gamma', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))), #4: Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
            'lam': hp.uniform('lam', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))) #4: Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
            }
    return space

# Draw a specified number of configurations randomly from the search space
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

# Initialize the model to be trained
def get_model(env, samples, config_num=0):
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
    """ 
    model = TRPO(MlpPolicy, env, 
              verbose=1,
              timesteps_per_batch=samples[config_num]['timesteps_per_batch'],
              vf_stepsize=samples[config_num]['vf_stepsize'],
              max_kl=samples[config_num]['max_kl'],
              gamma=samples[config_num]['gamma'],
              lam=samples[config_num]['lam']
            )
    return model

# Evaluation metric is calculated as the mean of the sum of episodic rewards for the given model.
def evaluate(env, model):
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


if __name__ == "__main__":

    num_configs = 3 #specify number of configurations to sample
    config_evals = [] #used to extract resulting metric after training of each config

    space = get_space()
    samples = sample(space, num_configs) 

    for config_num in range(num_configs):
        # Fixed random state
        rand_state = np.random.RandomState(1).get_state()
        np.random.set_state(rand_state)
        seed = np.random.randint(1, 2**31 - 1)
        tf.set_random_seed(seed)
        random.seed(seed)

        env = gym.make('CartPole-v1') #initialize environment
        env = DummyVecEnv([lambda: env])

        model = get_model(env, samples, config_num)

        model.learn(total_timesteps=10000)
        model.save("trpo_cartpole_" + str(config_num))
        
        result = evaluate(env, model)
        config_evals.append([config_num, result])

        best = max(config_evals, key=lambda x: x[1])
    print("best: {} \n".format(best)) #configuration number best[0]
    
    model = TRPO.load(("trpo_cartpole_" + str(best[0])), env)
    obs = env.reset()
    print("Running best model...")
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    env.close()

    # See info on best model
    # model_info = pickle.load(open(("trpo_cartpole" + str(config_num) + ".pkl"), "rb"))

       # print("\n model info: {} \n".format(model_info))


