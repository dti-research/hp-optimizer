import tensorflow as tf
import random
import gym
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hyperopt import hp
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO
from stable_baselines.common.vec_env import DummyVecEnv

def get_model(method, algorithm, env, config):
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
        
        TODO: Add more algorithms
    """ 
    if algorithm == "trpo":
        model = TRPO(MlpPolicy, env, 
             verbose=1,
             timesteps_per_batch=config['timesteps_per_batch'],
             vf_stepsize=config['vf_stepsize'],
             max_kl=config['max_kl'],
             gamma=config['gamma'],
             lam=config['lam']
            )

    return model

def set_seed(): #set_seed(config_num) to be able to generate and log specific seeds
    seed = (2**10)-534
    np.random.seed(seed)
#    seed = np.random.randint(1,2**31-1)
    tf.set_random_seed(seed)
    random.seed(seed)
    rand_state = np.random.RandomState(seed).get_state()
    np.random.set_state(rand_state)


def get_env(name): #Make an overwrite of this as user of library?
    """
        Initializes the environment to be evaluated. Compatible with openAI gym envs.
        
        
        Parameters:
        -----------
            name: name of environment. Currently needs to be name of env defined in gym.
        
        Returns:
        --------
            env: initialized environment to train agent in 
    """
    env = gym.make(name) #initialize environment
    env = DummyVecEnv([lambda: env])
    return env

def get_space(method):
    """
        Defines the search space to sample from for each hyperparameter for the hyperparameter 
        optimization. Define all parameters to tune in the given model here. 
        
        Returns:
        --------
            dict-like expression graph consisting of nested function expressions for all 
            hyperparameters to optimize.
    """   
    if method == "bayesian" or method == "random":
        space = {
            'timesteps_per_batch': hp.choice('timesteps_per_batch', [512, 1024, 2048, 4096, 8192]),
            'vf_stepsize': hp.loguniform('vf_stepsize', -5, -2),
            'max_kl' : hp.loguniform('max_kl', -2.5, -0.5),
            'gamma': hp.uniform('gamma', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))), #4: Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
            'lam': hp.uniform('lam', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))) #4: Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
            }
    
    elif method == "hyperband" or method == "bohb":
        space = CS.ConfigurationSpace()
        timesteps_per_batch=CSH.CategoricalHyperparameter('timesteps_per_batch', [512, 1024, 2048, 4096, 8192])
        vf_stepsize=CSH.UniformFloatHyperparameter('vf_stepsize', lower=2**-5, upper=2**-2, log=True)
        max_kl=CSH.UniformFloatHyperparameter('max_kl', lower=2**-2.5, upper=2**-0.5, log=True)
        gamma=CSH.UniformFloatHyperparameter('gamma', lower=(1-(1/((10**(-1))*4))), upper=(1-(1/((10**(1.5))*4))))
        lam=CSH.UniformFloatHyperparameter('lam', lower=(1-(1/((10**(-1))*4))), upper=(1-(1/((10**(1.5))*4))))

        space.add_hyperparameters([timesteps_per_batch, vf_stepsize, max_kl, gamma, lam])
    return space


def evaluate(env, model):
    """
        Computes evaluation metric. In this case, the metric chosen is the mean 
        of the sum of episodic rewards obtained during training
        
        
        Parameters:
        -----------
            env: environment to evaluate model in
            model: current model in a given state during training 
        
        Returns:
        --------
            mean of sum of episodic rewards for a full run of a given 
            configuration
    """ 
    reward_sum = 0
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        reward_sum += reward
    return reward_sum


