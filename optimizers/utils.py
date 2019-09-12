import gym
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hyperopt import hp

from stable_baselines.common.vec_env import DummyVecEnv

from optimizers.random_search import run_random_search
from optimizers.bayesian_optimization import run_bayesian_opt
from optimizers.hyperband import run_hyperband_opt
from optimizers.bohb import run_bohb_opt

def get_env(name):
    """
        Initializes the environment to be evaluated. Compatible with openAI gym envs.
        
        
        Parameters:
        -----------
            name: name of environment. Currently needs to be name of env defined in gym.
        
        Returns:
        --------
            env: initialized environment to train agent in 
    """
    env = gym.make('CartPole-v1') #initialize environment
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

def run_optimizer(env, method, num_configs, algorithm, space):
    if method == "random":
        best = run_random_search(env, num_configs, algorithm, space)

    elif method == "bayesian":
        best = run_bayesian_opt(env, num_configs, algorithm, space)
    
    elif method == "hyperband": # Should it be possible to choose budget here?
        best = run_hyperband_opt(env, num_configs,algorithm, space)

    elif method == "bohb": 
        best = run_bohb_opt(env, num_configs,algorithm, space)
        
    return best
