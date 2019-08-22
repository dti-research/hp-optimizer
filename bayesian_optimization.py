import tensorflow as tf
import numpy as np
import random
import gym
import math
import time
import pickle
import os
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines import TRPO

from builtins import range
import hyperopt.pyll.stochastic
from hyperopt.base import miscs_to_idxs_vals
from hyperopt import fmin, rand, hp, tpe, Trials, trials_from_docs, STATUS_OK

# Define the search space of each hyperparameter for the hyperparameter optimization
def get_space():
    space = {
            'timesteps_per_batch': hp.choice('timesteps_per_batch', [512, 1024, 2048, 4096, 8192]),
            'vf_stepsize': hp.loguniform('vf_stepsize', -5, -2),
            'max_kl' : hp.loguniform('max_kl', -2.5, -0.5),
            'gamma': hp.uniform('gamma', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))), #4:T. Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
            'lam': hp.uniform('lam', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))) #4:T. Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
            }
    return space

def objective(hyperparams):
    global iteration
    iteration += 1

    # Evaluate given model
    result = run_model(hyperparams, iteration)

    # Transform to loss in order to minimize
    loss = -result

    # Return loss, current hyperparameter configuration, iteration and key indicating if evaluation was succesful
    return {'loss': loss, 'hyperparams': hyperparams, 'iteration': iteration, 'status': STATUS_OK}

def run_model(hyperparams, iteration):
    # Fixed random state
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)
    seed = np.random.randint(1, 2**31 - 1)
    tf.set_random_seed(seed)
    random.seed(seed)


    env = gym.make('CartPole-v1')
    env = DummyVecEnv([lambda: env])

    # Get all the current hyperparameter values
    hyperparams['timesteps_per_batch'] = hyperparams['timesteps_per_batch']
    print("\ntimesteps_per_batch: {} \n".format(hyperparams['timesteps_per_batch']))
    for parameter_name in ['vf_stepsize', 'max_kl', 'gamma', 'lam']:
        hyperparams[parameter_name] = float(hyperparams[parameter_name])

    model = TRPO(MlpPolicy, env, 
                 verbose=1,
                 timesteps_per_batch=hyperparams['timesteps_per_batch'],
                 vf_stepsize=hyperparams['vf_stepsize'],
                 max_kl=hyperparams['max_kl'],
                 gamma=hyperparams['gamma'],
                 lam=hyperparams['lam']
                )

    model.learn(total_timesteps=10000)
    model.save("trpo_cartpole_" + str(iteration))
        
    result = evaluate(env, model)
    return result

def evaluate(env, model):
    """Return mean fitness (sum of episodic rewards) for the given model"""
    episode_rewards = []
    for _ in range(10):
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

    global iteration
    iteration = 0
    max_evaluations = 3
    space = get_space()

    # Keep track of results
    bayes_trials = Trials()

    # Optimization algorithm
    best = fmin(fn = objective,
		space = space, 
		algo = tpe.suggest, 
		max_evals = max_evaluations, 
		trials = bayes_trials)
    
    print(bayes_trials)
    print(best)



