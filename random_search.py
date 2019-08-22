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
from hyperopt import fmin, rand, hp, Trials, trials_from_docs, STATUS_OK

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

def sample(space, num_configs):
    timesteps_per_batch, vf_stepsize, max_kl, gamma, lam = ([] for i in range(5))
    for i in range(num_configs):
        sample = hyperopt.pyll.stochastic.sample(space)
        timesteps_per_batch.append(sample['timesteps_per_batch'])
        vf_stepsize.append(sample['vf_stepsize'])
        max_kl.append(sample['max_kl'])
        gamma.append(sample['gamma'])
        lam.append(sample['lam'])
    return timesteps_per_batch, vf_stepsize, max_kl, gamma, lam

def get_model(env, config_num=0):
    model = TRPO(MlpPolicy, env, 
              verbose=1,
              timesteps_per_batch=timesteps_per_batch[config_num],
              vf_stepsize=vf_stepsize[config_num],
              max_kl=max_kl[config_num],
              gamma=gamma[config_num],
              lam=lam[config_num]
            )
    return model

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

    num_configs = 3
    config_evals = []
    space = get_space()
    timesteps_per_batch, vf_stepsize, max_kl, gamma, lam = sample(space, num_configs)

    for config_num in range(num_configs):
        # Fixed random state
        rand_state = np.random.RandomState(1).get_state()
        np.random.set_state(rand_state)
        seed = np.random.randint(1, 2**31 - 1)
        tf.set_random_seed(seed)
        random.seed(seed)


        env = gym.make('CartPole-v1')
        env = DummyVecEnv([lambda: env])

        model = get_model(env, config_num)

        model.learn(total_timesteps=10000)
        model.save("trpo_cartpole_" + str(config_num))

        del model # remove to demonstrate saving and loading

        model = TRPO.load(("trpo_cartpole_" + str(config_num)), env)
        
        result = evaluate(env, model)
        config_evals.append([config_num, result])
        best = max(config_evals, key=lambda x: x[1])
    print("best: {} \n".format(best))

    #print("config: {}".format(best[0]))
    model = TRPO.load(("trpo_cartpole_" + str(config_num)), env)
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


