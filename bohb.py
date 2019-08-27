import tensorflow as tf
import numpy as np
import random
import gym
import math
import time
import os
import socket
import logging
import pickle


import traceback
import threading
import Pyro4

import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as opt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines import TRPO

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.read_and_write import json

#1. get space
#2. use random number generator to create random state
#3. Define cartpole env with max_episodes, avg_n_episodes and max_budget 

#bugdet in percentage of total run

def run_model(config, budget):
    # Fixed random state
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)
    seed = np.random.randint(1, 2**31 - 1)
    tf.set_random_seed(seed)
    random.seed(seed)


    env = gym.make('CartPole-v1')
    env = DummyVecEnv([lambda: env])

    # Get all the current hyperparameter values
    config['timesteps_per_batch'] = config['timesteps_per_batch']
    for parameter_name in ['vf_stepsize', 'max_kl', 'gamma', 'lam']:
        config[parameter_name] = float(config[parameter_name])

    model = TRPO(MlpPolicy, env, 
                 verbose=1,
                 timesteps_per_batch=config['timesteps_per_batch'],
                 vf_stepsize=config['vf_stepsize'],
                 max_kl=config['max_kl'],
                 gamma=config['gamma'],
                 lam=config['lam']
                )

    total_timesteps = 10000 #Might be transformed to iterations, in order to follow original author standards
    budget_steps = int(total_timesteps*budget)
    model.learn(total_timesteps=budget_steps)
        
    result = evaluate(env, model)
    return result

def run_experiment(space, num_iterations, nic_name, run_id, work_dir, worker, min_budget, max_budget, eta, dest_dir, store_all_runs=False):
    # make sure the working and dest directory exist
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(dest_dir, exist_ok=True)

    # setup a nameserver. Every run needs a nameserver. Here it will be started for the local machine with a random port
    NS = hpns.NameServer(run_id=run_id, host='localhost', port=0, working_directory=work_dir, nic_name=nic_name)
    ns_host, ns_port = NS.start() 

    # start worker in the background
    worker.load_nameserver_credentials(work_dir)
    worker.run(background=True)
    print("host: {}".format(ns_host))
    print("port: {}".format(ns_port))
    print("socket {}".format(socket))

    BOHB = opt(configspace = space, run_id=run_id, eta=eta, min_budget=min_budget, max_budget=max_budget,
             nameserver = ns_host,
             working_directory = dest_dir,
             host = ns_host,
             nameserver_port = ns_port,
             ping_interval=3600,
             result_logger=None
            )

    result = BOHB.run(n_iterations=num_iterations) 

    # shutdown the worker and the dispatcher
    BOHB.shutdown(shutdown_workers=True)
    NS.shutdown()

    # NEVER USED the number of iterations for the blackbox optimizers must be increased so they have comparable total budgets
#    bb_iterations = int(num_iterations * (1+(np.log(max_budget) - np.log(min_budget))/np.log(eta)))

   # with open(os.path.join(dest_dir, '{}_run_{}.pkl'.format(method, run_id)), 'wb') as fh:
   #     pickle.dump(extract_results_to_pickle(result), fh)
    
    #if store_all_runs:
    #    with open(os.path.join(dest_dir, '{}_full_run_{}.pkl'.format(method, run_id)), 'wb') as fh:
    #        pickle.dump(extract_results_to_pickle(result), fh)
    
    # in case one wants to inspect the complete run
    return(result)

# Define the search space of each hyperparameter for the hyperparameter optimization
def get_space(): #CartPoleReduced
    # First, define the hyperparameters and add them to the configuration space
    space = CS.ConfigurationSpace()
    timesteps_per_batch=CSH.CategoricalHyperparameter('timesteps_per_batch', [512, 1024, 2048, 4096, 8192])
    vf_stepsize=CSH.UniformFloatHyperparameter('vf_stepsize', lower=2**-5, upper=2**-2, log=True)
    max_kl=CSH.UniformFloatHyperparameter('max_kl', lower=2**-2.5, upper=2**-0.5, log=True)
    gamma=CSH.UniformFloatHyperparameter('gamma', lower=(1-(1/((10**(-1))*4))), upper=(1-(1/((10**(1.5))*4))))
    lam=CSH.UniformFloatHyperparameter('lam', lower=(1-(1/((10**(-1))*4))), upper=(1-(1/((10**(1.5))*4))))

    space.add_hyperparameters([timesteps_per_batch, vf_stepsize, max_kl, gamma, lam])

    # Store the defined configuration space to a json file
    #with open('configspace.json', 'w') as fh:
    #    fh.write(json.write(space))

    return space

def evaluate(env, model):
    #Return mean fitness (sum of episodic rewards) for the given model
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


class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, config, budget, **kwargs):
        """
        What this function does

        Args:
            config: sampled configuration by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        # Get all the current hyperparameter values
        config['timesteps_per_batch'] = config['timesteps_per_batch']
        for parameter_name in ['vf_stepsize', 'max_kl', 'gamma', 'lam']:
            config[parameter_name] = float(config[parameter_name])

        result = run_model(config, budget)

        # Transform to loss in order to minimize
        loss = (-result).item()

        return({
                    'loss': loss,  # this is the a mandatory field to run hyperband
                    'info': {'loss': loss, 'config': config} # can be used for any user-defined information - also mandatory
                })

    @staticmethod
    # Define the search space of each hyperparameter for the hyperparameter optimization
    def get_space(): #Why is this function here, when we already received configs?
        # First, define the hyperparameters and add them to the configuration space
        space = CS.ConfigurationSpace()
        timesteps_per_batch=CSH.CategoricalHyperparameter('timesteps_per_batch', [512, 1024, 2048, 4096, 8192])
        vf_stepsize=CSH.UniformFloatHyperparameter('vf_stepsize', lower=2**-5, upper=2**-2, log=True)
        max_kl=CSH.UniformFloatHyperparameter('max_kl', lower=2**-2.5, upper=2**-0.5, log=True)
        gamma=CSH.UniformFloatHyperparameter('gamma', lower=(1-(1/((10**(-1))*4))), upper=(1-(1/((10**(1.5))*4))))
        lam=CSH.UniformFloatHyperparameter('lam', lower=(1-(1/((10**(-1))*4))), upper=(1-(1/((10**(1.5))*4))))

        space.add_hyperparameters([timesteps_per_batch, vf_stepsize, max_kl, gamma, lam])

        # Store the defined configuration space to a json file
        #with open('configspace.json', 'w') as fh:
        #    fh.write(json.write(space))

        return space



if __name__ == "__main__":
    space = get_space()
    dest_dir = "results/"
    run_id = 0 # Every run has to have a unique (at runtime) id. for concurrent runs, i.e. when multiple. Here we pick '0'
    work_dir="tmp/"
    method ="hyperband"
    min_budget = 0.1 #minimum number of independent runs to estimate mean loss
    max_budget = 1 #maximum number of independent runs to estimate mean loss
    eta = 3
    num_iterations = 16 #number of Hyperband iterations performed.
    nic_name='lo'

    worker = MyWorker(run_id=run_id)

    # run experiment
    result = run_experiment(space, num_iterations, nic_name, run_id, work_dir, worker, min_budget, max_budget, eta, dest_dir, store_all_runs=False)

    print("Result: {} ".format(result))



