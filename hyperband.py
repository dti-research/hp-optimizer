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

from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving
from hpbandster.optimizers.config_generators import RandomSampling

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.read_and_write import json

from builtins import range
import hyperopt.pyll.stochastic
from hyperopt.base import miscs_to_idxs_vals
from hyperopt import fmin, rand, hp, tpe, Trials, trials_from_docs, STATUS_OK

"""
class HyperBand(Master):
	def __init__(self, configspace = None,
					eta=3, min_budget=0.01, max_budget=1,
					**kwargs ):
		
                #Hyperband implements hyperparameter optimization by sampling
                #candidates at random and "trying" them first, running them for
                #a specific budget. The approach is iterative, promising
                #candidates are run for a longer time, increasing the fidelity
                #for their performance. While this is a very efficient racing
                #approach, random sampling makes no use of the knowledge gained
                #about the candidates during optimization.

		#Parameters
		#----------
		#configspace: ConfigSpace object
		#	valid representation of the search space
		#eta : float
		#	In each iteration, a complete run of sequential halving is executed. In it,
		#	after evaluating each configuration on the same subset size, only a fraction of
		#	1/eta of them 'advances' to the next round.
		#	Must be greater or equal to 2.
		#min_budget : float
		#	The smallest budget to consider. Needs to be positive!
		#max_budget : float
		#	the largest budget to consider. Needs to be larger than min_budget!
		#	The budgets will be geometrically distributed $\sim \eta^k$ for
		#	$k\in [0, 1, ... , num_subsets - 1]$.
		


		# TODO: Proper check for ConfigSpace object!
		if configspace is None:
			raise ValueError("You have to provide a valid ConfigSpace object")

		super().__init__(config_generator=RandomSampling(configspace), **kwargs)

		# Hyperband related stuff
		self.eta = eta
		self.min_budget = min_budget
		self.max_budget = max_budget

		# precompute some HB stuff
		self.max_SH_iter = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
		self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter-1, 0, self.max_SH_iter))

		self.config.update({
						'eta'        : eta,
						'min_budget' : min_budget,
						'max_budget' : max_budget,
						'budgets'    : self.budgets,
						'max_SH_iter': self.max_SH_iter,
					})



	def get_next_iteration(self, iteration, iteration_kwargs={}):
		
		#Hyperband uses SuccessiveHalving for each iteration.
		#See Li et al. (2016) for reference.
		
		#Parameters
		#----------
		#	iteration: int
		#		the index of the iteration to be instantiated

		#Returns
		#-------
		#	SuccessiveHalving: the SuccessiveHalving iteration with the
		#		corresponding number of configurations
		
		
		# number of 'SH rungs'
		s = self.max_SH_iter - 1 - (iteration%self.max_SH_iter)
		# number of configurations in that bracket
		n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
		ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]

		return(SuccessiveHalving(HPB_iter=iteration, num_configs=ns, budgets=self.budgets[(-s-1):], config_sampler=self.config_generator.get_config, **iteration_kwargs))
"""

# Define the search space of each hyperparameter for the hyperparameter optimization
def get_space():
    # First, define the hyperparameters and add them to the configuration space
    space = CS.ConfigurationSpace()
    timesteps_per_batch=CSH.CategoricalHyperparameter('timesteps_per_batch', [512, 1024, 2048, 4096, 8192])
    vf_stepsize=CSH.UniformFloatHyperparameter('vf_stepsize', lower=2**-5, upper=2**-2, log=True)
    max_kl=CSH.UniformFloatHyperparameter('max_kl', lower=2**-2.5, upper=2**-0.5, log=True)
    gamma=CSH.UniformFloatHyperparameter('gamma', lower=(1-(1/((10**(-1))*4))), upper=(1-(1/((10**(1.5))*4))))
    lam=CSH.UniformFloatHyperparameter('lam', lower=(1-(1/((10**(-1))*4))), upper=(1-(1/((10**(1.5))*4))))

    space.add_hyperparameters([timesteps_per_batch, vf_stepsize, max_kl, gamma, lam])

    # To end this example, we store the defined configuration space to a json file
    with open('configspace.json', 'w') as fh:
        fh.write(json.write(space))

    return space
"""
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
"""

if __name__ == "__main__":
    space = get_space()
"""
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
    
    print("\ntrials info: {}\n".format(bayes_trials.trials)) #annoyingly enough prints out index for timesteps_per_batch
    print("\nbest model: {}".format(best))
"""



