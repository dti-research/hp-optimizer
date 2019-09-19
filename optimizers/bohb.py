
import numpy as np
import random
import gym
import math
import os
import pickle
import hpbandster.core.nameserver as hpns
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.read_and_write import json
from optimizers.utils import set_seed, get_env, get_model

from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as opt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO

def run_bohb_opt(env, method, num_configs, algorithm, space, total_timesteps, min_budget, max_budget, eta):
    dest_dir = "results/"
    run_id = 0 # Every run has to have a unique (at runtime) id. for concurrent runs, i.e. when multiple. Here we pick '0'
    work_dir="tmp/"
    num_iterations = num_configs #number of Hyperband iterations performed.
    nic_name='lo'

    worker = MyWorker(env, algorithm, method, total_timesteps, run_id=run_id)

    # run experiment
    result, loss = run_experiment(space, num_iterations, nic_name, run_id, work_dir, worker, min_budget, max_budget, eta, dest_dir, store_all_runs=False)

    id2config = result.get_id2config_mapping()
    incumbent = result.get_incumbent_id()
    all_runs = result.get_all_runs()

    #print('Best found configuration:', id2config[incumbent]['config'])
    #print('Best loss: ', loss)
    #print('Best found configuration:', id2config[incumbent]['config_info'])
    #print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    #print('A total of %i runs where executed.' % len(result.get_all_runs()))
    #print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/max_budget))
    #print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/max_budget))
    #print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

    best = -loss
    return best


def extract_results_to_pickle(results_object):
	"""
		Returns the best configurations over time, but also returns the cummulative budget
		
		
		Parameters:
		-----------
			all_budgets: bool
				If set to true all runs (even those not with the largest budget) can be the incumbent.
				Otherwise, only full budget runs are considered
		
		Returns:
		--------
			dict:
				dictionary with all the config IDs, the times the runs
				finished, their respective budgets, and corresponding losses
	"""
	all_runs = results_object.get_all_runs(only_largest_budget = False)
	all_runs.sort(key=lambda r: r.time_stamps['finished'])
	
	return_dict = { 'config_ids' : [],
					'times_finished': [],
					'budgets'    : [],
					'losses'     : [],
					'test_losses': [],
					'cummulative_budget' : [],
					'cummulative_cost' : []
	}

	cummulative_budget = 0
	cummulative_cost = 0
	current_incumbent = float('inf')
	incumbent_budget = -float('inf')
	
	for r in all_runs:
		
		cummulative_budget += r.budget
		try: cummulative_cost += r.info['cost']
		except: pass
		
		if r.loss is None: continue
		
		if (r.budget >= incumbent_budget and r.loss < current_incumbent):
			current_incumbent = r.loss
			incumbent_budget  = r.budget
			
			return_dict['config_ids'].append(r.config_id)
			return_dict['times_finished'].append(r.time_stamps['finished'])
			return_dict['budgets'].append(r.budget)
			return_dict['losses'].append(r.loss)
			return_dict['cummulative_budget'].append(cummulative_budget)
			return_dict['cummulative_cost'].append(cummulative_cost)
			try: return_dict['test_losses'].append(r.info['test_loss'])
			except: pass


	if current_incumbent != r.loss:
		r = all_runs[-1]
	
		return_dict['config_ids'].append(return_dict['config_ids'][-1])
		return_dict['times_finished'].append(r.time_stamps['finished'])
		return_dict['budgets'].append(return_dict['budgets'][-1])
		return_dict['losses'].append(return_dict['losses'][-1])
		return_dict['cummulative_budget'].append(cummulative_budget)
		return_dict['cummulative_cost'].append(cummulative_cost)
		try: return_dict['test_losses'].append(return_dict['test_losses'][-1])
		except: pass

	return_dict['configs'] = {}
	
	id2conf = results_object.get_id2config_mapping()
	
	
	for c in return_dict['config_ids']:
		return_dict['configs'][c] = id2conf[c]
	
	return_dict['HB_config'] = results_object.HB_config
	
	return (return_dict)

def run_model(config, budget, env, method, algorithm, total_timesteps):
    """
       Initializes the environment in which the model is evaluated, retrieves the values 
       for the current hyperparameter configuration, initializes and trains
       the given model. 


        Parameters:
        --------
            config: ConfigSpace object containing sampled values for a given hyperparameter configuration
            budget: how much of a full run is currently used to estimate mean loss
        
        Returns:
        --------
            A metric used to evaluate the performance of the current configuration. 
    """
    # Fixed random state
    set_seed()

    model = get_model(method, algorithm, env, config)
    budget_steps = int(total_timesteps*budget)  #I am not sure this is the right way to do it
    model.learn(total_timesteps=budget_steps)
        
    result = evaluate(env, model)
    return result

def run_experiment(space, num_iterations, nic_name, run_id, work_dir, worker, min_budget, max_budget, eta, dest_dir, store_all_runs=False):
    """
       Runs the optimization algorithm, and sets up a nameserver


        Parameters:
        --------
            space: hpbandster ConfigSpace object 
                containing the search space to sample from for each hyperparameter
                for hyperparameter optimization.
            num_iterations: int 
                number of iterations to run optimization algorithm
            nic_name: string
                name of the network interface used for communication. Note: default is only for 
                local execution on *nix!
            run_id: string
                A unique identifier of that Hyperband run. Use, for example, the cluster's JobID when 
                running multiple concurrent runs to separate them
            work_dir: string
                The top level working directory accessible to all compute nodes(shared filesystem).
            worker: modified hpbandster Worker object 
                Implements a user specified compute function, see MyWorker class.
            min_budget: float
                The smallest budget to consider. Needs to be positive!
            max_budget: float
                the largest budget to consider. Needs to be larger than min_budget!
                The budgets will be geometrically distributed $\sim \eta^k$ for
                $k\in [0, 1, ... , num_subsets - 1]$.
            eta: float
                In each iteration, a complete run of sequential halving is executed. In it,
                after evaluating each configuration on the same subset size, only a fraction of
                1/eta of them 'advances' to the next round. Must be greater or equal to 2.
            dest_dir: string
                the destination directory.
            store_all_runs: bool
                Specifies whether to store all the results of each run.
        
        Returns:
        --------
            A metric used to evaluate the performance of the current configuration. 
    """
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

    with open(os.path.join(dest_dir, '{}_run_{}.pkl'.format('bohb', run_id)), 'wb') as fh:
        pickle.dump(extract_results_to_pickle(result), fh)
    
    if store_all_runs:
        with open(os.path.join(dest_dir, '{}_full_run_{}.pkl'.format('bohb', run_id)), 'wb') as fh:
            pickle.dump(extract_results_to_pickle(result), fh)

    pickle_info = pickle.load(open((os.path.join(dest_dir, '{}_run_{}.pkl'.format("bohb", run_id))), 'rb'))
    best_loss = min(pickle_info['losses'])
    
    # in case one wants to inspect the complete run
    return(result, best_loss)

def get_space():
    """
        Defines the search space to sample from for each hyperparameter for the hyperparameter 
        optimization. Define all parameters to tune in the given model here. 
      
        Returns:
        --------
            ConfigSpace object containing the search space
    """
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

    def __init__(self, env, algorithm, method, total_timesteps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.algorithm = algorithm
        self.method = method
        self.total_timesteps = total_timesteps

    def compute(self, config, budget, **kwargs):
        """
        This function defines the objective function that we want to minimize

        Args:
            config: sampled configuration by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict) consisting of loss value and current configuration
        """

        result = run_model(config, budget, self.env, self.method, self.algorithm, self.total_timesteps)

        # Transform to loss in order to minimize
        loss = (-result).item()

        return({
                    'loss': loss,  # this is the a mandatory field to run bohb
                    'info': {'loss': loss, 'config': config} # can be used for any user-defined information - also mandatory
                })

