
import os
import gym
import pickle
import random
import numpy as np
import tensorflow as tf
import ConfigSpace as CS
import optimizers.utils as U
import matplotlib.pyplot as plt
import ConfigSpace.hyperparameters as CSH

from hyperopt import hp
from stable_baselines import TRPO
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy

best_mean_reward, n_steps, log_dir = -np.inf, 0, "logs/" #VERY UGLY GLOBAL VARS
os.makedirs(log_dir, exist_ok=True)

def main():
    method = "bohb"
    algorithm = "trpo"
    num_configs = 3 #specify number of configurations to sample
    total_timesteps = 10000 #specify total number of steps to run algorithm
    min_budget = 0.25
    max_budget = 1
    eta = 2

    try:
        space = get_space(method)
    except NameError:
                print('optimizer with name %s not implemented - must be "random",bayesian", "hyperband", or "bohb"'%method)    

    best, config_evals, total_time_spent, seeds = U.run_optimizer(method, num_configs, algorithm, space, total_timesteps, min_budget, max_budget, eta, log_dir)
    plt.savefig(log_dir + 'multiple_plots.pdf')
    plt.show()
    print("best: {} \n".format(best)) 
    print("config_evals: {} \n".format(config_evals)) 
    print("time spent optimizing: {} \n".format(total_time_spent)) 
    print("seeds: {} \n".format(seeds)) 


def run_model(method, algorithm, env, config, total_timesteps, seed, count):
    """
        Initializes and trains a model with a specified algorithm with 
        hyperparameter values from a given configuration. A callback function is
        defined to facilitate plotting of learning curves. Note that the 
        callback function is taken directly from a Stable-Baselines example.
        
        Parameters:
        -----------
            method: TODO not necessary 
            algorithm: The algorithm specified in main-function TODO necessary?
            env: environment to train model on, specified in get_env function.
            config: Configuration sampled by optimizer from search space, 
            specified in get_space function. 
            total_timesteps: total number of steps to train model, specified in
            main.
            seed: random seed initialized in optimizer.
        
        Returns:
        --------
            model: trained model.
        
        TODO: Add more algorithms
    """ 
    best_mean_reward, n_steps = -np.inf, 0 

    def callback(_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        NOTE: Implemented as-is from Stable-baselines example
        """
        global best_mean_reward, n_steps, log_dir
        # Print stats every 1000 calls
        if (n_steps + 1) % 1000 == 0:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                print(x[-1], 'timesteps')
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model")
                    _locals['self'].save(log_dir + 'best_model.pkl')
        n_steps += 1
        # Returning False will stop training early
        return True

    if algorithm == "trpo":
        model = TRPO(MlpPolicy, env, 
                 verbose=1,
                 timesteps_per_batch=config['timesteps_per_batch'],
                 vf_stepsize=config['vf_stepsize'],
                 max_kl=config['max_kl'],
                 gamma=config['gamma'],
                 lam=config['lam']
                )
        model.learn(total_timesteps,seed=seed, callback=callback)
        plot_results(log_dir, count, title="Learning curve for Cartpole TRPO ")
    return model

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    NOTE: implemented as-is from Stable-baselines example
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, count, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    NOTE: implemented directly from Stable-Baselines example
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    data = {}
    data[str(count)] = [x,y]

    name = log_folder + "learning_curves.pickle"
    with open(name, 'ab') as handle:
        pickle.dump(data, handle)

    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + "Smoothed")

def set_seed(): #TODO: set_seed(config_num) to be able to generate and log specific seeds
    """
    This function enables for fixed random states, and is used to keep track of 
    randomization seeds used throughout experiments.

    Returns:
    --------
        seed: randomization seed used for fixed random state
    """
    seed = np.random.randint(1,2**31-1)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)
    rand_state = np.random.RandomState(seed).get_state()
    np.random.set_state(rand_state)
    return seed


def get_env(log_dir): #Make an overwrite of this as user of package
    """
        Initializes the environment to be evaluated. Right now this package is 
        only guaranteed to be compatible with openAI gym envs.
        
        
        Parameters:
        -----------
            log_dir: name of directory to place monitor-files.
        
        Returns:
        --------
            env: initialized environment to train agent in 
    """
    env = gym.make("CartPole-v1") #initialize environment
    env = Monitor(env, log_dir, allow_early_resets=True)
#    env = DummyVecEnv([lambda: env]) #Stable baselines 
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


def evaluate(env, model): #Make an overwrite of this as user of package
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

if __name__ == "__main__": 
    main()
    