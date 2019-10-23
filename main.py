
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
from stable_baselines import TRPO, A2C, ACER, ACKTR, DDPG, DQN, PPO2
from stable_baselines.bench import Monitor
from stable_baselines.ddpg.policies import MlpPolicy as MlpPolicy2 #use this for DDPG only
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

best_mean_reward, n_steps, log_dir = -np.inf, 0, "logs/" #VERY UGLY GLOBAL VARS
os.makedirs(log_dir, exist_ok=True)
#TODO:bayesian error correction function doesnt work for PPO
#TODO: PROBLEM - VecEnvs, Stable Baselines, doesnt work for Hyperband and BOHB
#TODO: Now just printing to terminal 

def main():
    method = "bohb"
    algorithm = "ppo"
    num_configs = 3 #specify number of configurations to sample
    total_timesteps = 10000 #specify total number of steps to run algorithm
    min_budget = 0.25
    max_budget = 1
    eta = 2

    try:
        space = get_space(method, algorithm)
    except NameError:
        print('either optimizer with name %s or algorithm with name %s is not implemented' % (method, algorithm))    

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
        model = TRPO(MlpPolicy, env, verbose=1,
                timesteps_per_batch=int(config['timesteps_per_batch']),
                vf_stepsize=float(config['vf_stepsize']),
                max_kl=float(config['max_kl']),
                gamma=float(config['gamma']),
                lam=float(config['lam'])
                )
        model.learn(total_timesteps,seed=seed,callback=callback)
        plot_results(log_dir, count, title="Learning curve for Cartpole TRPO ")
       
    if algorithm == "ppo": #vec env, weird error nminibatches, maybe bad search space
        model = PPO2(MlpPolicy, env, verbose=1, 
                gamma=float(config['gamma']),
                n_steps=int(config['n_steps']),
                ent_coef=float(config['ent_coef']),
                learning_rate=float(config['learning_rate']),
                vf_coef=float(config['vf_coef']),
                max_grad_norm=float(config['max_grad_norm']),
                lam=float(config['lam']),
                nminibatches=int(config['nminibatches']),
                cliprange=float(config['cliprange']),
                cliprange_vf=float(config['cliprange_vf'])
                )
        model.learn(total_timesteps,seed=seed,callback=callback)
        plot_results(log_dir, count, title="Learning curve for Cartpole PPO ")
     
    if algorithm == "ddpg": #continuous env
        noisetypes = [None, NormalActionNoise, OrnsteinUhlenbeckActionNoise]
        action_noise = U.get_action_noise(env, config, noisetypes)

        model = DDPG(MlpPolicy2, env, verbose=1,
                gamma=float(config['gamma']),
                nb_train_steps=int(config['nb_train_steps']), #there must be a better way to make this an int
                nb_rollout_steps=int(config['nb_rollout_steps']), 
                param_noise=None,
                param_noise_adaption_interval=int(config['param_noise_adaption_interval']),
                tau=float(config['tau']),
                normalize_returns=bool(config['normalize_returns']),
                enable_popart=bool(config['enable_popart']), #if true then normalize_returns must be sent to true
                normalize_observations=bool(config['normalize_observations']),
                batch_size=int(config['batch_size']),
                observation_range=tuple(config['observation_range']),
                return_range=tuple(config['return_range']),
                critic_l2_reg=float(config['critic_l2_reg']),
                actor_lr=float(config['actor_lr']),
                critic_lr=float(config['critic_lr']),
                clip_norm=float(config['clip_norm']),
                reward_scale=float(config['reward_scale']),
                buffer_size=int(config['buffer_size']),
                action_noise=action_noise
                )
        model.learn(total_timesteps,seed=seed,callback=callback)
        plot_results(log_dir, count, title="Learning curve for Cartpole DDPG ")
    """
    if algorithm == "a2c":
        model = A2C(MlpPolicy, env,verbose=1,
                gamma=config['gamma'],
                n_steps=config['n_steps'],
                vf_coef=config['vf_coef'],
                ent_coef=config['ent_coef'],
                max_grad_norm=config['max_grad_norm'],
                learning_rate=config['learning_rate'],
                alpha=config['alpha'],
                epsilon=config['epsilon'],
                lr_schedule=config['lr_schedule']
                )
        model.learn(total_timesteps,seed=seed,callback=callback)
        plot_results(log_dir, count, title="Learning curve for Cartpole A2C")

    if algorithm == "acer":
        model = ACER(MlpPolicy, env, verbose=1,
                gamma=config['gamma'],
                n_steps=config['n_steps'],
                num_procs=config['num_procs'],
                q_coef=config['q_coef'],
                ent_coef=config['ent_coef'],
                max_grad_norm=config['max_grad_norm'],
                learning_rate=config['learning_rate'],
                lr_schedule=config['lr_schedule'],
                rprop_alpha=config['rprop_alpha'],
                rprop_epsilon=config['rprop_epsilon'],
                buffer_size=config['buffer_size'],
                replay_ratio=config['replay_ratio'],
                replay_start=config['replay_start'],
                correction_term=config['correction_term'],
                alpha=config['alpha'],
                delta=config['delta']
                )
        model.learn(total_timesteps,seed=seed,callback=callback)
        plot_results(log_dir, count, title="Learning curve for Cartpole ACER")
    
    if algorithm == "acktr":
        model = ACKTR(MlpPolicy, env, verbose=1,
                gamma=config['gamma'],
                nprocs=config['nprocs'],
                n_steps=config['n_steps'],
                ent_coef=config['ent_coef'],
                vf_coef=config['vf_coef'],
                vf_fisher_coef=config['vf_fisher_coef'],
                learning_rate=config['vf_learning_rate'],
                max_grad_norm=config['max_grad_norm'],
                kfac_clip=config['kfac_clip'],
                lr_schedule=config['lr_schedule'],
                )

    if algorithm == "dqn":
        model = DQN(MlpPolicy, env, verbose=1,
                gamma=config['gamma'],
                learning_rate=config['learning_rate'],
                buffer_size=config['buffer_size'],
                exploration_fraction=config['exploration_fraction'],
                exploration_final_eps=config['exploration_final_eps'],
                train_freq=config['train_freq'],
                batch_size=config['batch_size'],
                double_q=config['double_q'], 
                learning_starts=config['learning_starts'],
                target_network_update_freq=config['target_network_update_freq'],
                prioritized_replay=config['prioritized_replay'],
                prioritized_replay_alpha=config['prioritized_replay_alpha'],
                prioritized_replay_beta0=config['prioritized_replay_beta0'],
                prioritized_replay_beta_iters=config['prioritized_replay_beta_iters'],
                prioritized_replay_eps=config['prioritized_replay_eps'],
                param_noise=config['param_noise']
                )
    """
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
    #y = moving_average(y, window=10) #suddenly doesn't work?
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
    plt.title(title) # "Smoothed" if moving average is used

def set_seed(): #TODO: set_seed(config_num) to enable generate, log of specific
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


def get_env(log_dir, algorithm): #Make an overwrite of this as user of package
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
    env = gym.make("MountainCarContinuous-v0") #initialize environment
    env = Monitor(env, log_dir, allow_early_resets=True) 
    env = DummyVecEnv([lambda: env]) #Stable baselines
    return env

def get_space(method, algorithm):
    """
        Defines the search space to sample from for each hyperparameter for the 
        hyperparameter optimization. Define all parameters to tune in the given
        model here. 
        
        Returns
        --------
            dict-like expression graph consisting of nested function expressions 
            for all hyperparameters to optimize.
    """   
    #if method == "bayesian" or method == "random":
    if algorithm == "trpo":
        space = {
            'timesteps_per_batch': hp.choice('timesteps_per_batch', [512, 1024, 2048, 4096, 8192]),
            'vf_stepsize': hp.loguniform('vf_stepsize', -5, -2),
            'max_kl' : hp.loguniform('max_kl', -2.5, -0.5),
            'gamma': hp.uniform('gamma', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))), #4: Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
            'lam': hp.uniform('lam', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))) #4: Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
            }
    
    if algorithm == "ppo":
        space = {
            'gamma': hp.uniform('gamma', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))),
            'n_steps': hp.choice('n_steps', [512, 1024, 2048, 4096, 8192]),
            'ent_coef': hp.loguniform('ent_coef', -10, -2),
            'learning_rate': hp.uniform('learning_rate', 5*10**(-6), 0.003),
            'vf_coef': hp.uniform('vf_coef', 0.5, 1),
            'max_grad_norm': hp.choice('max_grad_norm', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            'lam': hp.uniform('lam', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))),
            'nminibatches': hp.choice('nminibatches', [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),
            'cliprange': hp.choice('cliprange', [0.1, 0.2, 0.3]),
            'cliprange_vf': hp.choice('cliprange_vf', [0.1, 0.2, 0.3])
            }
    
    if algorithm == "ddpg":
        space = {
            'gamma': hp.uniform('gamma', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))),
            'batch_size': hp.choice('batch_size', [512, 1024, 2048, 4096, 8192]),
            'nb_train_steps': hp.quniform('nb_train_steps', 10, 100, 1), 
            'nb_rollout_steps': hp.quniform('nb_rollout_steps', 10, 100, 1),
            #'param_noise': hp.choice('param_noise', [None, AdaptiveParamNoiseSpec]), #there seems to be an error in AdaptiveParamNoiseSpec implementation
            'action_noise': hp.choice('action_noise', [None, NormalActionNoise, OrnsteinUhlenbeckActionNoise]),
            'param_noise_adaption_interval': hp.choice('param_noise_adaption_interval', [100, 200, 300, 400, 500, 600, 700, 800, 900]),
            'tau': hp.loguniform('tau', -5, -2),
            'normalize_returns': hp.choice('normalize_returns', [True, False]),
            'enable_popart': hp.choice('enable_popart', [True, False]), #dependent on above var, how to fix?
            'normalize_observations': hp.choice('normalize_observations', [True, False]),
            'observation_range': hp.choice('observation_range',[(-2,2),(-3,3),(-4,4),(-5,5),(-6,6),(-7,7),(-8,8),(-9,9)]),
            'return_range': hp.choice('return_range', [(-np.inf, np.inf),(-500, 500),(-400, 400),(-300, 300),(-200, 200)]),
            'critic_l2_reg': hp.loguniform('critic_l2_reg',-8, -1),
            'actor_lr': hp.loguniform('actor_lr', -5, -2),
            'critic_lr': hp.loguniform('critic_lr', -5, -2),
            'clip_norm': hp.uniform('clip_norm', 1, 5), #absolutely no documentation on which values to set this one to.
            'reward_scale': hp.uniform('reward_scale', 0.2, 2),
            'buffer_size': hp.choice('buffer_size', [5000, 10000, 15000, 20000, 25000])
            }
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
        obs, reward, done, _ = env.step(action)
        reward_sum += int(reward) # Static cast to int needed for vec env
    return reward_sum

if __name__ == "__main__": 
    main()
    