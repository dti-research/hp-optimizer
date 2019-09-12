import tensorflow as tf
import numpy as np
import random
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK

# Define the search space of each hyperparameter for the hyperparameter
# optimization


def get_space():
    """Defines the search space to sample from for each hyperparameter for
       the hyperparameter optimization. Define all parameters to tune in the
       given model here.

    Returns:
        dict -- expression graph consisting of nested function expressions for
                all hyperparameters to optimize.
    """

    space = {
        'timesteps_per_batch': hp.choice('timesteps_per_batch',
                                         [512, 1024, 2048, 4096, 8192]),
        'vf_stepsize': hp.loguniform('vf_stepsize', -5, -2),
        'max_kl': hp.loguniform('max_kl', -2.5, -0.5),
        # 4: Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
        'gamma': hp.uniform('gamma', (1 - (1 / ((10**(-1)) * 4))),
                            (1 - (1 / ((10**(1.5)) * 4)))),
        # 4: Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
        'lam': hp.uniform('lam', (1 - (1 / ((10**(-1)) * 4))),
                          (1 - (1 / ((10**(1.5)) * 4))))
    }
    return space


def objective(hyperparams):
    """
        Defines the objective function that we want to minimize.


        Parameters:
        --------
            hyperparams: dictionary containing sampled values for a given
            hyperparameter configuration

        Returns:
        --------
            dictionary containing specified information. In this case, the
                loss, the current hyperparameter
            values, the iteration number, and a status flag, signalling if the
                run was successful
    """
    # necessary with a global variable because of implementation from
    # hyperopt.
    global iteration
    iteration += 1

    result = run_model(hyperparams, iteration)
    loss = -result  # transform to loss in order to minimize

    return {'loss': loss,
            'hyperparams': hyperparams,
            'iteration': iteration,
            'status': STATUS_OK}


def run_model(hyperparams, iteration):
    """
       This is the most important function of this script. Initializes the
       environment in which the model is evaluated, retrieves the values for
       the current hyperparameter configuration, initializes and trains the
       given model.


        Parameters:
        --------
            hyperparams: dictionary containing sampled values for a given
            hyperparameter configuration iteration: the iteration of running
            Bayesian optimization, i.e. configuration number

        Returns:
        --------
            A metric used to evaluate the performance of the current
            configuration.
    """
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

    # Initialize model
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
    """
        Computes evaluation metric. In this case, the metric chosen is the
        mean of the sum of episodic rewards obtained during training


        Parameters:
        -----------
            env: environment to evaluate model in
            model: current model in a given state during training

        Returns:
        --------
            mean of sum of episodic rewards for a full run of a given
            configuration
    """
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
    max_evaluations = 3  # maximum number of Bayesian optimization evaluations
    space = get_space()

    # Keep track of results
    bayes_trials = Trials()

    # Optimization algorithm
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evaluations,
                trials=bayes_trials)

    # annoyingly enough prints out index for timesteps_per_batch
    print("\ntrials info: {}\n".format(bayes_trials.trials))
    print("\nbest model: {}".format(best))
