import tensorflow as tf
import numpy as np
import random
import gym
import math
import time
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO

from builtins import range
import hyperopt.pyll.stochastic
from hyperopt.base import miscs_to_idxs_vals
from hyperopt import fmin, rand, hp, Trials, trials_from_docs, STATUS_OK

#num_configs=10

# Define the search space of each hyperparameter for the hyperparameter optimization
space = {
        'timesteps_per_batch': hp.choice('timesteps_per_batch', [512, 1024, 2048, 4096, 8192]),
        'vf_stepsize': hp.loguniform('vf_stepsize', -5, -2),
        'max_kl' : hp.loguniform('max_kl', -2.5, -0.5),
        'gamma': hp.uniform('gamma', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))), #4:T. Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
        'lam': hp.uniform('lam', (1-(1/((10**(-1))*4))), (1-(1/((10**(1.5))*4)))) #4:T. Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
        }



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
              timesteps_per_batch=timesteps_per_batch[config_num-1],
              vf_stepsize=vf_stepsize[config_num-1],
              max_kl=max_kl[config_num-1],
              gamma=gamma[config_num-1],
              lam=lam[config_num-1]
            )
    return model


if __name__ == "__main__":

    num_configs = 7
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
        model.save("trpo_cartpole")

        del model # remove to demonstrate saving and loading

        model = TRPO.load("trpo_cartpole", env)

        obs = env.reset()

        for i in range(50):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #break
            env.render()
        env.close()


"""
    

    # Create CartPole environment
    env = gym.make('CartPole-v0')

    # Start environment processes
    env.

    # Create TRPO policy function
    sess = U.single_threaded_session()
    sess.__enter__()

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=32, num_hid_layers=2)

    # Train baselines TRPO
    learn(env, policy_fn, max_timesteps=150000, timesteps_per_batch=2048, max_kl=0.05, cg_iters=10, cg_damping=0.1, vf_iters=5, vf_stepsize=0.001, gamma=0.995, lam=0.995)

    env.close()

def policy_fn(name, ob_space, ac_space):
        return policy_fn_class(name           = name,
                               ob_space       = ob_space,
                               ac_space       = ac_space,
                               hid_size       = cfg['algorithm']['hyperparameters']['hid_size'],
                               num_hid_layers = cfg['algorithm']['hyperparameters']['num_hid_layers'])

    learn(env, policy_fn,
          max_timesteps       = cfg['algorithm']['hyperparameters']['max_timesteps'],
          timesteps_per_batch = cfg['algorithm']['hyperparameters']['timesteps_per_batch'],
          max_kl              = cfg['algorithm']['hyperparameters']['max_kl'],
          cg_iters            = cfg['algorithm']['hyperparameters']['cg_iters'],
          cg_damping          = cfg['algorithm']['hyperparameters']['cg_damping'],
          vf_iters            = cfg['algorithm']['hyperparameters']['vf_iters'],
          vf_stepsize         = cfg['algorithm']['hyperparameters']['vf_stepsize'],
          gamma               = cfg['algorithm']['hyperparameters']['gamma'],
          lam                 = cfg['algorithm']['hyperparameters']['lam']
          )




if __name__ == '__main__':
    main()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters", [4, 2])   # Parameters
        state = tf.placeholder("float", [None, 4])              # World state
        actions = tf.placeholder("float", [None, 2])            # Actions - move left, or right
        advantages = tf.placeholder("float", [None, 1])         # Ooh, advantages
        linear = tf.matmul(state, params)                       # Combine
        probabilities = tf.nn.softmax(linear)                   # Probabilities
        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss) # Learning rate 0.1, aim to minimize loss
        return probabilities, state, actions, advantages, optimizer

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float", [None, 4])       # World state
        newvals = tf.placeholder("float", [None, 1])     
        w1 = tf.get_variable("w1", [4, 2])              # Value gradient is *w1+b1, Relu, *w2+b2. 4, 2, 1.
        b1 = tf.get_variable("b1", [2])
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        w2 = tf.get_variable("w2", [2, 1])
        b2 = tf.get_variable("b2", [1])
        calculated = tf.matmul(h1,w2) + b2
        diffs = calculated - newvals                     # How different did we do from expected?
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess, render=True):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []

    for t in range(200):
        # Render
        if render:
            env.render()

        # Calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated,feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0, 1) < probs[0][0] else 1

        # Record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)

        # Take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        # Done?
        if done:
            break

    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # Calculate discounted Monte Carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in range(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.99
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated, feed_dict={vl_state: obs_vector})[0][0]

        # Advantage: how much better was this action than normal?
        advantages.append(future_reward - currentval)

        # Update the value function towards new return
        update_vals.append(future_reward)

    # Update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    # Update policy function
    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    # Done
    return totalreward

# Go
env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'cartpole', force=True)
policy_grad = policy_gradient()
value_grad = value_gradient()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Learn
results = []
for i in range(200):
    reward = run_episode(env, policy_grad, value_grad, sess)
    results.append(reward)
    if reward < 200:
        print("Fail at {}".format(i))

# Run 100
print("Running 100 more.")
t = 0
for _ in range(100):
    reward = run_episode(env, policy_grad, value_grad, sess)
    t += reward
    results.append(reward)
print("Got {}".format(t / 100))

# Plot
plt.plot(results)
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('Rewards over time')
plt.show()

# Show runs
print("Showing 30")
for _ in range(30):
    reward = run_episode(env, policy_grad, value_grad, sess, True)
"""
