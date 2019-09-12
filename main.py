import numpy as np 
import random
import tensorflow as tf

from optimizers.utils import get_space, get_env, run_optimizer

if __name__ == "__main__":

    method = "bohb"
    algorithm = "trpo"
    num_configs = 3 #specify number of configurations to sample
    try:
        space = get_space(method)
    except NameError:
                print('optimizer with name %s not implemented - must be "random", "bayesian", "hyperband", or "bohb"'%method)    

    env_name = "CartPole-v1" 
    
    env = get_env(env_name)

    best = run_optimizer(env, method, num_configs, algorithm, space)
    print("best: {} \n".format(best)) #configuration number best[0]
    