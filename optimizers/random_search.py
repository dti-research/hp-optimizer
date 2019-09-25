import hyperopt.pyll.stochastic
from main import set_seed, run_model, evaluate, get_env
import numpy as np
import time

def run_random_search(method, num_configs, algorithm, space, total_timesteps,log_dir):
    seeds = []
    total_time_spent = 0
    start_time = time.time()
    samples = sample(space, num_configs) 
    config_evals = [] #store result metric after training of each config
    
    for config_num in range(num_configs):
        seed = set_seed()
        seeds.append(seed)
        env = get_env(log_dir)
        
         
        while True:
            try:
                config = samples[config_num]
                count = config_num
                model = run_model(method, algorithm, env, config,total_timesteps, seed, count)
                count += 1
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print("Run failed, trying again...")
                samples[config_num] = hyperopt.pyll.stochastic.sample(space)
        
#        config = samples[config_num]
#        model = run_model(method, algorithm, env, config,total_timesteps, seed, config_num)
            
        model.save(log_dir + "random_search_" + str(config_num)) 

        result = evaluate(env, model)
        config_evals.append([config_num, result, samples])

    best = max(config_evals, key=lambda x: x[1])
    end_time = time.time()
    total_time_spent = end_time - start_time
    #print("\n\ntotal optimizing time: {} s\n\n".format(total_time_spent))
    return int(best[1]), config_evals, total_time_spent, seeds

def sample(space, num_configs):
    """
        Returns a specified number of configurations drawn randomly from the search space
        
        
        Parameters:
        -----------
            space: dictionary containing function expressions for all hyperparameters to 
                   optimize
            num_configs: number of sample configurations to draw from the search space
        
        Returns:
        --------
            samples: nested lists containing all sampled hyperparameter configurations and the 
                     corresponding values of each hyperparameter.
    """   
    samples = []
    for i in range(num_configs):
        samples.append(hyperopt.pyll.stochastic.sample(space))
    return samples



