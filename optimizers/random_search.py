import hyperopt.pyll.stochastic
from optimizers.utils import set_seed, get_model, evaluate
import numpy as np

def run_random_search(env, method, num_configs, algorithm, space, total_timesteps):
    set_seed()
    samples = sample(space, num_configs) 
    config_evals = [] #store result metric after training of each config
    
    for config_num in range(num_configs):
        while True:
            try:
                config = samples[config_num]
                model = get_model(method, algorithm, env, config)
                model.learn(total_timesteps)
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print("Run failed, trying again...")
                samples[config_num] = hyperopt.pyll.stochastic.sample(space)

        model.save("trpo_cartpole_" + str(config_num)) #name_option

        result = evaluate(env, model)
        config_evals.append([config_num, result])

    best = max(config_evals, key=lambda x: x[1])
    return int(best[1])

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



