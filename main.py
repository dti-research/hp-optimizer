
from optimizers.utils import get_env, get_space, set_seed
from optimizers.random_search import run_random_search
from optimizers.bayesian_optimization import run_bayesian_opt
from optimizers.hyperband import run_hyperband_opt
from optimizers.bohb import run_bohb_opt


def run_optimizer(env, method, num_configs, algorithm, space, total_timesteps, min_budget, max_budget, eta):
    if method == "random":
        best = run_random_search(env, method, num_configs, algorithm, space, 
                                 total_timesteps
                                )

    elif method == "bayesian":
        best = run_bayesian_opt(env, method, num_configs, algorithm, space, 
                                total_timesteps
                               )
    
    elif method == "hyperband": # Should it be possible to choose budget here?
        best = run_hyperband_opt(env, method, num_configs, algorithm, space, 
                                 total_timesteps,
                                 min_budget,
                                 max_budget,
                                 eta
                                )

    elif method == "bohb": 
        best = run_bohb_opt(env, method, num_configs, algorithm, space,
                            total_timesteps,
                             min_budget,
                             max_budget,
                             eta
                           )
        
    return best

if __name__ == "__main__":
    method = "bayesian"
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

    env_name = "CartPole-v1" 
    env = get_env(env_name)

    best = run_optimizer(env, method, num_configs, algorithm, space, total_timesteps, min_budget, max_budget, eta)
    print("best: {} \n".format(best)) #configuration number best[0]
    