import numpy as np

def run_optimizer(method, num_configs, algorithm, space, total_timesteps, min_budget, max_budget, eta, log_dir):
    if method == "random":
        from optimizers.random_search import run_random_search
        best = run_random_search(method, num_configs, algorithm, space, 
                                 total_timesteps, log_dir
                                )

    elif method == "bayesian":
        from optimizers.bayesian_optimization import run_bayesian_opt
        best = run_bayesian_opt(method, num_configs, algorithm, space, 
                                total_timesteps, log_dir
                               )
    
    elif method == "hyperband": # Should it be possible to choose budget here?
        from optimizers.hyperband import run_hyperband_opt
        best = run_hyperband_opt(method, num_configs, algorithm, space, 
                                 total_timesteps,
                                 min_budget,
                                 max_budget,
                                 eta, log_dir
                                )

    elif method == "bohb": 
        from optimizers.bohb import run_bohb_opt
        best = run_bohb_opt(method, num_configs, algorithm, space,
                            total_timesteps,
                             min_budget,
                             max_budget,
                             eta, log_dir
                           )
        
    return best

def get_action_noise(env, config, ntypes):
    n_actions= env.action_space.shape[-1]
    action_noise = None
    if type(config['action_noise']) is str:
        stringlist = ["None", "NormalActionNoise", "OrnsteinUhlenbeckActionNoise"]
        for ntype in stringlist:
            if config['action_noise'] == str(ntype): #of some reason I must cast
                action_noise = ntypes[stringlist.index(ntype)]

    if action_noise != None:
        action_noise = action_noise(mean=np.zeros(n_actions),sigma=float(0.5)*np.ones(n_actions))

    return action_noise