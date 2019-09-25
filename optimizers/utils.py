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