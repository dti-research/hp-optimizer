from hyperopt import fmin, tpe, Trials, STATUS_OK
from main import set_seed, run_model, evaluate, get_env
import time

def run_bayesian_opt(method, num_configs, algorithm, space, total_timesteps, log_dir):
    total_time_spent = 0
    start_time = time.time()
    global iteration, seeds
    seeds = []
    iteration = 0

    # Keep track of results
    bayes_trials = Trials()

    def objective(hyperparams):
        """
            Defines the objective function that we want to minimize. 


            Parameters:
            --------
                hyperparams: dictionary containing sampled values for a given 
                hyperparameter configuration.
        
            Returns:
            --------
                dictionary containing specified information. In this case, the 
                loss, the current hyperparameter values, the iteration number, 
                and a status flag, signalling if the run was successful.
        """   
        global iteration, seeds #only necessary because of hyperopt implementation. 
        iteration += 1
        seed = set_seed()
        seeds.append(seed)

        result = run_current(hyperparams, iteration, seed)
        loss = -result #transform to loss in order to minimize

        return {'loss': loss, 
                'hyperparams': hyperparams, 
                'iteration': iteration, 
                'status': STATUS_OK,
                'seed': seed
               }

    def run_current(config, iteration, seed):
        """
            This is the most important function of this script. Initializes the 
            environment in which the model is evaluated, retrieves the values 
            for the current hyperparameter configuration, initializes and trains
            the given model. 


            Parameters:
            --------
                hyperparams: dictionary containing sampled values for a given 
                hyperparameter configuration.
                iteration: the iteration of running Bayesian optimization, i.e. 
                configuration number.
        
            Returns:
            --------
                A metric used to evaluate the performance of the current 
                configuration. 
        """ 
        env = get_env(log_dir, algorithm)
        fails = 0
        count = iteration

        while True:
            try:
                model = run_model(method, algorithm, env, config, total_timesteps, seed, count)
                model.save(log_dir + "bayesian_" + str(count))
                break
            except (KeyboardInterrupt, SystemExit):
                    raise
            except:
                print("Run failed, trying again...")
                try:
                    fails += 1
                    fmin(fn = objective,
		                 space = space, 
		                 algo = tpe.suggest, 
		                 max_evals = num_configs+fails, 
	                     trials = bayes_trials)
                    break
                except:
                    print("Run failed, trying again...")
                break
                
        
        result = evaluate(env, model)
        return result

    # Optimization algorithm
    fmin(fn = objective,
		space = space, 
		algo = tpe.suggest, 
		max_evals = num_configs, 
		trials = bayes_trials)
    
    #print("\ntrials info: {}\n".format(bayes_trials.trials)) #prints index for timesteps_per_batch
    #print("\nbest model: {}\n".format(best))
    #print("\ntrials: {}\n".format(bayes_trials.best_trial['result']['loss']))
    end_time = time.time()
    total_time_spent = end_time - start_time
    return -int(bayes_trials.best_trial['result']['loss']), bayes_trials.trials, total_time_spent, seeds


