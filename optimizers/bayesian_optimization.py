from hyperopt import fmin, tpe, Trials, STATUS_OK
from optimizers.utils import set_seed, get_model, evaluate
import time

def run_bayesian_opt(env, method, num_configs, algorithm, space, total_timesteps):
    total_time_spent = 0
    start_time = time.time()
    global iteration
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
        global iteration #only necessary because of hyperopt implementation. 
        iteration += 1

        result = run_model(hyperparams, iteration)
        loss = -result #transform to loss in order to minimize

        return {'loss': loss, 
                'hyperparams': hyperparams, 
                'iteration': iteration, 
                'status': STATUS_OK
               }

    def run_model(config, iteration):
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
        # Fixed random state
        set_seed()
        fails = 0

        while True:
            try:
                model = get_model(method, algorithm, env, config)
                model.learn(total_timesteps)
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
                
        
        model.save("trpo_cartpole_" + str(iteration)) #name should be input
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
    print("\n\ntotal optimizing time: {} s\n\n".format(total_time_spent))
    return -int(bayes_trials.best_trial['result']['loss'])


