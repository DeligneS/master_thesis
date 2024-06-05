import cma
import cmaes

import numpy as np
import os
from src.objective_functions import model_error, model_error_on_folder, objective_function_lite, model_error_on_list, model_error_estimation


# We use CMA-ES for model fitting https://cma-es.github.io 
# ----- ON ONE EXPERIMENT -----
def cma_free_on_model(df, parameters_fixed, delta_t, with_current, using_datasheet_param, initial_mean, initial_std, options):
    es = cma.CMAEvolutionStrategy(initial_mean, initial_std, options)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [model_error(df, x, parameters_fixed, delta_t, with_current, using_datasheet_param) for x in solutions])
        es.logger.add()  # write data to disc to be plotted
        es.disp()

    es.result_pretty()
    cma.plot()  # shortcut for es.logger.plot()


def cma_pos_on_model(df, parameters_fixed, delta_t, with_current, using_datasheet_param, initial_mean, initial_std, options):
    es = cma.CMAEvolutionStrategy(np.exp(initial_mean), initial_std, options)
    while not es.stop():
        log_solutions = es.ask()  # These are in log space
        positive_solutions = [np.exp(s) for s in log_solutions]  # Transform to positive solutions
        errors = [model_error(df, ps, parameters_fixed, delta_t, with_current, using_datasheet_param) for ps in positive_solutions]
        es.tell(log_solutions, errors)  # Provide feedback in log space
        es.logger.add()  # optional, for logging
        es.disp()

    # Extract the best solution and transform back
    best_parameters_log = es.result.xbest
    best_parameters = np.exp(best_parameters_log)  # Ensure parameters are positive

    print("Best parameters found:", best_parameters)


def cma_nor_on_model(df, parameters_fixed, delta_t, with_current, using_datasheet_param, initial_mean, initial_std, options):
    es = cma.CMAEvolutionStrategy(initial_mean, initial_std, options)
    while not es.stop():
        log_solutions = es.ask()  # These are in log space
        errors = [model_error(df, ps, parameters_fixed, delta_t, with_current, using_datasheet_param) for ps in log_solutions]
        es.tell(log_solutions, errors)  # Provide feedback in log space
        es.logger.add()  # optional, for logging
        es.disp()

    # Extract the best solution and transform back
    best_parameters = es.result.xbest

    print("Best parameters found:", best_parameters)


# ----- ON SEVERAL EXPERIMENTS -----
# def cma_free_on_model_folder(folder_path, with_current, using_datasheet_param, initial_mean, initial_std, options, dxl_model = ["M1", "M2", "M3", "M4", "Msolo"]):
#     es = cma.CMAEvolutionStrategy(initial_mean, initial_std, options)
#     while not es.stop():
#         solutions = es.ask()
#         # es.tell(solutions, [model_error_on_folder(folder_path, x, dxl_model, 
#         #                                           with_current=with_current, 
#         #                                           using_datasheet_param=using_datasheet_param
#         #                                           ) for x in solutions])
#         es.tell(solutions, [model_error_on_folder(folder_path, np.append(0, np.append(x, 0)), dxl_model, 
#                                                   with_current=with_current, 
#                                                   using_datasheet_param=using_datasheet_param
#                                                   ) for x in solutions])
#         es.logger.add()  # write data to disc to be plotted
#         es.disp()

#     es.result_pretty()
#     cma.plot()  # shortcut for es.logger.plot()


def model_error_wrapper(x, folder_path, dxl_model, with_current, using_datasheet_param):
    # Append the fixed parameters to the optimization parameters
    full_params = np.append([0, 0], np.append(x, 0))
    # Call your existing model error function
    return model_error_on_folder(folder_path, full_params, dxl_model, with_current=with_current, using_datasheet_param=using_datasheet_param)

def cma_free_on_model_folder(folder_path, with_current, using_datasheet_param, initial_mean, initial_std, options, dxl_model = ["M1", "M2", "M3", "M4", "Msolo"]):
    # Define additional arguments for the model_error_wrapper
    args = (folder_path, dxl_model, with_current, using_datasheet_param)

    # Use cma.fmin to perform the optimization
    res = cma.fmin(model_error_wrapper, initial_mean, initial_std, options, args=args, restarts=6, bipop=True)

    # Extract results
    best_parameters = res[0]  # Best found solution
    best_score = res[1]       # Best objective function value

    print("Best Parameters:", best_parameters)
    print("Best Score:", best_score)

    # Additional logging or plotting can be added here if needed








def cma_free_on_model_folder_no_friction(folder_path, fixed_parameters, with_current, using_datasheet_param, initial_mean, initial_std, options, dxl_model = ["M1", "M2", "M3", "M4", "Msolo"]):
    es = cma.CMAEvolutionStrategy(initial_mean, initial_std, options)
    while not es.stop():
        solutions = es.ask()
        # print(solutions)
        es.tell(solutions, [model_error_on_folder(folder_path, np.append(x, fixed_parameters), dxl_model, 
                                                  with_current=with_current, 
                                                  using_datasheet_param=using_datasheet_param
                                                  ) for x in solutions])
        es.logger.add()  # write data to disc to be plotted
        es.disp()

    es.result_pretty()
    cma.plot()  # shortcut for es.logger.plot()


def cmaes_free_on_model_folder(folder_path, with_current, using_datasheet_param, initial_mean, initial_sigma, bounds, dxl_model = ["M1", "M2", "M3", "M4", "Msolo"]):
    # Initialize the optimizer
    optimizer = cmaes.CMA(mean=initial_mean, sigma=initial_sigma, bounds=None, seed=0)

    # Define the number of generations or iterations
    num_generations = 100  # adjust as needed

    for _ in range(num_generations):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = model_error_on_folder(folder_path, x, dxl_model, 
                                                  with_current=with_current, 
                                                  using_datasheet_param=using_datasheet_param
                                                  )
            solutions.append((x, value))
        
        optimizer.tell(solutions)

    # Retrieve the best solution
    result = optimizer.result()
    best_parameters = result.x
    best_score = result.fun

    print("Best Parameters:", best_parameters)
    print("Best Score:", best_score)


def cma_on_list_of_df(dfs, parameters_fixed, external_inertia, initial_mean, initial_std, options):

    es = cma.CMAEvolutionStrategy(initial_mean, initial_std, options)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [model_error_estimation(dfs, x, parameters_fixed, external_inertia) for x in solutions])
        es.logger.add()  # write data to disc to be plotted
        es.disp()

    es.result_pretty()
    cma.plot()  # shortcut for es.logger.plot()


