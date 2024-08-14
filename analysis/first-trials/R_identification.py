import os
import cma
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.data_processing import process_file, replace_outliers, butter_lowpass_filter
from src.plotting import (
    plot_measured_I, 
    plot_measured_q, 
    plot_measured_q_dot,
    plot_measured_U
)

folder_path = 'data/exp20_01'
# Iterate over each file in the folder

def build_dfs(folder_path):
    dfs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            full_path = os.path.join(folder_path, file_name)
            # print(file_name)

            df = process_file(full_path)

            # Filter requirements.
            order = 2  # the order of the filter (higher order = sharper cutoff)
            fs = 1.0  # sample rate, Hz
            cutoff = 0.008  # desired cutoff frequency of the filter, Hz

            # Apply the filter
            df['DXL_Current'] = butter_lowpass_filter(df['DXL_Current'], cutoff, fs, order)
            df['DXL_Velocity'] = butter_lowpass_filter(df['DXL_Velocity'], cutoff, fs, order)
            df['U'] = butter_lowpass_filter(df['U'], cutoff, fs, order)
            dfs.append(df)
    return dfs

def objective_function(Rs):
    my_df = pd.DataFrame()
    i = 0
    R, x = Rs
    for df in dfs:
        col_name = f"C-", i
        my_df[col_name] = (df['U'] - R * df['DXL_Current'])/df['DXL_Velocity']
        i += 1
    
    row_variances = my_df.var(axis=1)
    total_variance = row_variances.sum()
    return total_variance


def objective_function(Rs):
    my_df = pd.DataFrame()
    i = 0
    R, x = Rs
    kphi = 3.8197
    for df in dfs:
        col_name = f"C-", i
        my_df[col_name] = (df['U'] - kphi * df['DXL_Velocity']) / df['DXL_Current']
        i += 1
    
    row_variances = my_df.var(axis=1)
    total_variance = row_variances.sum()
    return total_variance


def cma_free_on_model(dfs, initial_mean, initial_std, options):
    # es = cma.CMAEvolutionStrategy(initial_mean, initial_std, options)
    # while not es.stop():
    #     solutions = es.ask()
    #     es.tell(solutions, [objective_function(dfs, R) for R in solutions])
    #     es.logger.add()  # write data to disc to be plotted
    #     es.disp()

    # es.result_pretty()
    # cma.plot()  # shortcut for es.logger.plot()
    # Define additional arguments for the model_error_wrapper
    args = (dfs)

    # Use cma.fmin to perform the optimization
    res = cma.fmin(objective_function, initial_mean, initial_std, options, restarts=6, bipop=True)

    # Extract results
    best_parameters = res[0]  # Best found solution
    best_score = res[1]       # Best objective function value

    print("Best Parameters:", best_parameters)
    print("Best Score:", best_score)

dfs = build_dfs(folder_path)
initial_mean = [0, 0]
initial_std = 1
options = {'maxiter': 1000, 'tolx': 1e-4, 'tolfun': 1e-4, 'bounds' : [[0, 0], [None, 0.1]]}  # Adjust these options as needed
cma_free_on_model(dfs, initial_mean, initial_std, options)
