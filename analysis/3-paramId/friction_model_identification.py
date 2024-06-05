import cma
import numpy as np
from src.objective_functions import model_error
from src.data_processing import process_file
from src.model_fitting import (
    cma_pos_on_model, cma_free_on_model, 
    cma_free_on_model_folder, 
    cma_free_on_model_folder_no_friction,
    cmaes_free_on_model_folder,
    cma_on_list_of_df
)
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import glob
from src.data_processing import process_file, compute_physical, process_file_from_wizard, split_experiments

# folder_path = "data/Test_XH430W350_20240209/PWM_control/*.csv"

# # List to store each processed dataframe
# processed_dataframes = []

# # Iterate over all CSV files in the folder
# for file_path in glob.glob(folder_path):
#     # Apply processing function to the dataframe
#     processed_df = process_file_from_wizard(file_path)
#     experiments_split = split_experiments(processed_df)
    
#     # Store the processed dataframe in the list
#     processed_dataframes.extend(experiments_split)

with open('dataframes_list.pkl', 'rb') as file:
    processed_dataframes = pickle.load(file)

external_inertia = 0.0022421143208 # Msolo
Ra = 9.3756 # [Ohm]
kt = 2.6657
ke = 0.8594
ke = 3.62764

parameters_fixed = ke, kt, Ra

# We use CMA-ES for model fitting https://cma-es.github.io 
# Define initial mean and standard deviation for parameters

# First guess : Dynaban values
# q_dot_s, tau_c, tau_s, c_v, motor_inertia
initial_mean = [0.080964165, 0.0665140867408596, 0.203719639, 0.04217117990939209, 0.0155818308]
initial_std = 0.1  # Standard deviation

# Set up and run the CMA-ES algorithm
lower_bounds = [0] * 5  # n is the number of parameters
# upper_bounds = [100, 1000, 1, 1, 1, 10, 0.1]

options = {'maxiter': 1000, 'tolx': 1e-4, 'tolfun': 1e-4, 'bounds' : [lower_bounds, None]}  # Adjust these options as needed


cma_on_list_of_df(processed_dataframes, parameters_fixed,
                  external_inertia,
                  initial_mean=initial_mean, 
                  initial_std=initial_std, 
                  options=options
                  )
