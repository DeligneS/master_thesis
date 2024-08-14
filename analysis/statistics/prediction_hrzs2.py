import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib as mpl

# Set the font size globally
mpl.rcParams.update({'font.size': 18})  # Change 14 to your desired font size

def calculate_nrmse(real, predicted):
    """Calculate Normalized Root Mean Squared Error."""
    rmse = np.sqrt(np.mean((real - predicted) ** 2))
    range_of_data = real.max() - real.min()
    return rmse / range_of_data

def calc_nrmse(real, predicted):
    rmse = np.sqrt(np.mean((real - predicted) ** 2))
    nrmse_value = rmse / np.std(real)
    return nrmse_value

def fit_percentage(y_true, y_pred):
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    fit_perc = (1 - numerator/denominator) * 100
    return fit_perc

def calculate_metrics(horizons, experiment_numbers, method=1, model=2, percentile_name='99th percentile'):
    # Initialize a dictionary to store errors for each horizon
    errors_vel = {hrz: {'error': [], 'NRMSE': [], r'$R^2$': [], 'RMSE': [], 'MAE': [], 'real_vel': [], 'pred_vel': [], percentile_name: []} for hrz in horizons}
    errors_pos = {hrz: {'error': [], 'NRMSE': [], r'$R^2$': [], 'RMSE': [], 'MAE': [], 'real_pos': [], 'pred_pos': [], percentile_name: []} for hrz in horizons}
    for hrz in horizons:
        for exp_nbr in experiment_numbers:
            pred_filename = f'method{method}/model{model}/hrz{hrz}_exp{exp_nbr}.csv'
            real_filename = f'references/{exp_nbr}.csv'

            # Construct file paths
            pred_path = os.path.join(data_dir, pred_filename)
            real_path = os.path.join(data_dir, real_filename)
            
            # Check if files exist to avoid errors
            if not os.path.exists(pred_path) or not os.path.exists(real_path):
                continue

            # Read the prediction and real data files
            pred_data = pd.read_csv(pred_path)
            real_data = pd.read_csv(real_path)
            if method == 1 or hrz == 'inf':
                real_data = real_data[1:]
            else:
                real_data = real_data[1+hrz:]
            pred_data = pred_data[:len(real_data)]
            real_data = real_data.reset_index(drop=True)
            real_data['timestamp'] = real_data['timestamp'] - real_data['timestamp'][0]

            nrmse_pos = calc_nrmse(real_data['θ(t)'], pred_data['predicted_position'])
            r2_pos = r2_score(real_data['θ(t)'], pred_data['predicted_position'])
            rmse_pos = np.sqrt(mean_squared_error(real_data['θ(t)'], pred_data['predicted_position']))
            mae_pos = mean_absolute_error(real_data['θ(t)'], pred_data['predicted_position'])

            position_errors = np.abs(real_data['θ(t)'] - pred_data['predicted_position'])

            # Append errors to the respective lists
            errors_pos[hrz]['NRMSE'].append(nrmse_pos)
            errors_pos[hrz][r'$R^2$'].append(r2_pos)
            errors_pos[hrz]['RMSE'].append(rmse_pos)
            errors_pos[hrz]['MAE'].append(mae_pos)
            worst_case_bound = np.percentile(np.abs(position_errors), 99.9)
            errors_pos[hrz][percentile_name].append(worst_case_bound)
            # errors_pos[hrz]['percentiles'].append([np.percentile(position_errors, lower_percentile), np.percentile(position_errors, upper_percentile)])
            errors_pos[hrz]['error'].append(position_errors)
            errors_pos[hrz]['real_pos'].append(real_data['θ(t)'])
            errors_pos[hrz]['pred_pos'].append(pred_data['predicted_position'])

            real_data = real_data[2:]
            pred_data = pred_data[:len(real_data)]
            real_data = real_data.reset_index(drop=True)
            real_data['timestamp'] = real_data['timestamp'] - real_data['timestamp'][0]

            # Calculate error metrics
            rmse = np.sqrt(mean_squared_error(real_data['DXL_Velocity'], pred_data['predicted_velocity']))
            mae = mean_absolute_error(real_data['DXL_Velocity'], pred_data['predicted_velocity'])

            # Calculate error metrics
            nrmse = calc_nrmse(real_data['DXL_Velocity'], pred_data['predicted_velocity'])
            r2 = r2_score(real_data['DXL_Velocity'], pred_data['predicted_velocity'])
            velocity_errors = np.abs(real_data['DXL_Velocity'] - pred_data['predicted_velocity'])

            # Append errors to the respective lists
            errors_vel[hrz]['NRMSE'].append(nrmse)
            errors_vel[hrz][r'$R^2$'].append(r2)
            errors_vel[hrz]['RMSE'].append(rmse)
            errors_vel[hrz]['MAE'].append(mae)
            errors_vel[hrz]['error'].append(velocity_errors)
            worst_case_bound = np.percentile(np.abs(velocity_errors), 99)
            errors_vel[hrz][percentile_name].append(worst_case_bound)
            # errors_vel[hrz][percentile_name].append([np.percentile(velocity_errors, lower_percentile), np.percentile(velocity_errors, upper_percentile)])
            errors_vel[hrz]['real_vel'].append(real_data['DXL_Velocity'])
            errors_vel[hrz]['pred_vel'].append(pred_data['predicted_velocity'])

    return errors_vel, errors_pos


# Directory containing the prediction files and the real data files
data_dir = 'data/validation_exp/predictor/'

# Define the horizons and experiment numbers based on your experiment setup
experiment_numbers = [9, 10, 11, 12, 13, 14, 17, 18]
list1 = list(range(1, 20))  # From 1 to 19
list2 = list(range(20, 46, 5))  # From 20 to 45, step 5
list3 = list(range(50, 201, 10))  # From 50 to 200, step 10

# Concatenating lists
horizons = list1 + list2 + list3 + ['inf']
# horizons = ['inf']

percentile_name = '99.9th percentile'
method = 1
model = 1
errors_vel, errors_pos = calculate_metrics(horizons, experiment_numbers, method, model, percentile_name)
model = 2
errors_vel2, errors_pos2 = calculate_metrics(horizons, experiment_numbers, method, model, percentile_name)

def plotting(metric, hrzs, data, meh, model):
    inf_mean = np.mean(data["inf"][metric])
    # Calculate mean values for each horizon
    if metric == percentile_name:
        mean_values = [np.mean(np.mean(data[hrz][metric])) for hrz in hrzs if hrz != 'inf']
    else:
        mean_values = [np.mean(data[hrz][metric]) for hrz in hrzs if hrz != 'inf']
    horizons = [hrz for hrz in hrzs if hrz != 'inf']
    
    # Plot mean values with markers
    va = 'bottom'
    if model == 1:
        plt.plot(horizons, mean_values, label=f'{metric} PS1 (mean)', marker='o', color='royalblue', linestyle='-', linewidth=2)
        plt.axhline(inf_mean, color='cornflowerblue', linestyle='dashed', linewidth=2)
        if meh == 'pos' and metric == 'NRMSE':
            va = 'top'
        plt.text(max(horizons), inf_mean, f'{metric} PS1 (infinite horizon): {inf_mean:.3f}',
         color='cornflowerblue', ha='right', va=va)
    elif model == 2:
        plt.plot(horizons, mean_values, label=f'{metric} PS2 (mean)', marker='o', color='darkorange', linestyle='-', linewidth=2)
        plt.axhline(inf_mean, color='orange', linestyle='dashed', linewidth=2)
        if metric == r'$R^2$':
            va = 'top'
        plt.text(max(horizons), inf_mean, f'{metric} PS2 (infinite horizon): {inf_mean:.3f}',
         color='orange', ha='right', va=va)
    

    # Customize the axes
    plt.xlabel('Prediction Horizon')
    if metric == percentile_name:
        plt.ylabel(f'99.9th\nPercentile\nValue', labelpad=50, rotation=0)
    else:
        plt.ylabel(f'{metric}\nValue', labelpad=40, rotation=0)
    if meh=='vel':
        plt.title(f'{metric} Evolution of Velocity with\nPrediction Horizon for Parameter Set (PS) 1 & 2')
    else:
        plt.title(f'{metric} Evolution of Position with\nPrediction Horizon for Parameter Set (PS) 1 & 2')

    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, shadow=True, ncol=5)

    
    # Enhance grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout()

    plt.savefig(f'figures/{meh}_Model{model}_{metric}_V2.png', dpi=300)  # Save as PNG file

# NRMSE color model 1: royalblue, model 2:

hrzs = list1 + list2 + list3
plt.figure(figsize=(10, 6))
plotting('NRMSE', hrzs, errors_vel, 'vel', 1)
plotting('NRMSE', hrzs, errors_vel2, 'vel', 2)
plt.show()
plt.figure(figsize=(10, 6))
plotting(r'$R^2$', hrzs, errors_vel, 'vel', 1)
plotting(r'$R^2$', hrzs, errors_vel2, 'vel', 2)
plt.show()
plt.figure(figsize=(10, 6))
plotting('NRMSE', hrzs, errors_pos, 'pos', 1)
plotting('NRMSE', hrzs, errors_pos2, 'pos', 2)
plt.show()
plt.figure(figsize=(10, 6))
plotting(r'$R^2$', hrzs, errors_pos, 'pos', 1)
plotting(r'$R^2$', hrzs, errors_pos2, 'pos', 2)
plt.show()
