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
# Initialize a dictionary to store errors for each horizon
errors_vel = {hrz: {'error': [], 'NRMSE': [], r'$R^2$': [], 'RMSE': [], 'MAE': [], 'real_vel': [], 'pred_vel': [], percentile_name: []} for hrz in horizons}
errors_pos = {hrz: {'error': [], 'NRMSE': [], r'$R^2$': [], 'RMSE': [], 'MAE': [], 'real_pos': [], 'pred_pos': [], percentile_name: []} for hrz in horizons}

method = 1
model = 3

# for exp_nbr in experiment_numbers:
#     real_filename = f'references/{exp_nbr}.csv'

#     # Construct file paths
#     real_path = os.path.join(data_dir, real_filename)
#     real_data = pd.read_csv(real_path)
#     dt = real_data['timestamp'][1] - real_data['timestamp'][0]
#     velocities_computed = (real_data["θ(t)"][1:] - real_data["θ(t)"][:-1]) / (1 * dt)
#     velocities_computed = np.gradient(real_data['θ(t)'], real_data['timestamp'])
    
#     velocities_computed = np.insert(velocities_computed, 0, real_data["DXL_Velocity"][0])  # Insert 0 at start
#     velocities_computed = velocities_computed[:-1]  # Remove last element
#     # velocities_computed = np.append(velocities_computed, 0)  # Append 0 at end

#     residuals = real_data["DXL_Velocity"] - velocities_computed

#     # Statistical analysis
#     mean_residual = np.mean(residuals)
#     std_residual = np.std(residuals)

#     print(f"Mean of residuals: {mean_residual}")
#     print(f"Standard deviation of residuals: {std_residual}")

#     plt.figure(figsize=(10, 6))
#     plt.plot(real_data['timestamp'], real_data['DXL_Velocity'], label='Real Velocity')
#     plt.plot(real_data['timestamp'], velocities_computed, label='Computed Velocity')
#     plt.legend()
#     plt.show()




# Loop through each horizon and experiment
for hrz in horizons:
    full_velocities = []
    full_positions = []
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

        # real_data['DXL_Velocity'].plot()
        # pred_data['predicted_velocity'].plot()
        # real_data['DXL_Velocity'] = np.gradient(real_data['θ(t)'], real_data['timestamp'])
        # pred_data['predicted_velocity'] = np.gradient(pred_data['predicted_position'], pred_data['timestamp'])
        # real_data['DXL_Velocity'].plot()
        # pred_data['predicted_velocity'].plot()
        # plt.show()

        nrmse_pos = calc_nrmse(real_data['θ(t)'], pred_data['predicted_position'])
        r2_pos = r2_score(real_data['θ(t)'], pred_data['predicted_position'])
        rmse_pos = np.sqrt(mean_squared_error(real_data['θ(t)'], pred_data['predicted_position']))
        mae_pos = mean_absolute_error(real_data['θ(t)'], pred_data['predicted_position'])

        lower_percentile = 2.5
        upper_percentile = 97.5
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

        real_data = real_data[1:]
        pred_data = pred_data[:len(real_data)]
        real_data = real_data.reset_index(drop=True)
        real_data['timestamp'] = real_data['timestamp'] - real_data['timestamp'][0]

        # real_data['DXL_Velocity'] = np.gradient(real_data['θ(t)'], real_data['timestamp'])
        full_velocities.append(real_data['DXL_Velocity'])
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

print(np.mean(errors_vel['inf']['NRMSE']))
print(np.mean(errors_vel['inf'][r'$R^2$']))
print(np.mean(errors_pos['inf']['NRMSE']))
print(np.mean(errors_pos['inf'][r'$R^2$']))
print(max(errors_vel['inf']['real_vel'][6]))

def plotting(metric, hrzs, data, meh):
    plt.figure(figsize=(10, 6))

    # Calculate mean values for each horizon
    if metric == percentile_name:
        mean_values = [np.mean(np.mean(data[hrz][metric])) for hrz in hrzs if hrz != 'inf']
    else:
        mean_values = [np.mean(data[hrz][metric]) for hrz in hrzs if hrz != 'inf']
    horizons = [hrz for hrz in hrzs if hrz != 'inf']
    
    # Plot mean values with markers
    if metric=="NRMSE":
        plt.plot(horizons, mean_values, label=f'{metric} (mean)', marker='o', color='royalblue', linestyle='-', linewidth=2)
    else:
        plt.plot(horizons, mean_values, label=f'{metric} (mean)', marker='o', color='green', linestyle='-', linewidth=2)

    # Add a horizontal line for the 'inf' horizon
    inf_mean = np.mean(data["inf"][metric])
    plt.axhline(inf_mean, color='firebrick', linestyle='dashed', linewidth=2, label=f'{metric} (infinite horizon)')
    
    # Add annotation for the horizontal line
    # plt.text(max(horizons), inf_mean, f'{metric} (infinite horizon): {inf_mean:.3f}', color='firebrick', ha='right', va='top')
    vertical_offset = -0.004  # Adjust this value as needed
    if (meh == 'pos' and metric == '$R^2$') or (model==3 and meh=='vel' and metric == '$R^2$'):
        plt.text(max(horizons), inf_mean, f'{metric} (infinite horizon): {inf_mean:.3f}',
         color='firebrick', ha='right', va='bottom')
    elif metric == percentile_name and meh=='vel' and model == 2:
        plt.text(max(horizons), inf_mean, f'{metric} (infinite horizon): {inf_mean:.3f}',
            color='firebrick', ha='right', va='bottom')
        plt.ylim(0.12, 0.15)
    elif metric == percentile_name and meh=='vel' and model == 3:
        plt.text(max(horizons), inf_mean, f'{metric} (infinite horizon): {inf_mean:.3f}',
            color='firebrick', ha='right', va='bottom')
        plt.ylim(0.085, 0.145)
    elif metric == '$R^2$' and meh=='vel' and model == 2:
        plt.text(max(horizons), inf_mean, f'{metric} (infinite horizon): {inf_mean:.3f}',
            color='firebrick', ha='right', va='bottom')
        # plt.ylim(0.085, 0.145)
    else:
        if (model==2 and meh=='vel') or (model==3 and meh=='vel'):
            vertical_offset= -0.0004
        plt.text(max(horizons), inf_mean + vertical_offset, f'{metric} (infinite horizon): {inf_mean:.3f}',
            color='firebrick', ha='right', va='top')


    # Customize the axes
    plt.xlabel('Prediction Horizon')
    if metric == percentile_name:
        plt.ylabel(f'99.9th\nPercentile\nValue', labelpad=50, rotation=0)
    else:
        plt.ylabel(f'{metric}\nValue', labelpad=40, rotation=0)
    if meh=='vel':
        plt.title(f'{metric} Evolution of Velocity with\nPrediction Horizon for Parameter Set {model}')
    else:
        plt.title(f'{metric} Evolution of Position with\nPrediction Horizon for Parameter Set {model}')
    
    # Add legend
    plt.legend(loc='best')
    
    # Enhance grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout()

    plt.savefig(f'figures/{meh}_Model{model}_{metric}.png', dpi=300)  # Save as PNG file

    # Show the plot
    plt.show()


hrzs = list1 + list2 + list3
# plotting('NRMSE', hrzs, errors_vel, 'vel')
# plotting(r'$R^2$', hrzs, errors_vel, 'vel')
# plotting('NRMSE', hrzs, errors_pos, 'pos')
# plotting(r'$R^2$', hrzs, errors_pos, 'pos')

plotting(percentile_name, hrzs, errors_vel, 'vel')
plotting(percentile_name, hrzs, errors_pos, 'pos')

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import scipy.stats as stats

# Assuming errors2 and errors2_pos are dictionaries containing arrays of errors for each horizon

# Prepare the plotting function
def update_plot(index):
    """ Update the histograms for the selected horizon index. """
    # Convert float slider value to nearest integer index
    index = int(index)
    # Get the horizon from the list using the index
    horizon = horizons[index]

    # Clear current axes
    ax1.clear()
    ax2.clear()
    
    # Data for histograms
    velocity_errors = np.concatenate(errors_vel[horizon]['error'])
    position_errors = np.concatenate(errors_pos[horizon]['error'])
    # velocity_errors = errors_vel[horizon]['NRMSE']
    # position_errors = errors_pos[horizon]['NRMSE']
    # Calculate standard deviation
    velocity_errors_std = np.std(velocity_errors)
    position_errors_std = np.std(position_errors)

    # Define the error bound (e.g., 2 standard deviations)
    error_bound_vel = 2 * velocity_errors_std
    error_bound_pos = 2 * position_errors_std

    # Plot velocity errors histogram
    ax1.hist(velocity_errors, bins=200, color='blue', alpha=0.7)
    ax1.axvline(-error_bound_vel, color='red', linestyle='dashed', linewidth=2)
    ax1.axvline(error_bound_vel, color='red', linestyle='dashed', linewidth=2)

    # ax1.plot(errors_pos[horizon]['real_pos'][6])
    # ax1.plot(errors_pos[horizon]['pred_pos'][6])
    # ax1.set_title(f'Velocity NRMSE Distribution for Horizon {horizon}')
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Frequency')
    ax1.grid(True)

    # Plot position errors histogram
    ax2.hist(position_errors, bins=200, color='green', alpha=0.7)
    ax2.axvline(-error_bound_pos, color='red', linestyle='dashed', linewidth=2)
    ax2.axvline(error_bound_pos, color='red', linestyle='dashed', linewidth=2)
    # ax2.set_title(f'Position NRMSE Distribution for Horizon {horizon}')
    ax2.set_xlabel('Error')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)

    # Redraw the plot
    plt.draw()

# Initial plot setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Add slider for selecting the prediction horizon
axcolor = 'lightgoldenrodyellow'
ax_horizon = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)
slider_horizon = Slider(ax_horizon, 'Prediction Horizon', 0, len(horizons)-1, valinit=0, valfmt='%0.0f')

# Update the plot when the slider value is changed
def sliders_on_changed(val):
    update_plot(slider_horizon.val)

slider_horizon.on_changed(sliders_on_changed)

# Initial horizon to display
update_plot(0)

plt.show()

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Assuming errors_vel and errors_pos are dictionaries containing arrays of errors for each horizon

# Prepare the plotting function
def update_plot(index):
    """ Update the histograms for the selected horizon index. """
    # Convert float slider value to nearest integer index
    index = int(index)
    # Get the horizon from the list using the index
    horizon = horizons[index]

    # Clear current axes
    ax1.clear()
    ax2.clear()
    
    # Data for histograms
    velocity_errors = np.concatenate(errors_vel[horizon]['error'])
    position_errors = np.concatenate(errors_pos[horizon]['error'])

    # Define percentiles for the error bounds
    lower_percentile = 2.5
    upper_percentile = 97.5

    # Calculate the bounds
    error_bound_vel_lower = np.percentile(velocity_errors, lower_percentile)
    error_bound_vel_upper = np.percentile(velocity_errors, upper_percentile)
    error_bound_pos_lower = np.percentile(position_errors, lower_percentile)
    error_bound_pos_upper = np.percentile(position_errors, upper_percentile)

    # velocity_errors = errors_vel[horizon]['NRMSE']
    # position_errors = errors_pos[horizon]['NRMSE']

    # Plot velocity errors histogram
    # ax1.hist(velocity_errors, bins=200, color='blue', alpha=0.7)
    # ax1.plot(errors_pos[horizon]['real_pos'][6])
    # ax1.plot(errors_pos[horizon]['pred_pos'][6])
    ax1.plot(errors_vel[horizon]['real_vel'][0])
    ax1.plot(errors_vel[horizon]['pred_vel'][0])

    # Plot velocity errors histogram
    # ax1.hist(velocity_errors, bins=200, color='blue', alpha=0.7)
    ax1.axvline(error_bound_vel_lower, color='red', linestyle='dashed', linewidth=2)
    ax1.axvline(error_bound_vel_upper, color='red', linestyle='dashed', linewidth=2)
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Frequency')
    ax1.grid(True)

    # Plot position errors histogram
    ax2.hist(position_errors, bins=200, color='green', alpha=0.7)
    ax2.axvline(error_bound_pos_lower, color='red', linestyle='dashed', linewidth=2)
    ax2.axvline(error_bound_pos_upper, color='red', linestyle='dashed', linewidth=2)
    ax2.set_xlabel('Error')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)

    # Redraw the plot
    plt.draw()

# Initial plot setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Add slider for selecting the prediction horizon
axcolor = 'lightgoldenrodyellow'
ax_horizon = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)
slider_horizon = Slider(ax_horizon, 'Prediction Horizon', 0, len(horizons)-1, valinit=0, valfmt='%0.0f')

# Update the plot when the slider value is changed
def sliders_on_changed(val):
    update_plot(slider_horizon.val)

slider_horizon.on_changed(sliders_on_changed)

# Initial horizon to display
update_plot(0)

plt.show()
