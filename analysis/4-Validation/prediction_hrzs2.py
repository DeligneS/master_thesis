import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib as mpl

# Set the font size globally
mpl.rcParams.update({'font.size': 14})  # Change 14 to your desired font size

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
list2 = list(range(20, 36, 5))  # From 20 to 45, step 5
# list3 = list(range(50, 201, 10))  # From 50 to 200, step 10

# Concatenating lists
horizons = list1 + list2 #+ list3 #+ ['inf']

# Initialize a dictionary to store errors for each horizon
errors_vel = {hrz: {'error': [], 'NRMSE': [], 'R2': [], 'RMSE': [], 'MAE': [], 'real_vel': [], 'pred_vel': []} for hrz in horizons}
errors_pos = {hrz: {'error': [], 'NRMSE': [], 'R2': [], 'RMSE': [], 'MAE': [], 'real_pos': [], 'pred_pos': []} for hrz in horizons}


# Loop through each horizon and experiment
for hrz in horizons:
    full_velocities = []
    full_positions = []
    for exp_nbr in experiment_numbers:
        pred_filename = f'method2/hrz{hrz}_exp{exp_nbr}.csv'
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
        real_data = real_data[1+hrz:]
        pred_data = pred_data[:len(real_data)]
        real_data = real_data.reset_index(drop=True)
        real_data['timestamp'] = real_data['timestamp'] - real_data['timestamp'][0]

        nrmse_pos = calc_nrmse(real_data['θ(t)'], pred_data['predicted_position'])
        r2_pos = r2_score(real_data['θ(t)'], pred_data['predicted_position'])
        rmse_pos = np.sqrt(mean_squared_error(real_data['θ(t)'], pred_data['predicted_position']))
        mae_pos = mean_absolute_error(real_data['θ(t)'], pred_data['predicted_position'])

        # Append errors to the respective lists
        errors_pos[hrz]['NRMSE'].append(nrmse_pos)
        errors_pos[hrz]['R2'].append(r2_pos)
        errors_pos[hrz]['RMSE'].append(rmse_pos)
        errors_pos[hrz]['MAE'].append(mae_pos)
        errors_pos[hrz]['error'].append(real_data['θ(t)'] - pred_data['predicted_position'])
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

        # Append errors to the respective lists
        errors_vel[hrz]['NRMSE'].append(nrmse)
        errors_vel[hrz]['R2'].append(r2)
        errors_vel[hrz]['RMSE'].append(rmse)
        errors_vel[hrz]['MAE'].append(mae)
        errors_vel[hrz]['error'].append(real_data['DXL_Velocity'] - pred_data['predicted_velocity'])
        errors_vel[hrz]['real_vel'].append(real_data['DXL_Velocity'])
        errors_vel[hrz]['pred_vel'].append(pred_data['predicted_velocity'])

# Plotting the errors for each horizon
plt.figure(figsize=(10, 5))
# for metric in ['RMSE', 'MAE']:
#     plt.plot(horizons, [np.mean(errors[hrz][metric]) for hrz in horizons], label=metric, marker='o')

plt.plot(horizons, [calc_nrmse(np.concatenate(errors_vel[hrz]['real_vel']), np.concatenate(errors_vel[hrz]['pred_vel'])) for hrz in horizons], label='NRMSE', marker='o', color='blue')

for metric in ['NRMSE', 'R2']:
    plt.plot(horizons, [np.mean(errors_vel[hrz][metric]) for hrz in horizons], label=metric, marker='o')

plt.xlabel('Prediction Horizon')
plt.ylabel('Error Metric')
plt.title('Error Evolution with Prediction Horizon')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
# for metric in ['RMSE', 'MAE']:
#     plt.plot(horizons, [np.mean(errors[hrz][metric]) for hrz in horizons], label=metric, marker='o')
plt.plot(horizons, [calc_nrmse(np.concatenate(errors_pos[hrz]['real_pos']), np.concatenate(errors_pos[hrz]['pred_pos'])) for hrz in horizons], label='NRMSE', marker='o', color='blue')

for metric in ['NRMSE', 'R2']:
    plt.plot(horizons, [np.mean(errors_pos[hrz][metric]) for hrz in horizons], label=metric, marker='o')

plt.xlabel('Prediction Horizon')
plt.ylabel('Error Metric')
plt.title('Error Evolution with Prediction Horizon')
plt.legend()
plt.grid(True)
plt.show()


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

    # Plot velocity errors histogram
    ax1.hist(velocity_errors, bins=200, color='blue', alpha=0.7)
    # ax1.plot(errors_pos[horizon]['real_pos'][6])
    # ax1.plot(errors_pos[horizon]['pred_pos'][6])
    # ax1.set_title(f'Velocity NRMSE Distribution for Horizon {horizon}')
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Frequency')
    ax1.grid(True)

    # Plot position errors histogram
    ax2.hist(position_errors, bins=200, color='green', alpha=0.7)
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


horizon = 1
error_data = np.concatenate(errors_vel[horizon]['error'])
# position_errors = np.concatenate(errors_pos[horizon]['error'])
# Shapiro-Wilk Test
shapiro_stat, shapiro_p = stats.shapiro(error_data)
print(f"Shapiro-Wilk Test: Statistic={shapiro_stat}, p-value={shapiro_p}")

# Kolmogorov-Smirnov Test
ks_stat, ks_p = stats.kstest(error_data, 'norm', args=(np.mean(error_data), np.std(error_data)))
print(f"Kolmogorov-Smirnov Test: Statistic={ks_stat}, p-value={ks_p}")

# Q-Q Plot
plt.figure(figsize=(6, 6))
plt.plot(error_data, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()



##### PLOTTING EXPERIMENTS U #####

# Create a figure and a set of subplots
fig, axes = plt.subplots(4, 2, figsize=(12, 16))

# Flatten the axes array for easy iteration
axes = axes.flatten()
titles = ['Square 3V Input Signal', 'Square 1.5V Input Signal', 'Tooth 3V Input Signal', 'Tooth 1.5V Input Signal', 'Triangle 3V Input Signal', 'Triangle 1.5V Input Signal', 'Fast walking gait Input Signal', 'Slow walking gait Input Signal']
experiment_numbers2 = [9, 12, 10, 13, 11, 14, 17, 18]
# Plot each experiment in its own subplot
for i, exp in enumerate(experiment_numbers2):
    real_filename = f'references/{exp}.csv'
    real_path = os.path.join(data_dir, real_filename)
    real_data = pd.read_csv(real_path)
    axes[i].plot(real_data['timestamp'], real_data['U'])
    axes[i].set_title(titles[i])


    # axes[i].set_xlabel('Value')
    if i % 2 == 0:
        axes[i].set_ylabel('Voltage\n[V]', labelpad=20, rotation=0)
    axes[i].grid(True)
    axes[i].set_xlabel('Time [s]')

# Adjust y-limits for each row to be the same
for row in range(4):  # There are 4 rows
    row_axes = axes[row*2:(row+1)*2]  # Get the two subplots in the current row
    row_y_lims = [ax.get_ylim() for ax in row_axes]
    max_ylim = (min([ylim[0] for ylim in row_y_lims]), max([ylim[1] for ylim in row_y_lims]))
    for ax in row_axes:
        ax.set_ylim(max_ylim)

plt.subplots_adjust(hspace=0.6)  # Add more space between rows

# axes[6].set_xlabel('Time [s]')
# axes[7].set_xlabel('Time [s]')
plt.savefig('experiments_plot.png', dpi=300)  # Save as PNG file

# Adjust layout
plt.tight_layout()
plt.show()
