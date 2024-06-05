import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
dependencies_path = os.path.join(current_dir, '..')
sys.path.append(dependencies_path)

from src.modelisation import output_torque_motor_with_I
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.found_parameters.parameters import PARAMETERS_TO_FIND
from src.data_processing import process_file, replace_outliers, butter_lowpass_filter
from sklearn.linear_model import LinearRegression

folder_path = 'data/exp23_01'
# folder_path = 'data/all_cst_velocity'
# folder_path = 'data/all_cst_current'

# Iterate over each file in the folder
def build_dfs(folder_path):
    kphi = 3.6
    R = 20
    dfs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            full_path = os.path.join(folder_path, file_name)
            # print(file_name)

            df = process_file(full_path)
            df['tau_EM'] = (df['U']/df['DXL_Velocity']) * df['DXL_Current']
            # Filter requirements.
            order = 2  # the order of the filter (higher order = sharper cutoff)
            fs = 1.0  # sample rate, Hz
            cutoff = 0.008  # desired cutoff frequency of the filter, Hz

            # # Apply the filter
            apply_filter = False
            if apply_filter :
                I = butter_lowpass_filter(df['DXL_Current'], cutoff, fs, order)
                v = butter_lowpass_filter(df['DXL_Velocity'], cutoff, fs, order)
                u = butter_lowpass_filter(df['U'], cutoff, fs, order)
            else :
                I = df['DXL_Current']
                v = df['DXL_Velocity']
                u = df['U']
            # Compute the electromagnetic torque (ideal one)
            tau_em_ideal = (u/v) * I

            df['kphi'] = (u - R * I) / v 
            df['R'] = np.where(I == 0, None, (u - kphi * v) / I)
            # kphi = df['kphi']
            tau_em_alternate = kphi * ((u - kphi * v)/R)

            # Use alternate formula where I is zero
            # df['tau_EM'] = np.where(I == 0, (tau_em_alternate).round(3), (tau_em_ideal).round(3))
            # df['tau_EM'] = np.where(I == 0, None, (tau_em_ideal).round(3))

            # C(v) estimation
            kt = 2.6657
            tau_em_ideal = kt * I
            df['tau_EM'] = np.where(I == 0, None, (tau_em_ideal).round(3))

            # df['tau_EM'] = (tau_em_ideal).round(3)

            dfs.append(df[100:])
            # dfs.append(df)

    return dfs


# Assuming df1, df2, df3, ... are your DataFrames
# We will concatenate them into one large DataFrame
dfs = build_dfs(folder_path)
big_df = pd.concat(dfs, ignore_index=True)

# Group the data by 'DXL_Velocity', 'DXL_Current', and 'U' to find the recurrence
# grouped = big_df.groupby(['DXL_Velocity', 'R']).size().reset_index(name='counts')
# grouped = big_df.groupby(['DXL_Current', 'tau_EM']).size().reset_index(name='counts')
grouped = big_df.groupby(['tau_EM', 'DXL_Velocity']).size().reset_index(name='counts')

# mean_R = grouped['R'].mean()

# Now let's create the scatter plot
# plt.figure(figsize=(10, 8))
# sc = plt.scatter(grouped['tau_EM'], grouped['DXL_Current'], c=grouped['counts'], cmap='viridis', alpha=0.6, edgecolors='w', s=50)

# # plt.axhline(y=mean_R, color='r', linestyle='-', label=f'Average Current: {mean_R:.2f}')


# plt.xlabel('DXL Velocity [rad/s]')
# # plt.ylabel('Ideal EM Torque Nm')
# plt.ylabel('kphi value [V.s/rad]')
# cb = plt.colorbar(sc)
# cb.set_label('Number of Recurrences')
# # Adding the legend
# plt.legend()
# plt.show()



def linear_regression_tau_I():
    # Create linear regression objects
    reg_current = LinearRegression()

    # Fit the models

    weights = grouped['counts'].values
    # full_df = big_df.groupby(['DXL_Current', 'tau_EM'])
    reg_current.fit(grouped[['tau_EM']], grouped['DXL_Current'], sample_weight=weights)

    # Get the slope (coefficient) of the fit
    slope_current = 1/reg_current.coef_[0]

    # Create a new figure for plotting
    plt.figure(figsize=(12, 8))

    # Predictions for the linear approximation line
    current_line = reg_current.predict(grouped[['tau_EM']])

    # Plot the scatter points
    plt.scatter(grouped['tau_EM'], grouped['DXL_Current'], c='red', label='Current')

    # Plot the linear regression line for 'DXL_Current'
    plt.plot(grouped['tau_EM'], current_line, color='darkred', label=f'Current Slope: {slope_current:.2f}')

    # Add legend and labels
    plt.legend()
    plt.xlabel('DXL Torque')
    plt.ylabel('DXL Current')

    # Show the plot
    plt.show()

def linear_regression_tau_w():
    # Create linear regression objects
    reg_current = LinearRegression()

    # Fit the models

    weights = grouped['counts'].values
    reg_current.fit(grouped[['DXL_Velocity']], grouped['tau_EM'], sample_weight=weights)
    print(reg_current.coef_)
    print(reg_current.intercept_)
    # Get the slope (coefficient) of the fit
    slope_current = reg_current.coef_[0]

    # Create a new figure for plotting
    plt.figure(figsize=(12, 8))

    # Predictions for the linear approximation line
    current_line = reg_current.predict(grouped[['DXL_Velocity']])

    # Plot the scatter points
    plt.scatter(grouped['DXL_Velocity'], grouped['tau_EM'], c='red', label='Current')

    # Plot the linear regression line for 'DXL_Current'
    plt.plot(grouped['DXL_Velocity'], current_line, color='darkred', label=f'Current Slope: {slope_current:.6f}')

    # Add legend and labels
    plt.legend()
    plt.xlabel('DXL Velocity [rad/s]')
    plt.ylabel('DXL Torque [Nm]')

    # Show the plot
    plt.show()

# linear_regression_tau_I()
linear_regression_tau_w()




# # Create the weighted linear regression model for 'DXL_Current' using the count as weights
# weights = grouped['counts'].values
# reg_current = LinearRegression()
# reg_current.fit(grouped[['DXL_Velocity']], grouped['DXL_Current'], sample_weight=weights)
# slope_current = reg_current.coef_[0]

# # Predictions for the linear approximation line
# current_line = reg_current.predict(grouped[['DXL_Velocity']])

# # Plotting the scatter and linear approximation
# plt.figure(figsize=(12, 8))

# # Scatter plots
# plt.scatter(grouped['DXL_Velocity'], grouped['DXL_Current'], alpha=0.6, s=weights, c='red', label='Current')

# # Linear approximation lines
# plt.plot(grouped['DXL_Velocity'], current_line, color='darkred', linestyle='-', label=f'Current Slope: {slope_current:.2f}')

# # Color bar and labels
# plt.colorbar(label='Number of Recurrences')
# plt.xlabel('DXL Velocity')
# plt.ylabel('DXL Current / U (Voltage)')
# plt.legend()
# plt.show()
