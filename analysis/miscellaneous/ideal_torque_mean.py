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


folder_path = 'data/exp23_01'

# Iterate over each file in the folder
def build_dfs(folder_path):
    kphi = 3.6
    R = 5
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

            # # Apply the filter
            apply_filter = False
            if apply_filter :
                I = butter_lowpass_filter(df['DXL_Current'], cutoff, fs, order)
                v = butter_lowpass_filter(df['DXL_Velocity'], cutoff, fs, order)
                u = butter_lowpass_filter(df['U'], cutoff, fs, order)
            else :
                I = df['DXL_Current'][100:].mean()
                v = df['DXL_Velocity'][100:].mean()
                u = df['U'][100:].mean()
            # Compute the electromagnetic torque (ideal one)
            if I != 0 :
                tau_em_ideal = (u - R*I/v) * I
            else :
                tau_em_ideal = 0

            # Use alternate formula where I is zero
            df['tau_EM'] = tau_em_ideal
            # df['tau_EM'] = (tau_em_ideal).round(3)

            # dfs.append(df[100:])
            dfs.append(df)

    return dfs


# Assuming df1, df2, df3, ... are your DataFrames
# We will concatenate them into one large DataFrame
dfs = build_dfs(folder_path)
big_df = pd.concat(dfs, ignore_index=True)

# Group the data by 'DXL_Velocity', 'DXL_Current', and 'U' to find the recurrence
grouped = big_df.groupby(['DXL_Velocity', 'tau_EM']).size().reset_index(name='counts')
# mean_R = grouped['R'].mean()

# Now let's create the scatter plot
plt.figure(figsize=(10, 8))
sc = plt.scatter(grouped['DXL_Velocity'], grouped['tau_EM'], c=grouped['counts'], cmap='viridis', alpha=0.6, edgecolors='w', s=50)

# plt.axhline(y=mean_R, color='r', linestyle='-', label=f'Average Current: {mean_R:.2f}')


plt.xlabel('DXL Velocity [rad/s]')
plt.ylabel('Ideal EM Torque Nm')
# plt.ylabel('kphi value [V.s/rad]')
cb = plt.colorbar(sc)
cb.set_label('Number of Recurrences')
# Adding the legend
plt.legend()
plt.show()
