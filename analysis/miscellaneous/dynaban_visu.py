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


folder_path = 'data/exp24_01'
folder_path = 'data/exp08_02/cst'
# folder_path = 'data/exp08_02/sinus'
# folder_path = 'data/all_cst_velocity'

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

            # # Apply the filter
            # df['DXL_Current'] = butter_lowpass_filter(df['DXL_Current'], cutoff, fs, order)
            # df['DXL_Velocity'] = butter_lowpass_filter(df['DXL_Velocity'], cutoff, fs, order)
            # df['U'] = butter_lowpass_filter(df['U'], cutoff, fs, order)
            dfs.append(df[2:])
    return dfs

# We will concatenate them into one large DataFrame
dfs = build_dfs(folder_path)
big_df = pd.concat(dfs, ignore_index=True)

# Group the data by 'DXL_Velocity', 'DXL_Current', and 'U' to find the recurrence
grouped = big_df.groupby(['DXL_Velocity', 'DXL_Current', 'U']).size().reset_index(name='counts')

# Now let's create the scatter plot
plt.figure(figsize=(10, 8))
sc = plt.scatter(grouped['DXL_Velocity'], grouped['DXL_Current'], c=grouped['counts'], cmap='viridis', alpha=0.6, edgecolors='w', s=50)
plt.xlabel('DXL Velocity')
plt.ylabel('DXL Current')
cb = plt.colorbar(sc)
cb.set_label('Number of Recurrences')

# We'll use a separate plot to show Voltage vs Velocity
plt.figure(figsize=(10, 8))
sc = plt.scatter(grouped['DXL_Velocity'], grouped['U'], c=grouped['counts'], cmap='viridis', alpha=0.6, edgecolors='w', s=50)
plt.xlabel('DXL Velocity')
plt.ylabel('U (Voltage)')
cb = plt.colorbar(sc)
cb.set_label('Number of Recurrences')

plt.show()
