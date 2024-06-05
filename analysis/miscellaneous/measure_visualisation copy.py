from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

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


folder_path = 'data/exp20_01_ST'
folder_path = 'data/exp08_02'
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
            dfs.append(df[200:1200])
    return dfs

# Assuming df1, df2, df3, ... are your DataFrames
# We will concatenate them into one large DataFrame
dfs = build_dfs(folder_path)
big_df = pd.concat(dfs, ignore_index=True)

# Group the data to count recurrences
grouped = big_df.groupby(['DXL_Velocity', 'DXL_Current', 'U']).size().reset_index(name='counts')

# Create a new figure for plotting
plt.figure(figsize=(12, 8))

# Plot 'DXL_Current' vs 'DXL_Velocity'
plt.scatter(grouped['DXL_Velocity'], grouped['DXL_Current'], alpha=0.6, c=grouped['counts'], cmap='Reds', label='Current')

# Plot 'U' (voltage) vs 'DXL_Velocity' in the same figure with a different color
plt.scatter(grouped['DXL_Velocity'], grouped['U'], alpha=0.6, c=grouped['counts'], cmap='Blues', label='Voltage', marker='x')

# Creating a color bar
plt.colorbar().set_label('Number of Recurrences')

# Adding the legend
plt.legend()

# Adding labels for the axes
plt.xlabel('DXL Velocity')
plt.ylabel('DXL Current / U (Voltage)')

# Show the plot
plt.show()
