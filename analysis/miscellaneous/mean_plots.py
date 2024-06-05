import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
dependencies_path = os.path.join(current_dir, '..')
sys.path.append(dependencies_path)
import matplotlib.pyplot as plt
import numpy as np
from src.data_processing import process_file, replace_outliers, butter_lowpass_filter
from src.plotting import (
    plot_measured_I, 
    plot_measured_q, 
    plot_measured_q_dot,
    plot_measured_U
)

folder_path = 'data/exp23_01'
folder_path = 'data/all_cst_velocity'

data = {}
# Iterate over each file in the folder
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
        upper_bound = 2500
        lower_bound = 500
        if apply_filter :
            I = butter_lowpass_filter(df['DXL_Current'], cutoff, fs, order)
            v = butter_lowpass_filter(df['DXL_Velocity'], cutoff, fs, order)
            u = butter_lowpass_filter(df['U'], cutoff, fs, order)
        else :
            v = df['DXL_Velocity'][lower_bound:upper_bound].mean()
            I = df['DXL_Current'][lower_bound:upper_bound].mean()
            u = df['U'][lower_bound:upper_bound].mean()

        # Compute the electromagnetic torque (ideal one)
        # kphi = 3.8197
        kphi = 4

        if I != 0 :
            R = (u - kphi * v)/I
        else :
            R = 5
        R = 10

        # Compute the electromagnetic torque (ideal one)
        if I != 0 :
            tau_em_ideal = ((u - R*I)/v) * I
        else :
            tau_em_ideal = kphi * (u - kphi * v)/R
        
        data[file_name] = {"Current" : I,
                           "Voltage" : u, 
                           "Velocity" : v,
                           "R" : R,
                           "T_ideal" : tau_em_ideal
                           }
        # plt.plot(I * 1000, label=file_name)
        


# plt.xlabel('Time (sample)')
# plt.ylabel('Current [mA]')
# plt.title(f'Measured current')
# plt.legend()
# plt.show()


# Extract the data for plotting
currents = [data[file]["Current"] for file in data]
voltages = [data[file]["Voltage"] for file in data]
velocities = [data[file]["Velocity"] for file in data]
Rs = [data[file]["R"] for file in data]
tau = [data[file]["T_ideal"] for file in data]
# print(data)

# R = 20
# kphi = 3.5
# for i in range(len(currents)):
#     if currents[i] == 0 :
#         currents[i] = (voltages[i] - kphi * velocities[i]) / R    # (((10 * 0.113)/100) * 12 - 3.8197 * (0.229*2*np.pi/60)) / 10

# Create plots
plt.figure(figsize=(14, 6))

# Current vs. Velocity
plt.subplot(1, 2, 1)
plt.scatter(velocities, currents, color='blue')
plt.title('Current vs. Velocity')
plt.xlabel('Velocity')
plt.ylabel('Current')
plt.grid()
# Voltage vs. Velocity
plt.subplot(1, 2, 2)
plt.scatter(velocities, voltages, color='red')
plt.title('Voltage vs. Velocity')
plt.xlabel('Velocity')
plt.ylabel('Voltage')

# Show the plot
plt.tight_layout()
plt.grid()
plt.show()