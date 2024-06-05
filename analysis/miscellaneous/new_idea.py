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

# folder_path = 'data/exp24_01'
# folder_path = 'data/test_pwm_cst'
folder_path = 'data/all_cst_current'

# Iterate over each file in the folder
plt.figure(figsize=(10, 6))
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        full_path = os.path.join(folder_path, file_name)
        print(file_name)

        df = process_file(full_path)

        # Filter requirements.
        order = 2  # the order of the filter (higher order = sharper cutoff)
        fs = 1.0  # sample rate, Hz
        cutoff = 0.008  # desired cutoff frequency of the filter, Hz

        # Apply the filter
        df['DXL_Current'] = butter_lowpass_filter(df['DXL_Current'], cutoff, fs, order)
        df['DXL_Velocity'] = butter_lowpass_filter(df['DXL_Velocity'], cutoff, fs, order)
        df['U'] = butter_lowpass_filter(df['U'], cutoff, fs, order)

        # plt.plot(df['DXL_Current'] * 1000, label=file_name)
        # plt.plot(df['DXL_Position'], label=file_name)
        # plt.plot(df['DXL_Velocity'], label=file_name)
        # mean_I = df['DXL_Current'][100:].mean()
        # plt.axhline(y=mean_I, color='r', linestyle='-', label=f'Average Current: {mean_I:.6f}')
        # plt.plot(df['U'][1:], label=file_name)
        # plt.plot((df['U'] - 232 * df['DXL_Current'])/df['DXL_Velocity'], label=file_name)
        # plt.plot((df['U'])/df['DXL_Velocity'], label=file_name)
        plt.plot((df['U'])/df['DXL_Velocity'], label=file_name)
        kt =  2.6657
        cv = 0.019323
        # plt.plot(kt*df["DXL_Current"], label=file_name)
        # # Apply the filter
        # I = butter_lowpass_filter(df['DXL_Current'], cutoff, fs, order)
        # # v = butter_lowpass_filter(df['DXL_Velocity'], cutoff, fs, order)
        # u = butter_lowpass_filter(df['U'], cutoff, fs, order)

        # Compute the electromagnetic torque (ideal one)
        # kphi = 3.8197
        # R = 10
        # tau_em_ideal = (u/v) * I
        # tau_em_alternate = kphi * ((u - kphi * v)/R)

        # Use alternate formula where I is zero
        # df['tau_EM'] = np.where(I == 0, (tau_em_alternate).round(2), (tau_em_ideal).round(2))
        # df['tau_EM'] = butter_lowpass_filter(df['tau_EM'], cutoff, fs, order)
        # plt.plot(df['tau_EM'], label=file_name)
        # plt.plot(tau_em_ideal, label=file_name+"id")
        # plt.plot(tau_em_alternate, label=file_name+"alt")
        # plt.plot(I, label=file_name)

plt.xlabel('Time (sample)')
plt.ylabel('Current [mA]')
plt.title(f'Measured current')
plt.legend()
plt.show()


