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

folder_path = 'data/exp19_01'
folder_path = 'data/exp08_02/cst'
folder_path = 'data/exp05_02'
# folder_path = 'data/validation_exp/voltage_exp'
# folder_path = 'data/validation_exp/calibration'


# folder_path = 'data/test_pwm_cst'
# folder_path = 'data/all_cst_current'

# Iterate over each file in the folder
plt.figure(figsize=(10, 6))
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        full_path = os.path.join(folder_path, file_name)
        print(file_name)

        df = process_file(full_path, delta_t=20e-3)

        # Filter requirements.
        order = 2  # the order of the filter (higher order = sharper cutoff)
        fs = 1.0  # sample rate, Hz
        cutoff = 0.008  # desired cutoff frequency of the filter, Hz

        # Apply the filter
        # df['DXL_Current'] = butter_lowpass_filter(df['DXL_Current'], cutoff, fs, order)
        # df['DXL_Velocity'] = butter_lowpass_filter(df['DXL_Velocity'], cutoff, fs, order)
        # df['U'] = butter_lowpass_filter(df['U'], cutoff, fs, order)
        

        # plt.plot((df['DXL_Position'] * 180/np.pi - 90), label=file_name)
        plt.plot(df['t'], (df['DXL_Position'] + np.pi/2), label=file_name)
        plt.legend()
        # plt.plot(df['DXL_Velocity'], label=file_name)
        # mean_I = df['DXL_Current'][100:].mean()
        # plt.axhline(y=mean_I, color='r', linestyle='-', label=f'Average Current: {mean_I:.6f}')
        # plt.plot(df['U'][1:], label=file_name)
        # plt.plot((df['U'] - 232 * df['DXL_Current'])/df['DXL_Velocity'], label=file_name)
        # plt.plot((df['U'])/df['DXL_Velocity'], label=file_name)

        # # Apply the filter
        I = butter_lowpass_filter(df['DXL_Current'], cutoff, fs, order)
        v = butter_lowpass_filter(df['DXL_Velocity'], cutoff, fs, order)
        u = butter_lowpass_filter(df['U'], cutoff, fs, order)

        # plt.plot(df['DXL_Position'], label="Position")

        # Calculate Δt from the sampling rate
        delta_t = 80e-3

        # Calculate Δx (change in position)
        delta_x = np.diff(df['DXL_Position'])

        # Calculate velocity (Δx/Δt)
        velocity = delta_x / delta_t

        # Filter requirements
        order = 2  # the order of the filter (higher order = sharper cutoff)
        fs = 0.1  # sample rate, Hz
        cutoff = 0.008  # desired cutoff frequency of the filter, Hz
        velocity_filter = butter_lowpass_filter(velocity, cutoff, fs, order)
        delta_v = np.diff(velocity_filter)
        acceleration = delta_v / delta_t

        df['t'] = df['t'] #* 80e-3
        # plt.plot(df['t'], df['DXL_Velocity'], label="Velocity")
        # plt.plot(velocity, label="Velocity from position")
        # plt.plot(velocity_filter, label="Velocity from position")
        # plt.plot(acceleration, label="acceleration")
        # plt.plot(df['t'], df['U'], label="Voltage")
        # plt.plot(df['t'], df['DXL_Current'], label="Current")
        # Plot for velocities
        plt.figure(figsize=(10, 6))  # Create a new figure for velocities
        plt.plot(df['t'], df['DXL_Velocity'].shift(-1), label="Velocity")
        plt.plot(velocity, label="Velocity from position")
        plt.plot(velocity_filter, label="Filtered Velocity from position")
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Velocity Analysis')
        plt.legend()
        plt.show(block=False)  # Show the plot without blocking

        # Plot for acceleration
        plt.figure(figsize=(10, 6))  # Create a new figure for acceleration
        plt.plot(acceleration, label="Acceleration")
        plt.xlabel('Time')
        plt.ylabel('Acceleration')
        plt.title('Acceleration Analysis')
        plt.legend()
        plt.show(block=False)  # Show the plot without blocking

        # Plot for electrical properties (Voltage and Current)
        plt.figure(figsize=(10, 6))  # Create a new figure for electrical properties
        plt.plot(df['t'], df['U'], label="Voltage")
        plt.plot(df['t'], df['DXL_Current'], label="Current")
        plt.xlabel('Time')
        plt.ylabel('Electrical Properties')
        plt.title('Voltage and Current Analysis')
        plt.legend()
        plt.show(block=False)  # Show the plot without blocking

        # Keep the plots open
        plt.show()



        ke = 3.7
        # 3.8197, 2.6657
        # (u - ke*speed)*kt/r
        # plt.plot(df['DXL_Current'] * 2.6657, label="EM torque from I")
        # plt.plot((df['U'] - ke*df['DXL_Velocity'])*2.6657/5, label="current")
        # plt.plot((u - ke*v)*2.6657/5, label="current")
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
        

# plt.xlabel('Time (sample)')
# plt.ylabel('Current [mA]')
# plt.title(f'Measured current')
# plt.legend()
# plt.show()


        # # Perform FFT
        # fft_result = np.fft.fft(data)
        # frequencies = np.fft.fftfreq(len(fft_result))

        # # Plot the frequencies
        # plt.figure(figsize=(12, 6))
        # plt.plot(frequencies, np.abs(fft_result))
        # plt.title('Frequency Domain of the Data')
        # plt.xlabel('Frequency')
        # plt.ylabel('Amplitude')
        # plt.show()

        # plot_measured_I(df)
        # plot_measured_q(df)
        # plot_measured_q_dot(df)
        # plot_measured_U(df)
