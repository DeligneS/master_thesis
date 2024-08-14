from src.data_processing import process_file
from src.plotting import (
    plot_measured_I, 
    plot_measured_q, 
    plot_measured_q_dot,
    plot_measured_U
)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft
from scipy.signal import spectrogram, bode
from scipy.signal import cwt, ricker
import matplotlib as mpl

# Set the font size globally
mpl.rcParams.update({'font.size': 18})  # Change 14 to your desired font size

df_ref = pd.read_csv('data/used_reference_trajectories/non_trivial_ref_current_hip.csv')

def load_my_data(file_name):
    # file_path = f"data/validation_exp/calibration/{file_name}.txt"
    file_path = f"data/validation_exp/mesures/{file_name}.txt"

    df = process_file(file_path, delta_t=0.02)

    df['DXL_Position'] = df['DXL_Position'] - np.pi/2
    df['DXL_Position'] = (df['DXL_Position'] * 180/np.pi - 1.6) * np.pi/180

    # plot_measured_q(df)
    # plot_measured_U(df)
    # plot_measured_I(df)

    df = df.rename(columns={'t':'timestamp', 'DXL_Position':'θ(t)'})
    csv_filename = f'data/validation_exp/mesures/{file_name}.csv'
    df.to_csv(csv_filename, index=False)

    # Ensure 'U' and 'θ(t)' are numeric and sorted by 'timestamp'
    df.sort_values('timestamp', inplace=True)
    df['U'] = pd.to_numeric(df['U'], errors='coerce')
    df['θ(t)'] = pd.to_numeric(df['θ(t)'], errors='coerce')
    return df


file_name = "non_trivial_current" # dyn_210, dyn_210_2, non_trivial_current, dyn210_test1, dyn210_test2
df = load_my_data(file_name)
# df1 = load_my_data("dyn210_test2")

# Time-domain plot
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
# plt.plot(df['timestamp'], df['U'], label='Input Signal')
# plt.plot(df1['timestamp'], df1['U'], label='Input Signal')

# plt.plot(df['timestamp'], df['DXL_Current'], label='Input Signal')
plt.plot(df_ref['time'], df_ref['q1_l']*2.69/1000, label='Input Signal')
# plt.title('Time-domain Signal')
plt.ylabel('Current\n[A]', rotation=0, labelpad=30)
plt.legend()
# plt.xticks([0, 33, 66, 100, 133])
plt.xticks([0, 33, 66, 100, 133, 166, 200])
plt.xlim(0, 133)
plt.ylim(-0.15, 0.15)

# plt.yticks([-3, -2, -1, 0, 1, 2, 3])
plt.grid(which='major', axis='x', linestyle='--', color='gray')
plt.grid(which='major', axis='y', linestyle='--', color='gray')

plt.subplot(2, 1, 2)
plt.plot(df['timestamp'], df['θ(t)'], label='Output Signal', color='orange')
# plt.plot(df1['timestamp'], df1['θ(t)'], label='Output Signal', color='orange')

plt.xlabel('Time [s]')
plt.ylabel('Position\n[rad]', rotation=0, labelpad=30)
plt.legend()
plt.tight_layout()
# plt.xticks([0, 33, 66, 100, 133])
plt.xticks([0, 33, 66, 100, 133, 166, 200])
plt.xlim(0, 133)
plt.yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
plt.grid(which='major', axis='x', linestyle='--', color='gray')
plt.grid(which='major', axis='y', linestyle='--', color='gray')

plt.show()

# Plot measured pwm signal
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
# plt.plot(df['timestamp'], df['DXL_Current'], label='Input Signal')
plt.plot(df_ref['time'], df_ref['q1_l']*2.69/1000, label='Reference Signal')
# plt.title('Time-domain Signal')
plt.ylabel('Current [A]')
plt.legend()
plt.xticks([0, 33, 66, 100, 133])
plt.xlim(0, 133)
# plt.yticks([-3, -2, -1, 0, 1, 2, 3])
plt.grid(which='major', axis='x', linestyle='--', color='gray')
plt.grid(which='major', axis='y', linestyle='--', color='gray')

plt.subplot(2, 1, 2)
plt.plot(df['timestamp'], df['U'], label='Measured Voltage Signal', color='green')
plt.xlabel('Time [s]')
plt.ylabel('Measured\nvoltage\n[V]', rotation=0, labelpad=30)
plt.legend()
plt.xticks([0, 33, 66, 100, 133])
plt.xlim(0, 133)
plt.grid(which='major', axis='x', linestyle='--', color='gray')
plt.grid(which='major', axis='y', linestyle='--', color='gray')
plt.show()


# # Plot reference VS measured input signal
# df_ref = pd.read_csv('data/used_reference_trajectories/non_trivial_ref_current_hip.csv')
# # time,q1_l,q1_r,q2_l,q2_r
# plt.figure(figsize=(14, 7))
# # plt.subplot(2, 1, 1)
# plt.plot(df_ref['time']*0.996, df_ref['q1_l']*2.69/1000, label='Reference Signal')
# plt.plot(df['timestamp'], df['DXL_Current'], label='Measured Signal', color='orange')

# # plt.title('Time-domain Signal')
# plt.ylabel('Current [A]')
# plt.legend()
# plt.xticks([0, 33, 66, 100, 133])
# plt.xlim(0, 133)
# # plt.yticks([-3, -2, -1, 0, 1, 2, 3])
# plt.grid(which='major', axis='x', linestyle='--', color='gray')
# plt.grid(which='major', axis='y', linestyle='--', color='gray')

# # plt.subplot(2, 1, 2)
# # plt.plot(df['timestamp'], df['DXL_Current'], label='Measured Signal', color='orange')
# # plt.xlabel('Time [s]')
# # plt.ylabel('Current [A]')
# # plt.legend()
# # plt.xticks([0, 33, 66, 100, 133])
# # plt.xlim(0, 133)
# # # plt.yticks([-3, -2, -1, 0, 1, 2, 3])
# # plt.grid(which='major', axis='x', linestyle='--', color='gray')
# # plt.grid(which='major', axis='y', linestyle='--', color='gray')
# plt.show()

# Plot simulation with current
df_simu_I = pd.read_csv('data/validation_exp/mesures/current_ctrl_non_trivial.csv')
df_simu_I = pd.read_csv('data/validation_exp/mesures/pwm_ctrl_non_trivial.csv')

plt.figure(figsize=(14, 7))
plt.plot(df['timestamp'], df['θ(t)'], label='Measured Output', color='orange')
plt.plot(df_simu_I['time'], df_simu_I['pos_sim'], label='Simulated Output')

plt.xlabel('Time [s]')
plt.ylabel('Position\n[rad]', labelpad=30, rotation=0)
plt.legend()
plt.tight_layout()
plt.xticks([0, 33, 66, 100, 133])
plt.xlim(0, 133)
plt.yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
plt.ylim(-1, 1)
plt.grid(which='major', axis='x', linestyle='--', color='gray')
plt.grid(which='major', axis='y', linestyle='--', color='gray')

plt.show()

# Plot simulation with PWM
df_simu_pwm = pd.read_csv('data/validation_exp/mesures/pwm_ctrl_non_trivial.csv')

plt.figure(figsize=(14, 7))
plt.plot(df['timestamp'], df['θ(t)'], label='Output Signal', color='orange')
plt.plot(df_simu_pwm['time'], df_simu_pwm['pos_sim'], label='Output Signal')

plt.xlabel('Time [s]')
plt.ylabel('Position [rad]')
plt.legend()
plt.tight_layout()
plt.xticks([0, 33, 66, 100, 133])
plt.xlim(0, 133)
plt.yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
plt.grid(which='major', axis='x', linestyle='--', color='gray')
plt.grid(which='major', axis='y', linestyle='--', color='gray')

plt.show()
