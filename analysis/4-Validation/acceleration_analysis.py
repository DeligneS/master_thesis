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

# Select the measurements
file_name = "chirp_1"
file_name = "non_trivial_2"

file_path = f"data/validation_exp/calibration/{file_name}.txt"
file_path = "/Users/simondeligne/Dionysos/thesis_model_dxl/data/used_reference_trajectories/Xing_trajectories/measures_interpolated/pwm_ctrl_fast.txt"
file_path = "/Users/simondeligne/Dionysos/thesis_model_dxl/data/used_reference_trajectories/Xing_trajectories/measures_interpolated/pwm_ctrl_slow.txt"

df = process_file(file_path, delta_t=0.02)

df['DXL_Position'] = df['DXL_Position'] - np.pi/2
df['DXL_Position'] = (df['DXL_Position'] * 180/np.pi - 1.6) * np.pi/180

# plot_measured_q(df)
# plot_measured_U(df)

df = df.rename(columns={'t':'timestamp', 'DXL_Position':'θ(t)'})
csv_filename = f'data/validation_exp/calibration_processed/{file_name}.csv'
# df.to_csv(csv_filename, index=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import spectrogram, bode
from scipy.signal import cwt, ricker

# Ensure 'U' and 'θ(t)' are numeric and sorted by 'timestamp'
df.sort_values('timestamp', inplace=True)
df['U'] = pd.to_numeric(df['U'], errors='coerce')
df['θ(t)'] = pd.to_numeric(df['θ(t)'], errors='coerce')

file_names = ["exp2_square", "exp2_tooth", "exp2_triangle", "exp3_square_1_5", "exp3_tooth_1_5", "exp3_triangle_1_5", "pwm_ctrl_slow", "pwm_ctrl_fast"]
base_path = "/Users/simondeligne/Dionysos/thesis_model_dxl/data/validation_exp/acceleration_analysis/"
file_name = file_names[4]
# Process real data
real_data_path = f"{base_path}real_data/{file_name}.csv"
df = pd.read_csv(real_data_path)

# Calculate sampling interval and frequency
T = np.mean(np.diff(df['timestamp']))
fs = 1.0 / T

# Time-domain plot
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(df['timestamp'], df['U'], label='Input Signal')
# plt.title('Time-domain Signal')
plt.ylabel('Voltage [V]')
plt.legend()
plt.xticks([0, 33, 66, 100, 133, 166, 200])
plt.yticks([-3, -2, -1, 0, 1, 2, 3])
plt.grid(which='major', axis='x', linestyle='--', color='gray')
plt.grid(which='major', axis='y', linestyle='--', color='gray')

plt.subplot(2, 1, 2)
plt.plot(df['timestamp'], df['θ(t)'], label='Output Signal', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('Position [rad]')
plt.legend()
plt.tight_layout()
plt.xticks([0, 33, 66, 100, 133, 166, 200])
plt.yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
plt.grid(which='major', axis='x', linestyle='--', color='gray')
plt.grid(which='major', axis='y', linestyle='--', color='gray')

plt.show()


from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Filter requirements.
order = 6
fs = 1.0 / np.mean(np.diff(df['timestamp']))      # sample rate, Hz
cutoff = 1  # desired cutoff frequency of the filter, Hz

# Apply the filter to the position data
position_smoothed = butter_lowpass_filter(df['θ(t)'], cutoff, fs, order)

# Time-domain plot
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 2)
plt.plot(df['timestamp'], df['θ(t)'], label='Output Signal', color='orange')
plt.plot(df['timestamp'],position_smoothed, label='Output Signal smoothed')
plt.xlabel('Time [s]')
plt.ylabel('Position [rad]')
plt.legend()
plt.tight_layout()
plt.xticks([0, 33, 66, 100, 133, 166, 200])
plt.yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
plt.grid(which='major', axis='x', linestyle='--', color='gray')
plt.grid(which='major', axis='y', linestyle='--', color='gray')

plt.show()


# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot original and filtered data
axs[0].plot(df['timestamp'], df['θ(t)'], label='Original', alpha=0.7)
axs[0].plot(df['timestamp'], position_smoothed, label='Filtered', alpha=0.7)
axs[0].set_title('Original vs. Filtered Position')
axs[0].legend()

# Plot the difference
difference = df['θ(t)'] - position_smoothed
axs[1].plot(df['timestamp'], difference, label='Difference')
axs[1].set_title('Difference Between Original and Filtered')
axs[1].legend()

# Statistics
mean_diff = np.mean(difference)
std_diff = np.std(difference)
max_diff = np.max(np.abs(difference))
stats_text = f'Mean Difference: {mean_diff}\nStd Deviation: {std_diff}\nMax Difference: {max_diff}'
axs[2].text(0.5, 0.5, stats_text, fontsize=12, ha='center')
axs[2].axis('off')
axs[2].set_title('Statistical Summary of Differences')

# Display all plots
plt.tight_layout()
plt.show()






# Differentiate the smoothed position to get velocity
velocity_from_smoothed_position = np.gradient(position_smoothed, df['timestamp'])

# Differentiate the original position data to get velocity for comparison
velocity_from_original_position = np.gradient(df['θ(t)'], df['timestamp'])

# Plot the velocities for comparison
plt.figure(figsize=(14, 7))

# Velocity computed from smoothed position
plt.plot(df['timestamp'], velocity_from_smoothed_position, label='Velocity derived from Filtered Position', color='red')

# Velocity computed from original position
plt.plot(df['timestamp'], velocity_from_original_position, label='Velocity derived from Measured Position', color='green', linestyle='--')

# Actual velocity measurements (assuming 'DXL_Velocity' is the column with these measurements)
plt.plot(df['timestamp'][2:]-df['timestamp'][2], df['DXL_Velocity'][2:], label='Actual Measured Velocity', color='blue', alpha=0.5)

plt.xlabel('Time [s]')
plt.ylabel('Velocity [rad/s]')
plt.title('Comparison of Velocity Computations')
plt.legend()
plt.grid(True)
plt.show()


# Compute acceleration from the velocity derived from the filtered/smoothed position
acceleration_from_filtered_velocity = np.gradient(velocity_from_smoothed_position, df['timestamp'])

# Plot the computed acceleration
plt.figure(figsize=(14, 7))
plt.plot(df['timestamp'], acceleration_from_filtered_velocity, label='Acceleration from Filtered Velocity', color='green')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [rad/s²]')
plt.title('Derived Acceleration from Filtered Position Measures')
# plt.legend()
plt.grid(True)
plt.show()


file_path = "/Users/simondeligne/Dionysos/thesis_model_dxl/data/used_reference_trajectories/Xing_trajectories/measures_interpolated/simu_fast_smooth_tsit.csv"
file_path = "/Users/simondeligne/Dionysos/thesis_model_dxl/data/used_reference_trajectories/Xing_trajectories/measures_interpolated/simu_slow_smooth_tsit.csv"

sim_df = pd.read_csv(file_path)

# Process simulation data
simulation_path = f"{base_path}simulation/{file_name}.csv"
sim_df = pd.read_csv(simulation_path)


from scipy.signal import butter, filtfilt

# Define the Butterworth filter parameters
order = 4
cutoff = 25  # Assuming you want to cut off frequencies above 5 Hz

# Calculate the sampling frequency of the simulated data
fs_sim = 1.0 / np.mean(np.diff(sim_df['time']))

# Design the Butterworth filter
nyq = 0.5 * fs_sim
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)

# Apply the filter to the simulated acceleration
# sim_df['acc_sim'] = filtfilt(b, a, sim_df['acc_sim'])


# Interpolating simulated acceleration onto the real dataset's time stamps
acc_sim_interpolated = np.interp(df['timestamp'], sim_df['time'], sim_df['acc_sim'])
# Assuming acceleration_from_filtered_velocity is the real acceleration computed earlier
error_acc = acceleration_from_filtered_velocity - acc_sim_interpolated


# Calculate NRMSE
nrmse = np.sqrt(np.mean(error_acc ** 2)) / (np.max(acceleration_from_filtered_velocity) - np.min(acceleration_from_filtered_velocity))

# Define a threshold for the percentage fit calculation, e.g., within 10% of the max acceleration range
threshold = 0.1 * (np.max(acceleration_from_filtered_velocity) - np.min(acceleration_from_filtered_velocity))
percentage_fit = np.mean(np.abs(error_acc) < threshold) * 100

def fit_percentage(y_true, y_pred):
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    fit_perc = (1 - numerator/denominator) * 100
    return fit_perc

R2 = fit_percentage(acceleration_from_filtered_velocity, acc_sim_interpolated)
# Print NRMSE and Percentage Fit for verification
print(f"NRMSE: {nrmse:.4f}")
print(f"Percentage Fit within {threshold:.4f} rad/s² threshold: {percentage_fit:.2f}%")
print(f"Percentage Fit R-squared : {R2}%")


# Plotting
plt.figure(figsize=(15, 6))

# Plot Real vs. Filtered & Interpolated Simulated Acceleration
plt.plot(df['timestamp'], acceleration_from_filtered_velocity, label='Real Acceleration', color='green')
plt.plot(sim_df['time'], sim_df['acc_sim'], label='Simulated Acceleration', color='tab:purple')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [rad/s²]')
plt.title('Comparison of Accelerations')

# Annotations for NRMSE and Percentage Fit
metrics_text = f'NRMSE: {nrmse:.4f}\nPercentage Fit: {percentage_fit:.2f}% within {threshold:.2f} rad/s²\nR-squared : {R2:.2f}%'
plt.annotate(metrics_text, xy=(0.05, 0.15), xycoords='axes fraction', verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5), fontsize=9)

plt.legend(loc='lower right')

plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Error Distribution
plt.figure(figsize=(15, 6))
plt.hist(error_acc, bins=100, color='gray', alpha=0.7)
plt.xlabel('Error [rad/s²]')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.grid(True)
plt.xlim(-5, 5)
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(df['timestamp'], df['θ(t)'])
plt.plot(sim_df['time'], sim_df['pos_sim'])
plt.show()


# # Calculate the derivative of the position data to obtain velocity
# # Using the central difference method for all interior points and forward/backward difference for the endpoints
# velocity_calculated = np.gradient(df['θ(t)'], df['timestamp'])

# # Plot measured velocity vs. calculated velocity
# plt.figure(figsize=(14, 7))

# # Measured velocity
# plt.plot(df['timestamp'], df['DXL_Velocity'], label='Measured Velocity', color='blue')

# # Calculated velocity from position data
# plt.plot(df['timestamp'], velocity_calculated, label='Calculated Velocity from θ(t)', color='green', linestyle='--')

# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [rad/s]')
# plt.title('Measured vs. Calculated Velocity')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()


# from scipy.signal import savgol_filter

# # Calculate the mean of the measured and calculated velocities
# mean_velocity = (df['DXL_Velocity'] + velocity_calculated) / 2

# # Smooth the mean velocity using the Savitzky-Golay filter
# # Window length chosen as 51 and polynomial order as 3, adjust these parameters as needed
# window_length, poly_order = 51, 3
# smoothed_mean_velocity = savgol_filter(mean_velocity, window_length, poly_order)

# # Ensure the window length is odd and less than the size of the signal
# if window_length >= len(mean_velocity) or window_length % 2 == 0:
#     print("Adjust the window length to be odd and less than the size of the signal.")

# # Plot raw mean velocity and smoothed mean velocity
# plt.figure(figsize=(14, 7))
# plt.plot(df['timestamp'], mean_velocity, label='Mean Velocity', color='purple')
# plt.plot(df['timestamp'], smoothed_mean_velocity, label='Smoothed Mean Velocity', color='red', linestyle='--')

# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [rad/s]')
# plt.title('Mean vs. Smoothed Mean Velocity')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # Assuming 'df', 'smoothed_mean_velocity', and 'T' (sampling interval) are already defined

# # Integrate the smoothed mean velocity to get the position
# # Using the cumulative sum method to approximate integration
# position_from_smoothed_velocity = np.cumsum(smoothed_mean_velocity) * T

# # Adjust the integrated position so that its initial value matches the original position data
# position_from_smoothed_velocity += df['θ(t)'].iloc[0] - position_from_smoothed_velocity[0]

# # Plot the comparison
# plt.figure(figsize=(14, 7))
# plt.plot(df['timestamp'], df['θ(t)'], label='Original Position Evolution', color='orange')
# plt.plot(df['timestamp'], position_from_smoothed_velocity, label='Position from Smoothed Velocity', color='cyan', linestyle='--')

# plt.xlabel('Time [s]')
# plt.ylabel('Position [rad]')
# plt.title('Position Evolution: Original vs. From Smoothed Velocity')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()


# # Calculate the differences between the original and reconstructed positions
# differences = df['θ(t)'] - position_from_smoothed_velocity

# # Calculate the Normalized Root Mean Square Error (NRMSE)
# rmse = np.sqrt(np.mean(differences ** 2))
# nrmse = rmse / (df['θ(t)'].max() - df['θ(t)'].min())

# print(f"NRMSE: {nrmse:.4f}")

# # To calculate fit in percentage, we could consider explained variance
# # However, this requires a model-based approach. Instead, let's look at how often the differences are within a threshold
# # Define a reasonable threshold for your application
# threshold = 0.05 * (df['θ(t)'].max() - df['θ(t)'].min())  # For example, 5% of the range of the observed data
# fit_within_threshold = np.mean(np.abs(differences) < threshold) * 100  # Percentage of points within the threshold

# print(f"Fit within ±{threshold:.4f} rad: {fit_within_threshold:.2f}%")




# # Differentiate the smoothed velocity to get acceleration
# acceleration_from_smoothed_velocity = np.gradient(smoothed_mean_velocity, df['timestamp'])

# # Plot acceleration
# plt.figure(figsize=(14, 7))
# plt.plot(df['timestamp'], acceleration_from_smoothed_velocity, label='Acceleration from Smoothed Velocity', color='red')
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [rad/s²]')
# plt.title('Acceleration from Smoothed Velocity')
# plt.legend()
# plt.grid(True)
# plt.show()


# # Integrate the acceleration to get velocity
# velocity_from_acceleration = np.cumsum(acceleration_from_smoothed_velocity) * T

# # Adjust the integrated velocity so that its initial value matches the smoothed velocity data's initial value
# velocity_from_acceleration += smoothed_mean_velocity[0] - velocity_from_acceleration[0]

# # Integrate the velocity to get position
# position_from_acceleration = np.cumsum(velocity_from_acceleration) * T

# # Adjust the integrated position so that its initial value matches the original position data's initial value
# position_from_acceleration += df['θ(t)'].iloc[0] - position_from_acceleration[0]



# # Plot comparison
# plt.figure(figsize=(14, 7))
# plt.plot(df['timestamp'], df['θ(t)'], label='Original Position Evolution', color='orange')
# plt.plot(df['timestamp'], position_from_smoothed_velocity, label='Position from Smoothed Velocity', color='cyan', linestyle='--')
# plt.plot(df['timestamp'], position_from_acceleration, label='Position from Integrated Acceleration', color='green', linestyle='-.')
# plt.xlabel('Time [s]')
# plt.ylabel('Position [rad]')
# plt.title('Position Evolution: Comparison')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# # Zero-centering the acceleration data
# acceleration_centered = acceleration_from_smoothed_velocity - np.mean(acceleration_from_smoothed_velocity)

# # Placeholder for simplicity; replace with your actual acceleration data and timestamps
# acceleration = acceleration_centered  # Assuming this is your corrected acceleration data
# timestamps = df['timestamp']

# # Integrate acceleration to get velocity and position
# velocity = np.cumsum(acceleration) * T
# position_integrated = np.cumsum(velocity) * T

# # Adjust the integrated position using checkpoints
# # For demonstration, let's adjust every N data points (you can choose intervals based on your data)
# N = 50  # Example interval; adjust based on your dataset
# for i in range(N, len(position_integrated), N):
#     actual_position_at_checkpoint = df['θ(t)'].iloc[i]
#     integrated_position_at_checkpoint = position_integrated[i]
#     correction_factor = actual_position_at_checkpoint - integrated_position_at_checkpoint
#     # Apply correction factor to subsequent points until the next checkpoint
#     end_range = i + N if i + N < len(position_integrated) else len(position_integrated)
#     for j in range(i, end_range):
#         position_integrated[j] += correction_factor

# # Plotting the corrected position against the original
# plt.figure(figsize=(14, 7))
# plt.plot(timestamps, df['θ(t)'], label='Original Position', color='orange')
# plt.plot(timestamps, position_integrated, label='Corrected Integrated Position', color='green', linestyle='-.')
# plt.xlabel('Time [s]')
# plt.ylabel('Position [rad]')
# plt.title('Corrected Integrated Position vs. Original Position')
# plt.legend()
# plt.grid(True)
# plt.show()




# from scipy.signal import butter, filtfilt

# # Define the filter
# def butter_bandpass(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a

# def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = filtfilt(b, a, data)
#     return y

# # Apply the filter
# # Adjust 'lowcut' and 'highcut' to cover the frequency range of interest with some margin
# lowcut = 0.02  # Slightly below the lowest frequency of interest
# highcut = 5  # Slightly above the highest frequency of interest
# order = 4  # Filter order; adjust based on the required sharpness of the cutoff

# smoothed_mean_velocity_filtered = butter_bandpass_filter(mean_velocity, lowcut, highcut, fs, order)

# # Plot the filtered smoothed velocity
# plt.figure(figsize=(14, 7))
# plt.plot(df['timestamp'], mean_velocity, label='Smoothed Velocity', color='blue', alpha=0.5)
# plt.plot(df['timestamp'], smoothed_mean_velocity_filtered, label='Filtered Smoothed Velocity', color='red')
# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [rad/s]')
# plt.title('Filtered Smoothed Velocity')
# plt.legend()
# plt.grid(True)
# plt.show()
