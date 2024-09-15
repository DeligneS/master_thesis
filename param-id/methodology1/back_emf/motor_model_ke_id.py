import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('dataframes_list.pkl', 'rb') as file:
    processed_dataframes = pickle.load(file)

plt.figure(figsize=(10, 6))
count = 0
all_experiment = processed_dataframes
for df in all_experiment:
    # Calculate Δx (change in position)
    delta_x = np.diff(df['DXL_Position'])
    delta_x = np.insert(delta_x, 0, 0)
    df['delta_x'] = delta_x

    delta_t = np.diff(df['t'])
    delta_t = np.insert(delta_t, 0, 0)
    df['delta_t'] = delta_t

    # Calculate velocity (Δx/Δt)
    df['Velocity_from_position'] = delta_x / delta_t
    plt.plot(df['Velocity_from_position'])
    count += 1

plt.xlabel('Time (sample)')
plt.ylabel('Voltage [V]')
plt.title(f'Measured Voltage')
plt.show()

plt.figure(figsize=(10, 6))
count = 0
for df in all_experiment:
    plt.plot(df['DXL_Current'])
    count += 1

plt.xlabel('Time (sample)')
plt.ylabel('Voltage [V]')
plt.title(f'Measured Voltage')
plt.show()