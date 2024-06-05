import pandas as pd

import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('simu_slow_method1.csv')
data2 = pd.read_csv('simu_slow_method2.csv')
data3 = pd.read_csv('simu_slow_methodRB.csv')
data4 = pd.read_csv('simu_slow_methodCB.csv')
data5 = pd.read_csv('simu_slow_methodCB2.csv')
data6 = pd.read_csv('simu_slow_tsit_3.csv')


# Plot the elements
# plt.plot(data['time'], data['pos_sim'], label='Position')
# plt.plot(data['time'], data['vel_sim'], label='Velocity')
# plt.plot(data['time'], data['acc_sim'], label='Acceleration')
plt.plot(data2['time'], data2['acc_sim'], label='Acceleration2')
plt.plot(data3['time'], data3['acc_sim'], label='Acceleration3')
# plt.plot(data4['time'], data4['acc_sim'], label='Acceleration4')
plt.plot(data5['time'], data5['acc_sim'], label='Acceleration5')
plt.plot(data6['time'], data6['acc_sim'], label='Acceleration6')
# plt.plot(data3['time'], data3['input_tau'], label='Acceleration3')
# plt.plot(data3['time'], data3['tau_grav'], label='Acceleration3')
# plt.plot(data3['time'], data3['tau_f'], label='Acceleration3')
# plt.plot(data['time'], data['input'], label='Torque')

# plt.figure(figsize=(14, 7))
# # plt.title('Acceleration vs. Time')
# plt.subplot(2, 1, 1)
# plt.plot(data['time'], data['input'], label='Previous Input Voltage')
# plt.plot(data2['time'], data2['input'], label='New Input Voltage', color = 'tab:red')

# plt.xlim(2, 2.7)
# plt.ylim(-1, 2)
# plt.ylabel('Voltage [V]')
# plt.legend()
# plt.grid()

# plt.subplot(2, 1, 2)
# plt.plot(data['time'], data['acc_sim'], label='Acceleration Previous Method', color='green')
# plt.plot(data2['time'], data2['acc_sim'], label='Acceleration New Method', color='tab:purple')
# plt.xlim(2, 2.7)
# plt.ylim(-4, 2)
plt.legend()
# plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [rad/s²]')
plt.show()
