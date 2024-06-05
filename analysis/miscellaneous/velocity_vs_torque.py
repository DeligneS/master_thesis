import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
dependencies_path = os.path.join(current_dir, '..')
sys.path.append(dependencies_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.modelisation import friction_torque
from src.found_parameters.parameters import PARAMETERS_TO_FIND

# Define your model parameters
max_speed = 130 * (0.229) * (1/60) * 2*np.pi # from datasheet

num_points = 10000

# Create a DataFrame for a range of speeds
speeds = np.linspace(-max_speed, max_speed, num_points)
df = pd.DataFrame({'DXL_Velocity': speeds})
#q_dot_s, tau_c, tau_s, c_v, motor_inertia
PARAMETERS_TO_FIND = [1, 1, 0.080964165, 0.02, 0.0637042, 0.019323, 0.01] # DYNABAN VALUES
PARAMETERS_TO_FIND = [1, 1, 1.4346972207577002, 0.0004125317819209018, 0.09916091004229707, 0.0043450040003781215, 0.006879746042466658] # DYNABAN VALUES
PARAMETERS_TO_FIND = [1, 1, 0.12460979651121884, 1.6828569261486217e-07, 0.28473500620977843, 0.00404276701106158, 1.0155066637572764e-11]

# [1, 1, 0.12460979651121884, 0.0002759430569151436, 0.28473500620977843, 0.0008226599993493758, 0.0002586515292869222]
PARAMETERS_TO_FIND = [1, 1, 0.080964165, 0.0003854706405367776, 0.28473500620977843, 0.0008387596221714474, 0.0002645147933766714]
q_dot_s = 0.080964165
tau_c = 0.02
tau_s = 0.0637042
c_v = 0.019323

# Calculate the torque for each speed
friction_torque(df, PARAMETERS_TO_FIND)
k_gear = 353
# Plotting
plt.plot(df['DXL_Velocity'], df['tau_f'], label=r'$\tau_f$')

# Adding horizontal lines for tau_s and tau_c
plt.axhline(y=tau_s, color='g', linestyle='--', label=r'$\tau_s$')
# plt.axhline(y=tau_c, color='g', linestyle='--', label=r'$\tau_c$')
# plt.axvline(x=q_dot_s, color='b', linestyle=':', label=r'$\dot{q}_s$')

# plt.axhline(y=-tau_c, color='g', linestyle='--')  # tau_c is symmetric about the speed axis

# Adding a line for viscous friction with slope c_v
idx = int(num_points/2)
print(idx)
viscous_line = c_v * speeds[idx:] + tau_c*np.sign(speeds[idx:])
latex_string = r'$\tau_c + c^{(v)}\dot{q}$'
plt.plot(speeds[idx:], viscous_line, color='r', linestyle='-.', label=latex_string)

plt.xlabel('Speed (rad/s)')
plt.ylabel('Friction Torque (Nm)')
plt.title('Speed vs. Friction Torque')
plt.grid(True)
plt.legend()
plt.show()
