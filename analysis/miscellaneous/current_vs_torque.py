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


min_current = 0.084 # from datasheet
max_current = 1.3

num_points = 1000

# Create a DataFrame for a range of speeds
current = np.linspace(min_current, max_current, num_points)
df = pd.DataFrame({'DXL_Current': current})

# Calculate the torque for each speed
output_torque_motor_with_I(df, PARAMETERS_TO_FIND)

# Real curve : tau = 0.06 -> I = 0.084 A
# tau = 2.82 -> I = 1.12
# tau = m*I + b

# Given data points
tau_1, I_1 = 0.06, 0.084
tau_2, I_2 = 2.82, 1.12

# Calculating the slope (m)
m = (tau_2 - tau_1) / (I_2 - I_1)
print(m)

# Calculating the y-intercept (b)
b = tau_1 - m * I_1

df['tau_real'] = m*df['DXL_Current'] + b

# Plotting
plt.plot(df['tau_I'], df['DXL_Current'])
plt.plot(df['tau_real'], df['DXL_Current'])
plt.xlabel('Torque (Nm)')
plt.ylabel('Current (A)')
plt.title('Torque vs. Current')
plt.grid(True)
plt.show()