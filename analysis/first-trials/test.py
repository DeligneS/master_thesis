import numpy as np
import matplotlib.pyplot as plt

r = [0.128, 1.2465, 0., 0., 0.128, 0.07]
r = [0.11741464750529594, 1.2380507798987224, 0.38065022439819174, 0.11755681827219377, 0.12876092204679299, 0.029231003865808534]

# Parameters for demonstration
f = r[1]        # Viscous friction coefficient [N⋅m/(rad/s)]
tau_c = r[0]   # Coulomb friction torque [N⋅m]
w_brk = r[5]    # Breakaway friction velocity [rad/s]
tau_brk = r[4]  # Breakaway friction torque [N⋅m]

# Calculated parameters
str_scale = np.sqrt(2 * np.exp(1)) * (tau_brk - tau_c)
w_st = w_brk * np.sqrt(2)
w_coul = w_brk / 10

# Relative angular velocity range
w_rel = np.linspace(-1, 1, 50000)

# Friction torque equation
tau = str_scale * (np.exp(-(w_rel / w_st)**2) * w_rel / w_st) + \
      tau_c * np.tanh(w_rel / w_coul) + \
      f * w_rel

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(w_rel, tau, label='Friction Torque', color='blue')
plt.xlabel('Relative Angular Velocity [$rad/s$]')
plt.ylabel('Friction Torque [$N⋅m$]')
plt.title('Friction Torque vs. Relative Angular Velocity')
plt.grid(True)
plt.legend()
plt.show()

# Plotting with breakaway torque highlighted
plt.figure(figsize=(10, 6))
plt.plot(w_rel, tau, label='Friction Torque', color='blue')
plt.axhline(y=tau_brk, color='red', linestyle='--', label='Breakaway Torque ($\\tau_{brk}$)')

plt.xlabel('Relative Angular Velocity [$rad/s$]')
plt.ylabel('Friction Torque [$N⋅m$]')
plt.title('Friction Torque vs. Relative Angular Velocity with Breakaway Torque')
plt.grid(True)
plt.legend()
plt.show()
