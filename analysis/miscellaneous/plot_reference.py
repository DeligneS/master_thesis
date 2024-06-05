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
from src.data_processing import process_file
from src.prediction import model_prediction, schwarz_model_prediction, new_model_prediction

full_path1 = "data/used_reference_trajectories/walkingPattern_ref_slow.csv"
df1 = pd.read_csv(full_path1)

# plt.plot(df1['q1_l'] * 180/np.pi + 90)
plt.plot(df1['q1_r'])

plt.xlabel('Torque (Nm)')
plt.ylabel('Current (A)')
plt.title('Torque vs. Current')
plt.grid(True)
plt.show()
