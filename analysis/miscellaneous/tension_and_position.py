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

full_path = "data/Xing_trajectories/exp10_01_M1_fast.txt"
parameters_fixed = 353.5, 0.0188598

df = process_file(full_path)
model_prediction(df, PARAMETERS_TO_FIND, parameters_fixed, 4*40e-3)

# df.to_csv("prediction_cas1.csv")
# plt.plot(df['U'])
plt.plot(df['q_dot_pred'])
plt.plot(df['DXL_Velocity'])
plt.xlabel('Speed (rad/s)')
plt.ylabel('Torque (Nm)')
plt.title('Speed vs. Torque')
plt.grid(True)
plt.show()
