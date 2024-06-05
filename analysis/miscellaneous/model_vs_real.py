import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
dependencies_path = os.path.join(current_dir, '..')
sys.path.append(dependencies_path)

from src.data_processing import process_file, extract_dxl_model
from src.prediction import model_prediction, schwarz_model_prediction, new_model_prediction
from src.plotting import plot_data_q, plot_data_q_dot
from src.found_parameters.parameters import PARAMETERS_TO_FIND
import os

# parameters = [+, +, +, +, +, +, +]
# parameters = [k_tau, R, q_dot_s, tau_c, tau_s, c_v, motor_inertia]
# parameters for MX-106 : [0.19817, 0.49586, 0.13729, 0.03006] = [R/k_tau, (k_gear*k_tau + R*c_v/k_tau), tau_c*R/k_tau, tau_s*R/k_tau]

parameters = PARAMETERS_TO_FIND

"""
columns=['t', 'DXL_PWM', 'DXL_Current', 'DXL_Velocity', 'DXL_Position', 'DXL_Input_Voltage']
units = ['raw', '[%]', '[A]', '[rad/s]', '[rad]', '[V]']
"""

"""
delta_t need to be extracted from the database.
"""

# Folder containing the data files
folder_path = 'data/Xing_trajectories'
folder_path = 'data/exp19_01'
params = extract_dxl_model(folder_path)
print(params)
# Iterate over each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        full_path = os.path.join(folder_path, file_name)
        df = process_file(full_path)
        if df['DXL_Input_Voltage'].between(9.5, 16.0).all(): # else the collected data has errors
            motor_name = params[file_name]["motor_name"]
            # print(full_path)
            dt = 4e-3*float(params[file_name]["rate"])
            if (motor_name == "M1") or (motor_name == "M2"):
                parameters_fixed = 353.5, 0.0188598 # for Dynamixel XH430-W350 on the hips -> M1 & M2

            elif (motor_name == "M3") or (motor_name == "M4"):
                parameters_fixed = 212.6, 0.00235245 # for Dynamixel XH430-W210 on the knees -> M3 & M4

            elif (motor_name == "Msolo"):
                parameters_fixed = 353.5, 0         # for Dynamixel XH430-W350 without external inertia -> Msolo
            
            model_prediction(df, parameters, parameters_fixed, dt, with_current=False, model_identification=False)
            # df.to_csv("out.csv")
            # schwarz_model_prediction(df, parameters_fixed, dt)
            df['next_q'] = df['DXL_Position'].shift(-1)
            df['next_q_dot'] = df['DXL_Velocity'].shift(-1)
            plot_data_q(df)
            plot_data_q_dot(df)
