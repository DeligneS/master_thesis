import cma
import numpy as np
from src.objective_functions import model_error
from src.data_processing import process_file
from src.model_fitting import (
    cma_pos_on_model, cma_free_on_model, 
    cma_free_on_model_folder, 
    cma_free_on_model_folder_no_friction,
    cmaes_free_on_model_folder
)

# Import the experiment's data

#file_path = 'data/modelDXL_data_22_12_processed/exp22_12_M4_slow.txt'
file_path = 'data/modelDXL_data_22_12_processed/exp22_12_Msolo_fast_20ms.txt'

# Choose the right delta t
dt = 20e-3
# dt = 40e-3
delta_t = 4*dt # In the database : if it is written 40ms, then the 'real' rate is 4 times this value

"""
Pre-process the data, the output is in the following format :
columns=['t', 'DXL_PWM', 'DXL_Current', 'DXL_Velocity', 'DXL_Position', 'DXL_Input_Voltage']
units = ['raw', '[%]', '[A]', '[rad/s]', '[rad]', '[V]']
"""
df = process_file(file_path)


"""
parameters_to_find -> using CMA-ES
k_tau, R, q_dot_s, tau_c, tau_s, c_v, motor_inertia  = parameters_to_find
Should be like :

c_v = 0.0811          # Viscous friction coefficient
q_dot_s = 0.195       # Transition velocity coefficient
τ_c = 0.1174262       # Coulomb friction torque at transition speed coefficient
τ_s = 0.1212          # Static friction torque coefficient
i0 = 0.00353          # kg.m**2. Measured moment of inertia when empty (gear box)
r = 4.07              # Ohm. Datasheet says 5.86 ohm but is not reliable
ke = 1.399            # V.s/rad (voltage/rotational speed)
or
ke = 1.6 #V*s/rad
we can take as first guess the values of the Dynaban project for the MX64
parameters_to_find = [1.399, 4.07, 0.195, 0.1174262, 0.1212, 0.0811, 0.00353]
"""

"""
parameters_fixed
k_gear, external_inertia  = parameters_fixed
k_b : back-emf constant 
   -> the back EMF constant and the torque constant are equal due to the physical construction of the motor,
      but they are defined in different units.
external_inertia : RigidBodyDynamics.jl can compute it
"""
parameters_fixed = 353.5, 0.0188598 # for Dynamixel XH430-W350 on the hips -> M1 & M2
#parameters_fixed = 212.6, 0.00235245 # for Dynamixel XH430-W210 on the knees -> M3 & M4
parameters_fixed = 353.5, 0         # for Dynamixel XH... without external inertia -> Msolo


# We use CMA-ES for model fitting https://cma-es.github.io 
# Define initial mean and standard deviation for parameters

initial_mean = [0.5] * 7  # Adjust based on the number of parameters in your model
initial_mean = [1e-5] * 7
initial_mean = [0] * 7

# First guess : Dynaban values
initial_mean = [1.51294795, 4.39798614, 0.080964165, 0.0665140867408596, 0.203719639, 0.04217117990939209, 0.0155818308]
# initial_mean = [0.7760734059332939, 53.08438534503578, 0.3643458235916052, 1.53583629850428e-11, 0.050638932514598695, 0.0012799212082675578, 0.09997941698830523]
# initial_mean = [0.6923936804530714, 32.20489150344485, 0.0025740880638445963, 0.00013630955898567978, 2.1025323811096435, 0.0016890327193859178, 1.351235650475586]
# initial_mean = [4.39798614, 0.080964165, 0.0665140867408596, 0.203719639, 0.04217117990939209]
# initial_mean = [0.080964165, 0.0665140867408596, 0.203719639, 0.04217117990939209]


initial_std = 0.1  # Standard deviation
# initial_std = [0.5, 0.5, 0.05, 0.05, 0.5, 0.05, 0.05]
# Set up and run the CMA-ES algorithm
lower_bounds = [0] * 7  # n is the number of parameters
# lower_bounds[0] = 0.3
# initial_mean[0] = 0.37536231884057975
# upper_bounds = [100, 1000, 1, 1, 1, 10, 0.1]
# upper_bounds = [None, None, None, None, None, None, 0.05]


# initial_mean = [1.51294795, 4.39798614]

options = {'maxiter': 1000, 'tolx': 1e-4, 'tolfun': 1e-4, 'bounds' : [lower_bounds, None]}  # Adjust these options as needed
# options = {'maxiter': 1000, 'tolx': 1e-4, 'tolfun': 1e-4}

# cma_free_on_model(df, parameters_fixed, delta_t, 
#                   with_current=False, 
#                   initial_mean=initial_mean, 
#                   initial_std=initial_std, 
#                   options=options
#                   )

#folder_path = 'data/modelDXL_data_22_12_processed'
folder_path = 'data/exp20_01'
file_name = 'data/exp12_01/exp12_01_Msolo_sinus_fast_20ms.txt'
# fixed_params = [0.080964165, 0.0665140867408596, 0.203719639, 0.04217117990939209, 0.0155818308]



# premier test : prediction sur tension, validation sur position
# cma_free_on_model_folder(folder_path,
#                   # fixed_params,               
#                   with_current=False,
#                   using_datasheet_param = False,
#                   initial_mean=initial_mean, 
#                   initial_std=initial_std, 
#                   options=options, 
#                   dxl_model = ["Msolo", "M1", "M2"]
#                   # dxl_model = ["M3", "M4"]
#                   )


# # Other python package
# cmaes_free_on_model_folder(folder_path,              
#                   with_current=False,
#                   using_datasheet_param = False,
#                   initial_mean=initial_mean, 
#                   initial_sigma=initial_std, 
#                   bounds=[lower_bounds, None], 
#                   # dxl_model = ["Msolo", "M1", "M2"]
#                   # dxl_model = ["M3", "M4"]
#                   )




file_path = "data/exp08_02/cst/exp08_02_Msolo_pwm_cst_20min.txt"
df = process_file(file_path)
parameters_fixed = 353.5, 0.0022421143208         # for Dynamixel XH430-W350 with external inertia -> Msolo
dt = 20e-3
# dt = 40e-3
delta_t = 4*dt # In the database : if it is written 40ms, then the 'real' rate is 4 times this value



cma_free_on_model(df, parameters_fixed, delta_t, 
                  with_current=False, 
                  using_datasheet_param=False, 
                  initial_mean=initial_mean, 
                  initial_std=initial_std, 
                  options=options
                  )

# [0.1394365882433089, 5.502297769546699, 0.002852982486073203, 2.2457719146637638e-05, 0.3666787288950554, 0.00027217610370829364, 0.7845443466646911]
# [0.13261174824116306, 4.684333828268925, 0.0007180404224726566, 6.247739850661171e-05, 0.49568345775590006, 0.002246854889104875, 1.3175771617888046]