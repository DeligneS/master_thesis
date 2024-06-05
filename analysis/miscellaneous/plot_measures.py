import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
dependencies_path = os.path.join(current_dir, '..')
sys.path.append(dependencies_path)

from src.data_processing import process_file
from src.plotting import (
    plot_measured_I, 
    plot_measured_q, 
    plot_measured_q_dot,
    plot_measured_U
)

choice = 1

if choice == 1:
    file_path1 = "data/Xing_trajectories/exp10_01_M1_fast.txt"
    file_path2 = "data/Xing_trajectories/exp10_01_M1_slow.txt"
if choice == 2:
    file_path1 = "data/modelDXL_data_22_12_processed/exp22_12_M2_slow.txt"
    file_path2 = "data/modelDXL_data_22_12_processed/exp22_12_M2_slow.txt"
if choice == 3:
    file_path1 = "data/Xing_trajectories/exp10_01_M3_fast.txt"
    file_path2 = "data/Xing_trajectories/exp10_01_M3_slow.txt"
if choice == 4:
    file_path1 = "data/Xing_trajectories/exp22_12_M4_fast.txt"
    file_path2 = "data/Xing_trajectories/exp22_12_M4_slow.txt"
if choice == 5:
    file_path1 = "data/Xing_trajectories/exp22_12_Msolo_fast_20ms.txt"
    file_path2 = "data/Xing_trajectories/exp22_12_Msolo_slow_20ms.txt"
if choice == 6:
    file_path1 = "data/Xing_trajectories/exp22_12_Msolo_fast_40ms.txt"
    file_path2 = "data/Xing_trajectories/exp22_12_Msolo_slow_40ms.txt"

file_path1 = "data/exp10_01/exp12_01_Msolo_sinus_fast_20ms.txt"
file_path2 = "data/exp19_01/exp19_01_Msolo_pwm_cst_442.txt"
file_path1 = file_path2
df = process_file(file_path1)

# df2 = process_file(file_path2)
df2 = None
plot_measured_I(df, df2)
plot_measured_q(df, df2)
plot_measured_q_dot(df, df2)
plot_measured_U(df, df2)
