import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os
import re
from src.preprocessing.modelisation import output_torque_motor_with_I, friction_torque, output_torque_motor_with_U


def convert_decimal(euro_decimal):
    return float(euro_decimal.replace(',', '.'))


def correct_current_values(current_ma):
    """
    Corrects the current values assuming they are signed 16-bit integers and return the current in [A]
    """
    # Convert from mA to raw value
    raw_value = int(current_ma / 2.69)

    # Interpret as signed 16-bit integer
    if raw_value >= 32768:  # 2^15, for the sign bit
        raw_value = raw_value - 65536  # 2^16, to get the correct negative value

    # Convert back to mA
    corrected_current_ma = raw_value * 2.69

    # Convert to [A]
    current = corrected_current_ma / 1000
    return current


def correct_pwm_values(pwm_raw):
    """
    Corrects the pwm values assuming they are signed 16-bit integers and return the duty cycle in [%]
    """
    
    raw_value = pwm_raw

    # Interpret as signed 16-bit integer
    if raw_value >= 32768:  # 2^15, for the sign bit
        raw_value = raw_value - 65536  # 2^16, to get the correct negative value
    
    # if raw_value > 885 :
    #     print("The file contains wrong PWM values (too high).\n")

    duty_cycle = (raw_value * 0.113)/100       # [raw] * [%/raw]

    return duty_cycle


def process_file(file_path, delta_t=80e-3):
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = [line.strip().split('\t') for line in lines if line.strip()]
    data_rows = data[3:]

    df = pd.DataFrame(data_rows, columns=['t', 'DXL_PWM', 'DXL_Current', 'DXL_Velocity', 'DXL_Position', 'DXL_Input_Voltage'])
    df['t'] = df['t'].astype(int)
    df['t'] = df['t'] * delta_t

    for col in ['DXL_PWM', 'DXL_Current', 'DXL_Velocity', 'DXL_Position', 'DXL_Input_Voltage']:
        df[col] = df[col].apply(convert_decimal)

    # Apply the correction to the DataFrame
    df["DXL_Current"] = df["DXL_Current"].apply(correct_current_values)
    df["DXL_PWM"] = df["DXL_PWM"].apply(correct_pwm_values)
    df['DXL_Position'] = df['DXL_Position'] * 0.088 * np.pi/180  # [deg/pulse] * [rad/deg]
    df['DXL_Velocity'] = df['DXL_Velocity'] * (1/60) * 2*np.pi   # [rev/min] * [min/s] * [rad/rev] = [rad/s] (NB : V_max = pi [rad/s] (datasheet at 12V))
    df = replace_outliers(df)
    df['U'] = df['DXL_PWM'] * df['DXL_Input_Voltage']
    return df


def process_file_from_wizard(file_path):
    # Read the CSV file, skipping the first seven lines of metadata
    # Assuming the actual data starts from the 8th line
    df = pd.read_csv(file_path, skiprows=7)
    df["DXL_Current"] = df["Present Current"] * 2.69 / 1000 # raw to A
    df['DXL_PWM'] = df['Present PWM'] * 0.113/100
    try :
        df['DXL_Input_Voltage'] = df['Present Voltage']
    except:
         df['DXL_Input_Voltage'] = 12
    df["U"] = df["DXL_PWM"] * df['DXL_Input_Voltage']
    df['DXL_Velocity'] = df['Present Velocity'] * 0.229 * (1/60) * 2*np.pi
    df['DXL_Position'] = df['Present Position'] * 0.088 * np.pi/180
    df['t'] = df['Time[ms]'] / 1000 # in s

    delta_t = np.diff(df['t'])
    delta_t = np.insert(delta_t, 0, 0)
    df['delta_t'] = delta_t
    # Time[ms]	Realtime Tick
    return df


def is_measure_valid(voltage_measure, velocity_measure):
    """
    Function to check if all values in the 'DXL_Input_Voltage' column are within the specified range
    """
    voltage_threshold = (9.5, 16.0)
    velocity_threshold = (-25, 25) # rad/s
    if (voltage_threshold[0] <= voltage_measure <= voltage_threshold[1]) and (velocity_threshold[0] <= velocity_measure <= velocity_threshold[1]) :
        return True
    else :
        return False


def extract_dxl_model(directory_path):

    # Regex pattern to match motor name and time step
    pattern = r'_(M\d+|Msolo)(?:.*?(\d+)ms)?.*\.txt$'

    # List to store extracted information
    extracted_info = {}
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            # Match the pattern in the filename
            match = re.search(pattern, filename)
            if match:
                # Extract the motor type (e.g., M1, M2, M3, M4, Msolo)
                m_type = match.group(1)
                # Extract the ms value if present
                ms_value = match.group(2) if match.group(2) else '40'
                extracted_info[filename] = {"motor_name" : m_type, "rate" : ms_value}

    return extracted_info


def replace_outliers(df, voltage_threshold = (9.5, 16.0)):
    errors = 0
    for i in range(1, len(df)):
        if not is_measure_valid(df.loc[i, 'DXL_Input_Voltage'], df.loc[i, 'DXL_Velocity']):
            # Replace all columns except 'DXL_Position' with the previous value
            for column in ['DXL_PWM', 'DXL_Current', 'DXL_Velocity', 'DXL_Input_Voltage']:
                df.loc[i, column] = df.loc[i-1, column]

            # Special handling for 'DXL_Position'
            prev_pos = df.loc[i-1, 'DXL_Position']
            # Find the next valid 'DXL_Position' value
            for j in range(i+1, len(df)):
                if voltage_threshold[0] <= df.loc[j, 'DXL_Input_Voltage'] <= voltage_threshold[1]:
                    next_pos = df.loc[j, 'DXL_Position']
                    break
            else:
                # If no next valid value found, use the previous value
                next_pos = prev_pos

            # Calculate the single time step progression
            single_step_progression = (next_pos - prev_pos) / (j - (i-1))
            df.loc[i, 'DXL_Position'] = prev_pos + single_step_progression
            errors += 1
    
    # print(f"Number of measure errors : %d", errors)
    return df



# Define the low-pass filter
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y



def compute_physical(df, parameters, external_inertia=0):
    k_e, k_t, R, q_dot_s, tau_c, tau_s, c_v, motor_inertia = parameters
    parameters_to_find = k_t, R, q_dot_s, tau_c, tau_s, c_v, motor_inertia
    output_torque_motor_with_I(df, parameters_to_find)
    output_torque_motor_with_U(df, parameters)
    
    order = 2  # the order of the filter (higher order = sharper cutoff)
    fs = 1.0  # sample rate, Hz
    cutoff = 0.008  # desired cutoff frequency of the filter, Hz

    df['DXL_Current_filtered'] = butter_lowpass_filter(df['DXL_Current'], cutoff, fs, order)
    df['DXL_Velocity_filtered'] = butter_lowpass_filter(df['DXL_Velocity'], cutoff, fs, order)
    df['U_filtered'] = butter_lowpass_filter(df['U'], cutoff, fs, order)

    # Calculate Δx (change in position)
    delta_x = np.diff(df['DXL_Position'])
    delta_x = np.insert(delta_x, 0, 0)
    df['delta_x'] = delta_x

    delta_t = np.diff(df['t'])
    delta_t = np.insert(delta_t, 0, 0)
    df['delta_t'] = delta_t

    # Calculate velocity (Δx/Δt)
    df['Velocity_from_position'] = delta_x / delta_t
    df['Velocity_from_position_filtered'] = butter_lowpass_filter(df['Velocity_from_position'], cutoff, fs, order)

    # Acceleration computation
    delta_v = np.diff(df['Velocity_from_position'])
    delta_v = np.insert(delta_v, 0, 0)
    df['Acceleration_from_Velocity_from_position'] = delta_v / delta_t
    delta_v = np.diff(df['Velocity_from_position_filtered'])
    delta_v = np.insert(delta_v, 0, 0)
    df['Acceleration_from_Velocity_from_position_filtered'] = delta_v / delta_t
    delta_v = np.diff(df['DXL_Velocity'])
    delta_v = np.insert(delta_v, 0, 0)
    df['Acceleration_from_DXL_Velocity'] = delta_v / delta_t
    delta_v = np.diff(df['DXL_Velocity_filtered'])
    delta_v = np.insert(delta_v, 0, 0)
    df['Acceleration_from_DXL_Velocity_filtered'] = delta_v / delta_t

    friction_torque(df, parameters_to_find, source='Velocity_from_position')
    df['tau_f_from_position'] = df['tau_f']
    friction_torque(df, parameters_to_find)
    
    df['tau_f_from_inertia'] = df['tau_U'] - (motor_inertia + external_inertia) * df['Acceleration_from_Velocity_from_position']



def split_data(df):
    # Assuming df is your DataFrame and 't' contains time in seconds
    # Ensure 't' is sorted if not already
    df.sort_values('t', inplace=True)

    # Calculate relative time since the start
    start_time = df['t'].iloc[0]
    df['time_since_start'] = df['t'] - start_time

    # Determine the split points, starting with a 60-second interval, followed by 70-second intervals
    first_interval = 60
    subsequent_interval = 70
    split_points = [0, first_interval] + [(first_interval + i*subsequent_interval) for i in range(1, int((df['time_since_start'].iloc[-1] - first_interval) // subsequent_interval) + 1)]

    # Split the DataFrame
    list_of_dfs = []
    for i in range(len(split_points)-1):
        # Find the rows where 'time_since_start' falls into the current interval
        mask = (df['time_since_start'] >= split_points[i]) & (df['time_since_start'] < split_points[i+1])
        segment_df = df.loc[mask].copy()
        if not segment_df.empty:
            list_of_dfs.append(segment_df)

    # Handle the last segment if there is any remainder
    if df['time_since_start'].iloc[-1] >= split_points[-1]:
        mask = df['time_since_start'] >= split_points[-1]
        segment_df = df.loc[mask].copy()
        if not segment_df.empty:
            list_of_dfs.append(segment_df)

    # At this point, list_of_dfs contains your DataFrame split at the first 60 seconds and then every 70 seconds
    # You may choose to drop the 'time_since_start' column from each segment if not needed
    for df_segment in list_of_dfs:
        df_segment.drop(columns=['time_since_start'], inplace=True)
    return list_of_dfs


def process_df_for_split(df):
    # Calculate the index to split the dataframe into two halves
    split_index = len(df) // 2
    
    # Split the dataframe
    first_half = df.iloc[:split_index]
    second_half = df.iloc[split_index:]
    
    # Filter the second half
    filtered_second_half = second_half[second_half['Present PWM'] != 0]
    
    # Concatenate the two halves
    result_df = pd.concat([first_half, filtered_second_half]).reset_index(drop=True)
    
    return result_df

def split_experiments(df):
    """
    Function to split the experiments done on the Dynamixel XH430-W350 using the Wizard 2.0
    The experiments are of the type : 0 during x1[s], U1 during x2[s], 0 during x3[s], -U1 during x4[s], 0 during x5[s], etc...
    """
    experiments = []
    experiment_start = None
    for i, row in df.iterrows():
        # Check if we're at the start of an experiment (U transitions from 0 to non-zero)
        if row['U'] != 0 and (i == 0 or df.iloc[i-1]['U'] == 0):
            # Determine the start index, including 10 samples before, if possible
            start_index = max(i - 10, 0)
            if experiment_start is not None:
                # End the previous experiment here, but don't include the current row
                experiments.append(df.iloc[experiment_start:i].reset_index(drop=True))
            experiment_start = start_index
        elif row['U'] == 0 and experiment_start is not None and (i == len(df) - 1 or df.iloc[i+1]['U'] != 0):
            # We're at the end of an experiment, include up to the current row
            experiments.append(df.iloc[experiment_start:i+1].reset_index(drop=True))
            experiment_start = None

    # If the last row is part of an ongoing experiment, include it
    if experiment_start is not None and experiment_start < len(df):
        experiments.append(df.iloc[experiment_start:].reset_index(drop=True))

    final_experiments = []
    for experiment in experiments:
        final_experiments.append(process_df_for_split(experiment))
    return final_experiments
