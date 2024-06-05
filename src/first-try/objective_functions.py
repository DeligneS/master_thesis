import os
from src.prediction import model_prediction, new_model_prediction, model_prediction_dynaban, acc_prediction_dynaban
from src.data_processing import extract_dxl_model, process_file
from src.modelisation import friction_torque, output_torque_motor_with_U, output_torque_motor_with_I
import pandas as pd
import numpy as np

def model_error(df, parameters_to_find, parameters_fixed, delta_t, with_current = False, using_datasheet_param = False, verbose = False):
    """
    Compute the error of the model based on the given parameters,
    considering both position and speed.

    :param df: DataFrame with experimental data.
    :param parameters_to_find: Array of parameters for the model.
    :param parameters_fixed: Fixed parameters for the model.
    :param delta_t: Time difference between measurements.
    :param with_current: If True, compute the torque using the current model, else using voltage model
    :return: The combined error of the model.
    """

    # Check non-feasible values
    k_tau, R, q_dot_s, tau_c, tau_s, c_v, motor_inertia = parameters_to_find
    if tau_c > tau_s :
        return 1e100

    # Prediction using the model
    # new_model_prediction(df, parameters_to_find, parameters_fixed, delta_t, with_current = with_current, using_datasheet_param=using_datasheet_param)
    model_prediction(df, parameters_to_find, parameters_fixed, delta_t, with_current = with_current, using_datasheet_param=using_datasheet_param)

    # Compute the objective functions values
    error_position = objective_function_position(df)
    error_speed = objective_function_velocity(df)
    divergence_torques = objective_function_torques(df)
    objective_friction = objective_function_friction_torque(df)


    # Sum the errors
    weigth_position = 0.5
    total_error = error_position * (weigth_position) + error_speed * (1 - weigth_position)


    # Weighted Penalty for Torque Divergence
    divergence_penalty_weight = 1  # Adjust as needed
    torque_divergence_penalty = divergence_penalty_weight * divergence_torques

    # Penalize solutions where friction torque is always higher than other torques.
    friction_penalty_weight = 1000  # Adjust as needed
    excessive_friction_penalty = objective_friction * friction_penalty_weight

    # Total Objective
    total_objective = total_error + torque_divergence_penalty + excessive_friction_penalty

    if verbose :
        print(total_error)
        print(torque_divergence_penalty)
        print(excessive_friction_penalty)

    return total_objective


def model_error_on_folder(folder_path, parameters_to_find, dxl_model, with_current = False, using_datasheet_param = False):
    """
    Compute the error of the model based on the given parameters,
    considering both position and speed.

    :param df: DataFrame with experimental data.
    :param parameters_to_find: Array of parameters for the model.
    :param parameters_fixed: Fixed parameters for the model.
    :param delta_t: Time difference between measurements.
    :param with_current: If True, compute the torque using the current model, else using voltage model
    :return: The combined error of the model.
    """
    params = extract_dxl_model(folder_path)
    total_error = 0
    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            full_path = os.path.join(folder_path, file_name)
            df = process_file(full_path)
            motor_name = params[file_name]["motor_name"]
            if (motor_name in dxl_model) and  (df['DXL_Input_Voltage'].between(9.5, 16.0).all()): # else the collected data has errors
                dt = 4e-3*float(params[file_name]["rate"])
                if (motor_name == "M1") or (motor_name == "M2"):
                    parameters_fixed = 353.5, 0.0188598 # for Dynamixel XH430-W350 on the hips -> M1 & M2

                elif (motor_name == "M3") or (motor_name == "M4"):
                    parameters_fixed = 212.6, 0.00235245 # for Dynamixel XH430-W210 on the knees -> M3 & M4

                elif (motor_name == "Msolo"):
                    parameters_fixed = 353.5, 0         # for Dynamixel XH430-W350 without external inertia -> Msolo

                # total_error += model_error(df, parameters_to_find, parameters_fixed, dt, with_current=with_current, using_datasheet_param = using_datasheet_param)
                df = df.iloc[200:1200] # Need to be in steady state
                total_error += objective_function_lite(df, parameters_to_find, parameters_fixed)

    return total_error


def objective_function_velocity(df):
    # Shift the DataFrame to align next velocity
    df['next_q_dot'] = df['DXL_Velocity'].shift(-1)

    # Compute the squared error for position and speed
    error_speed = ((df['q_dot_pred'] - df['next_q_dot'])**2).mean()

    return error_speed


def objective_function_position(df):
    # Shift the DataFrame to align next position
    df['next_q'] = df['DXL_Position'].shift(-1)

    # Compute the squared error for position and speed
    error_position = ((df['q_pred'] - df['next_q']) ** 2).mean()

    return error_position


def objective_function_torques(df):
    # Compute the error on the torques (computed torques through I, U or datasheet should be relatively close)
    torques = df[['tau_I', 'tau_U', 'tau_datasheet']]

    # Calculating standard deviation for each row
    divergence = torques.std(axis=1).mean()

    return divergence


def objective_function_friction_torque(df):
    """ Compute a penalty for cases where friction torque exceeds other torque values. """
    # Identify instances where friction torque is higher than other torques
    # excessive_friction = (df['tau_f'].abs() > df['tau_I'].abs()) | \
    #                      (df['tau_f'].abs() > df['tau_U'].abs()) | \
    #                      (df['tau_f'].abs() > df['tau_datasheet'].abs()) & (df['DXL_Velocity'] != 0)
    excessive_friction = (df['tau_f'].abs() > df['tau_U'].abs()) & (df['DXL_Velocity'].abs() > 0.1)

    # Assign a penalty for such instances
    total_penalty = excessive_friction.sum()
    return total_penalty


def objective_function_on_folder(folder_path, parameters_to_find, dxl_model, with_current = False, using_datasheet_param = False):
    params = extract_dxl_model(folder_path)
    result = (0, 0, 0, 0)
    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            full_path = os.path.join(folder_path, file_name)
            df = process_file(full_path)
            motor_name = params[file_name]["motor_name"]
            if (motor_name in dxl_model) and  (df['DXL_Input_Voltage'].between(9.5, 16.0).all()): # else the collected data has errors
                dt = 4e-3*float(params[file_name]["rate"])
                if (motor_name == "M1") or (motor_name == "M2"):
                    parameters_fixed = 353.5, 0.0188598 # for Dynamixel XH430-W350 on the hips -> M1 & M2

                elif (motor_name == "M3") or (motor_name == "M4"):
                    parameters_fixed = 212.6, 0.00235245 # for Dynamixel XH430-W210 on the knees -> M3 & M4

                elif (motor_name == "Msolo"):
                    parameters_fixed = 353.5, 0         # for Dynamixel XH430-W350 without external inertia -> Msolo

                result = tuple(x + y for x, y in zip(result, multi_objective(df, parameters_to_find, parameters_fixed, dt, with_current=with_current, using_datasheet_param = using_datasheet_param)))
    return result


def multi_objective(df, parameters_to_find, parameters_fixed, delta_t, with_current = False, using_datasheet_param = False, verbose = False):
    # Check non-feasible values
    k_tau, R, q_dot_s, tau_c, tau_s, c_v, motor_inertia = parameters_to_find
    if tau_c > tau_s :
        return (1e100, 1e100, 1e100, 1e100)

    # Prediction using the model
    # new_model_prediction(df, parameters_to_find, parameters_fixed, delta_t, with_current = with_current, using_datasheet_param=using_datasheet_param)
    model_prediction(df, parameters_to_find, parameters_fixed, delta_t, with_current = with_current, using_datasheet_param=using_datasheet_param)

    # Compute the objective functions values
    error_position = objective_function_position(df)
    error_speed = objective_function_velocity(df)
    divergence_torques = objective_function_torques(df)
    objective_friction = objective_function_friction_torque(df)

    if verbose :
        print(error_position)
        print(error_speed)
        print(divergence_torques)
        print(objective_friction)

    return (error_position,
            error_speed,
            divergence_torques,
            objective_friction)


def electromagnetic_torque(df, parameters_to_find):
    k_tau, R, q_dot_s, tau_c, tau_s, c_v, motor_inertia = parameters_to_find
    # df['tau_em'] = (df['U'] - R * df['DXL_Current']) * df['DXL_Current'] / df['omega_m']
    df['tau_em'] = ((df['U'] - R * df['DXL_Current']) * df['DXL_Current'] / df['DXL_Velocity']).fillna(0)
    df['tau_em'] = df['tau_em'].mask(df['omega_m'] == 0, 0)
    # print(df['tau_em'])


def objective_function_lite(df, parameters_to_find, parameters_fixed):
    
    k_gear, external_inertia = parameters_fixed
    k_tau, R, q_dot_s, tau_c, tau_s, c_v, motor_inertia = parameters_to_find
    if tau_c > tau_s :
        return 1e100
    # Motor and Friction Model
    df['omega_m'] = df['DXL_Velocity'] * k_gear
    electromagnetic_torque(df, parameters_to_find)
    friction_torque(df, parameters_to_find)

    # At constant speed (+ no transient state), the EM torque is exactly equal to the friction torque
    error = ((df['tau_em'] - df['tau_f']) ** 2).sum()

    # print(error)
    return error


def model_error_on_list(dfs, parameters_to_find, parameters_fixed, external_inertia):
    errors = 0
    for df in dfs :
        q_dot_s, tau_c, tau_s, c_v, motor_inertia = parameters_to_find
        ke, kt, Ra = parameters_fixed
        parameters_friction = kt, 0, q_dot_s, tau_c, tau_s, c_v, motor_inertia
        parameters_U = ke, kt, Ra, q_dot_s, tau_c, tau_s, c_v, motor_inertia
        friction_torque(df, parameters_friction)
        output_torque_motor_with_U(df, parameters_U)
        output_torque_motor_with_I(df, parameters_friction)
        df['tau'] = df['tau_I']
        model_prediction_dynaban(df, motor_inertia, external_inertia)
        errors += ((df['q_pred'] - df['DXL_Position']) ** 2).sum()

    return errors


def model_error_estimation(dfs, parameters_to_find, parameters_fixed, external_inertia):
    errors = 0
    for df in dfs :
        q_dot_s, tau_c, tau_s, c_v, motor_inertia = parameters_to_find
        if tau_c > tau_s:
            return 1e9
        ke, kt, Ra = parameters_fixed
        parameters_friction = kt, 0, q_dot_s, tau_c, tau_s, c_v, motor_inertia
        parameters_U = ke, kt, Ra, q_dot_s, tau_c, tau_s, c_v, motor_inertia

        friction_torque(df, parameters_friction)
        output_torque_motor_with_U(df, parameters_U)
        # Estimation v_final
        q_i = df['DXL_Position'].iloc[0]
        q_f = df['DXL_Position'].iloc[-1]
        v_final = (q_f - q_i)/(df['t'].iloc[-1] - df['t'].iloc[0])

        output_torque_motor_with_I(df, parameters_friction)
        df['tau'] = df['tau_U']

        acc_prediction_dynaban(df, motor_inertia, external_inertia)

        # Step 1 : compute on transient
        errors += compute_error_on_transient(df[:100], v_final)

        df['tau'] = (kt / Ra) * (df['U'] - ke * v_final)

        # Step 2 : compute on steady-state
        if abs(v_final) > 0.5:
            errors += compute_error_on_steady_state(df[100:])

    return errors


def compute_error_on_transient(df, v_final):

    # Filter the dataframe for rows where acceleration is not zero
    acceleration_periods = df[df['a'] != 0]

    # Sum the 'delta_t' column of the filtered dataframe
    total_delta_t_acc = acceleration_periods['delta_t'].sum()

    v_initial = 0

    if total_delta_t_acc == 0 :
        return 1e10
    acc_real = (v_final - v_initial) / total_delta_t_acc

    error = ((df['a'] * df['delta_t']).sum() - acc_real) ** 2

    return error


def compute_error_on_steady_state(df):
    """
    Computes a custom error for a DataFrame where the goal is to make the 'tau' column
    as close as possible to 'tau_f', with a penalty if 'tau' exceeds 'tau_f'.
    
    Parameters:
    - df: DataFrame, must contain 'tau' and 'tau_f' columns.
    
    Returns:
    - error: float, the calculated error with penalties.
    """
    # Extracting columns
    tau = df['tau']
    tau_f = df['tau_f']
    
    # Base Error: Mean Absolute Error (MAE)
    base_error = np.mean(np.abs(tau - tau_f))
    
    # Penalty for tau > tau_f
    # penalty_factor = 10  # Adjust the penalty factor as needed
    # penalties = np.where(tau > tau_f, (tau - tau_f) * penalty_factor, 0)
    # penalty_error = np.mean(penalties)
    
    # Total Error
    total_error = base_error #+ penalty_error
    
    return total_error
