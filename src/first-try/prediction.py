from src.modelisation import output_torque_motor_with_I, friction_torque, output_torque_motor_with_U, output_torque_motor_from_datasheet
from src.schwarz_model import schwarz_model
import numpy as np

def model_prediction(df, parameters_to_find, parameters_fixed, delta_t, with_current=False, model_identification = False, using_datasheet_param = False):
    """
    Predict the servo motor output based on the given parameters and inputs.
    :param df: DataFrame containing the input data.
    :param parameters_to_find: List of model parameters [k_tau, R, q_dot_s, tau_c, tau_s, c_v, k_gear].
    :param parameters_fixed: Fixed parameters for the model.
    :param delta_t: Time difference between measurements.
    :param with_current: Boolean indicating whether to use current or voltage for torque calculation.
    :return: DataFrame with predicted velocity and position.
    """
    _, _, _, _, _, _, motor_inertia = parameters_to_find
    k_gear, external_inertia = parameters_fixed

    # Motor and Friction Model
    df['rot_speed'] = df['DXL_Velocity'] * k_gear

    output_torque_motor_with_I(df, parameters_to_find)
    output_torque_motor_with_U(df, parameters_to_find)
    output_torque_motor_from_datasheet(df)

    if using_datasheet_param :
        df['tau'] = df['tau_datasheet']
    if model_identification :
        weight_U = 0.5  # Example weight for voltage-based torque
        weight_I = 0.5  # Example weight for current-based torque
        df['tau'] = weight_U * df['tau_U'] + weight_I * df['tau_I']
    else :
        if with_current:
            df['tau'] = output_torque_motor_with_I(df, parameters_to_find)
        else:
            df['tau'] = output_torque_motor_with_U(df, parameters_to_find)
    
    friction_torque(df, parameters_to_find)

    # df['tau_o'] = df['tau']
    # Calculate the output torque, which should be zero if the friction torque (tau_f) is higher than the motor torque (tau)
    # df['tau_o'] = np.where(np.sign(df['tau']) == np.sign(df['tau_f']), 
    #                         np.where(np.abs(df['tau_f']) < np.abs(df['tau']), 
    #                                 df['tau'] - df['tau_f'], 
    #                                 0),
    #                         df['tau'] - df['tau_f']
    #                         )
    df['tau_o'] =  np.where(np.abs(df['tau_f']) < np.abs(df['tau']), 
                                    df['tau'] - df['tau_f'], 
                                    0)

    # Predicted acceleration, considering the inertia of the motor and potential external one
    df['a'] = df['tau_o'] / (motor_inertia + external_inertia)

    # We assume that during Δt the inputs of the motor are constant
    df['q_dot_pred'] = df['DXL_Velocity'] + df['a'] * delta_t  # Predicted velocity
    # df['q_pred'] = df['DXL_Position'] + df['DXL_Velocity'] * delta_t + 0.5 * df['a'] * (delta_t ** 2)  # Predicted position
    df['q_pred'] = df['DXL_Position'] + df['DXL_Velocity'] * delta_t   # Predicted position


def schwarz_model_prediction(df, parameters_fixed, delta_t):
    """
    Predict the servo motor output based on the given parameters and inputs.
    :param df: DataFrame containing the input data.
    :param parameters_to_find: List of model parameters [k_tau, R, q_dot_s, tau_c, tau_s, c_v, k_gear].
    :param parameters_fixed: Fixed parameters for the model.
    :param delta_t: Time difference between measurements.
    :param with_current: Boolean indicating whether to use current or voltage for torque calculation.
    :return: DataFrame with predicted velocity and position.
    """
    motor_inertia = 0.00353 # on récupère l'inertie de Dynaban
    k_gear, external_inertia = parameters_fixed

    # Motor and Friction Model
    df['rot_speed'] = df['DXL_Velocity'] * k_gear
    df['U'] = df['DXL_PWM'] * df['DXL_Input_Voltage']

    schwarz_model(df)

    # Predicted acceleration, considering the inertia of the motor and potential external one
    df['a'] = df['tau_o'] / (motor_inertia + external_inertia)

    # We assume that during Δt the inputs of the motor are constant
    df['q_dot_pred'] = df['DXL_Velocity'] + df['a'] * delta_t  # Predicted velocity
    df['q_pred'] = df['DXL_Position'] + df['DXL_Velocity'] * delta_t + 0.5 * df['a'] * (delta_t ** 2)  # Predicted position


## This prediction is inspired from Dynaban, but to use it we need a constant input (voltage/current)
def new_model_prediction(df, parameters_to_find, parameters_fixed, delta_t, with_current=False, model_identification = False, using_datasheet_param = False):
    """
    Predict the servo motor output based on the given parameters and inputs.
    :param df: DataFrame containing the input data.
    :param parameters_to_find: List of model parameters [k_tau, R, q_dot_s, tau_c, tau_s, c_v, k_gear].
    :param parameters_fixed: Fixed parameters for the model.
    :param delta_t: Time difference between measurements.
    :param with_current: Boolean indicating whether to use current or voltage for torque calculation.
    :return: DataFrame with predicted velocity and position.
    """
    _, _, _, _, _, _, motor_inertia = parameters_to_find
    k_gear, external_inertia = parameters_fixed
    df['rot_speed'] = df['DXL_Velocity'] * k_gear

    #Cas 1 : prediction sur tension, validation sur position
    df['tau'] = output_torque_motor_with_U(df, parameters_to_find)
    friction_torque(df, parameters_to_find)
    # df['tau_o'] = df['tau'] - df['tau_f']
    # Calculate the output torque, which should be zero if the friction torque (tau_f) is higher than the motor torque (tau)
    df['tau_o'] = np.where(np.sign(df['tau']) == np.sign(df['tau_f']), 
                            np.where(np.abs(df['tau_f']) < np.abs(df['tau']), 
                                    df['tau'] - df['tau_f'], 
                                    0),
                            df['tau'] - df['tau_f']
                            )
    
    df['a'] = df['tau_o'] / (motor_inertia + external_inertia)

    # Initialize predicted velocity and position with initial values
    initial_velocity = df['DXL_Velocity'].iloc[0]
    initial_position = df['DXL_Position'].iloc[0]

    df['q_dot_pred'] = 0
    df['q_pred'] = 0

    df['q_dot_pred'].iloc[0] = initial_velocity
    df['q_pred'].iloc[0] = initial_position

    # Iteratively update predicted velocity and position
    for i in range(1, len(df)):
        df.loc[i, 'q_dot_pred'] = df.loc[i - 1, 'q_dot_pred'] + df.loc[i - 1, 'a'] * delta_t  # Predicted velocity at t + 1
        df.loc[i, 'q_pred'] = df.loc[i - 1, 'q_pred'] + df.loc[i - 1, 'q_dot_pred'] * delta_t  # Predicted position at t + 1




def model_prediction_dynaban(df, motor_inertia, external_inertia):
    """
    Predict the servo motor output based on the given parameters and inputs.
    :param df: DataFrame containing the input data.
    :param parameters_to_find: List of model parameters [k_tau, R, q_dot_s, tau_c, tau_s, c_v, k_gear].
    :param parameters_fixed: Fixed parameters for the model.
    :param delta_t: Time difference between measurements.
    :param with_current: Boolean indicating whether to use current or voltage for torque calculation.
    :return: DataFrame with predicted velocity and position.
    """

    #Cas 1 : prediction sur tension, validation sur position
    # df['tau_o'] = df['tau'] - df['tau_f']
    # Calculate the output torque, which should be zero if the friction torque (tau_f) is higher than the motor torque (tau)
    # df['tau_o'] = np.where(np.sign(df['tau']) == np.sign(df['tau_f']), 
    #                         np.where(np.abs(df['tau_f']) < np.abs(df['tau']), 
    #                                 df['tau'] - df['tau_f'], 
    #                                 0),
    #                         0#df['tau'] - df['tau_f']
    #
    #                          )
    # if df['tau']*df['DXL_Velocity']>0:
    #     if df['DXL_Velocity'] == 0:
    #         if np.abs(df['tau_f']) > np.abs(df['tau']):
    #             df['tau_o'] = df['tau'] - df['tau_f']
    #         else:
    #             df['tau_o'] = 0
    #     else :
    #         if np.abs(df['tau_f']) > np.abs(df['tau']):
    #             df['tau_o'] = df['tau'] - df['tau_f']
    #         else:
    #             df['tau_o'] = 0
    # else:
    #     if df['DXL_Velocity'] == 0:
    #         if np.abs(df['tau_f']) > np.abs(df['tau']):
    #             df['tau_o'] = df['tau'] - df['tau_f']
    #         else:
    #             df['tau_o'] = 0
    #     else :
    #         if np.abs(df['tau_f']) > np.abs(df['tau']):
    #             df['tau_o'] = df['tau'] + df['tau_f']
    #         else:
    #             df['tau_o'] = 0
    df['tau_o'] = np.where(
        df['DXL_Velocity'] == 0,
        np.where(
            np.abs(df['tau_f']) >= np.abs(df['tau']),
            0,
            df['tau'] - np.sign(df['tau']) * df['tau_f']
        ),
        np.where(
            (df['tau'] * df['DXL_Velocity'] > 0) & (np.abs(df['tau_f']) < np.abs(df['tau'])),
            df['tau'] - np.sign(df['tau']) * df['tau_f'],
            0
        )
    )

    df['a'] = df['tau_o'] / (motor_inertia + external_inertia)
    
    # df['a'] = df['Acceleration_from_Velocity_from_position']
    # df['a'] = df['Acceleration_from_Velocity_from_position_filtered']
    # df['a'] = df['Acceleration_from_DXL_Velocity']
    # df['a'] = df['Acceleration_from_DXL_Velocity_filtered']

    # Initialize predicted velocity and position with initial values as float64 to avoid dtype warnings
    df['q_dot_pred'] = 0.0  # Initializes the column as float64 due to the floating-point zero
    df['q_pred'] = 0.0  # Same here

    # Correctly set initial values for 'q_dot_pred' and 'q_pred'
    df.loc[0, 'q_dot_pred'] = float(df['DXL_Velocity'].iloc[0])
    df.loc[0, 'q_pred'] = float(df['DXL_Position'].iloc[0])

    # Iteratively update predicted velocity and position
    # for i in range(1, len(df)):
    for i in range(1, 100):
        df.loc[i, 'q_dot_pred'] = df.loc[i - 1, 'q_dot_pred'] + df.loc[i - 1, 'a'] * df.loc[i - 1, 'delta_t']  # Predicted velocity at t + 1
        df.loc[i, 'q_pred'] = df.loc[i - 1, 'q_pred'] + df.loc[i - 1, 'q_dot_pred'] * df.loc[i - 1, 'delta_t']  # Predicted position at t + 1



def acc_prediction_dynaban(df, motor_inertia, external_inertia):
    """
    Predict the servo motor output based on the given parameters and inputs.
    :param df: DataFrame containing the input data.
    :param parameters_to_find: List of model parameters [k_tau, R, q_dot_s, tau_c, tau_s, c_v, k_gear].
    :param parameters_fixed: Fixed parameters for the model.
    :param delta_t: Time difference between measurements.
    :param with_current: Boolean indicating whether to use current or voltage for torque calculation.
    :return: DataFrame with predicted velocity and position.
    """

    df['tau_o'] = np.where(
        df['DXL_Velocity'] == 0,
        np.where(
            np.abs(df['tau_f']) >= np.abs(df['tau']),
            0,
            df['tau'] - np.sign(df['tau']) * df['tau_f']
        ),
        np.where(
            (df['tau'] * df['DXL_Velocity'] > 0) & (np.abs(df['tau_f']) < np.abs(df['tau'])),
            df['tau'] - np.sign(df['tau']) * df['tau_f'],
            0
        )
    )

    df['a'] = df['tau_o'] / (motor_inertia + external_inertia)
    