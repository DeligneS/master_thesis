import numpy as np

def friction_torque(df, parameters_to_find, source='DXL_Velocity'):
    """
    Calculate the friction torque based on servo velocity and parameters.

    :param parameters: Model parameters relevant to friction.
    :param q_dot: Current velocity of the servo at the DC motor (rad/s).
    :return: Friction torque.
    """
    k_tau, R, q_dot_s, tau_c, tau_s, c_v, motor_inertia = parameters_to_find

    ß = np.exp(-abs(df[source] / q_dot_s))  # assume ∂ = 1
    df['tau_f'] = np.sign(df[source]) * ((1 - ß) * tau_c + ß * tau_s) + c_v * df[source]


def output_torque_motor_with_U(df, parameters_to_find):
    """
    Calculate the output torque from the motor based on input voltage and rotational speed.

    :param parameters: List of model parameters.
    :param U: Input voltage to the motor.
    :param rot_speed: Rotational speed of the motor (rad/s).
    :return: Output torque from the motor.
    """
    k_e, k_t, R, q_dot_s, tau_c, tau_s, c_v, motor_inertia = parameters_to_find
    df['tau_U'] = (k_t / R) * (df['U'] - k_e * df['DXL_Velocity'])
    return df['tau_U']


def output_torque_motor_with_I(df, parameters_to_find):
    """
    Calculate the output torque from the motor based on input current.

    :param parameters: List of model parameters.
    :param I: Input current to the motor.
    :return: Output torque from the motor.
    """
    k_t, R, q_dot_s, tau_c, tau_s, c_v, motor_inertia = parameters_to_find

    df['tau_I'] = df['DXL_Current'] * k_t
    return df['tau_I']


def output_torque_motor_from_datasheet(df):
    """
    Calculate the output torque from the motor based on input current.

    :param parameters: List of model parameters.
    :param I: Input current to the motor.
    :return: Output torque from the motor.
    """
    tau_1, I_1 = 0.06, 0.084
    tau_2, I_2 = 2.82, 1.12

    # Calculating the slope (m)
    m = (tau_2 - tau_1) / (I_2 - I_1)

    # Calculating the y-intercept (b)
    b = tau_1 - m * I_1

    df['tau_datasheet'] = m*df['DXL_Current'] + b

    return df['tau_datasheet']